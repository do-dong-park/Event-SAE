"""Event-feature score matrix.

Joins per-event SAE top-k activations (from `event_sae.openvla.activations`
or `event_sae.openpi.activations` either online or via
`scripts/extract_topk.py`) with VLM-annotated event clusters, builds
event-centered temporal windows around each event, and projects three
time templates (pulse, step-up, step-down) onto the per-feature
trajectory. The per-feature score is the maximum positive projection
across templates; per-event scores are averaged within each
`(cluster, episode)` group and then across episodes to give one row per
cluster.

Three matrices are produced per call, mirroring openpi-mech's
`build_openpi_feature_score_matrix.py`:

  - ``matrix_raw``         — `max(pulse, step_up, step_down)` over the
                             3 templates (event_aligned score)
  - ``matrix_window_mean`` — mean activation over each event's window
                             (window_mean score)
  - ``matrix_task_mean``   — per-task mean activation over every cached
                             timestep (task_mean score)

The ``step_mapping`` argument controls how shard rows map to env
timesteps (mirrors openpi-mech):

  - ``action_executed`` (OpenPI AE default): one env step per row at
    ``chunk_start + token_idx`` when the token is executed.
  - ``chunk_executed`` (OpenPI PG default): broadcast each row to every
    executed env step of its chunk
    (``chunk_start..chunk_start+executed_chunk_len-1``).
  - ``inference_step`` (OpenVLA legacy / fallback): use ``step_in_episode``
    directly (no chunk semantics).

Output payload (single torch.save .pt):

  - `matrix_raw` / `matrix_window_mean` / `matrix_task_mean`
  - `matrix`               : alias of `matrix_raw` for backward compat
  - `row_keys`             : per-row cluster metadata
  - `row_results`          : per-row top-N feature summaries
  - `templates`            : the three time templates used
  - `selection_counts`     : join + filter accounting
  - `selected_events`      : per-event provenance after scoring
  - `step_mapping`         : the mapping used
  - `source`               : input paths + manifest summary
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from event_sae import resolve_groot_artifact_path
from event_sae import sha256_file as _sha256
from event_sae.events.io import load_jsonl


SPARSE_TOPK_FORMAT = "token_topk_sparse_v1"
SCORE_AUDIT_FORMAT = "groot_n15_pq3_stage4_score_audit_v3"


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Activation manifest must be a JSON object: {path}")
    if manifest.get("format") != SPARSE_TOPK_FORMAT:
        raise ValueError(
            f"Unsupported manifest format in {path}: "
            f"{manifest.get('format')!r}"
        )
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError(f"Activation manifest has no shard list: {path}")
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or not str(shard.get("path", "")).strip():
            raise ValueError(f"Invalid shard metadata at index {index} in {path}")
    return manifest


@dataclass(frozen=True)
class SparseTopKArtifact:
    """A manifest together with the directory context needed for its shards."""

    run_dir: Path
    manifest_path: Path
    manifest: dict

    def resolve_shard_path(self, shard_relpath: str | Path) -> Path:
        shard_path = Path(shard_relpath)
        if shard_path.is_absolute():
            if shard_path.is_file():
                return shard_path.resolve()
            raise FileNotFoundError(f"Activation shard not found: {shard_path}")

        search_roots = [self.manifest_path.parent, self.run_dir]
        nested_root = self.run_dir / "sae_activations"
        if nested_root.is_dir():
            search_roots.extend(
                path for path in sorted(nested_root.iterdir()) if path.is_dir()
            )

        seen: set[Path] = set()
        for root in search_roots:
            candidate = (root / shard_path).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Activation shard {str(shard_relpath)!r} referenced by "
            f"{self.manifest_path} was not found"
        )

    def iter_shards(
        self,
        *,
        desc: str | None = None,
    ) -> Iterator[tuple[dict, dict]]:
        """Yield ``(shard_metadata, payload)`` in manifest order."""
        shards = self.manifest["shards"]
        iterator = shards
        if tqdm is not None and desc:
            iterator = tqdm(shards, desc=desc, unit="shard")
        for shard_meta in iterator:
            shard_path = self.resolve_shard_path(shard_meta["path"])
            payload = torch.load(shard_path, map_location="cpu")
            yield shard_meta, payload


def open_sparse_topk_artifact(topk_run_dir: Path) -> SparseTopKArtifact:
    """Open a flat artifact or one nested under ``sae_activations/*``."""
    run_dir = resolve_groot_artifact_path(topk_run_dir).resolve()
    root_manifest_path = run_dir / "manifest.json"
    if root_manifest_path.is_file():
        return SparseTopKArtifact(
            run_dir=run_dir,
            manifest_path=root_manifest_path,
            manifest=_load_manifest(root_manifest_path),
        )

    nested_paths = sorted(
        (run_dir / "sae_activations").glob("*/manifest.json")
    )
    valid: list[tuple[Path, dict]] = []
    invalid: list[ValueError] = []
    for manifest_path in nested_paths:
        try:
            valid.append((manifest_path, _load_manifest(manifest_path)))
        except ValueError as error:
            invalid.append(error)
    if len(valid) > 1:
        paths = [str(path) for path, _ in valid]
        raise ValueError(
            "Multiple token_topk_sparse_v1 manifests found; pass the intended "
            f"activation subdirectory explicitly: {paths}"
        )
    if len(valid) == 1:
        manifest_path, manifest = valid[0]
        return SparseTopKArtifact(
            run_dir=run_dir,
            manifest_path=manifest_path,
            manifest=manifest,
        )
    if invalid:
        raise invalid[0]
    raise FileNotFoundError(
        f"No {SPARSE_TOPK_FORMAT} manifest under {run_dir}"
    )


# ---------------------------------------------------------------------------
# Templates + projections
# ---------------------------------------------------------------------------


def _normalize_template(template: torch.Tensor) -> torch.Tensor:
    template = template.to(dtype=torch.float32)
    template = template - torch.mean(template)
    norm = torch.linalg.vector_norm(template)
    if float(norm) == 0.0:
        raise ValueError("Template norm is zero after mean-centering.")
    return template / norm


def build_templates(window_size: int) -> dict[str, torch.Tensor]:
    """Build the three normalized time templates used for event scoring."""
    positions = torch.arange(-window_size, window_size + 1, dtype=torch.float32)
    pulse = 1.0 - torch.abs(positions) / float(window_size + 1)
    pulse = _normalize_template(pulse)
    step_up = torch.where(positions < 0, -torch.ones_like(positions), torch.ones_like(positions))
    step_up = _normalize_template(step_up)
    step_down = -step_up
    return {"pulse": pulse, "step_up": step_up, "step_down": step_down}


def _project_pattern_scores(
    centered_matrix: torch.Tensor,
    templates: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    centered_signal = centered_matrix - torch.mean(centered_matrix, dim=0, keepdim=True)
    return {
        name: torch.clamp(torch.matmul(centered_signal.transpose(0, 1), template), min=0.0)
        for name, template in templates.items()
    }


def _top_features(scores: torch.Tensor, top_n: int) -> tuple[list[int], list[float]]:
    k = min(top_n, int(scores.numel()))
    values, indices = torch.topk(scores, k=k, largest=True)
    return indices.tolist(), [float(x) for x in values.tolist()]


def _row_top_summary(scores: torch.Tensor, top_n: int) -> dict[str, list]:
    feature_ids, values = _top_features(scores, top_n)
    return {"feature_ids": feature_ids, "scores": values}


# ---------------------------------------------------------------------------
# Window placement + per-step action-token aggregation
# ---------------------------------------------------------------------------


def _fit_step_window(
    *,
    center_step: int,
    window_size: int,
    num_steps: int,
) -> tuple[list[int] | None, list[int], int | None]:
    """Shift a (2w+1)-step centered window to stay inside [0, num_steps)."""
    requested_steps = list(range(center_step - window_size, center_step + window_size + 1))
    if not requested_steps:
        return [], [], 0
    min_req, max_req = min(requested_steps), max(requested_steps)
    if (max_req - min_req) >= num_steps:
        return None, requested_steps, None
    shift = 0
    if max_req >= num_steps:
        shift -= max_req - (num_steps - 1)
    if min_req + shift < 0:
        shift += -(min_req + shift)
    return [s + shift for s in requested_steps], requested_steps, shift


def build_templates_at_event_idx(window_size: int, event_idx: int) -> dict[str, torch.Tensor]:
    """Boundary-aware templates: when the event window is shifted to stay
    inside ``[0, num_steps)``, the event position inside the window may not
    be the center any more. Templates are re-centered around
    ``event_idx`` so pulse / step-up / step-down stay anchored on the
    waypoint. Matches openpi-mech's ``_build_templates_at_event_idx``."""
    if event_idx < 0 or event_idx > 2 * window_size:
        raise ValueError(f"event_idx={event_idx} outside window length {2 * window_size + 1}.")
    positions = torch.arange(2 * window_size + 1, dtype=torch.float32) - float(event_idx)
    pulse = torch.clamp(1.0 - torch.abs(positions) / float(window_size + 1), min=0.0)
    pulse = _normalize_template(pulse)
    step_up = torch.where(positions < 0, -torch.ones_like(positions), torch.ones_like(positions))
    step_up = _normalize_template(step_up)
    step_down = -step_up
    return {"pulse": pulse, "step_up": step_up, "step_down": step_down}


def _effective_steps_for_row(
    *,
    step_mapping: str,
    step_in_episode: int,
    token_idx: int,
    chunk_start_step: int,
    executed_chunk_len: int,
) -> list[int]:
    """Translate a topk-shard row into the list of env steps it represents.

    Matches openpi-mech ``build_openpi_feature_score_matrix.py::_effective_steps``.
    ``chunk_start_step`` and ``executed_chunk_len`` may be ``-1`` sentinels
    on OpenVLA shards (no chunking); in that case only ``inference_step``
    mode is valid.
    """
    if step_mapping == "inference_step":
        return [step_in_episode] if step_in_episode >= 0 else []
    if step_mapping == "action_executed":
        if chunk_start_step < 0 or executed_chunk_len <= 0:
            return []
        if token_idx < 0 or token_idx >= executed_chunk_len:
            return []
        return [chunk_start_step + token_idx]
    if step_mapping == "chunk_executed":
        if chunk_start_step < 0 or executed_chunk_len <= 0:
            return []
        return [chunk_start_step + offset for offset in range(int(executed_chunk_len))]
    raise ValueError(f"Unsupported step_mapping={step_mapping!r}")


def resolve_activation_step_mapping(
    requested_mapping: str,
    capture_target: str | None,
) -> str:
    """Resolve ``auto`` to the row-to-environment-step mapping for a backend."""

    if requested_mapping != "auto":
        return requested_mapping
    if capture_target == "action_expert":
        return "action_executed"
    if capture_target == "paligemma":
        return "chunk_executed"
    return "inference_step"


# ---------------------------------------------------------------------------
# Cluster / event join
# ---------------------------------------------------------------------------


@dataclass
class _JoinResult:
    selected_events: list[dict]
    cluster_metadata_by_id: dict[str, dict]
    counts: dict[str, int]


def _merge_episode_task_ids(
    episode_to_task_id: dict[int, int],
    records: list[dict],
    *,
    source_name: str,
) -> None:
    """Merge episode ownership without allowing an existing task to change."""

    for record in records:
        episode_num = int(record["episode_num"])
        task_id = int(record["task_id"])
        previous_task_id = episode_to_task_id.get(episode_num)
        if previous_task_id is not None and previous_task_id != task_id:
            raise ValueError(
                f"episode_num={episode_num} maps to multiple task_id values "
                f"while merging {source_name}: {previous_task_id} and {task_id}"
            )
        episode_to_task_id[episode_num] = task_id


def join_cluster_events(
    *,
    event_features: list[dict],
    cluster_assignments: list[dict],
    cluster_annotations: list[dict],
) -> _JoinResult:
    """Join event_features + cluster_assignments + cluster_annotations, filter
    out clusters with API/parse errors or empty phrase/phase."""
    event_by_sample_id = {}
    episode_to_task_id: dict[int, int] = {}
    for record in event_features:
        sample_id = str(record["sample_id"])
        if sample_id in event_by_sample_id:
            raise ValueError(f"Duplicate sample_id in event_features.jsonl: {sample_id}")
        event_by_sample_id[sample_id] = record
        episode_num = int(record["episode_num"])
        task_id = int(record["task_id"])
        previous_task_id = episode_to_task_id.setdefault(episode_num, task_id)
        if previous_task_id != task_id:
            raise ValueError(
                f"episode_num={episode_num} maps to multiple task_id values: "
                f"{previous_task_id} and {task_id}"
            )

    counts = {
        "valid_clusters": 0,
        "joined_events": 0,
        "skipped_api_error": 0,
        "skipped_parse_error": 0,
        "skipped_empty_phrase": 0,
        "skipped_empty_phase": 0,
        "skipped_missing_cluster_annotation": 0,
        "skipped_missing_event_features": 0,
    }

    cluster_metadata_by_id: dict[str, dict] = {}
    annotation_task_by_cluster_id: dict[str, str] = {}
    annotation_cluster_ids: set[str] = set()
    for annotation in cluster_annotations:
        cluster_id = str(annotation["cluster_id"])
        if cluster_id in annotation_cluster_ids:
            raise ValueError(
                f"Duplicate cluster_id in cluster annotations: {cluster_id}"
            )
        annotation_cluster_ids.add(cluster_id)
        annotation_task_value = annotation.get("task_description")
        if annotation_task_value is not None:
            annotation_task = str(annotation_task_value)
            previous_task = annotation_task_by_cluster_id.setdefault(
                cluster_id,
                annotation_task,
            )
            if previous_task != annotation_task:
                raise ValueError(
                    f"cluster_id={cluster_id} has conflicting annotation tasks: "
                    f"{previous_task!r} and {annotation_task!r}"
                )
        if annotation.get("api_error") is not None:
            counts["skipped_api_error"] += 1
            continue
        if annotation.get("parse_error") is not None:
            counts["skipped_parse_error"] += 1
            continue
        phrase = str(annotation.get("phrase", "")).strip()
        if not phrase:
            counts["skipped_empty_phrase"] += 1
            continue
        phase = str(annotation.get("phase", "")).strip()
        if not phase:
            counts["skipped_empty_phase"] += 1
            continue
        annotation_task = str(annotation["task_description"])
        cluster_metadata_by_id[cluster_id] = {
            "cluster_id": cluster_id,
            "task_description": annotation_task,
            "phrase": phrase,
            "phase": phase,
            "episode_coverage": float(annotation.get("episode_coverage", 0.0)),
            "model": str(annotation.get("model", "")),
            "prompt_version": str(annotation.get("prompt_version", "")),
            "representative_sample_ids": list(annotation.get("representative_sample_ids", [])),
            "representative_clip_paths": list(annotation.get("representative_clip_paths", [])),
            "representative_frame_paths": list(annotation.get("representative_frame_paths", [])),
            "representative_progress_percents": list(annotation.get("representative_progress_percents", [])),
            "review_mode": annotation.get("review_mode"),
            "review_verdict": annotation.get("review_verdict"),
            "actual_human_review_completed": annotation.get("actual_human_review_completed"),
        }
    counts["valid_clusters"] = len(cluster_metadata_by_id)

    assignment_sample_ids: set[str] = set()
    cluster_to_task_id: dict[str, int] = {}
    for assignment in cluster_assignments:
        sample_id = str(assignment["sample_id"])
        if sample_id in assignment_sample_ids:
            raise ValueError(
                f"Duplicate sample_id in cluster assignments: {sample_id}"
            )
        assignment_sample_ids.add(sample_id)

        event = event_by_sample_id.get(sample_id)
        if event is None:
            continue
        cluster_id = str(assignment["cluster_id"])
        event_task = str(event["task_description"])
        assignment_task = str(assignment["task_description"])
        annotation_task = annotation_task_by_cluster_id.get(cluster_id)
        if event_task != assignment_task or (
            annotation_task is not None and event_task != annotation_task
        ):
            raise ValueError(
                f"Task description mismatch for sample_id={sample_id}: "
                f"event={event_task!r}, assignment={assignment_task!r}, "
                f"annotation={annotation_task!r}"
            )

        task_id = int(event["task_id"])
        previous_task_id = cluster_to_task_id.setdefault(cluster_id, task_id)
        if previous_task_id != task_id:
            raise ValueError(
                f"cluster_id={cluster_id} maps to multiple task_id values: "
                f"{previous_task_id} and {task_id}"
            )

    joined_events: list[dict] = []
    for assignment in cluster_assignments:
        cluster_id = str(assignment["cluster_id"])
        cluster_meta = cluster_metadata_by_id.get(cluster_id)
        if cluster_meta is None:
            counts["skipped_missing_cluster_annotation"] += 1
            continue
        sample_id = str(assignment["sample_id"])
        event = event_by_sample_id.get(sample_id)
        if event is None:
            counts["skipped_missing_event_features"] += 1
            continue
        joined_events.append(
            {
                "sample_id": sample_id,
                "task_description": str(event["task_description"]),
                "task_id": int(event["task_id"]),
                "task_episode_idx": int(event["task_episode_idx"]),
                "episode_num": int(event["episode_num"]),
                "waypoint_rank": int(event["waypoint_rank"]),
                "waypoint_step": int(event["waypoint_step"]),
                "progress_percent": float(event["progress_percent"]),
                "num_steps": int(event["num_steps"]),
                "cluster_id": cluster_id,
                "phrase": cluster_meta["phrase"],
                "phase": cluster_meta["phase"],
            }
        )
    counts["joined_events"] = len(joined_events)

    member_counts: dict[str, int] = defaultdict(int)
    episode_sets: dict[str, set[int]] = defaultdict(set)
    for event in joined_events:
        member_counts[event["cluster_id"]] += 1
        episode_sets[event["cluster_id"]].add(int(event["episode_num"]))
    for cluster_id, meta in cluster_metadata_by_id.items():
        meta["num_members"] = int(member_counts.get(cluster_id, 0))
        meta["num_episodes"] = int(len(episode_sets.get(cluster_id, set())))

    return _JoinResult(joined_events, cluster_metadata_by_id, counts)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def aggregate_sparse_activations_by_timestep(
    topk_run_dir: Path,
    *,
    step_mapping: str,
    episode_to_task_id: dict[int, int],
    task_id_set: set[int],
    dict_size: int,
) -> tuple[
    dict[tuple[int, int], torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, int],
    dict,
    dict[str, int],
]:
    """Walk topk shards and accumulate per-``(episode, env_step)`` dense
    vectors under ``step_mapping``. Mirrors openpi-mech's
    ``load_timestep_vectors`` reference implementation:

    * Filter by ``task_id`` (read EVERY episode in any selected task,
      not just episodes that produced clustered events). This matters
      because ``matrix_task_mean`` is the per-task mean over **all**
      rollout timesteps the cache covers, not only event-window ones.
    * Compute per-timestep means first, then aggregate task means as
      the mean of per-timestep means weighted equally by timestep — not
      by row count. Under ``chunk_executed`` a single row contributes to
      multiple timesteps; row-count weighting overcounts wide chunks.
    """
    artifact = open_sparse_topk_artifact(topk_run_dir)
    manifest = artifact.manifest
    counters = {
        "shards_loaded": 0,
        "rows_seen": 0,
        "rows_used": 0,
        "rows_skipped_unknown_task": 0,
        "rows_skipped_nonexecuted": 0,
        "rows_skipped_no_effective_step": 0,
    }
    timestep_sums: dict[tuple[int, int], torch.Tensor] = {}
    timestep_counts: dict[tuple[int, int], int] = defaultdict(int)
    timestep_task_ids: dict[tuple[int, int], int] = {}

    for _shard_meta, payload in artifact.iter_shards(
        desc="Loading topk shards"
    ):
        counters["shards_loaded"] += 1
        ep_arr = payload["episode_num"].to(dtype=torch.int64)
        step_arr = payload["step_in_episode"].to(dtype=torch.int64)
        tok_arr = payload["token_idx"].to(dtype=torch.int64)
        chunk_start_arr = payload.get("chunk_start_step")
        if chunk_start_arr is None:
            chunk_start_arr = torch.full_like(ep_arr, -1)
        else:
            chunk_start_arr = chunk_start_arr.to(dtype=torch.int64)
        exec_len_arr = payload.get("executed_chunk_len")
        if exec_len_arr is None:
            exec_len_arr = torch.full_like(ep_arr, -1)
        else:
            exec_len_arr = exec_len_arr.to(dtype=torch.int64)
        feat_ids = payload["top_feature_ids"].to(dtype=torch.int64)
        feat_vals = payload["top_feature_vals"].to(dtype=torch.float32)
        n_rows = int(ep_arr.shape[0])
        counters["rows_seen"] += n_rows

        for row_idx in range(n_rows):
            ep = int(ep_arr[row_idx])
            task_id = episode_to_task_id.get(ep)
            if task_id is None or task_id not in task_id_set:
                counters["rows_skipped_unknown_task"] += 1
                continue
            steps = _effective_steps_for_row(
                step_mapping=step_mapping,
                step_in_episode=int(step_arr[row_idx]),
                token_idx=int(tok_arr[row_idx]),
                chunk_start_step=int(chunk_start_arr[row_idx]),
                executed_chunk_len=int(exec_len_arr[row_idx]),
            )
            if not steps:
                if step_mapping == "action_executed":
                    counters["rows_skipped_nonexecuted"] += 1
                else:
                    counters["rows_skipped_no_effective_step"] += 1
                continue
            row_indices = feat_ids[row_idx]
            row_values = feat_vals[row_idx]
            for step in steps:
                if step < 0:
                    counters["rows_skipped_no_effective_step"] += 1
                    continue
                key = (ep, step)
                vec = timestep_sums.get(key)
                if vec is None:
                    vec = torch.zeros(dict_size, dtype=torch.float32)
                    timestep_sums[key] = vec
                    timestep_task_ids[key] = task_id
                vec.index_add_(0, row_indices, row_values)
                timestep_counts[key] += 1
                counters["rows_used"] += 1

    # Per-timestep mean.
    timestep_vectors: dict[tuple[int, int], torch.Tensor] = {}
    for key, vec_sum in timestep_sums.items():
        c = timestep_counts[key]
        if c > 0:
            timestep_vectors[key] = vec_sum / float(c)

    # Per-task mean of per-timestep vectors. Mirrors openpi-mech.
    task_sums: dict[int, torch.Tensor] = {}
    task_counts: dict[int, int] = defaultdict(int)
    for key, vec in timestep_vectors.items():
        tid = timestep_task_ids[key]
        if tid not in task_sums:
            task_sums[tid] = torch.zeros(dict_size, dtype=torch.float32)
        task_sums[tid] += vec
        task_counts[tid] += 1
    task_means: dict[int, torch.Tensor] = {}
    for tid, vec_sum in task_sums.items():
        c = task_counts[tid]
        if c > 0:
            task_means[tid] = vec_sum / float(c)

    return timestep_vectors, task_means, dict(task_counts), manifest, counters


def score_cluster_features(
    *,
    topk_run_dir: Path,
    event_features_path: Path,
    cluster_assignments_path: Path,
    cluster_annotations_path: Path,
    output_path: Path,
    window_size: int = 5,
    top_n: int = 20,
    step_mapping: str = "auto",
    event_step_scale: int = 1,
    prompt_records_path: Path | None = None,
) -> dict:
    """Build the event-feature score matrices and save a single `.pt`
    payload. Mirrors openpi-mech ``build_openpi_feature_score_matrix.py``.

    Three legacy matrices are produced (paper Table 4 / Fig 3):

      * ``matrix_raw``: max-over-templates projection (event_aligned)
      * ``matrix_window_mean``: mean activation over the event window
      * ``matrix_task_mean``: per-task mean activation across all
        timesteps the cache covers

    Directional matrices preserve the pulse, step-up, and step-down scores
    before their feature-wise maximum. At the episode-group level,
    ``episode_group_matrix_raw`` is exactly the maximum of those three
    matrices. At the row level, each directional matrix is the
    episode-balanced mean of its episode-group matrix, while legacy
    ``matrix_raw`` remains the episode-balanced mean of the group-level
    maxima. Consequently, ``matrix_raw`` and the maximum of the row-level
    directional means need not be equal: maximum and mean do not commute.

    ``step_mapping`` defaults to ``"auto"``: pick per ``manifest.capture_target``
    (``action_executed`` for ``action_expert``, ``chunk_executed`` for
    ``paligemma``, otherwise ``inference_step``).
    """
    topk_run_dir = resolve_groot_artifact_path(topk_run_dir).resolve()
    event_features_path = resolve_groot_artifact_path(
        event_features_path
    ).resolve()
    cluster_assignments_path = resolve_groot_artifact_path(
        cluster_assignments_path
    ).resolve()
    cluster_annotations_path = resolve_groot_artifact_path(
        cluster_annotations_path
    ).resolve()
    if prompt_records_path is not None:
        prompt_records_path = resolve_groot_artifact_path(
            prompt_records_path
        ).resolve()
    output_path = resolve_groot_artifact_path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing score artifact: {output_path}")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if event_step_scale <= 0:
        raise ValueError("event_step_scale must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Peek manifest for capture_target before joining, so step_mapping
    # auto-detect runs before any aggregation.
    manifest = open_sparse_topk_artifact(topk_run_dir).manifest
    dict_size = int(manifest["dict_size"])
    capture_target = manifest.get("capture_target")
    if step_mapping == "auto":
        step_mapping = resolve_activation_step_mapping(
            step_mapping,
            capture_target,
        )

    join = join_cluster_events(
        event_features=load_jsonl(event_features_path),
        cluster_assignments=load_jsonl(cluster_assignments_path),
        cluster_annotations=load_jsonl(cluster_annotations_path),
    )
    if not join.selected_events:
        raise RuntimeError("No usable clustered events after the join.")

    w = window_size
    usable_events: list[dict] = []
    skipped_window = 0
    shifted_window_count = 0
    event_episode_to_task_id: dict[int, int] = {}
    for event in join.selected_events:
        episode = int(event["episode_num"])
        center_step = int(event["waypoint_step"]) * event_step_scale
        num_steps = int(event["num_steps"]) * event_step_scale
        window_steps, requested, shift = _fit_step_window(
            center_step=center_step, window_size=w, num_steps=num_steps
        )
        if window_steps is None:
            skipped_window += 1
            continue
        event_idx_in_window = w - int(shift)
        if event_idx_in_window < 0 or event_idx_in_window > 2 * w:
            skipped_window += 1
            continue
        if shift != 0:
            shifted_window_count += 1
        event = dict(event)
        event.update(
            {
                "window_steps": window_steps,
                "requested_steps": requested,
                "window_shift": int(shift),
                "event_idx_in_window": int(event_idx_in_window),
                "event_center_step": center_step,
                "episode_num_steps": num_steps,
            }
        )
        usable_events.append(event)
        event_episode_to_task_id[episode] = int(event["task_id"])
    if not usable_events:
        raise RuntimeError("No events remained after centered-window filtering.")

    # Episode → task_id for the WHOLE run (not only event-window episodes).
    # Paper's matrix_task_mean is the per-task mean over every cached
    # timestep, so we need a full episode mapping. Prefer prompt_records;
    # fall back to the event-only mapping if not provided (paper-style
    # task_mean will be approximated).
    episode_to_task_id: dict[int, int] = dict(event_episode_to_task_id)
    if prompt_records_path is not None:
        _merge_episode_task_ids(
            episode_to_task_id,
            load_jsonl(prompt_records_path),
            source_name="prompt_records",
        )
    task_id_set: set[int] = set(event_episode_to_task_id.values())

    (
        timestep_vectors,
        task_means,
        task_counts,
        _manifest,
        load_counters,
    ) = aggregate_sparse_activations_by_timestep(
        topk_run_dir,
        step_mapping=step_mapping,
        episode_to_task_id=episode_to_task_id,
        task_id_set=task_id_set,
        dict_size=dict_size,
    )

    # ---- score per event ----
    episode_group_scores: dict[tuple[str, int], dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    episode_group_window_means: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
    episode_group_event_counts: dict[tuple[str, int], int] = defaultdict(int)
    selected_event_payloads: list[dict] = []
    skipped_missing_window_vectors = 0

    event_iter = tqdm(usable_events, desc="Scoring events", unit="event") if tqdm is not None else usable_events
    for event in event_iter:
        episode = int(event["episode_num"])
        centered = torch.zeros((2 * w + 1, dict_size), dtype=torch.float32)
        ok = True
        for row_idx, step in enumerate(event["window_steps"]):
            vec = timestep_vectors.get((episode, int(step)))
            if vec is None:
                ok = False
                break
            centered[row_idx] = vec
        if not ok:
            skipped_missing_window_vectors += 1
            continue
        templates_for_event = build_templates_at_event_idx(w, int(event["event_idx_in_window"]))
        pattern_scores = _project_pattern_scores(centered, templates_for_event)
        group_key = (str(event["cluster_id"]), episode)
        for name, vec in pattern_scores.items():
            episode_group_scores[group_key][name].append(vec)
        episode_group_window_means[group_key].append(centered.mean(dim=0))
        episode_group_event_counts[group_key] += 1
        selected_event_payloads.append(
            {
                "sample_id": event["sample_id"],
                "task_description": event["task_description"],
                "task_id": event["task_id"],
                "task_episode_idx": event["task_episode_idx"],
                "episode_num": episode,
                "waypoint_rank": event["waypoint_rank"],
                "waypoint_step": event["waypoint_step"],
                "event_center_step": event["event_center_step"],
                "progress_percent": event["progress_percent"],
                "num_steps": event["num_steps"],
                "episode_num_steps": event["episode_num_steps"],
                "cluster_id": event["cluster_id"],
                "phrase": event["phrase"],
                "phase": event["phase"],
                "requested_steps": event["requested_steps"],
                "window_steps": event["window_steps"],
                "window_shift": event["window_shift"],
                "event_idx_in_window": event["event_idx_in_window"],
            }
        )
    if not selected_event_payloads:
        raise RuntimeError("No events remained after activation-window filtering.")

    # ---- aggregate per (cluster, episode) → per cluster ----
    template_names = ("pulse", "step_up", "step_down")
    row_scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    row_template_scores: dict[
        str, dict[str, list[torch.Tensor]]
    ] = {
        name: defaultdict(list)
        for name in template_names
    }
    row_window_means: dict[str, list[torch.Tensor]] = defaultdict(list)
    row_episode_counts: dict[str, int] = defaultdict(int)
    row_event_counts: dict[str, int] = defaultdict(int)
    episode_group_keys: list[dict] = []
    episode_group_matrix_raw: list[torch.Tensor] = []
    episode_group_template_matrices: dict[str, list[torch.Tensor]] = {
        name: []
        for name in template_names
    }
    for group_key, pattern_lists in episode_group_scores.items():
        cluster_id, group_episode = group_key
        group_means = {
            name: torch.stack(score_list, dim=0).mean(dim=0)
            for name, score_list in pattern_lists.items()
        }
        combined = torch.maximum(
            group_means["pulse"], torch.maximum(group_means["step_up"], group_means["step_down"])
        )
        episode_group_keys.append(
            {
                "cluster_id": cluster_id,
                "episode_num": group_episode,
                "num_events": episode_group_event_counts[group_key],
            }
        )
        episode_group_matrix_raw.append(combined)
        row_scores[cluster_id].append(combined)
        for name in template_names:
            episode_group_template_matrices[name].append(group_means[name])
            row_template_scores[name][cluster_id].append(group_means[name])
        row_window_means[cluster_id].append(
            torch.stack(episode_group_window_means[group_key], dim=0).mean(dim=0)
        )
        row_episode_counts[cluster_id] += 1
        row_event_counts[cluster_id] += episode_group_event_counts[group_key]

    episode_group_matrix_raw_tensor = torch.stack(
        episode_group_matrix_raw,
        dim=0,
    )
    episode_group_template_tensors = {
        name: torch.stack(values, dim=0)
        for name, values in episode_group_template_matrices.items()
    }
    episode_group_template_max = torch.maximum(
        episode_group_template_tensors["pulse"],
        torch.maximum(
            episode_group_template_tensors["step_up"],
            episode_group_template_tensors["step_down"],
        ),
    )
    # Both tensors use the same already-computed group template means and
    # element-wise maximum operations, so exact equality is the appropriate
    # guard here; no floating-point reduction is repeated.
    if not torch.equal(
        episode_group_matrix_raw_tensor,
        episode_group_template_max,
    ):
        raise RuntimeError(
            "Episode-group raw scores differ from the feature-wise maximum "
            "of pulse, step-up, and step-down template means."
        )

    row_cluster_ids = sorted(
        row_scores,
        key=lambda cid: (
            str(join.cluster_metadata_by_id[cid]["task_description"]),
            str(cid),
        ),
    )
    num_rows = len(row_cluster_ids)
    matrix_raw = torch.zeros((num_rows, dict_size), dtype=torch.float32)
    matrix_templates = {
        name: torch.zeros((num_rows, dict_size), dtype=torch.float32)
        for name in template_names
    }
    matrix_window_mean = torch.zeros((num_rows, dict_size), dtype=torch.float32)
    matrix_task_mean = torch.zeros((num_rows, dict_size), dtype=torch.float32)
    cluster_to_task_id: dict[str, int] = {}
    for event in selected_event_payloads:
        cluster_to_task_id.setdefault(str(event["cluster_id"]), int(event["task_id"]))
    for row_idx, cluster_id in enumerate(row_cluster_ids):
        matrix_raw[row_idx] = torch.stack(row_scores[cluster_id], dim=0).mean(dim=0)
        for name in template_names:
            matrix_templates[name][row_idx] = torch.stack(
                row_template_scores[name][cluster_id],
                dim=0,
            ).mean(dim=0)
        matrix_window_mean[row_idx] = torch.stack(row_window_means[cluster_id], dim=0).mean(dim=0)
        task_id = cluster_to_task_id.get(cluster_id, -1)
        if task_id in task_means:
            matrix_task_mean[row_idx] = task_means[task_id]

    row_raw_from_episode_groups = torch.zeros_like(matrix_raw)
    group_indices_by_cluster: dict[str, list[int]] = defaultdict(list)
    for group_idx, group_key in enumerate(episode_group_keys):
        group_indices_by_cluster[str(group_key["cluster_id"])].append(group_idx)
    for row_idx, cluster_id in enumerate(row_cluster_ids):
        group_indices = torch.tensor(
            group_indices_by_cluster[cluster_id],
            dtype=torch.int64,
        )
        row_raw_from_episode_groups[row_idx] = (
            episode_group_matrix_raw_tensor.index_select(
                0,
                group_indices,
            ).mean(dim=0)
        )
    # This guard repeats an episode-group reduction through an independently
    # indexed tensor. A small allclose tolerance accommodates harmless CPU
    # reduction-order differences while still detecting a changed aggregation
    # contract. ``matrix_raw`` itself keeps the legacy computation above.
    row_raw_guard_rtol = 1e-6
    row_raw_guard_atol = 1e-7
    if not torch.allclose(
        matrix_raw,
        row_raw_from_episode_groups,
        rtol=row_raw_guard_rtol,
        atol=row_raw_guard_atol,
    ):
        max_abs = float(
            (matrix_raw - row_raw_from_episode_groups).abs().max()
        )
        raise RuntimeError(
            "Row raw scores differ from the episode-balanced mean of "
            f"episode-group raw scores (max_abs={max_abs})."
        )

    matrix_template_max = torch.maximum(
        matrix_templates["pulse"],
        torch.maximum(
            matrix_templates["step_up"],
            matrix_templates["step_down"],
        ),
    )
    raw_vs_template_max_abs = (
        matrix_raw - matrix_template_max
    ).abs()
    raw_vs_template_max_num_diff = int(
        torch.count_nonzero(matrix_raw != matrix_template_max)
    )

    row_results = []
    for row_idx, cluster_id in enumerate(row_cluster_ids):
        meta = join.cluster_metadata_by_id[cluster_id]
        row_results.append(
            {
                "task_description": meta["task_description"],
                "cluster_id": cluster_id,
                "phrase": meta["phrase"],
                "phase": meta["phase"],
                "num_episode_groups": row_episode_counts[cluster_id],
                "num_events": row_event_counts[cluster_id],
                "episode_coverage": meta["episode_coverage"],
                "raw_top_features": _row_top_summary(matrix_raw[row_idx], top_n),
                "pulse_top_features": _row_top_summary(
                    matrix_templates["pulse"][row_idx],
                    top_n,
                ),
                "step_up_top_features": _row_top_summary(
                    matrix_templates["step_up"][row_idx],
                    top_n,
                ),
                "step_down_top_features": _row_top_summary(
                    matrix_templates["step_down"][row_idx],
                    top_n,
                ),
                "template_max_top_features": _row_top_summary(
                    matrix_template_max[row_idx],
                    top_n,
                ),
                "window_mean_top_features": _row_top_summary(matrix_window_mean[row_idx], top_n),
                "task_mean_top_features": _row_top_summary(matrix_task_mean[row_idx], top_n),
            }
        )

    payload = {
        "source": {
            "contract_version": "event_feature_score_source_v2",
            "topk_run_dir": str(topk_run_dir),
            "topk_manifest_sha256": _sha256(topk_run_dir / "manifest.json"),
            "event_features_path": str(event_features_path),
            "event_features_sha256": _sha256(event_features_path),
            "cluster_assignments_path": str(cluster_assignments_path),
            "cluster_assignments_sha256": _sha256(
                cluster_assignments_path
            ),
            "cluster_annotations_path": str(cluster_annotations_path),
            "cluster_annotations_sha256": _sha256(
                cluster_annotations_path
            ),
            "prompt_records_path": (
                str(prompt_records_path)
                if prompt_records_path is not None
                else None
            ),
            "prompt_records_sha256": (
                _sha256(prompt_records_path)
                if prompt_records_path is not None
                else None
            ),
            "dict_size": dict_size,
            "topk": int(manifest["topk"]),
            "layer": manifest.get("layer"),
            "sae_path": manifest.get("sae_path"),
            "sae_sha256": manifest.get("sae_sha256"),
            "capture_target": capture_target,
            "event_step_scale": event_step_scale,
            "activation_source_manifest_sha256": manifest.get(
                "activation_source_manifest_sha256"
            ),
            "trajectory_manifest_sha256": manifest.get(
                "trajectory_manifest_sha256"
            ),
        },
        "window_size": window_size,
        "top_n": top_n,
        "step_mapping": step_mapping,
        "event_step_scale": event_step_scale,
        "row_semantics": "(task_description, cluster_id, phrase, phase)",
        "score_definitions": {
            "pulse": "positive projection onto a symmetric local-peak template (event-centered) after time-centering",
            "step_up": "positive projection onto a low-to-high step template (event-centered) after time-centering",
            "step_down": "positive projection onto a high-to-low step template (event-centered) after time-centering",
            "combined_score": (
                "within each (cluster, episode), average event projections "
                "separately for pulse, step_up, and step_down, then take the "
                "feature-wise maximum of those three template means"
            ),
            "matrix_raw": (
                "episode-balanced per-cluster mean of combined_score "
                "(== event_aligned ranking)"
            ),
            "matrix_window_mean": "per-cluster mean of window-mean activation",
            "matrix_task_mean": "per-cluster, broadcast the per-task mean activation over all cached timesteps",
        },
        "directional_score_definitions": {
            "episode_group_matrix_template": (
                "within each (cluster, episode), mean per-event projection "
                "for the named pulse, step_up, or step_down template"
            ),
            "episode_group_matrix_raw": (
                "feature-wise maximum of the three episode-group template "
                "mean matrices"
            ),
            "matrix_template": (
                "episode-balanced per-cluster mean of the corresponding "
                "episode-group template matrix"
            ),
            "matrix_template_max": (
                "feature-wise maximum of the three row-level template means; "
                "this is not generally equal to legacy matrix_raw because "
                "maximum and episode mean do not commute"
            ),
            "matrix_raw": (
                "unchanged legacy episode-balanced per-cluster mean of "
                "episode_group_matrix_raw"
            ),
        },
        "template_matrix_contract": {
            "template_names": list(template_names),
            "episode_group_raw_equals_template_max": True,
            "episode_group_raw_guard": "exact torch.equal",
            "row_raw_equals_episode_group_raw_mean": True,
            "row_raw_guard": {
                "comparison": "torch.allclose",
                "rtol": row_raw_guard_rtol,
                "atol": row_raw_guard_atol,
                "reason": (
                    "the guard repeats an episode-group reduction through "
                    "an independently indexed tensor"
                ),
            },
            "row_template_aggregation": (
                "equal mean across episode-group template means"
            ),
            "raw_vs_template_max": {
                "expected_relation": (
                    "not necessarily equal because mean(max(template)) and "
                    "max(mean(template)) do not commute"
                ),
                "max_abs": float(raw_vs_template_max_abs.max()),
                "num_diff": raw_vs_template_max_num_diff,
                "num_values": int(matrix_raw.numel()),
                "difference_test": "exact tensor inequality",
            },
        },
        "selection_counts": {
            **join.counts,
            "skipped_window": skipped_window,
            "shifted_window_count": shifted_window_count,
            "selected_events_before_activation_filter": len(usable_events),
            "selected_events_after_activation_filter": len(selected_event_payloads),
            "skipped_missing_window_vectors": skipped_missing_window_vectors,
            "task_timestep_counts": dict(task_counts),
            **load_counters,
        },
        "selected_events": selected_event_payloads,
        "episode_group_keys": episode_group_keys,
        "episode_group_matrix_pulse": episode_group_template_tensors["pulse"],
        "episode_group_matrix_step_up": episode_group_template_tensors[
            "step_up"
        ],
        "episode_group_matrix_step_down": episode_group_template_tensors[
            "step_down"
        ],
        "episode_group_matrix_raw": episode_group_matrix_raw_tensor,
        "row_keys": [
            {
                **join.cluster_metadata_by_id[cid],
                "num_episode_groups": row_episode_counts[cid],
                "num_events": row_event_counts[cid],
                "task_id": cluster_to_task_id.get(cid),
            }
            for cid in row_cluster_ids
        ],
        # Legacy paper-faithful matrices plus directional decomposition.
        "matrix_pulse": matrix_templates["pulse"],
        "matrix_step_up": matrix_templates["step_up"],
        "matrix_step_down": matrix_templates["step_down"],
        "matrix_template_max": matrix_template_max,
        "matrix_raw": matrix_raw,
        "matrix_window_mean": matrix_window_mean,
        "matrix_task_mean": matrix_task_mean,
        "matrix": matrix_raw,
        "row_results": row_results,
    }
    torch.save(payload, output_path)
    return {
        "output_path": str(output_path),
        "num_rows": num_rows,
        "dict_size": dict_size,
        "step_mapping": step_mapping,
        "event_step_scale": event_step_scale,
        "selected_events": len(selected_event_payloads),
    }


def _candidate_ids(rankings_dir: Path, ranking: str) -> list[int]:
    rows = load_jsonl(rankings_dir / "candidates.jsonl")
    return [int(row["feature_id"]) for row in rows if row["ranking"] == ranking]


def _rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Rank correlation requires equal vectors")
    left_rank = torch.empty_like(left)
    right_rank = torch.empty_like(right)
    left_rank[torch.argsort(left)] = torch.arange(len(left), dtype=left.dtype)
    right_rank[torch.argsort(right)] = torch.arange(len(right), dtype=right.dtype)
    return float(torch.corrcoef(torch.stack([left_rank, right_rank]))[0, 1])


def top_feature_records(
    vector: torch.Tensor,
    top_k: int,
) -> list[dict[str, float | int]]:
    """Return the highest-scoring feature IDs and scores from one vector."""

    if vector.ndim != 1:
        raise ValueError("Feature ranking requires a one-dimensional vector")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    values, indices = torch.topk(vector, k=min(top_k, int(vector.numel())))
    return [
        {"feature_id": int(feature_id), "score": float(score)}
        for feature_id, score in zip(
            indices.tolist(),
            values.tolist(),
            strict=True,
        )
    ]


def _bootstrap_topk(
    *,
    group_keys: list[dict],
    group_matrix: torch.Tensor,
    cluster_ids: list[str],
    canonical_top_ids: list[int],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for idx, key in enumerate(group_keys):
        by_cluster[str(key["cluster_id"])].append(idx)
    if set(by_cluster) != set(cluster_ids):
        raise ValueError("Episode-group matrix does not cover every cluster row")

    generator = torch.Generator().manual_seed(seed)
    selection_counts: dict[int, int] = defaultdict(int)
    for _ in range(repetitions):
        cluster_vectors = []
        for cluster_id in cluster_ids:
            indices = by_cluster[cluster_id]
            draws = torch.randint(
                0,
                len(indices),
                (len(indices),),
                generator=generator,
            )
            sampled = torch.tensor(indices, dtype=torch.int64)[draws]
            cluster_vectors.append(group_matrix[sampled].mean(dim=0))
        suite = torch.stack(cluster_vectors, dim=0).mean(dim=0)
        for feature_id in torch.topk(
            suite,
            k=len(canonical_top_ids),
        ).indices.tolist():
            selection_counts[int(feature_id)] += 1

    return {
        "repetitions": repetitions,
        "seed": seed,
        "canonical_top_selection_frequency": {
            str(feature_id): selection_counts.get(feature_id, 0) / repetitions
            for feature_id in canonical_top_ids
        },
        "most_frequent_features": [
            {
                "feature_id": feature_id,
                "selection_frequency": count / repetitions,
            }
            for feature_id, count in sorted(
                selection_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[: len(canonical_top_ids)]
        ],
    }


def _firing_coverage(
    *,
    score_payload: dict,
    timestep_vectors: dict[tuple[int, int], torch.Tensor],
    feature_ids: list[int],
) -> list[dict[str, Any]]:
    group_fired: dict[tuple[str, int], torch.Tensor] = {}
    for event in score_payload["selected_events"]:
        key = (str(event["cluster_id"]), int(event["episode_num"]))
        window = torch.stack(
            [
                timestep_vectors[
                    (int(event["episode_num"]), int(step))
                ][feature_ids]
                for step in event["window_steps"]
            ]
        )
        fired = (window > 0).any(dim=0)
        if key in group_fired:
            group_fired[key] |= fired
        else:
            group_fired[key] = fired
    matrix = torch.stack(list(group_fired.values()), dim=0)
    return [
        {
            "feature_id": feature_id,
            "episode_group_firing_coverage": float(
                matrix[:, idx].float().mean()
            ),
            "fired_episode_groups": int(matrix[:, idx].sum()),
            "episode_groups": len(matrix),
        }
        for idx, feature_id in enumerate(feature_ids)
    ]


def _render_heatmap(
    payload: dict,
    output_path: Path,
    top_n: int,
) -> list[int]:
    import matplotlib.pyplot as plt

    matrix = payload["matrix_raw"].to(dtype=torch.float32)
    suite = matrix.mean(dim=0)
    feature_ids = torch.topk(suite, k=top_n).indices.tolist()
    plotted = matrix[:, feature_ids]
    row_labels = [
        f"{row['task_id']} | {row.get('phase', '')} | {row['cluster_id']}"
        for row in payload["row_keys"]
    ]
    fig, ax = plt.subplots(figsize=(14, 10))
    image = ax.imshow(plotted.numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(
        range(len(feature_ids)),
        [str(value) for value in feature_ids],
        rotation=60,
    )
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=7)
    ax.set_xlabel("SAE feature ID (suite event-aligned top features)")
    ax.set_ylabel("Task | VLM phase | cluster")
    ax.set_title("GR00T PQ3 event-aligned phase-feature scores (W=5)")
    fig.colorbar(image, ax=ax, label="event-aligned score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return feature_ids


def audit_groot_phase_feature_scores(args: Any) -> dict[str, Any]:
    """Audit GR00T phase-feature score and ranking artifacts."""

    from event_sae.scoring.rankings import alive_feature_ids

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty audit output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_w5 = args.scores_w5.resolve()
    scores_w4 = args.scores_w4.resolve()
    rankings_w5 = args.rankings_w5.resolve()
    rankings_w4 = args.rankings_w4.resolve()
    topk_run_dir = args.topk_run_dir.resolve()
    payload5 = torch.load(scores_w5, map_location="cpu")
    payload4 = torch.load(scores_w4, map_location="cpu")

    expected_counts = {
        "valid_clusters": args.expected_clusters,
        "joined_events": args.expected_events,
        "selected_events_after_activation_filter": args.expected_events,
        "shards_loaded": args.expected_files,
        "rows_seen": args.expected_rows,
        "rows_used": args.expected_executed_rows,
        "rows_skipped_nonexecuted": (
            args.expected_rows - args.expected_executed_rows
        ),
        "skipped_missing_window_vectors": 0,
    }
    for key, expected in expected_counts.items():
        actual = int(payload5["selection_counts"].get(key, -1))
        if actual != expected:
            raise ValueError(
                f"W=5 count mismatch for {key}: {actual} != {expected}"
            )
    if (
        int(payload5["selection_counts"]["shifted_window_count"])
        != args.expected_w5_shifts
    ):
        raise ValueError("Unexpected W=5 boundary shift count")
    if (
        int(payload4["selection_counts"]["shifted_window_count"])
        != args.expected_w4_shifts
    ):
        raise ValueError("Unexpected W=4 boundary shift count")

    for name in ("matrix_raw", "matrix_window_mean", "matrix_task_mean"):
        for payload in (payload5, payload4):
            matrix = payload[name]
            if tuple(matrix.shape) != (
                args.expected_clusters,
                args.dict_size,
            ):
                raise ValueError(
                    f"Unexpected {name} shape: {tuple(matrix.shape)}"
                )
            if not torch.isfinite(matrix).all():
                raise ValueError(f"Non-finite values in {name}")
    if [row["cluster_id"] for row in payload5["row_keys"]] != [
        row["cluster_id"] for row in payload4["row_keys"]
    ]:
        raise ValueError("W=5 and W=4 row keys differ")

    event5 = _candidate_ids(rankings_w5, "event_aligned")
    event4 = _candidate_ids(rankings_w4, "event_aligned")
    window5 = _candidate_ids(rankings_w5, "window_mean")
    task5 = _candidate_ids(rankings_w5, "task_mean")
    random5 = _candidate_ids(rankings_w5, "random_alive")
    if any(
        len(values) != args.top_k
        for values in (event5, event4, window5, task5, random5)
    ):
        raise ValueError("A ranking does not contain the requested top-K")

    suite5 = payload5["matrix_raw"].mean(dim=0)
    suite4 = payload4["matrix_raw"].mean(dim=0)
    rows_by_task: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(payload5["row_keys"]):
        rows_by_task[str(row["task_description"])].append(idx)
    task_balanced = torch.stack(
        [
            payload5["matrix_raw"][indices].mean(dim=0)
            for indices in rows_by_task.values()
        ]
    ).mean(dim=0)

    episode_to_task = {
        int(row["episode_num"]): int(row["task_id"])
        for row in load_jsonl(args.prompt_records_path.resolve())
    }
    task_ids = {int(row["task_id"]) for row in payload5["row_keys"]}
    (
        timestep_vectors,
        _task_means,
        task_counts,
        _manifest,
        load_counts,
    ) = aggregate_sparse_activations_by_timestep(
        topk_run_dir,
        step_mapping="action_executed",
        episode_to_task_id=episode_to_task,
        task_id_set=task_ids,
        dict_size=args.dict_size,
    )
    if sum(task_counts.values()) != args.expected_environment_steps:
        raise ValueError(
            "Environment-step total does not match the expected count"
        )

    group_matrix = payload5["episode_group_matrix_raw"].to(
        dtype=torch.float32
    )
    group_keys = list(payload5["episode_group_keys"])
    bootstrap = _bootstrap_topk(
        group_keys=group_keys,
        group_matrix=group_matrix,
        cluster_ids=[
            str(row["cluster_id"]) for row in payload5["row_keys"]
        ],
        canonical_top_ids=event5,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    firing_coverage = _firing_coverage(
        score_payload=payload5,
        timestep_vectors=timestep_vectors,
        feature_ids=event5,
    )

    heatmap_features = _render_heatmap(
        payload5,
        output_dir / "event_feature_heatmap.png",
        args.heatmap_features,
    )
    audit = {
        "format": SCORE_AUDIT_FORMAT,
        "status": "pass",
        "source": {
            "scores_w5": str(scores_w5),
            "scores_w5_sha256": _sha256(scores_w5),
            "scores_w4": str(scores_w4),
            "scores_w4_sha256": _sha256(scores_w4),
            "topk_manifest_sha256": _sha256(
                topk_run_dir / "manifest.json"
            ),
        },
        "review_provenance": {
            "modes": sorted(
                {
                    str(row.get("review_mode"))
                    for row in payload5["row_keys"]
                }
            ),
            "actual_human_review_completed": all(
                bool(row.get("actual_human_review_completed"))
                for row in payload5["row_keys"]
            ),
        },
        "counts": {
            **expected_counts,
            "environment_steps": sum(task_counts.values()),
            "episode_groups": len(group_keys),
            "w5_shifted_windows": payload5["selection_counts"][
                "shifted_window_count"
            ],
            "w4_shifted_windows": payload4["selection_counts"][
                "shifted_window_count"
            ],
            "load_counts": load_counts,
        },
        "encoding": {
            "alive_features_all_16_tokens": len(
                alive_feature_ids(
                    topk_run_dir,
                    step_mapping="inference_step",
                )
            ),
            "alive_features_executed_tokens_0_to_4": len(
                alive_feature_ids(
                    topk_run_dir,
                    step_mapping="action_executed",
                )
            ),
        },
        "canonical_candidates": {
            "event_aligned": event5,
            "window_mean": window5,
            "task_mean": task5,
            "random_alive": random5,
        },
        "sensitivity": {
            "w5_w4_event_topk_overlap": len(set(event5) & set(event4)),
            "w5_w4_event_rank_correlation": _rank_correlation(
                suite5,
                suite4,
            ),
            "event_window_topk_overlap": len(set(event5) & set(window5)),
            "event_task_topk_overlap": len(set(event5) & set(task5)),
            "window_task_topk_overlap": len(set(window5) & set(task5)),
            "task_balanced_topk": top_feature_records(
                task_balanced,
                args.top_k,
            ),
            "cluster_uniform_task_balanced_overlap": len(
                set(event5)
                & {
                    int(row["feature_id"])
                    for row in top_feature_records(task_balanced, args.top_k)
                }
            ),
        },
        "bootstrap": bootstrap,
        "firing_coverage": firing_coverage,
        "heatmap_feature_ids": heatmap_features,
        "confound_audit": {
            "length": (
                "fail_unresolved; no success/failure comparison is reported"
            ),
            "task_identity": (
                "controlled by task-local rows and task-balanced sensitivity"
            ),
            "instruction_balance": "n_a; one instruction per task",
            "in_sample_rescue": (
                "n_a; no detector or intervention in this analysis"
            ),
            "rollout_pooling": (
                "pass; event scores average within episode then across episodes"
            ),
            "phase_dwell": (
                "fail_unresolved; Stage 3 descriptors include progress"
            ),
            "observation_not_causation": (
                "pass; outputs are candidates only"
            ),
            "scene_local_not_general": "fail; one cell per task",
            "claim_strength": "diagnostic evidence; confounded",
        },
    }
    (output_dir / "score_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return audit
