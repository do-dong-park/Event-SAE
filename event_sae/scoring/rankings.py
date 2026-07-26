"""Four feature-ranking strategies for SAE candidate selection.

Implements the rankings compared in paper Section 4.4:

- ``event_aligned`` — top-N features per cluster row from an existing
  event-feature score matrix (output of ``score_cluster_features.py``).
- ``window_mean`` — for each cluster row, mean SAE activation over the
  same event windows used by event-aligned, then top-N features.
- ``task_mean`` — for each task, mean SAE activation over every rollout
  timestep in the run, then top-N features.
- ``random_alive`` — uniform random sample from features that fire at
  least once in the run, excluding features already selected by any of
  the three informed rankings.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from event_sae import sha256_file as _sha256
from event_sae.events.io import load_jsonl
from event_sae.scoring.score_matrix import (
    open_sparse_topk_artifact,
    resolve_activation_step_mapping,
    top_feature_records,
)


# ---------------------------------------------------------------------------
# Ranking implementations
# ---------------------------------------------------------------------------


def _matrix_from_payload(
    *,
    payload: dict,
    key: str,
    scores_pt_path: Path,
) -> tuple[torch.Tensor, list[dict]]:
    if key in payload:
        return payload[key].to(dtype=torch.float32), list(payload["row_keys"])
    if "matrix" in payload:
        if key == "matrix_raw":
            return (
                payload["matrix"].to(dtype=torch.float32),
                list(payload["row_keys"]),
            )
        raise ValueError(
            "Legacy score artifact key 'matrix' is event_aligned-only and "
            f"cannot satisfy requested '{key}': {scores_pt_path}"
        )
    raise KeyError(f"Score artifact missing '{key}': {scores_pt_path}")


def _load_matrix(scores_pt_path: Path, key: str) -> tuple[torch.Tensor, list[dict]]:
    """Load a named matrix, allowing legacy ``matrix`` only for event-aligned."""
    scores_pt_path = Path(scores_pt_path).resolve()
    payload = torch.load(scores_pt_path, map_location="cpu")
    return _matrix_from_payload(
        payload=payload,
        key=key,
        scores_pt_path=scores_pt_path,
    )


def _load_event_aligned_matrix(scores_pt_path: Path) -> tuple[torch.Tensor, list[dict]]:
    return _load_matrix(scores_pt_path, "matrix_raw")


def event_aligned_top_features_per_row(scores_pt_path: Path, top_n: int) -> list[dict]:
    """Read the score matrix produced by ``score_cluster_features.py`` and
    return per-cluster-row top-N features."""
    matrix, row_keys = _load_event_aligned_matrix(scores_pt_path)
    out: list[dict] = []
    for row_idx, meta in enumerate(row_keys):
        out.append(
            {
                "ranking": "event_aligned",
                "task_description": str(meta["task_description"]),
                "cluster_id": str(meta["cluster_id"]),
                "phrase": str(meta.get("phrase", "")),
                "phase": str(meta.get("phase", "")),
                "top_features": top_feature_records(matrix[row_idx], top_n),
            }
        )
    return out


def event_aligned_suite_top_k(
    scores_pt_path: Path, top_k: int, *, min_coverage: float = 0.5
) -> list[dict]:
    """Suite-level top-K event-aligned features: mean of the score matrix
    over **canonical** rows (``episode_coverage >= min_coverage``), then
    top-K. Matches mechanistic-steering-vlas suite-config generator,
    which averages over the canonical-filtered matrix (default
    ``min_coverage=0.5``)."""
    matrix, row_keys = _load_event_aligned_matrix(scores_pt_path)
    keep_idx = [
        i
        for i, row in enumerate(row_keys)
        if float(row.get("episode_coverage", 0.0)) >= min_coverage
    ]
    if not keep_idx:
        raise RuntimeError(
            f"No canonical rows with episode_coverage >= {min_coverage}; "
            f"total rows={len(row_keys)}."
        )
    suite_vec = matrix[keep_idx].mean(dim=0)
    return top_feature_records(suite_vec, top_k)


def window_mean_top_features_per_row(
    *,
    scores_pt_path: Path,
    top_n: int,
) -> list[dict]:
    """Per cluster row, top-N features by mean SAE activation over the
    cluster's event windows. Reads the pre-computed ``matrix_window_mean``
    from the score artifact (built by
    ``event_sae.scoring.score_matrix.score_cluster_features``)."""
    matrix, row_keys = _load_matrix(scores_pt_path, "matrix_window_mean")
    out: list[dict] = []
    for row_idx, meta in enumerate(row_keys):
        if float(matrix[row_idx].abs().sum().item()) == 0.0:
            continue
        out.append(
            {
                "ranking": "window_mean",
                "task_description": str(meta["task_description"]),
                "cluster_id": str(meta["cluster_id"]),
                "phrase": str(meta.get("phrase", "")),
                "phase": str(meta.get("phase", "")),
                "top_features": top_feature_records(matrix[row_idx], top_n),
            }
        )
    return out


def window_mean_suite_top_k(
    *,
    scores_pt_path: Path,
    top_k: int,
    min_coverage: float = 0.5,
) -> list[dict]:
    """Suite-level top-K window-mean features: ``num_events``-weighted
    mean of per-row pre-computed ``matrix_window_mean`` vectors,
    restricted to canonical rows (``episode_coverage >= min_coverage``),
    then top-K."""
    matrix, row_keys = _load_matrix(scores_pt_path, "matrix_window_mean")
    keep_idx = [
        i
        for i, row in enumerate(row_keys)
        if float(row.get("episode_coverage", 0.0)) >= min_coverage
    ]
    if not keep_idx:
        raise RuntimeError(
            f"No canonical rows with episode_coverage >= {min_coverage}; "
            f"total rows={len(row_keys)}."
        )
    matrix = matrix[keep_idx]
    num_events_per_row = [int(row.get("num_events", 0)) for row in row_keys]
    weights = torch.tensor(
        [float(num_events_per_row[i]) for i in keep_idx], dtype=torch.float32
    )
    if float(weights.sum().item()) <= 0:
        raise RuntimeError("Total event-weight is zero for window-mean aggregation.")
    suite_vec = (matrix * weights[:, None]).sum(dim=0) / weights.sum()
    return top_feature_records(suite_vec, top_k)




def task_mean_top_features_per_task(
    *,
    scores_pt_path: Path,
    top_n: int,
) -> list[dict]:
    """Per task, top-N features by mean SAE activation across every rollout
    step. Reads pre-computed ``matrix_task_mean`` from the score artifact.
    Note: rows here are per-cluster (each cluster broadcasts its task's
    mean), so we dedupe by task_description for the per-task ranking."""
    matrix, row_keys = _load_matrix(scores_pt_path, "matrix_task_mean")
    seen: dict[str, int] = {}
    for i, meta in enumerate(row_keys):
        task = str(meta.get("task_description", ""))
        if task and task not in seen:
            seen[task] = i
    out: list[dict] = []
    for task, idx in sorted(seen.items()):
        out.append(
            {
                "ranking": "task_mean",
                "task_description": task,
                "top_features": top_feature_records(matrix[idx], top_n),
            }
        )
    return out


def task_mean_suite_top_k(
    *,
    scores_pt_path: Path,
    top_k: int,
    min_coverage: float = 0.5,
) -> list[dict]:
    """Suite-level top-K task-mean features. Mirrors openpi-mech's
    ``_task_mean_suite_vector``: dedupe canonical cluster rows by
    ``task_description``, then take a per-task-timestep-count-weighted
    mean across unique tasks."""
    scores_pt_path = Path(scores_pt_path).resolve()
    payload = torch.load(scores_pt_path, map_location="cpu")
    matrix, row_keys = _matrix_from_payload(
        payload=payload,
        key="matrix_task_mean",
        scores_pt_path=scores_pt_path,
    )
    keep_idx = [
        i
        for i, row in enumerate(row_keys)
        if float(row.get("episode_coverage", 0.0)) >= min_coverage
    ]
    if not keep_idx:
        raise RuntimeError(
            f"No canonical rows with episode_coverage >= {min_coverage}; "
            f"total rows={len(row_keys)}."
        )
    task_timestep_counts = (
        payload.get("selection_counts", {}).get("task_timestep_counts", {}) or {}
    )

    seen_tasks: set[str] = set()
    vectors: list[torch.Tensor] = []
    weights: list[float] = []
    for i in keep_idx:
        meta = row_keys[i]
        task_desc = str(meta.get("task_description", ""))
        if task_desc in seen_tasks:
            continue
        seen_tasks.add(task_desc)
        vectors.append(matrix[i])
        task_id = meta.get("task_id")
        weight = task_timestep_counts.get(task_id)
        if weight is None and task_id is not None:
            weight = task_timestep_counts.get(str(task_id))
        weights.append(float(weight) if weight else 1.0)
    if not vectors:
        raise RuntimeError("No tasks remained for task_mean suite aggregation.")
    stacked = torch.stack(vectors, dim=0)
    weight_t = torch.tensor(weights, dtype=torch.float32)
    if float(weight_t.sum().item()) <= 0:
        raise RuntimeError("Total task-timestep weight is zero for task_mean.")
    suite_vec = (stacked * weight_t[:, None]).sum(dim=0) / weight_t.sum()
    return top_feature_records(suite_vec, top_k)


def alive_feature_ids(topk_run_dir: Path, *, step_mapping: str = "auto") -> set[int]:
    """Features with a positive value on rows used by the chosen mapping."""
    topk_run_dir = Path(topk_run_dir).resolve()
    artifact = open_sparse_topk_artifact(topk_run_dir)
    manifest = artifact.manifest
    step_mapping = resolve_activation_step_mapping(
        step_mapping,
        manifest.get("capture_target"),
    )
    alive: set[int] = set()
    for _shard_meta, payload in artifact.iter_shards(desc="alive scan"):
        ids = payload["top_feature_ids"].to(dtype=torch.int64)
        vals = payload["top_feature_vals"].to(dtype=torch.float32)
        if ids.shape != vals.shape or ids.ndim != 2:
            raise ValueError("top_feature_ids/top_feature_vals shape mismatch")
        n_rows = int(ids.shape[0])
        if step_mapping == "inference_step":
            steps = payload["step_in_episode"].to(dtype=torch.int64)
            row_mask = steps >= 0
        else:
            chunk_start = payload.get("chunk_start_step")
            executed_len = payload.get("executed_chunk_len")
            if chunk_start is None or executed_len is None:
                row_mask = torch.zeros(n_rows, dtype=torch.bool)
            else:
                chunk_start = chunk_start.to(dtype=torch.int64)
                executed_len = executed_len.to(dtype=torch.int64)
                row_mask = (chunk_start >= 0) & (executed_len > 0)
                if step_mapping == "action_executed":
                    token_idx = payload["token_idx"].to(dtype=torch.int64)
                    row_mask &= (token_idx >= 0) & (token_idx < executed_len)
                elif step_mapping != "chunk_executed":
                    raise ValueError(f"Unsupported step_mapping={step_mapping!r}")
        if tuple(row_mask.shape) != (n_rows,):
            raise ValueError("Row mapping metadata has an unexpected shape")
        positive_ids = ids[(vals > 0) & row_mask[:, None]]
        alive.update(int(value) for value in positive_ids.tolist())
    return alive


def random_alive_features(
    *,
    topk_run_dir: Path,
    num_features: int,
    exclude_feature_ids: set[int],
    seed: int = 0,
    step_mapping: str = "auto",
) -> list[int]:
    """Uniform-random sample of ``num_features`` alive features, excluding
    any feature already selected by the informed rankings. Paper Section
    4.4 random-alive control."""
    alive = alive_feature_ids(topk_run_dir, step_mapping=step_mapping)
    candidates = sorted(alive - set(int(x) for x in exclude_feature_ids))
    if len(candidates) < num_features:
        raise RuntimeError(
            f"Not enough alive features after exclusion: have {len(candidates)}, need {num_features}"
        )
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, num_features))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _parse_runs(values: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"--run must be LABEL=RUN_DIR, got {value!r}"
            )
        label, raw_path = value.split("=", 1)
        if not label or label in runs:
            raise ValueError(
                f"run label must be unique and non-empty: {label!r}"
            )
        runs[label] = Path(raw_path).resolve()
    if len(runs) < 2:
        raise ValueError("at least two --run values are required")
    return runs


def _within_task_phase_exclusive_cells(
    rows: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_description"]].append(row)

    output: list[dict[str, Any]] = []
    for task, task_rows in sorted(by_task.items()):
        phases = sorted({row["phase"] for row in task_rows})
        for phase in phases:
            cell_rows = [
                row for row in task_rows if row["phase"] == phase
            ]
            other_rows = [
                row for row in task_rows if row["phase"] != phase
            ]
            other_ids = {
                int(item["feature_id"])
                for row in other_rows
                for item in row["top_features"][:top_k]
            }
            support = Counter(
                int(item["feature_id"])
                for row in cell_rows
                for item in row["top_features"][:top_k]
                if int(item["feature_id"]) not in other_ids
            )
            candidates = [
                {
                    "feature_id_within_checkpoint": feature_id,
                    "cluster_support": count,
                    "cluster_fraction": count / len(cell_rows),
                }
                for feature_id, count in sorted(
                    support.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ]
            output.append(
                {
                    "task_description": task,
                    "phase": phase,
                    "cluster_count": len(cell_rows),
                    "cluster_ids": sorted(
                        row["cluster_id"] for row in cell_rows
                    ),
                    "candidates": candidates,
                    "repeated_candidates": [
                        item
                        for item in candidates
                        if item["cluster_support"] >= 2
                    ],
                }
            )
    return output


def load_normalized_decoder_columns(sae_path: Path) -> torch.Tensor:
    """Load one SAE decoder as unit-length feature columns."""

    state = torch.load(
        sae_path,
        map_location="cpu",
        weights_only=False,
    )
    weight = state["decoder.weight"].float()
    return F.normalize(weight, dim=0)


def match_mutual_nearest_decoder_features(
    left: torch.Tensor,
    right: torch.Tensor,
) -> dict[str, Any]:
    """Match decoder features in both directions by cosine similarity."""

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("decoder matrices must be two-dimensional")
    if left.shape[0] != right.shape[0]:
        raise ValueError("decoder matrices must share the activation dimension")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("decoder matrices must contain only finite values")
    if torch.any(torch.linalg.vector_norm(left, dim=0) == 0):
        raise ValueError("left decoder contains a zero-norm feature")
    if torch.any(torch.linalg.vector_norm(right, dim=0) == 0):
        raise ValueError("right decoder contains a zero-norm feature")

    left = F.normalize(left.float(), dim=0)
    right = F.normalize(right.float(), dim=0)
    similarity = left.T @ right
    left_to_right = similarity.argmax(dim=1)
    right_to_left = similarity.argmax(dim=0)
    left_indices = torch.arange(left.shape[1])
    mutual = right_to_left[left_to_right] == left_indices
    cosine = similarity[left_indices, left_to_right]
    return {
        "left_to_right": left_to_right.cpu().numpy(),
        "right_to_left": right_to_left.cpu().numpy(),
        "mutual": mutual.cpu().numpy(),
        "cosine": cosine.cpu().numpy(),
        "mutual_count": int(mutual.sum().item()),
        "mutual_cosines": cosine[mutual].cpu().numpy(),
    }


def _pairwise_decoder_cosines(
    labels: list[str],
    decoders: dict[str, torch.Tensor],
    feature_ids: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(labels, 2):
        for feature_id in feature_ids:
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "feature_id_same_integer_only": feature_id,
                    "decoder_cosine": float(
                        torch.dot(
                            decoders[left][:, feature_id],
                            decoders[right][:, feature_id],
                        ).item()
                    ),
                }
            )
    return rows


def compare_checkpoint_ranking_sets(args: Any) -> dict[str, Any]:
    """Compare ranking sets and same-index decoder directions across SAE runs."""

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    runs = _parse_runs(args.run)
    labels = list(runs)

    checkpoint_rows: dict[str, dict[str, Any]] = {}
    phase_cells: dict[str, list[dict[str, Any]]] = {}
    decoders: dict[str, torch.Tensor] = {}
    suite_sets: dict[str, set[int]] = {}

    for label, run_dir in runs.items():
        ranking_path = (
            run_dir
            / "rankings"
            / args.ranking_id
            / "event_aligned.jsonl"
        )
        audit_path = (
            run_dir
            / "audit"
            / args.audit_id
            / "score_audit.json"
        )
        manifest_path = run_dir / "topk" / "manifest.json"
        for path in (ranking_path, audit_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        ranking_rows = load_jsonl(ranking_path)
        audit = _load_json(audit_path)
        manifest = _load_json(manifest_path)
        if len(ranking_rows) != audit["counts"]["valid_clusters"]:
            raise ValueError(
                f"{label}: ranking rows {len(ranking_rows)} != "
                f"valid clusters {audit['counts']['valid_clusters']}"
            )
        if audit["status"] != "pass":
            raise ValueError(
                f"{label}: audit status is {audit['status']!r}"
            )

        cells = _within_task_phase_exclusive_cells(
            ranking_rows,
            args.top_k,
        )
        phase_cells[label] = cells
        suite = [
            int(value)
            for value in audit["canonical_candidates"]["event_aligned"]
        ]
        suite_sets[label] = set(suite)
        sae_path = Path(manifest["sae_path"])
        decoders[label] = load_normalized_decoder_columns(sae_path)

        multi_cells = [
            cell for cell in cells if cell["cluster_count"] >= 2
        ]
        checkpoint_rows[label] = {
            "run_dir": str(run_dir),
            "sae_path": str(sae_path),
            "sae_sha256": manifest["sae_sha256"],
            "ranking_path": str(ranking_path),
            "ranking_sha256": _sha256(ranking_path),
            "audit_path": str(audit_path),
            "audit_sha256": _sha256(audit_path),
            "review_provenance": audit["review_provenance"],
            "scope": audit["counts"],
            "alive_features_all_16_tokens": audit["encoding"][
                "alive_features_all_16_tokens"
            ],
            "alive_features_executed_tokens_0_to_4": audit["encoding"][
                "alive_features_executed_tokens_0_to_4"
            ],
            "event_aligned_suite_top5": suite,
            "w5_w4_top5_overlap": audit["sensitivity"][
                "w5_w4_event_topk_overlap"
            ],
            "w5_w4_rank_correlation": audit["sensitivity"][
                "w5_w4_event_rank_correlation"
            ],
            "bootstrap_top5_selection_frequency": audit["bootstrap"][
                "canonical_top_selection_frequency"
            ],
            "instruction_phase_cells": len(cells),
            "cells_with_phase_only_candidates": sum(
                bool(cell["candidates"]) for cell in cells
            ),
            "multi_cluster_cells": len(multi_cells),
            "multi_cluster_cells_with_repeated_candidates": sum(
                bool(cell["repeated_candidates"])
                for cell in multi_cells
            ),
        }

    common_suite_ids = sorted(
        set.intersection(*(suite_sets[label] for label in labels))
    )

    cell_maps = {
        label: {
            (cell["task_description"], cell["phase"]): {
                int(item["feature_id_within_checkpoint"])
                for item in cell["candidates"]
            }
            for cell in cells
        }
        for label, cells in phase_cells.items()
    }
    common_cell_keys = sorted(
        set.intersection(*(set(cell_maps[label]) for label in labels))
    )
    common_phase_only: list[dict[str, Any]] = []
    for task, phase in common_cell_keys:
        shared = sorted(
            set.intersection(
                *(
                    cell_maps[label][(task, phase)]
                    for label in labels
                )
            )
        )
        if shared:
            common_phase_only.append(
                {
                    "task_description": task,
                    "phase": phase,
                    "same_integer_feature_ids": shared,
                }
            )

    same_integer_ids = sorted(
        set(common_suite_ids).union(
            feature_id
            for row in common_phase_only
            for feature_id in row["same_integer_feature_ids"]
        )
    )
    comparison = {
        "format": "groot_n15_pq3_stage4_checkpoint_comparison_v3",
        "status": "diagnostic_unreviewed",
        "ranking_id": args.ranking_id,
        "audit_id": args.audit_id,
        "top_k": args.top_k,
        "comparison_contract": {
            "feature_ids_comparable_across_checkpoints": (
                "only with decoder-direction evidence; same integer is not "
                "identity"
            ),
            "phase_only_definition": (
                "top-k in a task-phase cell and absent from top-k of the "
                "same task's other phase cells"
            ),
            "repeated_candidate_definition": "cluster support >= 2",
            "raw_scores_comparable_across_checkpoints": False,
        },
        "checkpoints": checkpoint_rows,
        "cross_checkpoint": {
            "same_integer_suite_top5_intersection": common_suite_ids,
            "same_integer_phase_only_candidate_cells": common_phase_only,
            "decoder_cosines": _pairwise_decoder_cosines(
                labels,
                decoders,
                same_integer_ids,
            ),
        },
    }
    phase_only = {
        "format": "groot_n15_pq3_stage4_phase_only_candidates_v1",
        "status": "diagnostic_unreviewed",
        "top_k": args.top_k,
        "definition": comparison["comparison_contract"][
            "phase_only_definition"
        ],
        "checkpoints": phase_cells,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "checkpoint_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "phase_only_candidates.json").write_text(
        json.dumps(phase_only, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(checkpoint_rows, indent=2, ensure_ascii=False))
    print(f"wrote {args.output_dir}")
    return comparison
