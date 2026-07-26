"""Extract simulator-oracle phase keyframes from GR00T rollout metadata.

The collector stores ``env_step_phases`` at state resolution:
``env_step_phases[k]`` labels state ``s_k`` after ``k`` environment actions.
For phase-conditioned SAE scoring, state ``s_k`` is aligned to the action token
executed from that state, namely action ``k``.  A separate causal-action index
keeps action ``k - 1`` that produced the transition into ``s_k``.

Trusted rollout PKLs remain supported for local use.  A portable JSON-sidecar
mode reads only phase metadata and does not require pickle trust.

The extractor also emits lightweight programmatic
``(task_description, oracle_phase)`` cluster inputs.  No vision embedding,
clustering, or VLM annotation is involved.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

ORACLE_EVENT_FORMAT = "event_sae_oracle_phase_keyframe_v1"
ORACLE_MANIFEST_FORMAT = "event_sae_oracle_phase_keyframes_v1"
ORACLE_EVENTS_NAME = "oracle_phase_events.jsonl"
ORACLE_CLUSTERS_NAME = "oracle_phase_clusters.jsonl"
ORACLE_ASSIGNMENTS_NAME = "oracle_phase_cluster_assignments.jsonl"
ORACLE_ANNOTATIONS_NAME = "oracle_phase_cluster_annotations.jsonl"
ORACLE_MANIFEST_NAME = "oracle_phase_keyframes_manifest.json"
ANCHOR_MODES = ("phase-entry", "labeler-event", "all")

ORACLE_LABEL_SOURCE = "env_step_phases"
ORACLE_PHASE_SOURCE = "simulator_oracle_env_step_phases"
ORACLE_CONFIDENCE_TIER = "simulator-oracle"
ORACLE_PROVENANCE = {
    "source": "trusted_rollout_env_step_phases",
    "label_resolution": "environment_state",
    "generation": "programmatic",
    "upper_bound": True,
}

_AUXILIARY_EVENT_FIELDS = {
    "env_step_grasp_steps": "grasp",
    "env_step_drop_steps": "drop",
    "env_step_wrong_grasp_steps": "wrong-grasp",
}


def oracle_phase_entry_invariant_errors(
    event: dict[str, Any],
    *,
    required_window_size: int | None = None,
) -> list[str]:
    """Return violations of the direct Oracle phase-entry clock contract.

    ``env_step_phases[k]`` labels state ``s_k``.  A non-terminal entry is
    scored on the action token for outgoing action ``a_k``; action ``a_{k-1}``
    is retained separately as the action that caused the transition.  When a
    window size is supplied, the entry must also admit an unshifted centered
    window of that half-width.
    """

    errors: list[str] = []

    def integer(field: str) -> int | None:
        value = event.get(field)
        try:
            return int(value)
        except (TypeError, ValueError):
            errors.append(f"{field}_not_integer")
            return None

    if str(event.get("anchor_source") or "") not in {
        "oracle_phase_entry",
        "oracle_phase_entry_and_labeler_event",
    }:
        errors.append("not_oracle_phase_entry")
    if event.get("is_phase_transition") is not True:
        errors.append("not_phase_transition")

    phase = str(event.get("phase") or "")
    if not phase:
        errors.append("empty_phase")
    if str(event.get("phase_after") or "") != phase:
        errors.append("phase_after_mismatch")

    state_step = integer("state_env_step_index")
    activation_step = integer("activation_env_step_index")
    waypoint_step = integer("waypoint_step")
    num_steps = integer("num_steps")
    n_action_steps = integer("n_action_steps")
    activation_record = integer("activation_record_index")
    action_token_offset = integer("action_token_offset")

    if num_steps is not None and num_steps <= 0:
        errors.append("num_steps_not_positive")
    if n_action_steps is not None and n_action_steps <= 0:
        errors.append("n_action_steps_not_positive")
    if (
        state_step is not None
        and num_steps is not None
        and not 0 <= state_step <= num_steps
    ):
        errors.append("state_step_out_of_range")

    if state_step is not None:
        phase_before = event.get("phase_before")
        if state_step == 0:
            if phase_before is not None:
                errors.append("initial_phase_before_not_null")
        elif not str(phase_before or "") or str(phase_before) == phase:
            errors.append("phase_before_not_distinct")

    if (
        state_step is not None
        and activation_step is not None
        and num_steps is not None
        and num_steps > 0
        and activation_step != min(state_step, num_steps - 1)
    ):
        errors.append("activation_step_mismatch")
    if (
        waypoint_step is not None
        and activation_step is not None
        and waypoint_step != activation_step
    ):
        errors.append("waypoint_step_mismatch")

    causal_action = event.get("causal_action_env_step_index")
    if state_step is not None:
        expected_causal_action = None if state_step == 0 else state_step - 1
        try:
            actual_causal_action = (
                None if causal_action is None else int(causal_action)
            )
        except (TypeError, ValueError):
            actual_causal_action = causal_action
        if actual_causal_action != expected_causal_action:
            errors.append("causal_action_step_mismatch")

    if (
        activation_step is not None
        and n_action_steps is not None
        and n_action_steps > 0
    ):
        if (
            activation_record is not None
            and activation_record != activation_step // n_action_steps
        ):
            errors.append("activation_record_mismatch")
        if (
            action_token_offset is not None
            and action_token_offset != activation_step % n_action_steps
        ):
            errors.append("action_token_offset_mismatch")

    if required_window_size is not None:
        window_size = int(required_window_size)
        if window_size <= 0:
            raise ValueError("required_window_size must be positive")
        if (
            state_step is not None
            and activation_step is not None
            and state_step != activation_step
        ):
            errors.append("terminal_state_no_outgoing_action")
        if state_step == 0 or causal_action is None:
            errors.append("initial_state_no_causal_action")
        if activation_step is not None and activation_step - window_size < 0:
            errors.append("left_window_out_of_range")
        if (
            activation_step is not None
            and num_steps is not None
            and activation_step + window_size >= num_steps
        ):
            errors.append("right_window_out_of_range")

    return errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _slug(value: str, *, max_length: int = 36) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return (slug or "unnamed")[:max_length]


def _cluster_id(task_id: int, task_description: str, phase: str) -> str:
    identity = f"{task_id}\0{task_description}\0{phase}".encode()
    digest = hashlib.sha256(identity).hexdigest()[:8]
    return (
        f"oracle_t{task_id}_{_slug(task_description, max_length=24)}_"
        f"phase_{_slug(phase, max_length=24)}_{digest}"
    )


def _relative_source_path(value: Any, *, episode_num: int) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"episode {episode_num}: source_file must be a safe relative path"
        )
    return path


def _safe_relative_path(value: Any, *, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} must be a safe relative path")
    return path


def _load_sidecar_mapping(
    metadata_root: Path,
) -> tuple[dict[str, Path], Path | None, str | None]:
    manifest_path = metadata_root / "source_manifest.json"
    if not manifest_path.is_file():
        return {}, None, None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict):
        raise ValueError(
            f"{manifest_path}: expected a JSON object with a 'files' mapping"
        )
    mapping: dict[str, Path] = {}
    for source_value, metadata_value in files.items():
        source_path = _safe_relative_path(
            source_value,
            label=f"{manifest_path}: source path",
        )
        metadata_path = _safe_relative_path(
            metadata_value,
            label=f"{manifest_path}: metadata path for {source_path}",
        )
        mapping[source_path.as_posix()] = metadata_path
    return mapping, manifest_path, _sha256(manifest_path)


def _sidecar_path(
    *,
    metadata_root: Path,
    source_file: Path,
    cell_id: Any,
    mapping: dict[str, Path],
    episode_num: int,
) -> tuple[Path, Path, str]:
    mapped_path = mapping.get(source_file.as_posix())
    if mapped_path is not None:
        relative_path = mapped_path
        resolution = "source_manifest"
    else:
        cell_path = _safe_relative_path(
            cell_id,
            label=f"episode {episode_num}: cell_id",
        )
        relative_path = cell_path / source_file.with_suffix(".json").name
        resolution = "cell_stem_fallback"
    candidate = (metadata_root / relative_path).resolve()
    try:
        candidate.relative_to(metadata_root)
    except ValueError as error:
        raise ValueError(
            f"episode {episode_num}: metadata path escapes root: {relative_path}"
        ) from error
    if not candidate.is_file():
        raise FileNotFoundError(
            f"episode {episode_num}: rollout metadata not found: {candidate}"
        )
    return candidate, relative_path, resolution


def _require_positive_int(value: Any, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{label} must be positive, got {result}")
    return result


def _validate_manifest_episode(payload: dict, episode: dict, pkl_path: Path) -> None:
    comparisons = {
        "task_id": (payload.get("task_id"), episode.get("task_id")),
        "episode_idx": (
            payload.get("episode_idx"),
            episode.get("task_episode_idx"),
        ),
        "episode_success": (
            payload.get("episode_success"),
            episode.get("success"),
        ),
        "cell_id": (payload.get("cell_id"), episode.get("cell_id")),
    }
    mismatches = {}
    for key, (actual, expected) in comparisons.items():
        if actual is None or expected is None:
            continue
        if key == "episode_success":
            actual, expected = bool(actual), bool(expected)
        elif key != "cell_id":
            actual, expected = int(actual), int(expected)
        else:
            actual, expected = str(actual), str(expected)
        if actual != expected:
            mismatches[key] = {"pkl": actual, "manifest": expected}
    if mismatches:
        raise ValueError(f"{pkl_path}: manifest metadata mismatch: {mismatches}")


def _add_anchor(
    anchors: dict[int, dict[str, Any]],
    *,
    env_step_index: int,
    source: str,
    event_label: str | None = None,
) -> None:
    anchor = anchors.setdefault(
        env_step_index,
        {"sources": set(), "event_labels": set()},
    )
    anchor["sources"].add(source)
    if event_label is not None:
        anchor["event_labels"].add(event_label)


def _candidate_anchors(payload: dict, phases: list[str]) -> dict[int, dict[str, Any]]:
    anchors: dict[int, dict[str, Any]] = {}
    _add_anchor(anchors, env_step_index=0, source="phase-entry")
    for env_step_index in range(1, len(phases)):
        if phases[env_step_index] != phases[env_step_index - 1]:
            _add_anchor(
                anchors,
                env_step_index=env_step_index,
                source="phase-entry",
            )

    event_steps = payload.get("env_step_event_steps") or {}
    if not isinstance(event_steps, dict):
        raise ValueError("env_step_event_steps must be a mapping")
    for event_label, env_step_index in event_steps.items():
        _add_anchor(
            anchors,
            env_step_index=int(env_step_index),
            source="labeler-event",
            event_label=str(event_label),
        )

    for field, event_label in _AUXILIARY_EVENT_FIELDS.items():
        steps = payload.get(field) or []
        if not isinstance(steps, (list, tuple)):
            raise ValueError(f"{field} must be a list")
        for env_step_index in steps:
            _add_anchor(
                anchors,
                env_step_index=int(env_step_index),
                source="labeler-event",
                event_label=event_label,
            )
    return anchors


def _anchor_is_selected(anchor: dict[str, Any], anchor_mode: str) -> bool:
    sources = anchor["sources"]
    if anchor_mode == "phase-entry":
        return "phase-entry" in sources
    if anchor_mode == "labeler-event":
        return "labeler-event" in sources
    if anchor_mode == "all":
        return True
    raise ValueError(f"Unsupported anchor_mode={anchor_mode!r}")


def _anchor_source_label(sources: list[str]) -> str:
    source_set = set(sources)
    if source_set == {"phase-entry"}:
        return "oracle_phase_entry"
    if source_set == {"labeler-event"}:
        return "oracle_labeler_event"
    return "oracle_phase_entry_and_labeler_event"


def _episode_events(
    *,
    payload: dict,
    episode: dict,
    source_file: Path,
    anchor_mode: str,
) -> list[dict[str, Any]]:
    episode_num = int(episode["episode_num"])
    feature_phases = [str(phase) for phase in (payload.get("feature_phases") or [])]
    env_step_phases = [
        str(phase) for phase in (payload.get("env_step_phases") or [])
    ]
    if not feature_phases:
        raise ValueError(f"{source_file}: missing feature_phases")
    if not env_step_phases:
        raise ValueError(f"{source_file}: missing env_step_phases")

    num_records = int(episode["num_records"])
    if len(feature_phases) != num_records:
        raise ValueError(
            f"{source_file}: feature_phases={len(feature_phases)} "
            f"!= manifest num_records={num_records}"
        )
    hidden_states = payload.get("hidden_states")
    if hidden_states is not None and len(hidden_states) != num_records:
        raise ValueError(
            f"{source_file}: hidden_states={len(hidden_states)} "
            f"!= manifest num_records={num_records}"
        )

    n_action_steps = _require_positive_int(
        payload.get(
            "env_step_n_action_steps",
            payload.get("n_action_steps", episode.get("n_action_steps")),
        ),
        label=f"{source_file}: n_action_steps",
    )
    manifest_n_action_steps = _require_positive_int(
        episode.get("n_action_steps"),
        label=f"{source_file}: manifest n_action_steps",
    )
    if n_action_steps != manifest_n_action_steps:
        raise ValueError(
            f"{source_file}: n_action_steps={n_action_steps} "
            f"!= manifest={manifest_n_action_steps}"
        )
    model_action_horizon = payload.get("model_action_horizon")
    if (
        model_action_horizon is not None
        and int(model_action_horizon) < n_action_steps
    ):
        raise ValueError(
            f"{source_file}: model_action_horizon={model_action_horizon} "
            f"< n_action_steps={n_action_steps}"
        )

    num_env_steps = len(env_step_phases) - 1
    if num_env_steps <= 0:
        raise ValueError(f"{source_file}: env_step_phases must include s0 and a step")
    if num_env_steps > num_records * n_action_steps:
        raise ValueError(
            f"{source_file}: env steps={num_env_steps} exceed "
            f"records*n_action_steps={num_records * n_action_steps}"
        )

    candidates = _candidate_anchors(payload, env_step_phases)
    selected = [
        (env_step_index, anchor)
        for env_step_index, anchor in sorted(candidates.items())
        if _anchor_is_selected(anchor, anchor_mode)
    ]

    task_id = int(episode["task_id"])
    task_episode_idx = int(episode["task_episode_idx"])
    task_description = str(episode["task_description"])
    prompt_task_description = str(
        episode.get("prompt_task_description") or task_description
    )
    events: list[dict[str, Any]] = []
    for waypoint_rank, (env_step_index, anchor) in enumerate(selected):
        if not 0 <= env_step_index <= num_env_steps:
            raise ValueError(
                f"{source_file}: labeler anchor env_step={env_step_index} "
                f"outside [0,{num_env_steps}]"
            )

        # Phase s_k governs action a_k. If s_k is the final state with no
        # outgoing action, use the last executed action as an explicit fallback.
        activation_env_step = min(env_step_index, num_env_steps - 1)
        causal_action_env_step = (
            None if env_step_index == 0 else env_step_index - 1
        )
        activation_record_index = activation_env_step // n_action_steps
        action_token_offset = activation_env_step % n_action_steps
        if activation_record_index >= num_records:
            raise ValueError(
                f"{source_file}: activation record {activation_record_index} "
                f"outside {num_records} records"
            )
        observation_record_index = (
            env_step_index // n_action_steps
            if env_step_index % n_action_steps == 0
            and env_step_index // n_action_steps < num_records
            else None
        )

        phase = env_step_phases[env_step_index]
        sources = sorted(anchor["sources"])
        event_labels = sorted(anchor["event_labels"])
        sample_id = (
            f"oracle_ep{episode_num:04d}_kf{waypoint_rank:03d}_"
            f"s{env_step_index:04d}"
        )
        events.append(
            {
                "format": ORACLE_EVENT_FORMAT,
                "sample_id": sample_id,
                "episode_num": episode_num,
                "task_id": task_id,
                "task_episode_idx": task_episode_idx,
                "task_description": task_description,
                "prompt_task_description": prompt_task_description,
                "cell_id": episode.get("cell_id"),
                "success": bool(episode["success"]),
                "source_file": source_file.as_posix(),
                "phase_scheme": str(payload.get("phase_scheme") or ""),
                "phase": phase,
                "phase_before": (
                    None
                    if env_step_index == 0
                    else env_step_phases[env_step_index - 1]
                ),
                "phase_after": phase,
                "is_phase_transition": bool(
                    env_step_index == 0
                    or env_step_phases[env_step_index - 1] != phase
                ),
                "anchor_source": _anchor_source_label(sources),
                "anchor_sources": sources,
                "event_labels": event_labels,
                "waypoint_rank": waypoint_rank,
                # Existing scorer field: exact executed action-token time.
                "waypoint_step": activation_env_step,
                "progress_percent": float(
                    activation_env_step / max(num_env_steps - 1, 1)
                ),
                "num_steps": num_env_steps,
                "state_env_step_index": env_step_index,
                "activation_env_step_index": activation_env_step,
                "causal_action_env_step_index": causal_action_env_step,
                "activation_record_index": activation_record_index,
                "action_token_offset": action_token_offset,
                "observation_record_index": observation_record_index,
                "activation_record_phase": feature_phases[
                    activation_record_index
                ],
                "observation_record_phase": (
                    None
                    if observation_record_index is None
                    else feature_phases[observation_record_index]
                ),
                "n_action_steps": n_action_steps,
                "num_records": num_records,
                "oracle_upper_bound": True,
            }
        )
    return events


def build_oracle_phase_score_inputs(
    events: list[dict[str, Any]],
    *,
    task_episode_counts: dict[tuple[int, str], int],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Build deterministic source clusters for direct Oracle phase scoring."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    assignments: list[dict[str, Any]] = []
    cluster_identity: dict[str, tuple[int, str, str]] = {}
    phases_by_task: dict[tuple[int, str], set[str]] = defaultdict(set)

    for event in events:
        task_id = int(event["task_id"])
        task_description = str(event["task_description"])
        phase = str(event["phase"])
        phases_by_task[(task_id, task_description)].add(phase)
        cluster_id = _cluster_id(task_id, task_description, phase)
        identity = (task_id, task_description, phase)
        previous = cluster_identity.setdefault(cluster_id, identity)
        if previous != identity:
            raise RuntimeError(f"Oracle cluster ID collision: {cluster_id}")
        grouped[cluster_id].append(event)
        assignments.append(
            {
                "sample_id": event["sample_id"],
                "cluster_id": cluster_id,
                "task_description": task_description,
                "task_id": task_id,
                "task_episode_idx": event["task_episode_idx"],
                "episode_num": event["episode_num"],
                "waypoint_rank": event["waypoint_rank"],
                "waypoint_step": event["waypoint_step"],
                "progress_percent": event["progress_percent"],
                "num_steps": event["num_steps"],
                "cell_id": event["cell_id"],
                "success": event["success"],
                "anchor_source": event["anchor_source"],
                "anchor_sources": list(event["anchor_sources"]),
                "phase": phase,
                "phase_before": event["phase_before"],
                "phase_after": event["phase_after"],
                "is_phase_transition": bool(event["is_phase_transition"]),
                "state_env_step_index": int(event["state_env_step_index"]),
                "activation_env_step_index": int(
                    event["activation_env_step_index"]
                ),
                "causal_action_env_step_index": event[
                    "causal_action_env_step_index"
                ],
                "activation_record_index": int(
                    event["activation_record_index"]
                ),
                "activation_record_phase": str(
                    event["activation_record_phase"]
                ),
                "action_token_offset": int(event["action_token_offset"]),
                "observation_record_index": event[
                    "observation_record_index"
                ],
                "observation_record_phase": event[
                    "observation_record_phase"
                ],
                "n_action_steps": int(event["n_action_steps"]),
                "num_records": int(event["num_records"]),
                "source_cluster_id": cluster_id,
                "confidence_tier": ORACLE_CONFIDENCE_TIER,
                "phase_source": ORACLE_PHASE_SOURCE,
                "label_source": ORACLE_LABEL_SOURCE,
                "oracle_upper_bound": True,
                "oracle_provenance": dict(ORACLE_PROVENANCE),
            }
        )

    clusters: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for cluster_id, members in sorted(grouped.items()):
        first = members[0]
        task_id = int(first["task_id"])
        task_description = str(first["task_description"])
        phase = str(first["phase"])
        episode_count = len({int(member["episode_num"]) for member in members})
        task_total = task_episode_counts[(task_id, task_description)]
        representative_members = members[:5]
        member_sample_ids = [str(member["sample_id"]) for member in members]
        member_episode_nums = sorted(
            {int(member["episode_num"]) for member in members}
        )
        phase_schemes = sorted(
            {
                str(member.get("phase_scheme") or "")
                for member in members
            }
        )
        phase_scheme = (
            phase_schemes[0] if len(phase_schemes) == 1 else "mixed"
        )
        clusters.append(
            {
                "format": "event_sae_oracle_phase_cluster_v1",
                "cluster_id": cluster_id,
                "task_description": task_description,
                "task_id": task_id,
                "phase": phase,
                "phase_scheme": phase_scheme,
                "num_members": len(members),
                "total_task_episodes": task_total,
                "episode_coverage": float(episode_count / task_total),
                "member_sample_ids": member_sample_ids,
                "member_episode_nums": member_episode_nums,
                "representative_sample_ids": [
                    str(member["sample_id"])
                    for member in representative_members
                ],
                "representative_clip_paths": [],
                "representative_frame_paths": [],
                "representative_progress_percents": [
                    float(member["progress_percent"])
                    for member in representative_members
                ],
                "is_canonical": True,
                "meets_min_coverage": True,
                "confidence_tier": ORACLE_CONFIDENCE_TIER,
                "phase_source": ORACLE_PHASE_SOURCE,
                "label_source": ORACLE_LABEL_SOURCE,
                "oracle_upper_bound": True,
                "oracle_provenance": dict(ORACLE_PROVENANCE),
            }
        )
        annotations.append(
            {
                "format": "event_sae_oracle_phase_annotation_v1",
                "cluster_id": cluster_id,
                "task_description": task_description,
                "phrase": f"simulator-oracle anchor in {phase}",
                "phase": phase,
                "episode_coverage": float(episode_count / task_total),
                "num_members": len(members),
                "num_episodes": episode_count,
                "member_sample_ids": member_sample_ids,
                "source_cluster_ids": [cluster_id],
                "model": "simulator_oracle_labeler",
                "prompt_version": "none",
                "review_mode": "programmatic_oracle",
                "review_verdict": "oracle_generated",
                "actual_human_review_completed": False,
                "status": "programmatic-simulator-oracle",
                "confidence_tier": ORACLE_CONFIDENCE_TIER,
                "phase_source": ORACLE_PHASE_SOURCE,
                "label_source": ORACLE_LABEL_SOURCE,
                "oracle_upper_bound": True,
                "oracle_provenance": dict(ORACLE_PROVENANCE),
                "allowed_phase_labels": sorted(
                    phases_by_task[(task_id, task_description)]
                ),
                "phase_scheme": phase_scheme,
                "api_error": None,
                "parse_error": None,
                "representative_sample_ids": [
                    member["sample_id"] for member in representative_members
                ],
                "representative_clip_paths": [],
                "representative_frame_paths": [],
                "representative_progress_percents": [
                    member["progress_percent"]
                    for member in representative_members
                ],
            }
        )
    return clusters, assignments, annotations


def extract_oracle_phase_keyframes(
    *,
    trajectory_manifest_path: Path,
    output_dir: Path,
    trust_pkl: bool = False,
    raw_rollouts_dir: Path | None = None,
    rollout_metadata_dir: Path | None = None,
    anchor_mode: str = "phase-entry",
    progress_every: int = 10,
) -> dict[str, Any]:
    """Write Oracle keyframes and direct source-cluster scoring inputs."""

    if raw_rollouts_dir is not None and rollout_metadata_dir is not None:
        raise ValueError(
            "raw_rollouts_dir and rollout_metadata_dir are mutually exclusive"
        )
    if anchor_mode not in ANCHOR_MODES:
        raise ValueError(
            f"anchor_mode must be one of {ANCHOR_MODES}, got {anchor_mode!r}"
        )

    trajectory_manifest_path = Path(trajectory_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    manifest = json.loads(trajectory_manifest_path.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{trajectory_manifest_path}: missing non-empty episodes")

    if rollout_metadata_dir is not None:
        source_mode = "json_sidecar"
        source_root = Path(rollout_metadata_dir).resolve()
        source_label = "Rollout metadata root"
    else:
        if not trust_pkl:
            raise ValueError(
                "Refusing pickle.load without explicit trust_pkl=True"
            )
        source_mode = "trusted_pickle"
        source_root = (
            Path(raw_rollouts_dir).resolve()
            if raw_rollouts_dir is not None
            else Path(str(manifest["source_root"])).expanduser().resolve()
        )
        source_label = "Raw rollout root"
    if not source_root.is_dir():
        raise FileNotFoundError(f"{source_label} not found: {source_root}")

    (
        sidecar_mapping,
        sidecar_manifest_path,
        sidecar_manifest_sha256,
    ) = (
        _load_sidecar_mapping(source_root)
        if source_mode == "json_sidecar"
        else ({}, None, None)
    )

    task_episode_counts: Counter[tuple[int, str]] = Counter()
    for episode in episodes:
        task_episode_counts[
            (int(episode["task_id"]), str(episode["task_description"]))
        ] += 1

    all_events: list[dict[str, Any]] = []
    phase_schemes: Counter[str] = Counter()
    source_payloads: list[dict[str, Any]] = []
    episodes_with_keyframes = 0
    for index, episode in enumerate(episodes, 1):
        episode_num = int(episode["episode_num"])
        source_file = _relative_source_path(
            episode.get("source_file"),
            episode_num=episode_num,
        )
        if source_mode == "json_sidecar":
            (
                payload_path,
                relative_payload_path,
                path_resolution,
            ) = _sidecar_path(
                metadata_root=source_root,
                source_file=source_file,
                cell_id=episode.get("cell_id"),
                mapping=sidecar_mapping,
                episode_num=episode_num,
            )
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            source_payloads.append(
                {
                    "source_file": source_file.as_posix(),
                    "metadata_file": relative_payload_path.as_posix(),
                    "path_resolution": path_resolution,
                    "sha256": _sha256(payload_path),
                }
            )
        else:
            payload_path = source_root / source_file
            if not payload_path.is_file():
                raise FileNotFoundError(
                    f"Missing rollout PKL: {payload_path}"
                )
            with payload_path.open("rb") as handle:
                payload = pickle.load(handle)  # noqa: S301 -- explicit trust gate.
            source_payloads.append(
                {
                    "source_file": source_file.as_posix(),
                    "metadata_file": source_file.as_posix(),
                    "path_resolution": "trajectory_manifest_source_file",
                    "sha256": None,
                }
            )
        if not isinstance(payload, dict):
            raise ValueError(f"{payload_path}: expected a mapping payload")
        _validate_manifest_episode(payload, episode, payload_path)
        episode_events = _episode_events(
            payload=payload,
            episode=episode,
            source_file=source_file,
            anchor_mode=anchor_mode,
        )
        if episode_events:
            episodes_with_keyframes += 1
            all_events.extend(episode_events)
        phase_schemes[str(payload.get("phase_scheme") or "")] += 1
        del payload
        if progress_every > 0 and (
            index % progress_every == 0 or index == len(episodes)
        ):
            LOGGER.info(
                "oracle keyframes: episodes=%d/%d events=%d",
                index,
                len(episodes),
                len(all_events),
            )

    if not all_events:
        raise RuntimeError(
            f"No oracle keyframes selected for anchor_mode={anchor_mode}"
        )
    sample_ids = [str(event["sample_id"]) for event in all_events]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Oracle keyframe sample IDs are not unique")

    clusters, assignments, annotations = build_oracle_phase_score_inputs(
        all_events,
        task_episode_counts=dict(task_episode_counts),
    )
    phase_counts = Counter(str(event["phase"]) for event in all_events)
    anchor_source_counts: Counter[str] = Counter()
    event_label_counts: Counter[str] = Counter()
    for event in all_events:
        anchor_source_counts.update(event["anchor_sources"])
        event_label_counts.update(event["event_labels"])

    output_dir.mkdir(parents=True)
    events_path = output_dir / ORACLE_EVENTS_NAME
    clusters_path = output_dir / ORACLE_CLUSTERS_NAME
    assignments_path = output_dir / ORACLE_ASSIGNMENTS_NAME
    annotations_path = output_dir / ORACLE_ANNOTATIONS_NAME
    manifest_path = output_dir / ORACLE_MANIFEST_NAME
    _write_jsonl(events_path, all_events)
    _write_jsonl(clusters_path, clusters)
    _write_jsonl(assignments_path, assignments)
    _write_jsonl(annotations_path, annotations)

    output_manifest = {
        "format": ORACLE_MANIFEST_FORMAT,
        "source_trajectory_manifest_path": str(trajectory_manifest_path),
        "source_trajectory_manifest_sha256": _sha256(
            trajectory_manifest_path
        ),
        "source_mode": source_mode,
        "source_root": str(source_root),
        "source_raw_rollouts_root": (
            str(source_root) if source_mode == "trusted_pickle" else None
        ),
        "source_rollout_metadata_root": (
            str(source_root) if source_mode == "json_sidecar" else None
        ),
        "source_metadata_manifest_path": (
            str(sidecar_manifest_path)
            if sidecar_manifest_path is not None
            else None
        ),
        "source_metadata_manifest_sha256": sidecar_manifest_sha256,
        "source_payload_inventory_sha256": _canonical_sha256(
            source_payloads
        ),
        "source_payload_content_hashes_recorded": (
            source_mode == "json_sidecar"
        ),
        "source_payloads": source_payloads,
        "anchor_mode": anchor_mode,
        "label_source": ORACLE_LABEL_SOURCE,
        "phase_source": ORACLE_PHASE_SOURCE,
        "confidence_tier": ORACLE_CONFIDENCE_TIER,
        "oracle_provenance": dict(ORACLE_PROVENANCE),
        "label_resolution": "environment_state",
        "claim_scope": "simulator-oracle diagnostic upper bound",
        "activation_alignment": {
            "state_index": "env_step_phases[k] labels state s_k",
            "phase_action_index": (
                "state s_k maps to executed action min(k, num_env_steps-1)"
            ),
            "causal_action_index": (
                "transition into s_k was produced by action k-1; null for s_0"
            ),
            "record_index": "activation_env_step_index // n_action_steps",
            "action_token_offset": (
                "activation_env_step_index % n_action_steps"
            ),
            "initial_state": (
                "s_0 maps to record 0 token 0 as the activation computed "
                "from the reset observation"
            ),
        },
        "score_grouping": "(task_id, task_description, oracle_phase)",
        "num_source_episodes": len(episodes),
        "num_episodes_with_keyframes": episodes_with_keyframes,
        "num_keyframes": len(all_events),
        "num_phase_groups": len(annotations),
        "phase_counts": dict(sorted(phase_counts.items())),
        "anchor_source_counts": dict(sorted(anchor_source_counts.items())),
        "event_label_counts": dict(sorted(event_label_counts.items())),
        "phase_scheme_episode_counts": dict(sorted(phase_schemes.items())),
        "files": {
            "events": ORACLE_EVENTS_NAME,
            "events_sha256": _sha256(events_path),
            "clusters": ORACLE_CLUSTERS_NAME,
            "clusters_sha256": _sha256(clusters_path),
            "cluster_assignments": ORACLE_ASSIGNMENTS_NAME,
            "cluster_assignments_sha256": _sha256(assignments_path),
            "cluster_annotations": ORACLE_ANNOTATIONS_NAME,
            "cluster_annotations_sha256": _sha256(annotations_path),
        },
    }
    manifest_path.write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_manifest
