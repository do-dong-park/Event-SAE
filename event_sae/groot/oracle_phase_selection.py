"""Build a causal direct-Oracle phase bundle for SAE scoring and ranking."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from event_sae import sha256_file
from event_sae.events.cluster import build_phase_groups
from event_sae.events.io import load_jsonl, write_jsonl
from event_sae.groot.oracle_phase_keyframes import (
    ORACLE_CONFIDENCE_TIER,
    ORACLE_LABEL_SOURCE,
    ORACLE_PHASE_SOURCE,
    ORACLE_PROVENANCE,
    build_oracle_phase_score_inputs,
    oracle_phase_entry_invariant_errors,
)


FORMAT = "event_sae_oracle_phase_scoring_selection_v1"
EVENTS_NAME = "oracle_phase_events.jsonl"
FEATURES_NAME = "event_features.jsonl"
SOURCE_CLUSTERS_DIR_NAME = "source_clusters"
CLUSTERS_NAME = "clusters.jsonl"
ASSIGNMENTS_NAME = "cluster_assignments.jsonl"
ANNOTATIONS_NAME = "accepted_annotations.jsonl"
PHASE_GROUPS_DIR_NAME = "phase_groups"
PHASE_GROUPS_NAME = "phase_groups.jsonl"
PHASE_ASSIGNMENTS_NAME = "phase_group_assignments.jsonl"
MANIFEST_NAME = "selection_manifest.json"


def _unique_index(rows: list[dict], *, label: str) -> dict[str, dict]:
    output = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in output:
            raise ValueError(f"Duplicate sample_id in {label}: {sample_id}")
        output[sample_id] = row
    return output


def _generated_event_feature(event: dict[str, Any]) -> dict[str, Any]:
    """Return the metadata-only event row consumed by the SAE scorer."""

    return {
        "format": "event_sae_oracle_scoring_event_feature_v1",
        "sample_id": str(event["sample_id"]),
        "task_description": str(event["task_description"]),
        "task_id": int(event["task_id"]),
        "task_episode_idx": int(event["task_episode_idx"]),
        "episode_num": int(event["episode_num"]),
        "waypoint_rank": int(event["waypoint_rank"]),
        "waypoint_step": int(event["activation_env_step_index"]),
        "waypoint_clock": "environment_action_step",
        "progress_percent": float(event["progress_percent"]),
        "num_steps": int(event["num_steps"]),
        "cell_id": event.get("cell_id"),
        "success": bool(event["success"]),
        "anchor_source": event.get("anchor_source"),
        "phase": str(event["phase"]),
        "confidence_tier": ORACLE_CONFIDENCE_TIER,
        "phase_source": ORACLE_PHASE_SOURCE,
        "label_source": ORACLE_LABEL_SOURCE,
        "oracle_upper_bound": True,
        "oracle_provenance": dict(ORACLE_PROVENANCE),
    }


def _provided_feature_clock(
    *,
    events: list[dict[str, Any]],
    feature_by_id: dict[str, dict],
) -> str:
    record_clock_matches = True
    env_step_clock_matches = True
    for event in events:
        sample_id = str(event["sample_id"])
        waypoint_step = int(feature_by_id[sample_id]["waypoint_step"])
        record_clock_matches &= waypoint_step == int(
            event["activation_record_index"]
        )
        env_step_clock_matches &= waypoint_step == int(
            event["activation_env_step_index"]
        )
    if env_step_clock_matches:
        return "environment_action_step"
    if record_clock_matches:
        return "policy_inference_record"
    raise ValueError(
        "Provided event-feature waypoint clock is neither uniformly "
        "activation_record_index nor activation_env_step_index"
    )


def _task_episode_counts(
    events: list[dict[str, Any]],
) -> dict[tuple[int, str], int]:
    episodes_by_task: dict[tuple[int, str], set[int]] = {}
    for event in events:
        key = (int(event["task_id"]), str(event["task_description"]))
        episodes_by_task.setdefault(key, set()).add(int(event["episode_num"]))
    return {
        key: len(episode_nums)
        for key, episode_nums in episodes_by_task.items()
    }


def _add_oracle_phase_group_provenance(phase_groups_dir: Path) -> None:
    """Retain the common grouping contract while making Oracle origin explicit."""

    groups_path = phase_groups_dir / PHASE_GROUPS_NAME
    groups = load_jsonl(groups_path)
    for group in groups:
        group.update(
            {
                "model": "simulator_oracle_labeler",
                "review_mode": "programmatic_oracle_phase_group",
                "review_verdict": "oracle_generated",
                "actual_human_review_completed": False,
                "status": "programmatic-simulator-oracle",
                "confidence_tier": ORACLE_CONFIDENCE_TIER,
                "phase_source": ORACLE_PHASE_SOURCE,
                "label_source": ORACLE_LABEL_SOURCE,
                "oracle_upper_bound": True,
                "oracle_provenance": dict(ORACLE_PROVENANCE),
            }
        )
    write_jsonl(groups_path, groups)

    summary_path = phase_groups_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["outputs"]["phase_groups_sha256"] = sha256_file(groups_path)
    summary.update(
        {
            "review_modes": ["programmatic_oracle_phase_group"],
            "actual_human_review_completed": False,
            "result_status": "programmatic_simulator_oracle",
            "confidence_tier": ORACLE_CONFIDENCE_TIER,
            "phase_source": ORACLE_PHASE_SOURCE,
            "label_source": ORACLE_LABEL_SOURCE,
            "oracle_upper_bound": True,
            "passed": True,
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _link_sources_to_phase_groups(
    *,
    source_clusters_dir: Path,
    phase_groups_dir: Path,
) -> None:
    """Record the builder-resolved phase-group ID on each source artifact."""

    groups = load_jsonl(phase_groups_dir / PHASE_GROUPS_NAME)
    phase_group_by_source_cluster: dict[str, str] = {}
    for group in groups:
        phase_group_id = str(group["phase_group_id"])
        for source_cluster_id in group["source_cluster_ids"]:
            source_cluster_id = str(source_cluster_id)
            if source_cluster_id in phase_group_by_source_cluster:
                raise RuntimeError(
                    "Source cluster maps to multiple phase groups: "
                    f"{source_cluster_id}"
                )
            phase_group_by_source_cluster[source_cluster_id] = phase_group_id

    assignments_path = source_clusters_dir / ASSIGNMENTS_NAME
    assignments = load_jsonl(assignments_path)
    for assignment in assignments:
        source_cluster_id = str(assignment["source_cluster_id"])
        assignment["phase_group_id"] = phase_group_by_source_cluster[
            source_cluster_id
        ]
    write_jsonl(assignments_path, assignments)

    annotations_path = source_clusters_dir / ANNOTATIONS_NAME
    annotations = load_jsonl(annotations_path)
    for annotation in annotations:
        source_cluster_id = str(annotation["cluster_id"])
        annotation["phase_group_id"] = phase_group_by_source_cluster[
            source_cluster_id
        ]
    write_jsonl(annotations_path, annotations)

    summary_path = phase_groups_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["inputs"]["assignments"]["sha256"] = sha256_file(assignments_path)
    summary["inputs"]["finalized_annotations"]["sha256"] = sha256_file(
        annotations_path
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def select_causal_oracle_phase_entries(
    *,
    oracle_events_path: Path,
    output_dir: Path,
    event_features_path: Path | None = None,
    window_size: int = 5,
) -> dict[str, Any]:
    """Emit a complete direct-Oracle bundle with a full centered window."""

    oracle_events_path = Path(oracle_events_path).resolve()
    event_features_path = (
        Path(event_features_path).resolve()
        if event_features_path is not None
        else None
    )
    output_dir = Path(output_dir).resolve()
    if not oracle_events_path.is_file():
        raise FileNotFoundError(f"oracle events not found: {oracle_events_path}")
    if event_features_path is not None and not event_features_path.is_file():
        raise FileNotFoundError(
            f"event features not found: {event_features_path}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty selection directory: {output_dir}"
        )
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    events = load_jsonl(oracle_events_path)
    event_by_id = _unique_index(events, label="oracle events")
    if event_features_path is None:
        features = [_generated_event_feature(event) for event in events]
        feature_by_id = _unique_index(
            features,
            label="generated event features",
        )
        feature_mode = "programmatic_oracle_metadata"
        waypoint_clock = "environment_action_step"
    else:
        features = load_jsonl(event_features_path)
        feature_by_id = _unique_index(features, label="event features")
        if set(event_by_id) != set(feature_by_id):
            raise ValueError(
                "Oracle event and feature sample_id sets do not exactly match"
            )
        feature_mode = "provided_event_features"
        waypoint_clock = _provided_feature_clock(
            events=events,
            feature_by_id=feature_by_id,
        )

    selected_ids: list[str] = []
    exclusions: Counter[str] = Counter()
    for event in events:
        sample_id = str(event["sample_id"])
        if event.get("oracle_upper_bound") is not True:
            raise ValueError(f"{sample_id}: oracle_upper_bound must be true")
        invariant_errors = oracle_phase_entry_invariant_errors(event)
        if invariant_errors:
            raise ValueError(
                f"{sample_id}: invalid Oracle phase-entry contract: "
                f"{', '.join(invariant_errors)}"
            )

        center = int(event["activation_env_step_index"])
        num_steps = int(event["num_steps"])
        reasons = []
        if event.get("causal_action_env_step_index") is None:
            reasons.append("initial_state_no_causal_action")
        if int(event["state_env_step_index"]) >= num_steps:
            reasons.append("terminal_state_no_outgoing_action")
        if center - window_size < 0:
            reasons.append("left_window_out_of_range")
        if center + window_size >= num_steps:
            reasons.append("right_window_out_of_range")
        if reasons:
            exclusions.update(reasons)
            continue
        selected_ids.append(sample_id)

    if not selected_ids:
        raise ValueError("No causal fully centered oracle phase entries remain")
    selected_set = set(selected_ids)
    selected_events = [
        row for row in events if str(row["sample_id"]) in selected_set
    ]
    for event in selected_events:
        invariant_errors = oracle_phase_entry_invariant_errors(
            event,
            required_window_size=window_size,
        )
        if invariant_errors:
            raise RuntimeError(
                f"{event['sample_id']}: selected Oracle phase entry violates "
                f"the centered-window contract: {', '.join(invariant_errors)}"
            )
    selected_features = [
        row for row in features if str(row["sample_id"]) in selected_set
    ]
    if len(selected_events) != len(selected_features):
        raise RuntimeError("Selected event/feature coverage diverged")

    task_episode_counts = _task_episode_counts(events)
    clusters, assignments, annotations = build_oracle_phase_score_inputs(
        selected_events,
        task_episode_counts=task_episode_counts,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    events_output = output_dir / EVENTS_NAME
    features_output = output_dir / FEATURES_NAME
    source_clusters_dir = output_dir / SOURCE_CLUSTERS_DIR_NAME
    source_clusters_dir.mkdir()
    clusters_output = source_clusters_dir / CLUSTERS_NAME
    assignments_output = source_clusters_dir / ASSIGNMENTS_NAME
    annotations_output = source_clusters_dir / ANNOTATIONS_NAME
    write_jsonl(events_output, selected_events)
    write_jsonl(features_output, selected_features)
    write_jsonl(clusters_output, clusters)
    write_jsonl(assignments_output, assignments)
    write_jsonl(annotations_output, annotations)

    phase_groups_dir = output_dir / PHASE_GROUPS_DIR_NAME
    build_phase_groups(
        clusters_path=clusters_output,
        assignments_path=assignments_output,
        finalized_annotations_path=annotations_output,
        output_dir=phase_groups_dir,
        require_human_review=False,
    )
    _link_sources_to_phase_groups(
        source_clusters_dir=source_clusters_dir,
        phase_groups_dir=phase_groups_dir,
    )
    _add_oracle_phase_group_provenance(phase_groups_dir)
    phase_group_summary = json.loads(
        (phase_groups_dir / "summary.json").read_text(encoding="utf-8")
    )
    phase_groups_output = phase_groups_dir / PHASE_GROUPS_NAME
    phase_assignments_output = phase_groups_dir / PHASE_ASSIGNMENTS_NAME

    provided_scale_values = {
        int(event["n_action_steps"])
        for event in events
        if event.get("n_action_steps") is not None
    }
    recommended_event_step_scale = (
        1
        if waypoint_clock == "environment_action_step"
        else (
            next(iter(provided_scale_values))
            if len(provided_scale_values) == 1
            else None
        )
    )
    manifest = {
        "format": FORMAT,
        "selection": (
            "causal phase-entry with outgoing action and unshifted centered "
            f"W={window_size} env-step window"
        ),
        "window_size": window_size,
        "source_oracle_events_path": str(oracle_events_path),
        "source_oracle_events_sha256": sha256_file(oracle_events_path),
        "source_event_features_path": (
            str(event_features_path)
            if event_features_path is not None
            else None
        ),
        "source_event_features_sha256": (
            sha256_file(event_features_path)
            if event_features_path is not None
            else None
        ),
        "event_features_mode": feature_mode,
        "waypoint_clock": waypoint_clock,
        "recommended_event_step_scale": recommended_event_step_scale,
        "exact_phase_entry_contract_validated": True,
        "exact_phase_entry_contract": {
            "state_label": "env_step_phases[k] labels state s_k",
            "scored_action": "outgoing action a_k",
            "causal_action": "action a_(k-1) enters state s_k",
            "selected_window": (
                f"unshifted [{-window_size}, +{window_size}] environment-action "
                "steps around a_k"
            ),
        },
        "confidence_tier": ORACLE_CONFIDENCE_TIER,
        "phase_source": ORACLE_PHASE_SOURCE,
        "label_source": ORACLE_LABEL_SOURCE,
        "oracle_upper_bound": True,
        "oracle_provenance": dict(ORACLE_PROVENANCE),
        "num_input_events": len(events),
        "num_selected_events": len(selected_events),
        "num_excluded_events": len(events) - len(selected_events),
        "num_source_clusters": len(clusters),
        "num_phase_groups": int(phase_group_summary["num_phase_groups"]),
        "exclusion_reason_counts": dict(sorted(exclusions.items())),
        "phase_counts": dict(
            sorted(Counter(str(row["phase"]) for row in selected_events).items())
        ),
        "scoring_inputs": {
            "event_features_path": str(features_output),
            "cluster_assignments_path": str(phase_assignments_output),
            "cluster_annotations_path": str(phase_groups_output),
        },
        "ranking_inputs": {
            "accepted_annotations_path": str(annotations_output),
            "phase_groups_path": str(phase_groups_output),
            "phase_assignments_path": str(phase_assignments_output),
        },
        "outputs": {
            EVENTS_NAME: sha256_file(events_output),
            FEATURES_NAME: sha256_file(features_output),
            f"{SOURCE_CLUSTERS_DIR_NAME}/{CLUSTERS_NAME}": sha256_file(
                clusters_output
            ),
            f"{SOURCE_CLUSTERS_DIR_NAME}/{ASSIGNMENTS_NAME}": sha256_file(
                assignments_output
            ),
            f"{SOURCE_CLUSTERS_DIR_NAME}/{ANNOTATIONS_NAME}": sha256_file(
                annotations_output
            ),
            f"{PHASE_GROUPS_DIR_NAME}/{PHASE_GROUPS_NAME}": sha256_file(
                phase_groups_output
            ),
            f"{PHASE_GROUPS_DIR_NAME}/{PHASE_ASSIGNMENTS_NAME}": sha256_file(
                phase_assignments_output
            ),
            f"{PHASE_GROUPS_DIR_NAME}/summary.json": sha256_file(
                phase_groups_dir / "summary.json"
            ),
        },
        "passed": True,
    }
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
