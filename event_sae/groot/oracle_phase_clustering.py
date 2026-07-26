"""Cluster SigLIP/state descriptors inside simulator-oracle phase partitions.

The generic event-feature builder uses ``waypoint_step`` both as a policy
record index for physical-state lookup and as the timestep consumed by the SAE
scorer.  Oracle phase events intentionally separate those clocks:

* ``activation_record_index`` selects the saved policy state.
* ``activation_env_step_index`` selects the executed action token for scoring.

This module first performs an exact ``sample_id`` join that restores the
env-step clock while preserving descriptor provenance.  It then hard-partitions
records by ``(task_id, task_description, oracle_phase)`` and delegates each
partition to the existing :func:`event_sae.events.cluster.cluster_events`
implementation.  The hard partition includes ``phase_scheme`` so differently
defined oracle phases can never share a cluster.  The merged annotations are
programmatic simulator-oracle labels, so no VLM annotation call is needed.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from event_sae import sha256_file
from event_sae.events.cluster import cluster_events
from event_sae.events.io import load_jsonl, write_jsonl


ALIGNED_FEATURES_NAME = "aligned_oracle_phase_event_features.jsonl"
ASSIGNMENTS_NAME = "cluster_assignments.jsonl"
CLUSTERS_NAME = "clusters.jsonl"
ANNOTATIONS_NAME = "cluster_annotations.jsonl"
SUMMARY_NAME = "summary.json"

FORMAT = "event_sae_oracle_phase_state_clustering_v1"
ALIGNED_FORMAT = "event_sae_oracle_phase_aligned_features_v1"
COVERAGE_SCOPE = "episodes_with_oracle_phase_keyframe"


def _slug(value: str, *, max_length: int = 28) -> str:
    result = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return (result or "unnamed")[:max_length]


def _unique_index(
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in output:
            raise ValueError(f"Duplicate sample_id in {label}: {sample_id}")
        output[sample_id] = row
    return output


def _require_same_identity(feature: dict, event: dict) -> None:
    sample_id = str(event["sample_id"])
    for field in (
        "task_id",
        "task_description",
        "task_episode_idx",
        "episode_num",
        "waypoint_rank",
    ):
        feature_value = feature.get(field)
        event_value = event.get(field)
        if str(feature_value) != str(event_value):
            raise ValueError(
                f"{sample_id}: {field} mismatch between descriptor feature "
                f"({feature_value!r}) and oracle event ({event_value!r})"
            )


def align_oracle_event_features(
    *,
    event_features_path: Path,
    oracle_events_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Exact-join descriptor features with oracle events and restore env time.

    Input descriptor features must have been built with
    ``waypoint_step=activation_record_index``.  The aligned output replaces
    ``waypoint_step`` with the exact oracle ``activation_env_step_index`` used
    by the feature scorer, while keeping the former value as
    ``descriptor_record_index``.
    """

    event_features_path = Path(event_features_path).resolve()
    oracle_events_path = Path(oracle_events_path).resolve()
    output_path = Path(output_path).resolve()
    for label, path in (
        ("event features", event_features_path),
        ("oracle events", oracle_events_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite aligned features: {output_path}")

    feature_rows = load_jsonl(event_features_path)
    event_rows = load_jsonl(oracle_events_path)
    feature_by_id = _unique_index(feature_rows, label="event features")
    event_by_id = _unique_index(event_rows, label="oracle events")
    if set(feature_by_id) != set(event_by_id):
        missing_features = sorted(set(event_by_id).difference(feature_by_id))
        missing_events = sorted(set(feature_by_id).difference(event_by_id))
        raise ValueError(
            "Descriptor feature and oracle event sample_id sets do not exactly "
            f"match; missing_features={missing_features[:5]}, "
            f"missing_oracle_events={missing_events[:5]}"
        )
    if not event_rows:
        raise ValueError("No oracle events were provided")

    aligned: list[dict[str, Any]] = []
    phase_schemes: set[str] = set()
    for event in event_rows:
        sample_id = str(event["sample_id"])
        feature = feature_by_id[sample_id]
        _require_same_identity(feature, event)
        if event.get("oracle_upper_bound") is not True:
            raise ValueError(f"{sample_id}: oracle_upper_bound must be true")

        phase = str(event.get("phase", "")).strip()
        if not phase:
            raise ValueError(f"{sample_id}: oracle phase must be non-empty")
        phase_scheme = str(event.get("phase_scheme", "")).strip()
        if not phase_scheme:
            raise ValueError(f"{sample_id}: phase_scheme must be non-empty")
        phase_schemes.add(phase_scheme)

        descriptor_record_index = int(feature["waypoint_step"])
        activation_record_index = int(event["activation_record_index"])
        if descriptor_record_index != activation_record_index:
            raise ValueError(
                f"{sample_id}: descriptor waypoint_step={descriptor_record_index} "
                f"!= oracle activation_record_index={activation_record_index}"
            )

        n_action_steps = int(event["n_action_steps"])
        action_token_offset = int(event["action_token_offset"])
        if n_action_steps <= 0:
            raise ValueError(f"{sample_id}: n_action_steps must be positive")
        if not 0 <= action_token_offset < n_action_steps:
            raise ValueError(
                f"{sample_id}: action_token_offset={action_token_offset} outside "
                f"[0,{n_action_steps})"
            )
        activation_env_step_index = int(event["activation_env_step_index"])
        expected_env_step = (
            descriptor_record_index * n_action_steps + action_token_offset
        )
        if activation_env_step_index != expected_env_step:
            raise ValueError(
                f"{sample_id}: activation env-step identity mismatch; "
                f"{activation_env_step_index} != {expected_env_step}"
            )

        row = dict(feature)
        row.update(
            {
                "source_format": ALIGNED_FORMAT,
                "descriptor_record_index": descriptor_record_index,
                "descriptor_progress_percent": float(
                    feature["progress_percent"]
                ),
                "descriptor_num_records": int(feature["num_steps"]),
                "state_vector_record_index": descriptor_record_index,
                "state_vector_env_step_index": (
                    descriptor_record_index * n_action_steps
                ),
                "state_vector_env_step_lag": action_token_offset,
                "phase": phase,
                "oracle_phase": phase,
                "phase_scheme": phase_scheme,
                "oracle_upper_bound": True,
                "state_env_step_index": int(event["state_env_step_index"]),
                "activation_env_step_index": activation_env_step_index,
                "causal_action_env_step_index": event.get(
                    "causal_action_env_step_index"
                ),
                "action_token_offset": action_token_offset,
                "n_action_steps": n_action_steps,
                "observation_record_index": event.get(
                    "observation_record_index"
                ),
                "waypoint_step": activation_env_step_index,
                "progress_percent": float(event["progress_percent"]),
                "num_steps": int(event["num_steps"]),
            }
        )
        aligned.append(row)

    if len(phase_schemes) != 1:
        raise ValueError(
            f"Expected one phase_scheme, found {sorted(phase_schemes)}"
        )
    aligned.sort(
        key=lambda row: (
            int(row["episode_num"]),
            int(row["state_env_step_index"]),
            str(row["sample_id"]),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, aligned)
    return {
        "format": ALIGNED_FORMAT,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "event_features_path": str(event_features_path),
        "event_features_sha256": sha256_file(event_features_path),
        "oracle_events_path": str(oracle_events_path),
        "oracle_events_sha256": sha256_file(oracle_events_path),
        "num_events": len(aligned),
        "phase_scheme": next(iter(phase_schemes)),
        "num_action_token_offset_zero": sum(
            int(row["action_token_offset"]) == 0 for row in aligned
        ),
        "max_state_vector_env_step_lag": max(
            int(row["state_vector_env_step_lag"]) for row in aligned
        ),
        "passed": True,
    }


def _partition_identity(
    task_id: int,
    task_description: str,
    phase_scheme: str,
    phase: str,
) -> str:
    raw = f"{task_id}\0{task_description}\0{phase_scheme}\0{phase}".encode()
    digest = hashlib.sha256(raw).hexdigest()[:8]
    return (
        f"oracle_t{task_id}_{_slug(task_description)}_"
        f"phase_{_slug(phase)}_{digest}"
    )


def _programmatic_annotation(cluster: dict[str, Any]) -> dict[str, Any]:
    phase = str(cluster["phase"])
    local_cluster_index = int(cluster["phase_cluster_index"])
    return {
        "cluster_id": str(cluster["cluster_id"]),
        "task_id": int(cluster["task_id"]),
        "task_description": str(cluster["task_description"]),
        "phase_scheme": str(cluster["phase_scheme"]),
        "phase": phase,
        "phrase": (
            f"simulator-oracle {phase} / state cluster "
            f"{local_cluster_index:02d}"
        ),
        "episode_coverage": float(cluster["episode_coverage"]),
        "coverage_scope": COVERAGE_SCOPE,
        "total_phase_episodes": int(cluster["total_phase_episodes"]),
        "total_task_episodes": int(cluster["total_task_episodes"]),
        "phase_episode_coverage": float(cluster["phase_episode_coverage"]),
        "task_episode_coverage": float(cluster["task_episode_coverage"]),
        "num_members": int(cluster["num_members"]),
        "representative_sample_ids": list(
            cluster["representative_sample_ids"]
        ),
        "representative_clip_paths": list(
            cluster["representative_clip_paths"]
        ),
        "representative_frame_paths": list(
            cluster["representative_frame_paths"]
        ),
        "representative_progress_percents": list(
            cluster["representative_progress_percents"]
        ),
        "model": "simulator_oracle_labeler",
        "prompt_version": "none",
        "review_mode": "programmatic_oracle_phase_partition",
        "review_verdict": "oracle_generated",
        "actual_human_review_completed": False,
        "label_source": "env_step_phases",
        "oracle_upper_bound": True,
        "api_error": None,
        "parse_error": None,
    }


def cluster_oracle_phase_features(
    *,
    aligned_features_path: Path,
    output_dir: Path,
    vision_weight: float = 1.0,
    state_weight: float = 0.5,
    progress_weight: float = 0.0,
    block_normalization: str = "balanced",
    distance_threshold: float = 0.18,
    min_coverage: float = 0.5,
    num_exemplars: int = 5,
    expected_samples: int | None = None,
) -> dict[str, Any]:
    """Hard-partition aligned descriptors by oracle phase, then cluster state."""

    aligned_features_path = Path(aligned_features_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not aligned_features_path.is_file():
        raise FileNotFoundError(
            f"Aligned oracle features not found: {aligned_features_path}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty cluster directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(aligned_features_path)
    if expected_samples is not None and len(records) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} aligned events, found {len(records)}"
        )
    sample_ids = [str(row["sample_id"]) for row in records]
    if not records or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            "Aligned oracle features must be non-empty with unique sample_id values"
        )

    aligned_by_id = _unique_index(records, label="aligned oracle features")
    partitions: dict[
        tuple[int, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    task_episode_sets: dict[tuple[int, str], set[int]] = defaultdict(set)
    for row in records:
        if row.get("oracle_upper_bound") is not True:
            raise ValueError(
                f"{row['sample_id']}: oracle_upper_bound must be true"
            )
        phase = str(row.get("oracle_phase", row.get("phase", ""))).strip()
        if not phase:
            raise ValueError(f"{row['sample_id']}: oracle phase is empty")
        if str(row.get("phase", "")).strip() != phase:
            raise ValueError(
                f"{row['sample_id']}: phase and oracle_phase disagree"
            )
        phase_scheme = str(row.get("phase_scheme", "")).strip()
        if not phase_scheme:
            raise ValueError(f"{row['sample_id']}: phase_scheme is empty")
        task_key = (int(row["task_id"]), str(row["task_description"]))
        task_episode_sets[task_key].add(int(row["episode_num"]))
        partitions[(*task_key, phase_scheme, phase)].append(row)

    assignments: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []
    partition_summaries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="event_sae_oracle_phase_cluster_"
    ) as temporary_root:
        temporary_root_path = Path(temporary_root)
        for partition_index, (
            (task_id, task_description, phase_scheme, phase),
            partition_records,
        ) in enumerate(sorted(partitions.items())):
            partition_dir = temporary_root_path / f"p{partition_index:04d}"
            partition_input = partition_dir / "event_features.jsonl"
            partition_output = partition_dir / "clusters"
            partition_dir.mkdir(parents=True)
            write_jsonl(partition_input, partition_records)
            local_summary = cluster_events(
                event_features_path=partition_input,
                output_dir=partition_output,
                vision_weight=vision_weight,
                state_weight=state_weight,
                progress_weight=progress_weight,
                block_normalization=block_normalization,
                distance_threshold=distance_threshold,
                min_coverage=min_coverage,
                num_exemplars=num_exemplars,
                expected_samples=len(partition_records),
            )
            local_clusters = load_jsonl(partition_output / "clusters.jsonl")
            local_assignments = load_jsonl(
                partition_output / "cluster_assignments.jsonl"
            )
            partition_id = _partition_identity(
                task_id,
                task_description,
                phase_scheme,
                phase,
            )
            phase_episode_nums = {
                int(row["episode_num"]) for row in partition_records
            }
            total_phase_episodes = len(phase_episode_nums)
            total_task_episodes = len(
                task_episode_sets[(task_id, task_description)]
            )
            id_map: dict[str, str] = {}
            for phase_cluster_index, cluster in enumerate(local_clusters):
                old_cluster_id = str(cluster["cluster_id"])
                new_cluster_id = (
                    f"{partition_id}_cluster_{phase_cluster_index:02d}"
                )
                id_map[old_cluster_id] = new_cluster_id
                member_episode_nums = {
                    int(value) for value in cluster["member_episode_nums"]
                }
                phase_episode_coverage = (
                    len(member_episode_nums) / total_phase_episodes
                )
                task_episode_coverage = (
                    len(member_episode_nums) / total_task_episodes
                )
                if not np.isclose(
                    float(cluster["episode_coverage"]),
                    phase_episode_coverage,
                ):
                    raise RuntimeError(
                        f"{new_cluster_id}: local coverage denominator is not "
                        "the oracle-phase episode set"
                    )
                rewritten = dict(cluster)
                rewritten.update(
                    {
                        "cluster_id": new_cluster_id,
                        "task_id": task_id,
                        "phase_scheme": phase_scheme,
                        "phase": phase,
                        "oracle_phase": phase,
                        "oracle_partition_id": partition_id,
                        "phase_cluster_index": phase_cluster_index,
                        "total_task_episodes": total_task_episodes,
                        "total_phase_episodes": total_phase_episodes,
                        "episode_coverage": phase_episode_coverage,
                        "phase_episode_coverage": phase_episode_coverage,
                        "task_episode_coverage": task_episode_coverage,
                        "coverage_scope": COVERAGE_SCOPE,
                        "oracle_upper_bound": True,
                    }
                )
                clusters.append(rewritten)

            for assignment in local_assignments:
                sample_id = str(assignment["sample_id"])
                source = aligned_by_id[sample_id]
                rewritten = dict(assignment)
                rewritten.update(
                    {
                        "cluster_id": id_map[str(assignment["cluster_id"])],
                        "task_id": task_id,
                        "task_description": task_description,
                        "phase_scheme": phase_scheme,
                        "phase": phase,
                        "oracle_phase": phase,
                        "oracle_partition_id": partition_id,
                        "state_env_step_index": int(
                            source["state_env_step_index"]
                        ),
                        "activation_env_step_index": int(
                            source["activation_env_step_index"]
                        ),
                        "causal_action_env_step_index": source.get(
                            "causal_action_env_step_index"
                        ),
                        "activation_record_index": int(
                            source["descriptor_record_index"]
                        ),
                        "action_token_offset": int(
                            source["action_token_offset"]
                        ),
                        "num_steps": int(source["num_steps"]),
                        "oracle_upper_bound": True,
                    }
                )
                assignments.append(rewritten)

            partition_summaries.append(
                {
                    "oracle_partition_id": partition_id,
                    "task_id": task_id,
                    "task_description": task_description,
                    "phase_scheme": phase_scheme,
                    "phase": phase,
                    "num_events": len(partition_records),
                    "num_phase_episodes": total_phase_episodes,
                    "num_task_episodes": total_task_episodes,
                    "num_clusters": int(local_summary["num_clusters"]),
                    "num_singleton_clusters": int(
                        local_summary["num_singleton_clusters"]
                    ),
                }
            )

    assignments.sort(
        key=lambda row: (
            int(row["episode_num"]),
            int(row["waypoint_step"]),
            str(row["sample_id"]),
        )
    )
    clusters.sort(key=lambda row: str(row["cluster_id"]))
    annotations = [_programmatic_annotation(cluster) for cluster in clusters]

    if {str(row["sample_id"]) for row in assignments} != set(sample_ids):
        raise RuntimeError(
            "Merged oracle phase assignments do not exactly cover aligned features"
        )
    if len(assignments) != len(records):
        raise RuntimeError("Merged oracle phase assignments contain duplicates")
    cluster_ids = [str(row["cluster_id"]) for row in clusters]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise RuntimeError("Merged oracle phase cluster IDs are not unique")

    write_jsonl(output_dir / ASSIGNMENTS_NAME, assignments)
    write_jsonl(output_dir / CLUSTERS_NAME, clusters)
    write_jsonl(output_dir / ANNOTATIONS_NAME, annotations)
    cluster_sizes = [int(row["num_members"]) for row in clusters]
    summary = {
        "format": FORMAT,
        "aligned_features_path": str(aligned_features_path),
        "aligned_features_sha256": sha256_file(aligned_features_path),
        "partition_key": [
            "task_id",
            "task_description",
            "phase_scheme",
            "oracle_phase",
        ],
        "coverage_scope": COVERAGE_SCOPE,
        "claim_scope": "simulator-oracle diagnostic upper bound",
        "annotation_mode": "programmatic_oracle_no_vlm",
        "success_used_for_fitting_or_selection": False,
        "progress_used_for_fitting": bool(progress_weight),
        "num_events": len(records),
        "num_tasks": len(
            {(int(row["task_id"]), str(row["task_description"])) for row in records}
        ),
        "num_phase_partitions": len(partitions),
        "num_clusters": len(clusters),
        "num_singleton_clusters": sum(size == 1 for size in cluster_sizes),
        "singleton_event_fraction": float(
            sum(size == 1 for size in cluster_sizes) / len(records)
        ),
        "median_cluster_size": float(np.median(cluster_sizes)),
        "max_cluster_size": max(cluster_sizes),
        "num_clusters_meeting_min_coverage": sum(
            bool(cluster["meets_min_coverage"]) for cluster in clusters
        ),
        "vision_weight": float(vision_weight),
        "state_weight": float(state_weight),
        "progress_weight": float(progress_weight),
        "block_normalization": block_normalization,
        "distance_threshold": float(distance_threshold),
        "min_coverage": float(min_coverage),
        "num_exemplars": int(num_exemplars),
        "partition_summaries": partition_summaries,
        "outputs": {
            ASSIGNMENTS_NAME: sha256_file(output_dir / ASSIGNMENTS_NAME),
            CLUSTERS_NAME: sha256_file(output_dir / CLUSTERS_NAME),
            ANNOTATIONS_NAME: sha256_file(output_dir / ANNOTATIONS_NAME),
        },
        "passed": True,
    }
    (output_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def align_and_cluster_oracle_phase_features(
    *,
    event_features_path: Path,
    oracle_events_path: Path,
    output_dir: Path,
    vision_weight: float = 1.0,
    state_weight: float = 0.5,
    progress_weight: float = 0.0,
    block_normalization: str = "balanced",
    distance_threshold: float = 0.18,
    min_coverage: float = 0.5,
    num_exemplars: int = 5,
    expected_samples: int | None = None,
) -> dict[str, Any]:
    """Write aligned features and phase×state clusters in one directory."""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty cluster directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = output_dir / ALIGNED_FEATURES_NAME
    alignment = align_oracle_event_features(
        event_features_path=event_features_path,
        oracle_events_path=oracle_events_path,
        output_path=aligned_path,
    )
    cluster_output = output_dir / "phase_state_clusters"
    clustering = cluster_oracle_phase_features(
        aligned_features_path=aligned_path,
        output_dir=cluster_output,
        vision_weight=vision_weight,
        state_weight=state_weight,
        progress_weight=progress_weight,
        block_normalization=block_normalization,
        distance_threshold=distance_threshold,
        min_coverage=min_coverage,
        num_exemplars=num_exemplars,
        expected_samples=expected_samples,
    )
    result = {
        "format": FORMAT,
        "alignment": alignment,
        "clustering": clustering,
        "aligned_features_path": str(aligned_path),
        "cluster_output_dir": str(cluster_output),
        "passed": True,
    }
    (output_dir / "oracle_phase_clustering_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
