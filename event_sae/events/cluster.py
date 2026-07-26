"""Task-local agglomerative clustering of event features.

For each unique `task_description`, builds a single concatenated feature
vector per sample from [normalized vision embedding, z-scored state vector,
z-scored progress percent] and runs agglomerative clustering with cosine
distance threshold. Exemplars are the members closest to the cluster
centroid (preferring unique episodes). Clusters at or above `min_coverage`
of unique task episodes are marked canonical.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from event_sae import sha256_file as _sha256
from event_sae.events.io import load_jsonl, write_jsonl


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "task"


def _unique_index(rows: list[dict], key: str, label: str) -> dict[str, dict]:
    output = {}
    for row in rows:
        value = str(row[key])
        if value in output:
            raise ValueError(f"{label} contains duplicate {key} values")
        output[value] = row
    return output


@dataclass(frozen=True)
class _ClusterBundleIndex:
    """Validated indexes shared by clustering, review, and freeze workflows."""

    clusters_by_id: dict[str, dict]
    assignments_by_sample_id: dict[str, dict]
    members_by_cluster: dict[str, set[str]]
    features_by_sample_id: dict[str, dict] | None


def _validate_cluster_bundle(
    *,
    clusters: list[dict],
    assignments: list[dict],
    features: list[dict] | None = None,
) -> _ClusterBundleIndex:
    """Validate and index the structural cluster-artifact joins.

    Annotation/media selection is deliberately outside this helper. Callers
    retain their own coverage, review-state, vocabulary, and provenance rules.
    """

    clusters_by_id = {
        str(row["cluster_id"]): row for row in clusters
    }
    if len(clusters_by_id) != len(clusters):
        raise ValueError("Clusters contain duplicate cluster IDs")

    assignments_by_sample_id = {
        str(row["sample_id"]): row for row in assignments
    }
    if len(assignments_by_sample_id) != len(assignments):
        raise ValueError("Cluster assignments contain duplicate sample IDs")

    features_by_sample_id = None
    if features is not None:
        features_by_sample_id = {
            str(row["sample_id"]): row for row in features
        }
        if len(features_by_sample_id) != len(features):
            raise ValueError("Event features contain duplicate sample IDs")
        if set(features_by_sample_id) != set(assignments_by_sample_id):
            raise ValueError(
                "Feature and cluster-assignment sample IDs do not exactly match"
            )

    members_by_cluster: dict[str, set[str]] = defaultdict(set)
    for sample_id, assignment in assignments_by_sample_id.items():
        cluster_id = str(assignment["cluster_id"])
        if cluster_id not in clusters_by_id:
            raise ValueError(f"Assignment references unknown cluster: {cluster_id}")
        members_by_cluster[cluster_id].add(sample_id)

    for cluster_id, cluster in clusters_by_id.items():
        declared_members = [
            str(value) for value in cluster["member_sample_ids"]
        ]
        if len(declared_members) != len(set(declared_members)):
            raise ValueError(
                f"Cluster contains duplicate member sample IDs: {cluster_id}"
            )
        if (
            "num_members" in cluster
            and int(cluster["num_members"]) != len(declared_members)
        ):
            raise ValueError(f"Cluster num_members mismatch: {cluster_id}")
        if set(declared_members) != members_by_cluster[cluster_id]:
            raise ValueError(f"Cluster membership mismatch: {cluster_id}")

    return _ClusterBundleIndex(
        clusters_by_id=clusters_by_id,
        assignments_by_sample_id=assignments_by_sample_id,
        members_by_cluster=dict(members_by_cluster),
        features_by_sample_id=features_by_sample_id,
    )


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _zscore(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    return (matrix - mean) / std


def _normalize_zscored_block(matrix: np.ndarray) -> np.ndarray:
    """Z-score dimensions, then give the task-level block unit RMS norm."""
    standardized = _zscore(matrix)
    block_rms = float(
        np.sqrt(np.mean(np.sum(np.square(standardized), axis=1)))
    )
    if block_rms <= 1e-12:
        return standardized
    return standardized / block_rms


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - np.clip(a @ b.T, -1.0, 1.0)


EXEMPLAR_SELECTION_LEGACY = "centroid"
EXEMPLAR_SELECTION_CENTROID_DIVERSITY = "centroid3_maximin2"
_CENTROID_DIVERSITY_EPISODE_POLICIES = {"error", "fallback"}


def _stable_exemplar_key(record: dict) -> tuple[int, int, int, str]:
    """Return the content-based tie-break key for opt-in exemplar selection."""

    return (
        int(record["episode_num"]),
        int(record.get("waypoint_step", -1)),
        int(record.get("waypoint_rank", -1)),
        str(record["sample_id"]),
    )


def select_centroid_diversity_exemplars(
    member_records: list[dict],
    member_vectors: np.ndarray,
    *,
    insufficient_unique_episodes: str = "error",
) -> list[dict]:
    """Select three centroid-nearest and two maximin-diverse exemplars.

    All five representatives come from different episodes when at least five
    unique episodes are available. ``insufficient_unique_episodes="error"``
    fails closed otherwise. The explicit ``"fallback"`` policy relaxes only
    episode uniqueness: it retains centroid ordering for the first three
    positions and maximin ordering thereafter, returning at most five members.

    Distance ties are resolved by episode, waypoint step/rank, and sample ID,
    so selection does not depend on the input row order.
    """

    if (
        insufficient_unique_episodes
        not in _CENTROID_DIVERSITY_EPISODE_POLICIES
    ):
        raise ValueError(
            "insufficient_unique_episodes must be 'error' or 'fallback'"
        )
    if not member_records:
        raise ValueError("member_records must not be empty")

    vectors = np.asarray(member_vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(member_records):
        raise ValueError(
            "member_vectors must be a 2D matrix aligned with member_records"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("member_vectors must contain only finite values")

    sample_ids = [str(record["sample_id"]) for record in member_records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            "centroid3_maximin2 requires unique member sample_id values"
        )
    episode_nums = [int(record["episode_num"]) for record in member_records]
    unique_episode_count = len(set(episode_nums))
    if (
        insufficient_unique_episodes == "error"
        and unique_episode_count < 5
    ):
        raise ValueError(
            "centroid3_maximin2 requires at least 5 unique episodes; "
            f"found {unique_episode_count}"
        )

    normalized_vectors = _l2_normalize_rows(vectors)
    centroid = _l2_normalize_rows(
        normalized_vectors.mean(axis=0, keepdims=True)
    )
    centroid_distances = _cosine_distance(
        normalized_vectors,
        centroid,
    ).reshape(-1)
    centroid_order = sorted(
        range(len(member_records)),
        key=lambda idx: (
            float(centroid_distances[idx]),
            *_stable_exemplar_key(member_records[idx]),
        ),
    )

    selected_indices: list[int] = []
    selected_set: set[int] = set()
    used_episodes: set[int] = set()

    for idx in centroid_order:
        episode_num = episode_nums[idx]
        if episode_num in used_episodes:
            continue
        selected_indices.append(idx)
        selected_set.add(idx)
        used_episodes.add(episode_num)
        if len(selected_indices) == min(3, len(member_records)):
            break

    if (
        len(selected_indices) < min(3, len(member_records))
        and insufficient_unique_episodes == "fallback"
    ):
        for idx in centroid_order:
            if idx in selected_set:
                continue
            selected_indices.append(idx)
            selected_set.add(idx)
            used_episodes.add(episode_nums[idx])
            if len(selected_indices) == min(3, len(member_records)):
                break

    target_count = min(5, len(member_records))
    while len(selected_indices) < target_count:
        candidate_indices = [
            idx
            for idx in range(len(member_records))
            if (
                idx not in selected_set
                and episode_nums[idx] not in used_episodes
            )
        ]
        if not candidate_indices:
            if insufficient_unique_episodes == "error":
                raise RuntimeError(
                    "centroid3_maximin2 could not satisfy unique episodes"
                )
            candidate_indices = [
                idx
                for idx in range(len(member_records))
                if idx not in selected_set
            ]

        def diversity_order_key(
            idx: int,
        ) -> tuple[float, float, int, int, int, str]:
            distances_to_selected = _cosine_distance(
                normalized_vectors[idx : idx + 1],
                normalized_vectors[selected_indices],
            )
            maximin_score = float(np.min(distances_to_selected))
            return (
                -maximin_score,
                -float(centroid_distances[idx]),
                *_stable_exemplar_key(member_records[idx]),
            )

        selected_idx = min(candidate_indices, key=diversity_order_key)
        selected_indices.append(selected_idx)
        selected_set.add(selected_idx)
        used_episodes.add(episode_nums[selected_idx])

    return [member_records[idx] for idx in selected_indices]


def build_task_vectors(
    task_records: list[dict],
    *,
    vision_weight: float = 1.0,
    state_weight: float = 0.5,
    progress_weight: float = 0.4,
    block_normalization: str = "legacy",
) -> np.ndarray:
    """Concatenate weighted normalized [vision, state, progress] per record, then L2-normalize rows."""
    vision = np.asarray([record["vision_embedding"] for record in task_records], dtype=np.float32)
    state = np.asarray([record["state_vector"] for record in task_records], dtype=np.float32)
    progress = np.asarray([[record["progress_percent"]] for record in task_records], dtype=np.float32)

    vision_norm = _l2_normalize_rows(vision)
    if block_normalization == "legacy":
        state_norm = _l2_normalize_rows(_zscore(state))
        progress_norm = _zscore(progress)
    elif block_normalization == "balanced":
        state_norm = _normalize_zscored_block(state)
        progress_norm = _normalize_zscored_block(progress)
    else:
        raise ValueError(f"Unknown block_normalization={block_normalization!r}")

    combined = np.concatenate(
        [
            vision_weight * vision_norm,
            state_weight * state_norm,
            progress_weight * progress_norm,
        ],
        axis=1,
    )
    return _l2_normalize_rows(combined)


def select_exemplars(
    member_records: list[dict],
    member_vectors: np.ndarray,
    *,
    num_exemplars: int,
    strategy: str = EXEMPLAR_SELECTION_LEGACY,
    insufficient_unique_episodes: str = "error",
) -> list[dict]:
    """Select up to `num_exemplars` cluster members closest to the centroid,
    preferring unique source episodes.

    The default ``centroid`` strategy is the historical implementation.
    ``centroid3_maximin2`` is an opt-in five-representative strategy.
    """
    if strategy == EXEMPLAR_SELECTION_CENTROID_DIVERSITY:
        if num_exemplars != 5:
            raise ValueError(
                "centroid3_maximin2 requires num_exemplars=5"
            )
        return select_centroid_diversity_exemplars(
            member_records,
            member_vectors,
            insufficient_unique_episodes=insufficient_unique_episodes,
        )
    if strategy != EXEMPLAR_SELECTION_LEGACY:
        raise ValueError(f"Unknown exemplar selection strategy: {strategy!r}")

    centroid = _l2_normalize_rows(member_vectors.mean(axis=0, keepdims=True))
    distances = _cosine_distance(member_vectors, centroid).reshape(-1)
    order = np.argsort(distances)

    exemplars: list[dict] = []
    used_episodes: set[int] = set()
    for idx in order:
        record = member_records[int(idx)]
        episode_num = int(record["episode_num"])
        if episode_num in used_episodes:
            continue
        exemplars.append(record)
        used_episodes.add(episode_num)
        if len(exemplars) >= num_exemplars:
            return exemplars
    for idx in order:
        record = member_records[int(idx)]
        if record in exemplars:
            continue
        exemplars.append(record)
        if len(exemplars) >= num_exemplars:
            break
    return exemplars


def cluster_events(
    event_features_path: Path,
    output_dir: Path,
    *,
    vision_weight: float = 1.0,
    state_weight: float = 0.5,
    progress_weight: float = 0.4,
    block_normalization: str = "legacy",
    distance_threshold: float = 0.18,
    min_coverage: float = 0.5,
    num_exemplars: int = 5,
    expected_samples: int | None = None,
) -> dict:
    """Cluster event features task-locally and write assignments + summaries.

    Outputs (under output_dir):
      - cluster_assignments.jsonl  (one row per sample -> cluster_id)
      - clusters.jsonl             (one row per cluster + exemplars + coverage)
      - summary.json
    """
    event_features_path = Path(event_features_path).resolve()
    if not event_features_path.is_file():
        raise FileNotFoundError(f"event_features.jsonl not found: {event_features_path}")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty cluster directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(event_features_path)
    if expected_samples is not None and len(records) != expected_samples:
        raise ValueError(f"Expected {expected_samples} event features, found {len(records)}")
    sample_ids = [str(record["sample_id"]) for record in records]
    if not records or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Event features must be non-empty with unique sample_id values")
    by_task: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_task[record["task_description"]].append(record)

    assignments: list[dict] = []
    cluster_summaries: list[dict] = []
    for task_description, task_records in sorted(by_task.items()):
        task_records = sorted(
            task_records,
            key=lambda item: (
                int(item["episode_num"]),
                int(item["waypoint_step"]),
                int(item["waypoint_rank"]),
            ),
        )
        task_vectors = build_task_vectors(
            task_records,
            vision_weight=vision_weight,
            state_weight=state_weight,
            progress_weight=progress_weight,
            block_normalization=block_normalization,
        )
        if not np.isfinite(task_vectors).all():
            raise ValueError(f"Non-finite task vectors for task={task_description!r}")
        if len(task_records) == 1:
            labels = np.asarray([0], dtype=np.int32)
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=distance_threshold,
            )
            labels = clustering.fit_predict(task_vectors)

        task_slug = _slugify(task_description)
        total_episodes = len({int(record["episode_num"]) for record in task_records})
        member_ids_by_label: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            member_ids_by_label[int(label)].append(idx)

        cluster_id_by_label: dict[int, str] = {}
        for local_cluster_idx, label in enumerate(sorted(member_ids_by_label)):
            member_indices = member_ids_by_label[label]
            member_records = [task_records[idx] for idx in member_indices]
            member_vectors = task_vectors[member_indices]
            exemplars = select_exemplars(member_records, member_vectors, num_exemplars=num_exemplars)
            episode_nums = sorted({int(record["episode_num"]) for record in member_records})
            coverage = len(episode_nums) / max(total_episodes, 1)
            anchor_source_counts: dict[str, int] = defaultdict(int)
            for record in member_records:
                source = record.get("anchor_source")
                if source is not None:
                    anchor_source_counts[str(source)] += 1
            cluster_id = f"{task_slug}_cluster_{local_cluster_idx:02d}"
            cluster_id_by_label[int(label)] = cluster_id
            cluster_summaries.append(
                {
                    "cluster_id": cluster_id,
                    "task_description": task_description,
                    "cluster_label": int(label),
                    "num_members": len(member_records),
                    "total_task_episodes": int(total_episodes),
                    "episode_coverage": float(coverage),
                    "is_canonical": bool(coverage >= min_coverage),
                    "meets_min_coverage": bool(coverage >= min_coverage),
                    "member_sample_ids": [record["sample_id"] for record in member_records],
                    "member_episode_nums": episode_nums,
                    "anchor_source_counts": dict(sorted(anchor_source_counts.items())),
                    "representative_sample_ids": [record["sample_id"] for record in exemplars],
                    "representative_clip_paths": [record["clip_path"] for record in exemplars],
                    "representative_frame_paths": [record["frame_paths"] for record in exemplars],
                    "representative_waypoint_steps": [int(record["waypoint_step"]) for record in exemplars],
                    "representative_progress_percents": [
                        float(record["progress_percent"]) for record in exemplars
                    ],
                    "cluster_mean_progress_percent": float(
                        np.mean([record["progress_percent"] for record in member_records])
                    ),
                }
            )

        for record, label in zip(task_records, labels, strict=True):
            assignments.append(
                {
                    "sample_id": record["sample_id"],
                    "task_description": task_description,
                    "task_id": int(record["task_id"]),
                    "task_episode_idx": int(record["task_episode_idx"]),
                    "episode_num": int(record["episode_num"]),
                    "waypoint_rank": int(record["waypoint_rank"]),
                    "waypoint_step": int(record["waypoint_step"]),
                    "progress_percent": float(record["progress_percent"]),
                    "num_steps": int(record["num_steps"]),
                    "cell_id": record.get("cell_id"),
                    "success": record.get("success"),
                    "anchor_source": record.get("anchor_source"),
                    "boundary_shift_category": record.get("boundary_shift_category"),
                    "cluster_label": int(label),
                    "cluster_id": cluster_id_by_label[int(label)],
                }
            )

    assigned_ids = [str(record["sample_id"]) for record in assignments]
    if len(assignments) != len(records) or set(assigned_ids) != set(sample_ids):
        raise RuntimeError("Cluster assignments do not exactly cover event features")
    write_jsonl(output_dir / "cluster_assignments.jsonl", assignments)
    write_jsonl(output_dir / "clusters.jsonl", cluster_summaries)

    cluster_sizes = [int(cluster["num_members"]) for cluster in cluster_summaries]
    task_summaries = {}
    for task_description in sorted(by_task):
        task_clusters = [
            cluster
            for cluster in cluster_summaries
            if cluster["task_description"] == task_description
        ]
        task_sizes = [int(cluster["num_members"]) for cluster in task_clusters]
        task_summaries[task_description] = {
            "num_events": len(by_task[task_description]),
            "num_clusters": len(task_clusters),
            "num_singleton_clusters": sum(size == 1 for size in task_sizes),
            "median_cluster_size": float(np.median(task_sizes)),
            "num_clusters_meeting_min_coverage": sum(
                bool(cluster["meets_min_coverage"]) for cluster in task_clusters
            ),
        }
    summary = {
        "event_features_path": str(event_features_path),
        "num_events": len(records),
        "num_tasks": len(by_task),
        "num_clusters": len(cluster_summaries),
        "num_singleton_clusters": sum(size == 1 for size in cluster_sizes),
        "singleton_event_fraction": float(
            sum(size == 1 for size in cluster_sizes) / len(records)
        ),
        "median_cluster_size": float(np.median(cluster_sizes)),
        "max_cluster_size": max(cluster_sizes),
        "num_clusters_meeting_min_coverage": sum(
            bool(cluster["meets_min_coverage"]) for cluster in cluster_summaries
        ),
        "vision_weight": float(vision_weight),
        "state_weight": float(state_weight),
        "progress_weight": float(progress_weight),
        "block_normalization": block_normalization,
        "distance_threshold": float(distance_threshold),
        "min_coverage": float(min_coverage),
        "num_exemplars": int(num_exemplars),
        "task_summaries": task_summaries,
        "passed": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


CLUSTER_SWEEP_CONFIGS = {
    "c0": {
        "vision_weight": 1.0,
        "state_weight": 0.5,
        "progress_weight": 0.4,
    },
    "c1": {
        "vision_weight": 1.0,
        "state_weight": 0.5,
        "progress_weight": 0.0,
    },
    "c2": {
        "vision_weight": 1.0,
        "state_weight": 0.0,
        "progress_weight": 0.0,
    },
}


def _threshold_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def run_clustering_sweep(
    event_features_path: Path,
    output_root: Path,
    *,
    config_ids: list[str],
    thresholds: list[float],
    expected_samples: int | None = None,
    min_coverage: float = 0.5,
    num_exemplars: int = 5,
    block_normalization: str = "legacy",
) -> dict:
    """Run the fixed Stage 3 C0/C1/C2 clustering sensitivity sweep."""

    event_features_path = Path(event_features_path).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty sweep directory: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    feature_records = load_jsonl(event_features_path)
    sample_ids = sorted(
        str(record["sample_id"]) for record in feature_records
    )
    if (
        expected_samples is not None
        and len(feature_records) != expected_samples
    ):
        raise ValueError(
            f"Expected {expected_samples} event features, "
            f"found {len(feature_records)}"
        )
    if not feature_records or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            "Event features must be non-empty with unique sample_id values"
        )

    unknown_configs = sorted(
        set(config_ids).difference(CLUSTER_SWEEP_CONFIGS)
    )
    if unknown_configs:
        raise ValueError(f"Unknown clustering configs: {unknown_configs}")
    thresholds = sorted(set(float(value) for value in thresholds))
    if not thresholds or any(value <= 0.0 for value in thresholds):
        raise ValueError(
            f"Distance thresholds must be positive: {thresholds}"
        )

    run_summaries = []
    for config_id in config_ids:
        previous_assignments = None
        weights = CLUSTER_SWEEP_CONFIGS[config_id]
        for threshold in thresholds:
            run_id = f"{config_id}_d{_threshold_tag(threshold)}"
            run_dir = output_root / run_id
            summary = cluster_events(
                event_features_path=event_features_path,
                output_dir=run_dir,
                distance_threshold=threshold,
                min_coverage=min_coverage,
                num_exemplars=num_exemplars,
                expected_samples=expected_samples,
                block_normalization=block_normalization,
                **weights,
            )
            assignments = {
                str(record["sample_id"]): str(record["cluster_id"])
                for record in load_jsonl(
                    run_dir / "cluster_assignments.jsonl"
                )
            }
            if sorted(assignments) != sample_ids:
                raise RuntimeError(
                    f"Assignment coverage mismatch for run_id={run_id}"
                )
            adjacent_ari = None
            if previous_assignments is not None:
                adjacent_ari = float(
                    adjusted_rand_score(
                        [
                            previous_assignments[sample_id]
                            for sample_id in sample_ids
                        ],
                        [
                            assignments[sample_id]
                            for sample_id in sample_ids
                        ],
                    )
                )
            summary.update(
                {
                    "run_id": run_id,
                    "config_id": config_id,
                    "run_dir": str(run_dir),
                    "adjacent_threshold_ari": adjacent_ari,
                }
            )
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            run_summaries.append(summary)
            previous_assignments = assignments

    report = {
        "format": "event_sae_cluster_sweep_v1",
        "event_features_path": str(event_features_path),
        "event_features_sha256": _sha256(event_features_path),
        "num_events": len(feature_records),
        "config_ids": config_ids,
        "thresholds": thresholds,
        "min_coverage": float(min_coverage),
        "num_exemplars": int(num_exemplars),
        "block_normalization": block_normalization,
        "success_used_for_fitting_or_selection": False,
        "runs": run_summaries,
        "passed": (
            len(run_summaries) == len(config_ids) * len(thresholds)
        ),
    }
    (output_root / "sweep_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def build_phase_groups(
    *,
    clusters_path: Path,
    assignments_path: Path,
    finalized_annotations_path: Path,
    output_dir: Path,
    require_human_review: bool = True,
) -> dict:
    """Union reviewed clusters that share ``(task_description, phase)``."""
    input_paths = {
        "clusters": Path(clusters_path).resolve(),
        "assignments": Path(assignments_path).resolve(),
        "finalized_annotations": Path(finalized_annotations_path).resolve(),
    }
    for label, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty phase-group directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = load_jsonl(input_paths["clusters"])
    assignments = load_jsonl(input_paths["assignments"])
    annotations = load_jsonl(input_paths["finalized_annotations"])
    bundle = _validate_cluster_bundle(
        clusters=clusters,
        assignments=assignments,
    )
    clusters_by_id = bundle.clusters_by_id
    assignment_by_sample_id = bundle.assignments_by_sample_id
    annotations_by_id = _unique_index(
        annotations,
        "cluster_id",
        "annotations",
    )
    unknown_annotations = sorted(set(annotations_by_id).difference(clusters_by_id))
    if unknown_annotations:
        raise ValueError(
            f"Annotations reference unknown clusters: {unknown_annotations[:10]}"
        )

    grouped_cluster_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    for cluster_id, annotation in annotations_by_id.items():
        if (
            annotation.get("api_error") is not None
            or annotation.get("parse_error") is not None
        ):
            raise ValueError(f"Unresolved model annotation: {cluster_id}")
        phase = str(annotation.get("phase", "")).strip()
        if not phase:
            raise ValueError(f"Annotation has empty phase: {cluster_id}")
        allowed = {
            str(value) for value in annotation.get("allowed_phase_labels", [])
        }
        if allowed and phase not in allowed:
            raise ValueError(f"Phase outside annotation vocabulary: {cluster_id}")
        if require_human_review:
            if annotation.get("review_verdict") not in {"approved", "corrected"}:
                raise ValueError(
                    f"Cluster is not resolved by human review: {cluster_id}"
                )
            if annotation.get("actual_human_review_completed") is not True:
                raise ValueError(f"Cluster lacks actual human review: {cluster_id}")

        cluster = clusters_by_id[cluster_id]
        task_description = str(annotation["task_description"])
        if task_description != str(cluster["task_description"]):
            raise ValueError(f"Task mismatch for cluster: {cluster_id}")
        grouped_cluster_ids[(task_description, phase)].append(cluster_id)

    phase_groups = []
    phase_group_assignments = []
    used_sample_ids: set[str] = set()
    for (task_description, phase), cluster_ids in sorted(
        grouped_cluster_ids.items()
    ):
        cluster_ids = sorted(cluster_ids)
        task_slug = _slugify(task_description)
        phase_group_id = f"{task_slug}_phase_{_slugify(phase)}"
        cluster_rows = [clusters_by_id[cluster_id] for cluster_id in cluster_ids]
        annotation_rows = [
            annotations_by_id[cluster_id] for cluster_id in cluster_ids
        ]
        member_sample_ids = sorted(
            {
                str(sample_id)
                for cluster in cluster_rows
                for sample_id in cluster["member_sample_ids"]
            }
        )
        representative_sample_ids = list(
            dict.fromkeys(
                str(sample_id)
                for cluster in cluster_rows
                for sample_id in cluster["representative_sample_ids"]
            )
        )
        member_episode_nums = sorted(
            {
                int(assignment_by_sample_id[sample_id]["episode_num"])
                for sample_id in member_sample_ids
            }
        )
        total_episode_values = {
            int(cluster["total_task_episodes"]) for cluster in cluster_rows
        }
        if len(total_episode_values) != 1:
            raise ValueError(
                f"Inconsistent task episode totals for phase group {phase_group_id}"
            )
        total_task_episodes = next(iter(total_episode_values))
        episode_coverage = len(member_episode_nums) / max(total_task_episodes, 1)
        phrases = list(
            dict.fromkeys(
                str(row["phrase"]).strip() for row in annotation_rows
            )
        )
        allowed_phase_labels = list(
            annotation_rows[0].get("allowed_phase_labels", [])
        )
        if any(
            list(row.get("allowed_phase_labels", [])) != allowed_phase_labels
            for row in annotation_rows[1:]
        ):
            raise ValueError(
                f"Inconsistent phase vocabulary in group {phase_group_id}"
            )
        source_review_modes = sorted(
            {
                str(row.get("review_mode"))
                for row in annotation_rows
                if row.get("review_mode")
            }
        )
        actual_human_review_completed = all(
            row.get("actual_human_review_completed") is True
            for row in annotation_rows
        )
        if actual_human_review_completed:
            phase_group_review_mode = (
                "phase_group_of_human_reviewed_clusters"
            )
            phase_group_review_verdict = "approved"
        else:
            phase_group_review_mode = (
                "phase_group_of_user_authorized_assumed_review_clusters"
            )
            phase_group_review_verdict = "assumed_approved"

        phase_groups.append(
            {
                "format": "event_sae_phase_group_v1",
                "cluster_id": phase_group_id,
                "phase_group_id": phase_group_id,
                "task_description": task_description,
                "phase": phase,
                "phrase": f"{phase} phase group",
                "source_phrases": phrases,
                "source_cluster_ids": cluster_ids,
                "num_source_clusters": len(cluster_ids),
                "member_sample_ids": member_sample_ids,
                "num_members": len(member_sample_ids),
                "member_episode_nums": member_episode_nums,
                "total_task_episodes": total_task_episodes,
                "episode_coverage": float(episode_coverage),
                "representative_sample_ids": representative_sample_ids,
                "representative_clip_paths": list(
                    dict.fromkeys(
                        str(path)
                        for row in annotation_rows
                        for path in row.get("representative_clip_paths", [])
                    )
                ),
                "representative_frame_paths": [
                    group
                    for row in annotation_rows
                    for group in row.get("representative_frame_paths", [])
                ],
                "representative_progress_percents": [
                    float(value)
                    for row in annotation_rows
                    for value in row.get("representative_progress_percents", [])
                ],
                "allowed_phase_labels": allowed_phase_labels,
                "phase_scheme": annotation_rows[0].get("phase_scheme"),
                "model": "reviewed_cluster_union",
                "prompt_version": annotation_rows[0].get("prompt_version"),
                "review_mode": phase_group_review_mode,
                "source_review_modes": source_review_modes,
                "review_verdict": phase_group_review_verdict,
                "actual_human_review_completed": (
                    actual_human_review_completed
                ),
                "api_error": None,
                "parse_error": None,
            }
        )

        for sample_id in member_sample_ids:
            if sample_id in used_sample_ids:
                raise RuntimeError(
                    f"Sample assigned to multiple phase groups: {sample_id}"
                )
            used_sample_ids.add(sample_id)
            source_assignment = assignment_by_sample_id[sample_id]
            phase_group_assignments.append(
                {
                    **source_assignment,
                    "source_cluster_id": str(source_assignment["cluster_id"]),
                    "cluster_id": phase_group_id,
                    "phase_group_id": phase_group_id,
                    "phase": phase,
                }
            )

    phase_groups_path = output_dir / "phase_groups.jsonl"
    phase_assignments_path = output_dir / "phase_group_assignments.jsonl"
    write_jsonl(phase_groups_path, phase_groups)
    write_jsonl(phase_assignments_path, phase_group_assignments)
    summary = {
        "format": "event_sae_phase_groups_v1",
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in input_paths.items()
        },
        "outputs": {
            "phase_groups_path": str(phase_groups_path),
            "phase_groups_sha256": _sha256(phase_groups_path),
            "phase_group_assignments_path": str(phase_assignments_path),
            "phase_group_assignments_sha256": _sha256(phase_assignments_path),
        },
        "grouping_scope": "(task_description, reviewed_phase)",
        "cross_instruction_merge": False,
        "require_human_review": bool(require_human_review),
        "num_input_clusters": len(clusters),
        "num_reviewed_clusters": len(annotations),
        "num_phase_groups": len(phase_groups),
        "num_grouped_samples": len(phase_group_assignments),
        "review_modes": sorted(
            {str(row["review_mode"]) for row in phase_groups}
        ),
        "actual_human_review_completed": all(
            row["actual_human_review_completed"] is True
            for row in phase_groups
        ),
        "result_status": (
            "canonical_human_reviewed"
            if all(
                row["actual_human_review_completed"] is True
                for row in phase_groups
            )
            else "provisional_automatic"
        ),
        "passed": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


FORBIDDEN_ANNOTATION_FIELDS = {
    "success",
    "anchor_source",
    "qpos",
    "gripper_qpos",
    "oracle_event",
    "simulator_predicate",
}


def audit_and_freeze_annotation_bundle(
    *,
    event_features_path: Path,
    assignments_path: Path,
    clusters_path: Path,
    media_clusters_path: Path,
    annotations_path: Path,
    output_annotations_path: Path,
    audit_path: Path,
    expected_events: int = 1278,
    expected_annotation_clusters: int = 20,
    min_episode_coverage: float = 0.3,
) -> dict[str, Any]:
    """Audit feature/cluster/media joins and freeze an annotation bundle."""
    inputs = {
        "event_features": Path(event_features_path).resolve(),
        "assignments": Path(assignments_path).resolve(),
        "clusters": Path(clusters_path).resolve(),
        "media_clusters": Path(media_clusters_path).resolve(),
        "annotations": Path(annotations_path).resolve(),
    }
    output_annotations_path = Path(output_annotations_path).resolve()
    audit_path = Path(audit_path).resolve()
    for label, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    for path in (output_annotations_path, audit_path):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite annotation audit output: {path}"
            )

    features = load_jsonl(inputs["event_features"])
    assignments = load_jsonl(inputs["assignments"])
    clusters = load_jsonl(inputs["clusters"])
    media_clusters = load_jsonl(inputs["media_clusters"])
    annotations = load_jsonl(inputs["annotations"])
    bundle = _validate_cluster_bundle(
        clusters=clusters,
        assignments=assignments,
        features=features,
    )
    features_by_id = bundle.features_by_sample_id
    assert features_by_id is not None
    assignments_by_id = bundle.assignments_by_sample_id
    clusters_by_id = bundle.clusters_by_id
    members_by_cluster = bundle.members_by_cluster
    media_by_id = _unique_index(media_clusters, "cluster_id", "media clusters")
    annotations_by_id = _unique_index(
        annotations,
        "cluster_id",
        "annotations",
    )

    if len(features) != expected_events:
        raise ValueError(
            f"Expected {expected_events} features, found {len(features)}"
        )

    selected_ids = {
        cluster_id
        for cluster_id, cluster in clusters_by_id.items()
        if float(cluster["episode_coverage"]) >= min_episode_coverage
    }
    if len(selected_ids) != expected_annotation_clusters:
        raise ValueError(
            f"Expected {expected_annotation_clusters} selected clusters, "
            f"found {len(selected_ids)}"
        )
    if set(media_by_id) != selected_ids:
        raise ValueError("Triptych media IDs do not exactly match selected clusters")
    if set(annotations_by_id) != selected_ids:
        raise ValueError("Annotation IDs do not exactly match selected clusters")

    phase_cluster_counts: Counter[str] = Counter()
    phase_event_counts: Counter[str] = Counter()
    phase_anchor_event_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts_all: Counter[str] = Counter()
    source_counts_selected: Counter[str] = Counter()
    task_phase_cluster_counts: dict[str, Counter[str]] = defaultdict(Counter)
    annotation_prompt_versions: set[str] = set()
    annotation_layouts: set[str] = set()

    for assignment in assignments:
        source_counts_all[str(assignment.get("anchor_source"))] += 1

    for cluster_id in sorted(selected_ids):
        cluster = clusters_by_id[cluster_id]
        media = media_by_id[cluster_id]
        annotation = annotations_by_id[cluster_id]
        if annotation.get("api_error") is not None:
            raise ValueError(f"Annotation API error: {cluster_id}")
        if annotation.get("parse_error") is not None:
            raise ValueError(f"Annotation parse error: {cluster_id}")
        phase = str(annotation.get("phase", ""))
        phrase = str(annotation.get("phrase", "")).strip()
        allowed = {str(value) for value in annotation["allowed_phase_labels"]}
        if not phrase or phase not in allowed:
            raise ValueError(f"Invalid phase or phrase: {cluster_id}")
        leaked = sorted(FORBIDDEN_ANNOTATION_FIELDS.intersection(annotation))
        if leaked:
            raise ValueError(f"Forbidden annotation fields in {cluster_id}: {leaked}")
        if str(annotation["task_description"]) != str(cluster["task_description"]):
            raise ValueError(f"Task mismatch: {cluster_id}")
        representative_ids = [
            str(value) for value in cluster["representative_sample_ids"]
        ]
        if representative_ids != [
            str(value) for value in media["representative_sample_ids"]
        ]:
            raise ValueError(f"Media representative mismatch: {cluster_id}")
        if representative_ids != [
            str(value) for value in annotation["representative_sample_ids"]
        ]:
            raise ValueError(f"Annotation representative mismatch: {cluster_id}")
        if len(media["representative_frame_paths"]) != len(representative_ids):
            raise ValueError(f"Media sequence count mismatch: {cluster_id}")

        phase_cluster_counts[phase] += 1
        task_phase_cluster_counts[str(cluster["task_description"])][phase] += 1
        annotation_prompt_versions.add(str(annotation["prompt_version"]))
        annotation_layouts.add(str(annotation["annotation_media_layout"]))
        for sample_id in members_by_cluster[cluster_id]:
            source = str(assignments_by_id[sample_id].get("anchor_source"))
            phase_event_counts[phase] += 1
            phase_anchor_event_counts[phase][source] += 1
            source_counts_selected[source] += 1

    output_annotations_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(inputs["annotations"], output_annotations_path)
    report = {
        "format": "event_sae_v9_stage3_exact_join_audit_v1",
        "status": "canonical_unreviewed",
        "human_review_required": True,
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in inputs.items()
        },
        "canonical_annotations": {
            "path": str(output_annotations_path),
            "sha256": _sha256(output_annotations_path),
            "byte_identical_to_attempt": (
                _sha256(output_annotations_path) == _sha256(inputs["annotations"])
            ),
        },
        "contract": {
            "expected_events": expected_events,
            "min_episode_coverage": float(min_episode_coverage),
            "expected_annotation_clusters": expected_annotation_clusters,
            "success_used_for_clustering_or_selection": False,
            "raw_clusters_preserved": True,
            "forbidden_annotation_fields_absent": True,
        },
        "counts": {
            "events": len(features),
            "assignments": len(assignments),
            "raw_clusters": len(clusters),
            "selected_clusters": len(selected_ids),
            "selected_cluster_events": sum(source_counts_selected.values()),
            "unselected_cluster_events": (
                len(features) - sum(source_counts_selected.values())
            ),
            "triptych_clusters": len(media_clusters),
            "annotations": len(annotations),
            "annotation_api_errors": 0,
            "annotation_parse_errors": 0,
        },
        "source_counts_all_events": dict(sorted(source_counts_all.items())),
        "source_counts_selected_cluster_events": dict(
            sorted(source_counts_selected.items())
        ),
        "phase_cluster_counts": dict(sorted(phase_cluster_counts.items())),
        "phase_event_counts": dict(sorted(phase_event_counts.items())),
        "phase_anchor_source_event_counts": {
            phase: dict(sorted(counts.items()))
            for phase, counts in sorted(phase_anchor_event_counts.items())
        },
        "task_phase_cluster_counts": {
            task: dict(sorted(counts.items()))
            for task, counts in sorted(task_phase_cluster_counts.items())
        },
        "prompt_versions": sorted(annotation_prompt_versions),
        "annotation_media_layouts": sorted(annotation_layouts),
        "passed": True,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
