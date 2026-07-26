"""Read-only controlled-ablation result dataset, service, and HTTP adapter.

Historical ``stage4`` artifact paths, payload formats, and HTTP routes are kept
as serialized compatibility contracts; active Python names describe their
phase-feature meaning.
"""

from __future__ import annotations

import json
import math
import mimetypes
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from event_sae import resolve_groot_artifact_path
from event_sae.events.io import load_jsonl
from event_sae.events.review import build_blind_cluster_review_dataset
from event_sae.groot.oracle_results import (
    ORACLE_PHASE_VIEWS,
    build_oracle_waiting_payload,
    build_oracle_phase_results_dataset,
)
from event_sae.groot.phase_feature_results import (
    RESULT_RANKINGS,
    build_phase_feature_heatmap,
    build_phase_feature_overview,
    compact_ranking_row,
    decorate_score_task_identities,
    format_checkpoint_label,
    load_controlled_task_identity_registry,
    load_phase_feature_score_matrix,
)


CONDITION_LABELS = {
    "e0_rel_pos_cluster_left_label_left": (
        "Relative position · left clustering · left labels"
    ),
    "e1_rel_pos_cluster_left_label_multiview": (
        "Relative position · left clustering · 3-view labels"
    ),
    "e2_rel_pos_cluster_multiview_label_multiview": (
        "Relative position · 3-view clustering · 3-view labels"
    ),
    "e3_rel_pos_gripper_cluster_multiview_label_multiview": (
        "Relative position + gripper · 3-view clustering"
    ),
    "e4_abs_pos_gripper_cluster_multiview_label_multiview": (
        "Absolute position + gripper · 3-view clustering"
    ),
}


COMPARISON_LABELS = {
    "q1_annotation_view": "Annotation view · left vs 3-view",
    "q2_clustering_view": "Clustering view · left vs 3-view",
    "q3_gripper_anchor": "Anchor signal · position vs position + gripper",
    "q4_relative_absolute": "Anchor frame · relative vs absolute",
}


PARTITION_LABELS = {
    "p0_r_pos_left": "Relative position · left view",
    "p1_r_pos_3view": "Relative position · 3-view",
    "p2_r_pos_gripper_3view": "Relative position + gripper · 3-view",
    "p3_a_pos_gripper_3view": "Absolute position + gripper · 3-view",
}


PARTITION_COMPARISON_LABELS = {
    "q2_p0_vs_p1_all_r_pos": "Clustering view · left vs 3-view",
    "q3_p1_vs_p2_common_position": "Anchor signal · position vs + gripper",
    "q4_p2_vs_p3_common_anchors": "Anchor frame · relative vs absolute",
}


def _result_coverage_value(coverage_id: str) -> float:
    match = re.fullmatch(r"cov(\d+)p(\d+)", coverage_id)
    if match is None:
        raise ValueError(f"Invalid coverage directory ID: {coverage_id}")
    return float(f"{match.group(1)}.{match.group(2)}")


def _condition_explorer_paths(
    experiment_root: Path,
    condition_id: str,
) -> dict[str, Path | str]:
    root = resolve_groot_artifact_path(experiment_root).resolve()
    historical_root = (
        root.parent / "v9_abs_position_gripper_3view_action_phase_v1"
    )
    shared = {
        "e0_rel_pos_cluster_left_label_left": {
            "event_features": root / "features/r_pos/event_features_left.jsonl",
            "partition": root / "partitions/p0_r_pos_left",
            "media_clusters": (
                root / "annotation_media/p0_left/clusters_left.jsonl"
            ),
            "media_layout": "five-frame LEFT sequence",
        },
        "e1_rel_pos_cluster_left_label_multiview": {
            "event_features": root / "features/r_pos/event_features_left.jsonl",
            "partition": root / "partitions/p0_r_pos_left",
            "media_clusters": (
                root / "annotation_media/p0_multiview/clusters_multiview.jsonl"
            ),
            "media_layout": "synchronized LEFT | RIGHT | WRIST triptych",
        },
        "e2_rel_pos_cluster_multiview_label_multiview": {
            "event_features": (
                root / "features/r_pos/event_features_multiview.jsonl"
            ),
            "partition": root / "partitions/p1_r_pos_3view",
            "media_clusters": (
                root / "annotation_media/p1_multiview/clusters_multiview.jsonl"
            ),
            "media_layout": "synchronized LEFT | RIGHT | WRIST triptych",
        },
        "e3_rel_pos_gripper_cluster_multiview_label_multiview": {
            "event_features": (
                root / "features/r_pos_gripper/event_features_multiview.jsonl"
            ),
            "partition": root / "partitions/p2_r_pos_gripper_3view",
            "media_clusters": (
                root / "annotation_media/p2_multiview/clusters_multiview.jsonl"
            ),
            "media_layout": "synchronized LEFT | RIGHT | WRIST triptych",
        },
        "e4_abs_pos_gripper_cluster_multiview_label_multiview": {
            "event_features": (
                historical_root
                / "stage3_features/event_features_multiview_equal_concat_v9.jsonl"
            ),
            "partition": (
                historical_root
                / "stage3_clusters/c0_balanced_v9/c0_d0p18"
            ),
            "media_clusters": (
                root / "annotation_media/p3_multiview/clusters_multiview.jsonl"
            ),
            "media_layout": "synchronized LEFT | RIGHT | WRIST triptych",
        },
    }
    config = shared.get(condition_id)
    if config is None:
        raise KeyError(condition_id)
    partition = Path(config["partition"])
    return {
        "event_features": Path(config["event_features"]),
        "clusters": partition / "clusters.jsonl",
        "assignments": partition / "cluster_assignments.jsonl",
        "annotations": (
            root
            / "annotations"
            / condition_id
            / "provisional_finalized_annotations.jsonl"
        ),
        "media_clusters": Path(config["media_clusters"]),
        "media_layout": str(config["media_layout"]),
    }


def build_condition_cluster_explorer_dataset(
    *,
    experiment_root: Path,
    condition_id: str,
    projection_method: str = "pca",
) -> tuple[dict, dict[str, tuple[Path, ...]]]:
    """Build one read-only point-and-rollout explorer for a result condition."""

    paths = _condition_explorer_paths(experiment_root, condition_id)
    label = CONDITION_LABELS.get(condition_id, condition_id.replace("_", " "))
    payload, media_paths, annotations = build_blind_cluster_review_dataset(
        event_features_path=Path(paths["event_features"]),
        clusters_path=Path(paths["clusters"]),
        assignments_path=Path(paths["assignments"]),
        annotations_path=Path(paths["annotations"]),
        representative_media_clusters_path=Path(paths["media_clusters"]),
        condition_id=condition_id,
        condition_label=label,
        media_layout=str(paths["media_layout"]),
        condition_metadata={"result_explorer": "read_only"},
        projection_method=projection_method,
        validate_media=True,
        block_normalization="balanced",
    )
    for cluster in payload["clusters"]:
        annotation = annotations.get(cluster["cluster_id"])
        if annotation is None:
            continue
        cluster["annotation"] = {
            "phrase": str(annotation["phrase"]),
            "phase": str(annotation["phase"]),
            "review_mode": str(annotation.get("review_mode", "unknown")),
            "actual_human_review_completed": bool(
                annotation.get("actual_human_review_completed", False)
            ),
        }
    payload["format"] = "event_sae_cluster_result_explorer_v1"
    payload["meta"]["read_only"] = True
    return payload, media_paths


def build_experiment_results_dataset(experiment_root: Path) -> dict:
    """Index the complete Stage 4 experiment grid for browser exploration.

    Only compact JSON ranking artifacts are exposed. The larger torch score
    matrices are checked for completeness but are never loaded or sent to the
    browser.
    """

    root = resolve_groot_artifact_path(experiment_root).resolve()
    manifest_path = root / "experiment_manifest.json"
    audit_path = root / "audits/final_audit.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"Final audit not found: {audit_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    scope = audit.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("Final audit is missing its Stage 4 scope")
    conditions = [str(value) for value in scope.get("conditions", [])]
    coverages = [
        f"cov{str(float(value)).replace('.', 'p')}"
        for value in scope.get("coverage_thresholds", [])
    ]
    sae_ids = [
        str(value) for value in scope.get("sae_sensitivity_checkpoints", [])
    ]
    if not conditions or not coverages or not sae_ids:
        raise ValueError("Final audit has an empty Stage 4 experiment axis")

    condition_payload = [
        {
            "id": condition_id,
            "code": f"E{index}",
            "label": CONDITION_LABELS.get(
                condition_id,
                condition_id.replace("_", " "),
            ),
        }
        for index, condition_id in enumerate(conditions)
    ]
    condition_by_id = {
        row["id"]: row for row in condition_payload
    }
    expected_runs = len(conditions) * len(coverages) * len(sae_ids)
    runs: list[dict] = []
    phase_sets: list[dict] = []

    for condition_id in conditions:
        raw_clusters_path = Path(
            _condition_explorer_paths(root, condition_id)["clusters"]
        )
        if not raw_clusters_path.is_file():
            raise FileNotFoundError(
                f"Raw cluster artifact not found: {raw_clusters_path}"
            )
        raw_cluster_rows = load_jsonl(raw_clusters_path)
        raw_clusters_by_id = {
            str(row["cluster_id"]): row for row in raw_cluster_rows
        }
        if len(raw_clusters_by_id) != len(raw_cluster_rows):
            raise ValueError(
                f"Raw cluster artifact contains duplicate IDs: "
                f"{raw_clusters_path}"
            )
        for coverage_id in coverages:
            phase_groups_dir = (
                root / "stage4" / condition_id / coverage_id / "phase_groups"
            )
            phase_groups_path = phase_groups_dir / "phase_groups.jsonl"
            phase_summary_path = phase_groups_dir / "summary.json"
            if not phase_groups_path.is_file():
                raise FileNotFoundError(
                    f"Phase-group artifact not found: {phase_groups_path}"
                )
            if not phase_summary_path.is_file():
                raise FileNotFoundError(
                    f"Phase-group summary not found: {phase_summary_path}"
                )
            raw_phase_groups = load_jsonl(phase_groups_path)
            phase_summary = json.loads(
                phase_summary_path.read_text(encoding="utf-8")
            )
            compact_phase_groups = [
                {
                    "phase_group_id": str(
                        row.get("phase_group_id") or row["cluster_id"]
                    ),
                    "task_description": str(row["task_description"]),
                    "phase": str(row["phase"]),
                    "phrase": str(row["phrase"]),
                    "source_phrases": [
                        str(value)
                        for value in row.get("source_phrases", [])
                    ],
                    "source_cluster_ids": [
                        str(value)
                        for value in row.get("source_cluster_ids", [])
                    ],
                    "num_source_clusters": int(
                        row.get(
                            "num_source_clusters",
                            len(row.get("source_cluster_ids", [])),
                        )
                    ),
                    "num_members": int(row["num_members"]),
                    "episode_coverage": float(row["episode_coverage"]),
                    "total_task_episodes": int(row["total_task_episodes"]),
                    "review_mode": str(
                        row.get("review_mode", "unknown")
                    ),
                    "review_verdict": str(
                        row.get("review_verdict", "unknown")
                    ),
                    "actual_human_review_completed": bool(
                        row.get("actual_human_review_completed", False)
                    ),
                }
                for row in raw_phase_groups
            ]
            declared_input_clusters = int(
                phase_summary.get("num_input_clusters", 0)
            )
            if declared_input_clusters != len(raw_cluster_rows):
                raise ValueError(
                    f"Phase-group input-cluster count does not match raw "
                    f"clusters: {phase_groups_dir}"
                )
            declared_phase_groups = int(
                phase_summary.get(
                    "num_phase_groups",
                    len(compact_phase_groups),
                )
            )
            if declared_phase_groups != len(compact_phase_groups):
                raise ValueError(
                    f"Phase-group summary count mismatch: {phase_groups_dir}"
                )
            phase_keys = [
                (
                    row["task_description"],
                    row["phase_group_id"],
                )
                for row in compact_phase_groups
            ]
            if len(set(phase_keys)) != len(phase_keys):
                raise ValueError(
                    f"Phase groups contain duplicate task/group keys: "
                    f"{phase_groups_path}"
                )
            summary_human_review_completed = bool(
                phase_summary.get("actual_human_review_completed", False)
            )
            if any(
                phase_group["actual_human_review_completed"]
                != summary_human_review_completed
                for phase_group in compact_phase_groups
            ):
                raise ValueError(
                    f"Phase-group human-review status does not match summary: "
                    f"{phase_groups_dir}"
                )
            coverage_threshold = _result_coverage_value(coverage_id)
            expected_source_cluster_ids = {
                str(row["cluster_id"])
                for row in raw_cluster_rows
                if float(row["episode_coverage"]) + 1e-12
                >= coverage_threshold
            }
            filtered_annotations_path = (
                phase_groups_dir.parent / "filtered_finalized_annotations.jsonl"
            )
            if not filtered_annotations_path.is_file():
                raise FileNotFoundError(
                    "Filtered finalized annotations not found: "
                    f"{filtered_annotations_path}"
                )
            filtered_annotation_rows = load_jsonl(filtered_annotations_path)
            filtered_annotations_by_id = {
                str(row["cluster_id"]): row
                for row in filtered_annotation_rows
            }
            if len(filtered_annotations_by_id) != len(filtered_annotation_rows):
                raise ValueError(
                    "Filtered finalized annotations contain duplicate cluster "
                    f"IDs: {filtered_annotations_path}"
                )
            if set(filtered_annotations_by_id) != expected_source_cluster_ids:
                raise ValueError(
                    "Filtered finalized annotation IDs do not exactly match "
                    f"coverage-qualified raw clusters: {filtered_annotations_path}"
                )
            for phase_group in compact_phase_groups:
                source_cluster_ids = phase_group["source_cluster_ids"]
                if (
                    phase_group["num_source_clusters"]
                    != len(source_cluster_ids)
                ):
                    raise ValueError(
                        f"Phase-group source-cluster count mismatch: "
                        f"{phase_groups_path}"
                    )
                if (
                    not source_cluster_ids
                    or len(set(source_cluster_ids)) != len(source_cluster_ids)
                ):
                    raise ValueError(
                        f"Phase group has missing or duplicate source clusters: "
                        f"{phase_groups_path}"
                    )
                for source_cluster_id in source_cluster_ids:
                    raw_cluster = raw_clusters_by_id.get(source_cluster_id)
                    if raw_cluster is None:
                        raise ValueError(
                            f"Phase group references an unknown raw cluster "
                            f"{source_cluster_id}: {phase_groups_path}"
                        )
                    if (
                        str(raw_cluster["task_description"])
                        != phase_group["task_description"]
                    ):
                        raise ValueError(
                            f"Phase group source cluster task mismatch: "
                            f"{phase_groups_path}"
                        )
                    if (
                        float(raw_cluster["episode_coverage"])
                        + 1e-12
                        < coverage_threshold
                    ):
                        raise ValueError(
                            f"Phase group source cluster is below coverage "
                            f"{coverage_id}: {phase_groups_path}"
                        )
                    annotation = filtered_annotations_by_id[source_cluster_id]
                    if (
                        str(annotation["task_description"])
                        != phase_group["task_description"]
                        or str(annotation["phase"]) != phase_group["phase"]
                    ):
                        raise ValueError(
                            "Phase group source annotation task/phase mismatch: "
                            f"{phase_groups_path}"
                        )
                source_annotation_phrases = {
                    str(filtered_annotations_by_id[source_id]["phrase"])
                    for source_id in source_cluster_ids
                }
                if source_annotation_phrases != set(
                    phase_group["source_phrases"]
                ):
                    raise ValueError(
                        "Phase group source phrases do not match filtered "
                        f"annotations: {phase_groups_path}"
                    )
            source_cluster_ids = [
                source_cluster_id
                for phase_group in compact_phase_groups
                for source_cluster_id in phase_group["source_cluster_ids"]
            ]
            if len(set(source_cluster_ids)) != len(source_cluster_ids):
                raise ValueError(
                    f"Raw source cluster appears in multiple phase groups: "
                    f"{phase_groups_path}"
                )
            if len(source_cluster_ids) != int(
                phase_summary.get("num_reviewed_clusters", 0)
            ):
                raise ValueError(
                    f"Phase-group source union does not match reviewed-cluster "
                    f"count: {phase_groups_dir}"
                )
            if set(source_cluster_ids) != expected_source_cluster_ids:
                missing_sources = (
                    expected_source_cluster_ids - set(source_cluster_ids)
                )
                extra_sources = (
                    set(source_cluster_ids) - expected_source_cluster_ids
                )
                raise ValueError(
                    "Phase-group source union does not exactly match "
                    "coverage-qualified raw clusters "
                    f"under {phase_groups_dir}: "
                    f"missing={len(missing_sources)}, "
                    f"extra={len(extra_sources)}"
                )
            expected_phase_keys = set(phase_keys)
            canonical_phase_by_key = {
                (
                    row["task_description"],
                    row["phase_group_id"],
                ): row
                for row in compact_phase_groups
            }
            phase_sets.append(
                {
                    "id": f"{condition_id}-{coverage_id}",
                    "condition_id": condition_id,
                    "coverage_id": coverage_id,
                    "coverage": _result_coverage_value(coverage_id),
                    "result_status": str(
                        phase_summary.get("result_status", "unknown")
                    ),
                    "actual_human_review_completed": bool(
                        summary_human_review_completed
                    ),
                    "require_human_review": bool(
                        phase_summary.get("require_human_review", False)
                    ),
                    "num_input_clusters": declared_input_clusters,
                    "num_reviewed_clusters": int(
                        phase_summary.get("num_reviewed_clusters", 0)
                    ),
                    "num_grouped_samples": int(
                        phase_summary.get("num_grouped_samples", 0)
                    ),
                    "num_phase_groups": len(compact_phase_groups),
                    "phase_groups": compact_phase_groups,
                }
            )
            for sae_id in sae_ids:
                run_dir = root / "stage4" / condition_id / coverage_id / sae_id
                score_path = run_dir / "event_feature_scores.pt"
                rankings_dir = run_dir / "rankings"
                config_path = rankings_dir / "ranking_config.json"
                candidates_path = rankings_dir / "candidates.jsonl"
                ranking_paths = {
                    ranking: rankings_dir / f"{ranking}.jsonl"
                    for ranking in RESULT_RANKINGS
                }
                required_paths = [
                    score_path,
                    config_path,
                    candidates_path,
                    *ranking_paths.values(),
                ]
                missing = [
                    path.name for path in required_paths if not path.is_file()
                ]
                if missing:
                    raise FileNotFoundError(
                        f"Incomplete result run {condition_id}/{coverage_id}/"
                        f"{sae_id}: {', '.join(missing)}"
                    )

                config = json.loads(config_path.read_text(encoding="utf-8"))
                configured_score_path = Path(
                    str(config.get("scores_pt", ""))
                ).expanduser()
                if (
                    not configured_score_path.is_absolute()
                    or configured_score_path.resolve() != score_path.resolve()
                ):
                    raise ValueError(
                        "Ranking score provenance does not match the current "
                        f"checkpoint: {config_path}"
                    )
                configured_topk_dir = Path(
                    str(config.get("topk_run_dir", ""))
                ).expanduser()
                if (
                    not configured_topk_dir.is_absolute()
                    or configured_topk_dir.parent.name != sae_id
                ):
                    raise ValueError(
                        "Ranking Top-K provenance does not match the current "
                        f"checkpoint: {config_path}"
                    )
                top_n_per_row = int(config["top_n_per_row"])
                if top_n_per_row < 5:
                    raise ValueError(
                        "Phase-wise feature verification requires "
                        f"top_n_per_row >= 5: {config_path}"
                    )
                candidate_rows = [
                    compact_ranking_row(row)
                    for row in load_jsonl(candidates_path)
                ]
                candidates = {
                    ranking: [
                        row
                        for row in candidate_rows
                        if row.get("ranking") == ranking
                    ]
                    for ranking in RESULT_RANKINGS
                }
                if any(not rows for rows in candidates.values()):
                    raise ValueError(
                        "Candidate artifact is missing a ranking family: "
                        f"{candidates_path}"
                    )

                rankings = {
                    ranking: [
                        compact_ranking_row(row)
                        for row in load_jsonl(path)
                    ]
                    for ranking, path in ranking_paths.items()
                }
                if any(not rows for rows in rankings.values()):
                    raise ValueError(
                        f"Ranking artifact is empty under {rankings_dir}"
                    )
                expected_top_features = 5
                for ranking in ("event_aligned", "window_mean"):
                    phase_ranking_rows = rankings[ranking]
                    ranking_keys = [
                        (
                            str(row["task_description"]),
                            str(row["cluster_id"]),
                        )
                        for row in phase_ranking_rows
                    ]
                    if len(set(ranking_keys)) != len(ranking_keys):
                        raise ValueError(
                            f"{ranking} contains duplicate phase keys: "
                            f"{ranking_paths[ranking]}"
                        )
                    if set(ranking_keys) != expected_phase_keys:
                        missing_keys = expected_phase_keys - set(ranking_keys)
                        extra_keys = set(ranking_keys) - expected_phase_keys
                        raise ValueError(
                            f"{ranking} phase keys do not match phase groups "
                            f"under {run_dir}: missing={len(missing_keys)}, "
                            f"extra={len(extra_keys)}"
                        )
                    if any(
                        len(row.get("top_features", []))
                        < expected_top_features
                        for row in phase_ranking_rows
                    ):
                        raise ValueError(
                            f"{ranking} has fewer than "
                            f"{expected_top_features} features for a phase row: "
                            f"{ranking_paths[ranking]}"
                        )
                    for row in phase_ranking_rows:
                        key = (
                            str(row["task_description"]),
                            str(row["cluster_id"]),
                        )
                        canonical_phase = canonical_phase_by_key[key]
                        if (
                            str(row.get("phase", ""))
                            != canonical_phase["phase"]
                            or str(row.get("phrase", ""))
                            != canonical_phase["phrase"]
                        ):
                            raise ValueError(
                                f"{ranking} phase metadata does not match the "
                                f"canonical phase group: "
                                f"{ranking_paths[ranking]}"
                            )
                        top_features = row["top_features"][
                            :expected_top_features
                        ]
                        feature_ids = [
                            str(feature["feature_id"])
                            for feature in top_features
                        ]
                        if len(set(feature_ids)) != expected_top_features:
                            raise ValueError(
                                f"{ranking} has duplicate IDs in its Top-5 "
                                f"features: {ranking_paths[ranking]}"
                            )
                        try:
                            feature_scores = [
                                float(feature["score"])
                                for feature in top_features
                            ]
                        except (KeyError, TypeError, ValueError) as error:
                            raise ValueError(
                                f"{ranking} has a missing or invalid Top-5 "
                                f"feature score: {ranking_paths[ranking]}"
                            ) from error
                        if not all(
                            math.isfinite(score) for score in feature_scores
                        ):
                            raise ValueError(
                                f"{ranking} has a non-finite Top-5 feature "
                                f"score: {ranking_paths[ranking]}"
                            )

                condition = condition_by_id[condition_id]
                coverage_value = _result_coverage_value(coverage_id)
                runs.append(
                    {
                        "id": f"{condition['code']}-{coverage_id}-{sae_id}",
                        "condition_id": condition_id,
                        "condition_code": condition["code"],
                        "condition_label": condition["label"],
                        "coverage_id": coverage_id,
                        "coverage": coverage_value,
                        "coverage_label": f"{coverage_value:.1f}",
                        "sae_id": sae_id,
                        "sae_label": format_checkpoint_label(sae_id),
                        "status": "complete",
                        "score_artifact_available": True,
                        "top_k": int(config["top_k"]),
                        "top_n_per_row": top_n_per_row,
                        "phase_group_count": len(compact_phase_groups),
                        "task_count": len(rankings["task_mean"]),
                        "candidates": candidates,
                        "rankings": rankings,
                    }
                )

    for condition_id in conditions:
        condition_phase_sets = sorted(
            (
                phase_set
                for phase_set in phase_sets
                if phase_set["condition_id"] == condition_id
            ),
            key=lambda phase_set: phase_set["coverage"],
        )
        for lower, higher in zip(
            condition_phase_sets,
            condition_phase_sets[1:],
            strict=False,
        ):
            lower_sources = {
                source_cluster_id
                for phase_group in lower["phase_groups"]
                for source_cluster_id in phase_group["source_cluster_ids"]
            }
            higher_sources = {
                source_cluster_id
                for phase_group in higher["phase_groups"]
                for source_cluster_id in phase_group["source_cluster_ids"]
            }
            if not higher_sources.issubset(lower_sources):
                raise ValueError(
                    f"Higher coverage phase sources are not a subset for "
                    f"{condition_id}: {lower['coverage']} -> "
                    f"{higher['coverage']}"
                )

    declared_counts = manifest.get("artifact_counts", {})
    declared_scores = int(declared_counts.get("score_artifacts", expected_runs))
    declared_rankings = int(
        declared_counts.get("ranking_artifacts", expected_runs)
    )
    if declared_scores != expected_runs or declared_rankings != expected_runs:
        raise ValueError(
            "Experiment manifest artifact counts do not match the audit scope"
        )

    phase_feature_audit = audit.get("stage4_ranking", {})
    aggregates = phase_feature_audit.get("aggregates", {})
    comparison_payload = {
        comparison_id: {
            "id": comparison_id,
            "label": COMPARISON_LABELS.get(
                comparison_id,
                comparison_id.replace("_", " "),
            ),
            "rankings": rankings,
        }
        for comparison_id, rankings in aggregates.items()
    }
    mechanical = audit.get("mechanical", {})
    raw_partitions = mechanical.get("partitions", {})
    partitions = [
        {
            "id": partition_id,
            "code": f"P{index}",
            "label": PARTITION_LABELS.get(
                partition_id,
                partition_id.replace("_", " "),
            ),
            **metrics,
        }
        for index, (partition_id, metrics) in enumerate(raw_partitions.items())
    ]
    raw_pairwise = mechanical.get("pairwise_partition", {})
    partition_comparisons = [
        {
            "id": comparison_id,
            "label": PARTITION_COMPARISON_LABELS.get(
                comparison_id,
                comparison_id.replace("_", " "),
            ),
            **metrics,
        }
        for comparison_id, metrics in raw_pairwise.items()
    ]
    return {
        "format": "event_sae_stage4_results_browser_v1",
        "meta": {
            "title": "Anchor / view controlled ablation",
            "result_status": str(audit.get("result_status", "unknown")),
            "claim_strength": str(audit.get("claim_strength", "unspecified")),
            "generated_at": audit.get("generated_at"),
            "passed": bool(audit.get("passed", False)),
            "expected_runs": expected_runs,
            "available_runs": len(runs),
            "score_artifacts": declared_scores,
            "ranking_artifacts": declared_rankings,
            "candidate_rows": int(
                declared_counts.get(
                    "candidate_rows",
                    sum(
                        len(rows)
                        for run in runs
                        for rows in run["candidates"].values()
                    ),
                )
            ),
        },
        "facets": {
            "conditions": condition_payload,
            "coverages": [
                {
                    "id": coverage_id,
                    "value": _result_coverage_value(coverage_id),
                    "label": f"{_result_coverage_value(coverage_id):.1f}",
                }
                for coverage_id in coverages
            ],
            "checkpoints": [
                {
                    "id": sae_id,
                    "label": format_checkpoint_label(sae_id),
                }
                for sae_id in sae_ids
            ],
            "rankings": list(RESULT_RANKINGS),
        },
        "runs": runs,
        "phase_sets": phase_sets,
        "clustering": {
            "partitions": partitions,
            "comparisons": partition_comparisons,
            "gripper_only": mechanical.get("q3_gripper_only", {}),
            "anchor_sets": mechanical.get("q4_anchor_sets", {}),
        },
        "comparisons": comparison_payload,
        "comparison_cells": phase_feature_audit.get("cells", []),
    }


class ExperimentResultsService:
    """Read-only controlled-ablation result browser."""

    def __init__(
        self,
        *,
        payload: dict,
        ui_path: Path,
        experiment_root: Path,
        oracle_experiment_root: Path | None = None,
    ) -> None:
        self.payload = payload
        self.ui = Path(ui_path).read_bytes()
        self.experiment_root = resolve_groot_artifact_path(
            experiment_root
        ).resolve()
        self._explorer_lock = threading.Lock()
        self._explorer_payloads: dict[str, dict] = {}
        self._explorer_media: dict[str, dict[str, tuple[Path, ...]]] = {}
        self._heatmap_lock = threading.Lock()
        self._heatmap_score_cache: dict[tuple[str, int, int, str], dict] = {}
        self._task_identity_registry = load_controlled_task_identity_registry(
            self.experiment_root
        )
        self._phase_feature_overview_lock = threading.Lock()
        self._phase_feature_overview_payload: dict | None = None
        self.oracle_experiment_root = (
            Path(oracle_experiment_root).expanduser().resolve()
            if oracle_experiment_root is not None
            else None
        )
        self._oracle_lock = threading.Lock()
        self._oracle_payload: dict | None = None
        self._oracle_media: dict[
            tuple[str, str],
            tuple[Path, ...],
        ] = {}

    def data(self) -> dict:
        return self.payload

    def phase_feature_overview(self) -> dict:
        """Return a cached full-matrix phase-feature evidence overview."""

        with self._phase_feature_overview_lock:
            if self._phase_feature_overview_payload is None:
                self._phase_feature_overview_payload = build_phase_feature_overview(
                    experiment_root=self.experiment_root,
                    results_payload=self.payload,
                    task_identity_registry=self._task_identity_registry,
                )
            return self._phase_feature_overview_payload

    def _cached_heatmap_score_data(
        self,
        *,
        score_path: Path,
        ranking: str,
    ) -> dict:
        score_path = Path(score_path).expanduser().resolve()
        stat = score_path.stat()
        cache_key = (
            str(score_path),
            int(stat.st_mtime_ns),
            int(stat.st_size),
            ranking,
        )
        with self._heatmap_lock:
            cached = self._heatmap_score_cache.get(cache_key)
        if cached is not None:
            return cached
        loaded = load_phase_feature_score_matrix(
            score_path=score_path,
            ranking=ranking,
        )
        with self._heatmap_lock:
            if len(self._heatmap_score_cache) >= 12:
                oldest_key = next(iter(self._heatmap_score_cache))
                self._heatmap_score_cache.pop(oldest_key)
            self._heatmap_score_cache[cache_key] = loaded
        return loaded

    def _controlled_heatmap_source(
        self,
        run_id: str,
    ) -> tuple[Path, dict]:
        runs = [
            run for run in self.payload["runs"] if str(run["id"]) == run_id
        ]
        if len(runs) != 1:
            raise KeyError(run_id)
        run = runs[0]
        score_path = (
            self.experiment_root
            / "stage4"
            / str(run["condition_id"])
            / str(run["coverage_id"])
            / str(run["sae_id"])
            / "event_feature_scores.pt"
        ).resolve()
        if self.experiment_root not in score_path.parents:
            raise ValueError("Controlled score path escapes experiment root")
        return score_path, {
            "analysis_id": run_id,
            "dataset_label": "45개 controlled 실험",
            "condition_id": str(run["condition_id"]),
            "condition_code": str(run["condition_code"]),
            "condition_label": str(run["condition_label"]),
            "coverage_id": str(run["coverage_id"]),
            "coverage": float(run["coverage"]),
            "checkpoint_id": str(run["sae_id"]),
            "checkpoint_label": str(run["sae_label"]),
            "aggregation_scope": "phase",
            "claim_scope": str(self.payload["meta"]["claim_strength"]),
        }

    def _oracle_heatmap_source(
        self,
        package_id: str,
    ) -> tuple[Path, dict]:
        if self.oracle_experiment_root is None:
            raise KeyError(package_id)
        payload = self.oracle()
        packages = [
            package
            for package in payload.get("ranking_packages", [])
            if (
                str(package.get("id")) == package_id
                and package.get("source_mapping_verified") is True
            )
        ]
        if len(packages) != 1:
            raise KeyError(package_id)
        package = packages[0]
        relative_ranking_dir = Path(str(package["label"]))
        if (
            relative_ranking_dir.is_absolute()
            or ".." in relative_ranking_dir.parts
        ):
            raise ValueError("Oracle ranking path is not a safe relative path")
        config_path = (
            self.oracle_experiment_root
            / relative_ranking_dir
            / "ranking_config.json"
        ).resolve()
        if self.oracle_experiment_root not in config_path.parents:
            raise ValueError("Oracle ranking config escapes experiment root")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        score_path = Path(str(config.get("scores_pt", ""))).expanduser().resolve()
        if self.oracle_experiment_root not in score_path.parents:
            raise ValueError("Oracle score path escapes experiment root")
        return score_path, {
            "analysis_id": str(package["analysis_id"]),
            "package_id": package_id,
            "dataset_label": "Simulator-oracle phase 실험",
            "source_variant_id": str(package["source_variant_id"]),
            "checkpoint_id": str(package["checkpoint_id"]),
            "checkpoint_label": str(package["checkpoint_label"]),
            "aggregation_scope": str(package.get("scope", "state_cluster")),
            "window_size": package.get("window_size"),
            "claim_scope": str(payload["claim_scope"]),
            "source_mapping_verified": True,
        }

    def feature_heatmap(
        self,
        *,
        dataset: str,
        analysis_id: str,
        ranking: str,
        mode: str,
        limit: int,
        task_description: str | None = None,
        pinned_feature_id: int | None = None,
    ) -> dict:
        """Return one exact, checkpoint-local phase×feature heatmap."""

        if dataset == "controlled":
            score_path, context = self._controlled_heatmap_source(analysis_id)
        elif dataset == "oracle":
            score_path, context = self._oracle_heatmap_source(analysis_id)
        else:
            raise ValueError(f"Unsupported heatmap dataset: {dataset}")
        score_data = self._cached_heatmap_score_data(
            score_path=score_path,
            ranking=ranking,
        )
        score_data = decorate_score_task_identities(
            score_data,
            self._task_identity_registry if dataset == "controlled" else None,
        )
        declared_window = context.get("window_size")
        if (
            declared_window is not None
            and score_data.get("window_size") is not None
            and int(declared_window) != int(score_data["window_size"])
        ):
            raise ValueError(
                "Heatmap score window does not match ranking package"
            )
        return build_phase_feature_heatmap(
            score_data=score_data,
            dataset=dataset,
            context=context,
            mode=mode,
            limit=limit,
            task_description=task_description,
            pinned_feature_id=pinned_feature_id,
        )

    def oracle(self) -> dict:
        """Re-read the optional oracle experiment so in-flight work can appear."""

        with self._oracle_lock:
            try:
                payload, media = build_oracle_phase_results_dataset(
                    self.oracle_experiment_root
                )
            except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
                payload = build_oracle_waiting_payload(
                    status="invalid_artifact",
                    message=(
                        "Oracle phase 산출물의 형식 또는 연결을 확인해야 합니다. "
                        "기존 45개 실험 결과에는 영향이 없습니다."
                    ),
                )
                media = {}
            self._oracle_payload = payload
            self._oracle_media = media
            return payload

    def oracle_frame(
        self,
        sample_id: str,
        view: str,
        frame_index: int,
    ) -> Path:
        if view not in ORACLE_PHASE_VIEWS:
            raise KeyError((sample_id, view, frame_index))
        with self._oracle_lock:
            paths = self._oracle_media.get((sample_id, view))
        if paths is None:
            self.oracle()
            with self._oracle_lock:
                paths = self._oracle_media.get((sample_id, view))
        if paths is None or not 0 <= frame_index < len(paths):
            raise KeyError((sample_id, view, frame_index))
        return paths[frame_index]

    def explorer(self, condition_id: str) -> dict:
        valid_conditions = {
            str(condition["id"])
            for condition in self.payload["facets"]["conditions"]
        }
        if condition_id not in valid_conditions:
            raise KeyError(condition_id)
        with self._explorer_lock:
            if condition_id not in self._explorer_payloads:
                payload, media_paths = build_condition_cluster_explorer_dataset(
                    experiment_root=self.experiment_root,
                    condition_id=condition_id,
                    projection_method="pca",
                )
                self._explorer_payloads[condition_id] = payload
                self._explorer_media[condition_id] = media_paths
            return self._explorer_payloads[condition_id]

    def explorer_frame(
        self,
        condition_id: str,
        sample_id: str,
        frame_index: int,
    ) -> Path:
        self.explorer(condition_id)
        paths = self._explorer_media[condition_id].get(sample_id)
        if paths is None or not 0 <= frame_index < len(paths):
            raise KeyError((sample_id, frame_index))
        return paths[frame_index]


def build_experiment_results_service(
    *,
    experiment_root: Path,
    ui_path: Path,
    oracle_experiment_root: Path | None = None,
) -> ExperimentResultsService:
    return ExperimentResultsService(
        payload=build_experiment_results_dataset(experiment_root),
        ui_path=ui_path,
        experiment_root=experiment_root,
        oracle_experiment_root=oracle_experiment_root,
    )


def make_experiment_results_http_handler(
    application: ExperimentResultsService,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "EventSAEResults/1.0"

        def _headers(self, status: int, content_type: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; "
                "style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; "
                "connect-src 'self'",
            )
            self.end_headers()

        def _send_bytes(
            self,
            data: bytes,
            *,
            status: int = HTTPStatus.OK,
            content_type: str = "application/octet-stream",
        ) -> None:
            self._headers(int(status), content_type, len(data))
            self.wfile.write(data)

        def _send_json(self, value: dict, *, status: int = HTTPStatus.OK) -> None:
            self._send_bytes(
                json.dumps(value, ensure_ascii=False).encode("utf-8"),
                status=status,
                content_type="application/json; charset=utf-8",
            )

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path == "/":
                self._send_bytes(
                    application.ui,
                    content_type="text/html; charset=utf-8",
                )
                return
            if path == "/api/results":
                self._send_json(application.data())
                return
            if path == "/api/stage4-overview":
                try:
                    self._send_json(application.phase_feature_overview())
                except (
                    FileNotFoundError,
                    ValueError,
                    json.JSONDecodeError,
                    OSError,
                    RuntimeError,
                ):
                    self._send_json(
                        {"error": "Could not build Stage 4 overview"},
                        status=HTTPStatus.UNPROCESSABLE_ENTITY,
                    )
                return
            if path == "/api/oracle":
                self._send_json(application.oracle())
                return
            if path == "/api/phase-feature-heatmap":
                dataset = query.get("dataset", [""])[0]
                analysis_id = query.get("analysis_id", [""])[0]
                ranking = query.get("ranking", ["event_aligned"])[0]
                mode = query.get("mode", ["selectivity"])[0]
                task_description = query.get("task", [""])[0].strip() or None
                try:
                    limit = int(query.get("limit", ["24"])[0])
                    raw_pinned_feature_id = query.get(
                        "pinned_feature_id", [""]
                    )[0].strip()
                    pinned_feature_id = (
                        int(raw_pinned_feature_id)
                        if raw_pinned_feature_id
                        else None
                    )
                    payload = application.feature_heatmap(
                        dataset=dataset,
                        analysis_id=analysis_id,
                        ranking=ranking,
                        mode=mode,
                        limit=limit,
                        task_description=task_description,
                        pinned_feature_id=pinned_feature_id,
                    )
                except KeyError:
                    self._send_json(
                        {"error": "Unknown heatmap analysis or task"},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                except (
                    FileNotFoundError,
                    ValueError,
                    json.JSONDecodeError,
                    OSError,
                    RuntimeError,
                ):
                    self._send_json(
                        {"error": "Could not load phase feature heatmap"},
                        status=HTTPStatus.UNPROCESSABLE_ENTITY,
                    )
                    return
                self._send_json(payload)
                return
            if path == "/api/oracle/frame":
                sample_id = query.get("sample_id", [""])[0]
                view = query.get("view", [""])[0]
                try:
                    frame_index = int(query.get("index", ["-1"])[0])
                    frame_path = application.oracle_frame(
                        sample_id,
                        view,
                        frame_index,
                    )
                except (KeyError, ValueError):
                    self._send_json(
                        {"error": "Unknown oracle sample, view, or frame"},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                self._send_bytes(
                    frame_path.read_bytes(),
                    content_type=(
                        mimetypes.guess_type(frame_path.name)[0]
                        or "image/jpeg"
                    ),
                )
                return
            if path == "/api/explorer":
                condition_id = query.get("condition_id", [""])[0]
                try:
                    self._send_json(application.explorer(condition_id))
                except KeyError:
                    self._send_json(
                        {"error": "Unknown experiment condition"},
                        status=HTTPStatus.NOT_FOUND,
                    )
                except (FileNotFoundError, ValueError):
                    self._send_json(
                        {"error": "Could not load condition explorer artifacts"},
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                return
            if path == "/api/explorer/frame":
                condition_id = query.get("condition_id", [""])[0]
                sample_id = query.get("sample_id", [""])[0]
                try:
                    frame_index = int(query.get("index", ["-1"])[0])
                    frame_path = application.explorer_frame(
                        condition_id,
                        sample_id,
                        frame_index,
                    )
                except (KeyError, ValueError):
                    self._send_json(
                        {"error": "Unknown condition, sample, or frame"},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                self._send_bytes(
                    frame_path.read_bytes(),
                    content_type=(
                        mimetypes.guess_type(frame_path.name)[0]
                        or "image/jpeg"
                    ),
                )
                return
            if path == "/healthz":
                meta = application.payload["meta"]
                self._send_json(
                    {
                        "status": "ok",
                        "available_runs": meta["available_runs"],
                        "expected_runs": meta["expected_runs"],
                        "oracle_configured": (
                            application.oracle_experiment_root is not None
                        ),
                    }
                )
                return
            self._send_json(
                {"error": "Not found"},
                status=HTTPStatus.NOT_FOUND,
            )

        def log_message(self, format: str, *args: object) -> None:
            if self.path != "/healthz":
                super().log_message(format, *args)

    return Handler


__all__ = [
    "RESULT_RANKINGS",
    "ExperimentResultsService",
    "build_condition_cluster_explorer_dataset",
    "build_experiment_results_dataset",
    "build_experiment_results_service",
    "make_experiment_results_http_handler",
]
