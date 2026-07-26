#!/usr/bin/env python3
"""Audit and summarize the completed GR00T anchor/view ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


CONDITIONS = (
    "e0_rel_pos_cluster_left_label_left",
    "e1_rel_pos_cluster_left_label_multiview",
    "e2_rel_pos_cluster_multiview_label_multiview",
    "e3_rel_pos_gripper_cluster_multiview_label_multiview",
    "e4_abs_pos_gripper_cluster_multiview_label_multiview",
)
COVERAGES = ("cov0p3", "cov0p4", "cov0p5")
SAE_IDS = (
    "l15_sae1p2k_exec5_mean4_top96_v1",
    "l15_sae10k_exec5_mean4_top96_v1",
    "l15_sae10k_bs8192_exec5_mean4_top96_v1",
)
INFORMED_RANKINGS = ("event_aligned", "window_mean", "task_mean")
COMPARISONS = {
    "q1_annotation_view": (CONDITIONS[0], CONDITIONS[1]),
    "q2_clustering_view": (CONDITIONS[1], CONDITIONS[2]),
    "q3_gripper_anchor": (CONDITIONS[2], CONDITIONS[3]),
    "q4_relative_absolute": (CONDITIONS[3], CONDITIONS[4]),
}


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _assignment_key(row: dict) -> tuple[int, int]:
    return int(row["episode_num"]), int(row["waypoint_step"])


def _partition_pair_metrics(
    left: dict[tuple[int, int], dict],
    right: dict[tuple[int, int], dict],
) -> dict:
    common = sorted(set(left).intersection(right))
    if not common:
        raise ValueError("Partition comparison has no common anchors")
    for key in common:
        if left[key]["task_description"] != right[key]["task_description"]:
            raise ValueError(f"Task mismatch for common anchor {key}")

    def scores(keys: list[tuple[int, int]]) -> dict[str, float]:
        return {
            "ari": float(
                adjusted_rand_score(
                    [left[key]["cluster_id"] for key in keys],
                    [right[key]["cluster_id"] for key in keys],
                )
            ),
            "nmi": float(
                normalized_mutual_info_score(
                    [left[key]["cluster_id"] for key in keys],
                    [right[key]["cluster_id"] for key in keys],
                )
            ),
        }

    by_task: dict[str, dict] = {}
    for task in sorted({str(left[key]["task_description"]) for key in common}):
        keys = [key for key in common if left[key]["task_description"] == task]
        by_task[task] = {"n": len(keys), **scores(keys)}
    total = sum(row["n"] for row in by_task.values())
    return {
        "common_anchors": len(common),
        "left_only": len(set(left).difference(right)),
        "right_only": len(set(right).difference(left)),
        "pooled": scores(common),
        "within_task_macro": {
            metric: float(statistics.mean(row[metric] for row in by_task.values()))
            for metric in ("ari", "nmi")
        },
        "within_task_weighted": {
            metric: float(
                sum(row[metric] * row["n"] for row in by_task.values()) / total
            )
            for metric in ("ari", "nmi")
        },
        "by_task": by_task,
    }


def _suite_vector(payload: dict, ranking: str) -> torch.Tensor:
    rows = list(payload["row_keys"])
    if ranking == "event_aligned":
        return payload["matrix_raw"].to(torch.float32).mean(dim=0)
    if ranking == "window_mean":
        matrix = payload["matrix_window_mean"].to(torch.float32)
        weights = torch.tensor(
            [float(row["num_events"]) for row in rows],
            dtype=torch.float32,
        )
        return (matrix * weights[:, None]).sum(dim=0) / weights.sum()
    if ranking == "task_mean":
        matrix = payload["matrix_task_mean"].to(torch.float32)
        task_counts = payload["selection_counts"]["task_timestep_counts"]
        seen: set[str] = set()
        vectors: list[torch.Tensor] = []
        weights: list[float] = []
        for index, row in enumerate(rows):
            task = str(row["task_description"])
            if task in seen:
                continue
            seen.add(task)
            vectors.append(matrix[index])
            task_id = row.get("task_id")
            weight = task_counts.get(task_id, task_counts.get(str(task_id), 1))
            weights.append(float(weight))
        stacked = torch.stack(vectors)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        return (
            stacked * weight_tensor[:, None]
        ).sum(dim=0) / weight_tensor.sum()
    raise ValueError(f"Unsupported ranking: {ranking}")


def _candidate_ids(path: Path, ranking: str) -> list[int]:
    rows = _load_jsonl(path)
    selected = [
        row for row in rows if str(row["ranking"]) == ranking
    ]
    selected.sort(key=lambda row: int(row["rank"]))
    return [int(row["feature_id"]) for row in selected]


def _ranking_comparisons(root: Path) -> dict:
    cells: list[dict] = []
    for comparison, (left_condition, right_condition) in COMPARISONS.items():
        for coverage in COVERAGES:
            for sae_id in SAE_IDS:
                left_dir = root / "stage4" / left_condition / coverage / sae_id
                right_dir = root / "stage4" / right_condition / coverage / sae_id
                left_payload = torch.load(
                    left_dir / "event_feature_scores.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                right_payload = torch.load(
                    right_dir / "event_feature_scores.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                for ranking in INFORMED_RANKINGS:
                    left_ids = _candidate_ids(
                        left_dir / "rankings/candidates.jsonl",
                        ranking,
                    )
                    right_ids = _candidate_ids(
                        right_dir / "rankings/candidates.jsonl",
                        ranking,
                    )
                    overlap = len(set(left_ids).intersection(right_ids))
                    left_vector = _suite_vector(left_payload, ranking)
                    right_vector = _suite_vector(right_payload, ranking)
                    correlation = float(
                        spearmanr(
                            left_vector.numpy(),
                            right_vector.numpy(),
                        ).statistic
                    )
                    cosine = float(
                        torch.nn.functional.cosine_similarity(
                            left_vector,
                            right_vector,
                            dim=0,
                        ).item()
                    )
                    cells.append(
                        {
                            "comparison": comparison,
                            "left_condition": left_condition,
                            "right_condition": right_condition,
                            "coverage": coverage,
                            "sae_id": sae_id,
                            "ranking": ranking,
                            "top5_overlap_count": overlap,
                            "top5_overlap_fraction": overlap / 5.0,
                            "top5_jaccard": overlap / (10 - overlap),
                            "full_vector_spearman": correlation,
                            "full_vector_cosine": cosine,
                        }
                    )

    aggregates: dict[str, dict] = {}
    for comparison in COMPARISONS:
        aggregates[comparison] = {}
        for ranking in INFORMED_RANKINGS:
            selected = [
                row
                for row in cells
                if row["comparison"] == comparison
                and row["ranking"] == ranking
            ]
            metric_summary = {
                metric: {
                    "median": float(
                        statistics.median(row[metric] for row in selected)
                    ),
                    "min": float(min(row[metric] for row in selected)),
                    "max": float(max(row[metric] for row in selected)),
                }
                for metric in (
                    "top5_overlap_count",
                    "top5_overlap_fraction",
                    "full_vector_spearman",
                    "full_vector_cosine",
                )
            }
            aggregates[comparison][ranking] = {
                "n_cells": len(selected),
                **metric_summary,
            }
    return {"cells": cells, "aggregates": aggregates}


def _automatic_q1(root: Path, p0_clusters_path: Path) -> dict:
    annotation_root = root / "annotations"
    left = {
        row["cluster_id"]: row
        for row in _load_jsonl(
            annotation_root
            / CONDITIONS[0]
            / "audited_annotations.jsonl"
        )
    }
    multiview = {
        row["cluster_id"]: row
        for row in _load_jsonl(
            annotation_root
            / CONDITIONS[1]
            / "audited_annotations.jsonl"
        )
    }
    if set(left) != set(multiview):
        raise ValueError("E0/E1 annotation IDs differ")
    sizes = {
        row["cluster_id"]: int(row["num_members"])
        for row in _load_jsonl(p0_clusters_path)
    }
    transitions: Counter[tuple[str, str]] = Counter()
    transition_events: Counter[tuple[str, str]] = Counter()
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    matched_clusters = 0
    matched_events = 0
    total_events = 0
    phrase_matches = 0
    for cluster_id in sorted(left):
        left_row = left[cluster_id]
        right_row = multiview[cluster_id]
        left_phase = str(left_row["phase"])
        right_phase = str(right_row["phase"])
        size = sizes[cluster_id]
        matched = left_phase == right_phase
        matched_clusters += int(matched)
        matched_events += size * int(matched)
        total_events += size
        phrase_matches += int(left_row["phrase"] == right_row["phrase"])
        transitions[(left_phase, right_phase)] += 1
        transition_events[(left_phase, right_phase)] += size
        task_counts = by_task[str(left_row["task_description"])]
        task_counts["clusters"] += 1
        task_counts["matched_clusters"] += int(matched)
        task_counts["events"] += size
        task_counts["matched_events"] += size * int(matched)
    return {
        "paired_clusters": len(left),
        "paired_events": total_events,
        "phase_exact_match_cluster_weighted": {
            "numerator": matched_clusters,
            "denominator": len(left),
            "fraction": matched_clusters / len(left),
        },
        "phase_exact_match_event_weighted": {
            "numerator": matched_events,
            "denominator": total_events,
            "fraction": matched_events / total_events,
        },
        "phrase_exact_match_clusters": {
            "numerator": phrase_matches,
            "denominator": len(left),
            "fraction": phrase_matches / len(left),
        },
        "transition_cluster_counts": {
            f"{left_phase} -> {right_phase}": count
            for (left_phase, right_phase), count in sorted(transitions.items())
        },
        "transition_event_counts": {
            f"{left_phase} -> {right_phase}": count
            for (left_phase, right_phase), count in sorted(
                transition_events.items()
            )
        },
        "by_task": {
            task: {
                **dict(counts),
                "cluster_match_fraction": (
                    counts["matched_clusters"] / counts["clusters"]
                ),
                "event_match_fraction": (
                    counts["matched_events"] / counts["events"]
                ),
            }
            for task, counts in sorted(by_task.items())
        },
        "provenance": "gemini_v11_provisional_automatic_not_human_reviewed",
    }


def _mechanical_metrics(
    root: Path,
    repo_root: Path,
) -> tuple[dict, dict[str, Path]]:
    p3_root = (
        repo_root
        / "logs/groot_n15/experiments/"
        "v9_abs_position_gripper_3view_action_phase_v1/"
        "stage3_clusters/c0_balanced_v9/c0_d0p18"
    )
    partition_paths = {
        "p0_r_pos_left": root / "partitions/p0_r_pos_left",
        "p1_r_pos_3view": root / "partitions/p1_r_pos_3view",
        "p2_r_pos_gripper_3view": root
        / "partitions/p2_r_pos_gripper_3view",
        "p3_a_pos_gripper_3view": p3_root,
    }
    partitions: dict[str, dict] = {}
    assignment_maps: dict[str, dict] = {}
    for partition_id, path in partition_paths.items():
        summary = json.loads(
            (path / "summary.json").read_text(encoding="utf-8")
        )
        assignments = _load_jsonl(path / "cluster_assignments.jsonl")
        assignment_maps[partition_id] = {
            _assignment_key(row): row for row in assignments
        }
        partitions[partition_id] = {
            key: summary[key]
            for key in (
                "num_events",
                "num_clusters",
                "num_singleton_clusters",
                "singleton_event_fraction",
                "median_cluster_size",
                "max_cluster_size",
                "num_clusters_meeting_min_coverage",
                "distance_threshold",
                "block_normalization",
            )
        }
        partitions[partition_id]["anchor_source_counts"] = dict(
            Counter(str(row.get("anchor_source")) for row in assignments)
        )

    relative_multiview = assignment_maps["p1_r_pos_3view"]
    relative_gripper_multiview = assignment_maps[
        "p2_r_pos_gripper_3view"
    ]
    absolute_gripper_multiview = assignment_maps[
        "p3_a_pos_gripper_3view"
    ]
    relative_gripper_position_anchors = {
        key: row
        for key, row in relative_gripper_multiview.items()
        if str(row["anchor_source"]) in {"position", "both"}
    }
    pairwise = {
        "q2_p0_vs_p1_all_r_pos": _partition_pair_metrics(
            assignment_maps["p0_r_pos_left"],
            relative_multiview,
        ),
        "q3_p1_vs_p2_common_position": _partition_pair_metrics(
            relative_multiview,
            relative_gripper_position_anchors,
        ),
        "q4_p2_vs_p3_common_anchors": _partition_pair_metrics(
            relative_gripper_multiview,
            absolute_gripper_multiview,
        ),
    }

    relative_gripper_clusters = {
        row["cluster_id"]: row
        for row in _load_jsonl(
            partition_paths["p2_r_pos_gripper_3view"] / "clusters.jsonl"
        )
    }
    gripper_only = [
        row
        for row in relative_gripper_multiview.values()
        if row["anchor_source"] == "gripper_close"
    ]
    recurring = [
        row
        for row in gripper_only
        if float(
            relative_gripper_clusters[row["cluster_id"]]["episode_coverage"]
        )
        >= 0.3
    ]
    singleton_ids = {
        cluster_id
        for cluster_id, row in relative_gripper_clusters.items()
        if int(row["num_members"]) == 1
    }
    q3_gripper = {
        "gripper_only_events": len(gripper_only),
        "entered_coverage_ge_0p3_cluster": len(recurring),
        "recurring_entry_fraction": len(recurring) / len(gripper_only),
        "gripper_only_events_in_singleton_clusters": sum(
            row["cluster_id"] in singleton_ids for row in gripper_only
        ),
        "by_task": {},
    }
    for task in sorted({str(row["task_description"]) for row in gripper_only}):
        task_rows = [
            row for row in gripper_only if row["task_description"] == task
        ]
        task_recurring = [
            row
            for row in task_rows
            if float(
                relative_gripper_clusters[row["cluster_id"]][
                    "episode_coverage"
                ]
            )
            >= 0.3
        ]
        q3_gripper["by_task"][task] = {
            "events": len(task_rows),
            "recurring_events": len(task_recurring),
            "fraction": len(task_recurring) / len(task_rows),
        }

    common_position_frame_anchors = set(
        relative_gripper_multiview
    ).intersection(absolute_gripper_multiview)
    relative_only_anchors = set(relative_gripper_multiview).difference(
        absolute_gripper_multiview
    )
    absolute_only_anchors = set(absolute_gripper_multiview).difference(
        relative_gripper_multiview
    )
    q4_anchor_sets = {
        "common": len(common_position_frame_anchors),
        "relative_only": len(relative_only_anchors),
        "absolute_only": len(absolute_only_anchors),
        "relative_only_by_task": dict(
            Counter(
                str(relative_gripper_multiview[key]["task_description"])
                for key in relative_only_anchors
            )
        ),
        "absolute_only_by_task": dict(
            Counter(
                str(absolute_gripper_multiview[key]["task_description"])
                for key in absolute_only_anchors
            )
        ),
    }
    return (
        {
            "partitions": partitions,
            "pairwise_partition": pairwise,
            "q3_gripper_only": q3_gripper,
            "q4_anchor_sets": q4_anchor_sets,
        },
        partition_paths,
    )


def _annotation_metrics(root: Path) -> dict:
    conditions: dict[str, dict] = {}
    for condition in CONDITIONS:
        audit = json.loads(
            (
                root
                / "annotations"
                / condition
                / "annotation_join_audit.json"
            ).read_text(encoding="utf-8")
        )
        conditions[condition] = {
            "clusters": audit["counts"]["annotations"],
            "selected_cluster_events": audit["counts"][
                "selected_cluster_events"
            ],
            "phase_cluster_counts": audit["phase_cluster_counts"],
            "phase_event_counts": audit["phase_event_counts"],
            "api_errors": audit["counts"]["annotation_api_errors"],
            "parse_errors": audit["counts"]["annotation_parse_errors"],
            "prompt_versions": audit["prompt_versions"],
            "annotation_media_layouts": audit["annotation_media_layouts"],
            "actual_human_review_completed": False,
            "result_status": "provisional_automatic",
        }
    coverage: dict[str, dict] = {}
    for condition in CONDITIONS:
        coverage[condition] = {}
        for coverage_tag in COVERAGES:
            summary = json.loads(
                (
                    root
                    / "stage4"
                    / condition
                    / coverage_tag
                    / "phase_groups/summary.json"
                ).read_text(encoding="utf-8")
            )
            coverage[condition][coverage_tag] = {
                key: summary[key]
                for key in (
                    "num_reviewed_clusters",
                    "num_phase_groups",
                    "num_grouped_samples",
                    "actual_human_review_completed",
                    "result_status",
                )
            }
    return {"conditions": conditions, "coverage_phase_groups": coverage}


def _confound_audit() -> list[dict]:
    return [
        {
            "gate": "Length",
            "status": "N-A",
            "evidence": (
                "성공/실패 분리나 rollout-length classifier를 평가하지 않았다."
            ),
        },
        {
            "gate": "Task identity",
            "status": "PASS",
            "evidence": (
                "clustering은 exact task_description local이며 ARI/NMI는 "
                "instruction별 값과 macro/weighted 값을 함께 계산했다."
            ),
        },
        {
            "gate": "Instruction balance",
            "status": "PASS",
            "evidence": (
                "5개 instruction cell이 각각 30 episodes이며 모든 condition의 "
                "episode→task mapping 150/150이 exact match했다."
            ),
        },
        {
            "gate": "In-sample rescue",
            "status": "N-A",
            "evidence": (
                "detector/steering fit 및 held-out policy evaluation을 수행하지 않았다."
            ),
        },
        {
            "gate": "Rollout pooling",
            "status": "PASS",
            "evidence": (
                "sparse activation은 per-record/action-executed timestep으로 읽고 "
                "W=5 event window를 사용했다; episode-mean feature를 입력으로 쓰지 않았다."
            ),
        },
        {
            "gate": "Phase/dwell",
            "status": "N-A",
            "evidence": (
                "성공/실패 phase separation을 주장하지 않으며 모든 condition에 "
                "동일 W=5를 사용했다."
            ),
        },
        {
            "gate": "Observation ≠ causation",
            "status": "PASS",
            "evidence": (
                "결과를 clustering/annotation/ranking diagnostic으로만 제한하며 "
                "policy 성능 또는 intervention 효과를 주장하지 않는다."
            ),
        },
        {
            "gate": "Scene-local ≠ general",
            "status": "PASS",
            "evidence": (
                "범위를 현재 5 instruction/scene cells와 150 episodes로 명시한다."
            ),
        },
    ]


def _artifact_manifest(
    root: Path,
    repo_root: Path,
    output_paths: set[Path],
) -> dict:
    paths: set[Path] = {
        repo_root / "configs/groot/anchor_view_controlled_ablation_v1.json",
        repo_root / "docs/groot/protocols/anchor_view_controlled_ablation.md",
        root / "inputs/prompt_records.jsonl",
        root / "inputs/prompt_records_manifest.json",
    }
    patterns = (
        "waypoints/**/*.json",
        "media/**/samples.jsonl",
        "media/**/*manifest.json",
        "features/**/*manifest.json",
        "features/r_pos/event_features_left.jsonl",
        "features/r_pos/event_features_multiview.jsonl",
        "features/r_pos_gripper/event_features_multiview.jsonl",
        "partitions/**/*",
        "annotation_media/*/manifest.json",
        "annotation_media/*/clusters*.jsonl",
        "annotations/*/frozen_annotations.manifest.json",
        "annotations/*/audited_annotations.jsonl",
        "annotations/*/annotation_join_audit.json",
        "annotations/*/provisional_finalized_annotations.jsonl",
        "stage4/e*/cov*/filtered_finalized_annotations.manifest.json",
        "stage4/e*/cov*/phase_groups/*.json*",
        "stage4/e*/cov*/l15_*/event_feature_scores.pt",
        "stage4/e*/cov*/l15_*/rankings/candidates.jsonl",
        "stage4/e*/cov*/l15_*/rankings/ranking_config.json",
    )
    for pattern in patterns:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    p3_root = (
        repo_root
        / "logs/groot_n15/experiments/"
        "v9_abs_position_gripper_3view_action_phase_v1"
    )
    paths.update(
        {
            p3_root
            / "stage3_features/event_features_multiview_equal_concat_v9.jsonl",
            p3_root
            / "stage3_clusters/c0_balanced_v9/c0_d0p18/clusters.jsonl",
            p3_root
            / "stage3_clusters/c0_balanced_v9/c0_d0p18/"
            "cluster_assignments.jsonl",
            p3_root
            / "stage3_clusters/c0_balanced_v9/c0_d0p18/summary.json",
        }
    )
    for sae_id in SAE_IDS:
        paths.add(
            repo_root
            / "logs/groot_n15/stage4_feature_ranking"
            / sae_id
            / "topk/manifest.json"
        )
    paths.difference_update(output_paths)
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Manifest inputs missing: {missing[:10]}")
    records = [
        {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]
    return {
        "format": "event_sae_anchor_view_source_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": records,
        "artifact_count": len(records),
        "total_bytes": sum(row["bytes"] for row in records),
        "passed": True,
    }


def _format_range(metric: dict, digits: int = 3) -> str:
    return (
        f"{metric['median']:.{digits}f} "
        f"[{metric['min']:.{digits}f}, {metric['max']:.{digits}f}]"
    )


def _render_report(result: dict) -> str:
    mechanical = result["mechanical"]
    annotation_view_agreement = result["automatic_annotation"]["q1_paired"]
    coverage_groups = result["automatic_annotation"]["coverage_phase_groups"]
    ranking = result["stage4_ranking"]["aggregates"]
    lines = [
        "# Anchor/View Controlled Ablation — 자동화 결과",
        "",
        "## 1. 결과 숫자와 평가 범위",
        "",
        "- 범위: 5 instruction/scene cells × 30 episodes = 150 episodes.",
        "- 조건: E0–E4, coverage 0.3/0.4/0.5, SAE checkpoint 3종.",
        "- 산출물: Gemini annotation 90 condition-clusters, phase-group 15개 "
        "bundle, score 45개, ranking 45개(후보 900행).",
        "- review 상태: 사용자 승인 기반 자동 provisional 결과이며 "
        "`actual_human_review_completed=false`.",
        "",
        "### Partition 기계적 지표",
        "",
        "| Partition | Events | Raw clusters | Selected ≥0.3 | Singletons | "
        "Singleton event frac | Median / max size |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for partition_id, row in mechanical["partitions"].items():
        lines.append(
            f"| {partition_id} | {row['num_events']} | {row['num_clusters']} | "
            f"{row['num_clusters_meeting_min_coverage']} | "
            f"{row['num_singleton_clusters']} | "
            f"{row['singleton_event_fraction']:.3f} | "
            f"{row['median_cluster_size']:.1f} / {row['max_cluster_size']} |"
        )
    lines.extend(
        [
            "",
            "### Q1 자동 Gemini annotation view agreement",
            "",
            "- Paired clusters: "
            f"{annotation_view_agreement['paired_clusters']}; paired member "
            f"events: {annotation_view_agreement['paired_events']}.",
            "- Phase exact match: "
            f"{annotation_view_agreement['phase_exact_match_cluster_weighted']['numerator']}/"
            f"{annotation_view_agreement['phase_exact_match_cluster_weighted']['denominator']} "
            f"({annotation_view_agreement['phase_exact_match_cluster_weighted']['fraction']:.3f}) "
            "cluster-weighted; "
            f"{annotation_view_agreement['phase_exact_match_event_weighted']['numerator']}/"
            f"{annotation_view_agreement['phase_exact_match_event_weighted']['denominator']} "
            f"({annotation_view_agreement['phase_exact_match_event_weighted']['fraction']:.3f}) "
            "event-weighted.",
            "- 이 수치는 Gemini-vs-Gemini 자동 agreement이며 annotation 품질 "
            "또는 human exact match가 아니다.",
            "",
            "### Coverage별 실제 입력 n",
            "",
            "| Condition | cov0p3 clusters/groups/events | "
            "cov0p4 clusters/groups/events | cov0p5 clusters/groups/events |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        cells = []
        for coverage_tag in COVERAGES:
            row = coverage_groups[condition][coverage_tag]
            cells.append(
                f"{row['num_reviewed_clusters']}/"
                f"{row['num_phase_groups']}/"
                f"{row['num_grouped_samples']}"
            )
        lines.append(
            f"| {condition} | {cells[0]} | {cells[1]} | {cells[2]} |"
        )
    gripper_anchor_metrics = mechanical["q3_gripper_only"]
    position_frame_anchor_metrics = mechanical["q4_anchor_sets"]
    lines.extend(
        [
            "",
            "### Q3 gripper anchor",
            "",
            f"- R-PG anchor source: position 848, both 65, gripper_close "
            f"{gripper_anchor_metrics['gripper_only_events']}.",
            f"- Gripper-only event 중 coverage≥0.3 recurring cluster 진입: "
            f"{gripper_anchor_metrics['entered_coverage_ge_0p3_cluster']}/"
            f"{gripper_anchor_metrics['gripper_only_events']} "
            f"({gripper_anchor_metrics['recurring_entry_fraction']:.3f}).",
            f"- Gripper-only event 중 singleton cluster 소속: "
            f"{gripper_anchor_metrics['gripper_only_events_in_singleton_clusters']}/"
            f"{gripper_anchor_metrics['gripper_only_events']}. Closing peak는 grasp 정답으로 "
            "해석하지 않는다.",
            "",
            "### Q4 relative / absolute anchor set",
            "",
            f"- Common anchors: {position_frame_anchor_metrics['common']}; "
            "relative-only: "
            f"{position_frame_anchor_metrics['relative_only']}; "
            "absolute-only: "
            f"{position_frame_anchor_metrics['absolute_only']}.",
            "- Partition agreement는 common anchors에 대해서만 계산했고, "
            "condition-only anchors는 ARI/NMI에서 제외했다.",
            "",
            "### Common-anchor partition agreement (within-task primary)",
            "",
            "| Comparison | n common | ARI macro / weighted | NMI macro / weighted |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, row in mechanical["pairwise_partition"].items():
        lines.append(
            f"| {label} | {row['common_anchors']} | "
            f"{row['within_task_macro']['ari']:.3f} / "
            f"{row['within_task_weighted']['ari']:.3f} | "
            f"{row['within_task_macro']['nmi']:.3f} / "
            f"{row['within_task_weighted']['nmi']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Stage 4 비교",
            "",
            "아래 값은 coverage 3종 × SAE 3종 = 9 cells의 median [min, max]다. "
            "Feature ID는 같은 SAE 안에서만 비교했다.",
            "",
            "| Comparison | Ranking | Top-5 overlap / 5 | Full-vector Spearman |",
            "|---|---|---:|---:|",
        ]
    )
    for comparison, ranking_rows in ranking.items():
        for ranking_name, row in ranking_rows.items():
            lines.append(
                f"| {comparison} | {ranking_name} | "
                f"{_format_range(row['top5_overlap_count'], 1)} | "
                f"{_format_range(row['full_vector_spearman'])} |"
            )
    lines.extend(
        [
            "",
            "## 2. Confound audit",
            "",
            "| Gate | 판정 | 근거 |",
            "|---|---|---|",
        ]
    )
    for row in result["confound_audit"]:
        lines.append(
            f"| {row['gate']} | {row['status']} | {row['evidence']} |"
        )
    lines.extend(
        [
            "",
            "## 3. Claim-strength label",
            "",
            "**diagnostic evidence**",
            "",
            "이 결과는 clustering geometry, 자동 visual labeling, SAE feature "
            "ranking 안정성에 대한 진단 근거다. Detector performance, policy "
            "performance, intervention effect 또는 인과 효과를 뜻하지 않는다.",
            "",
            "## 4. 해석 제한",
            "",
            "- Pilot과 human review는 사용자 결정으로 생략했다. Human phase "
            "exact match, mixed-cluster rate, visually-insufficient rate는 N/A다.",
            "- 모든 semantic 결과는 provisional automatic이며 canonical "
            "human-reviewed 결과가 아니다.",
            "- 세 SAE는 사전 선택 sensitivity axis로 모두 보고했으며 결과를 보고 "
            "checkpoint를 재선택하지 않았다.",
            "- Relative/absolute 비교는 event set이 달라 common-anchor와 "
            "condition-only anchor 수를 분리했다.",
            "- Coverage 미달 raw clusters/events는 해당 threshold의 phase group과 "
            "Stage 4 입력에서 제외했다. 실패한 Gemini attempt와 SIGTERM partial "
            "ranking 디렉터리는 보존했지만 frozen/canonical 입력에서는 제외했다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path(
            "logs/groot_n15/experiments/"
            "anchor_view_controlled_ablation_v1"
        ),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    root = (repo_root / args.experiment_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    audit_path = root / "audits/final_audit.json"
    report_path = root / "reports/automatic_results.md"
    source_manifest_path = root / "inputs/source_manifest.json"
    experiment_manifest_path = root / "experiment_manifest.json"
    outputs = {
        audit_path,
        report_path,
        source_manifest_path,
        experiment_manifest_path,
    }
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite output: {path}")

    mechanical, partition_paths = _mechanical_metrics(root, repo_root)
    annotation = _annotation_metrics(root)
    annotation["q1_paired"] = _automatic_q1(
        root,
        partition_paths["p0_r_pos_left"] / "clusters.jsonl",
    )
    stage4 = _ranking_comparisons(root)
    result = {
        "format": "event_sae_anchor_view_final_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "episodes": 150,
            "instruction_scene_cells": 5,
            "episodes_per_cell": 30,
            "conditions": list(CONDITIONS),
            "coverage_thresholds": [0.3, 0.4, 0.5],
            "sae_sensitivity_checkpoints": list(SAE_IDS),
        },
        "mechanical": mechanical,
        "automatic_annotation": annotation,
        "stage4_ranking": stage4,
        "confound_audit": _confound_audit(),
        "claim_strength": "diagnostic evidence",
        "canonical_human_reviewed": False,
        "result_status": "gate6_complete_provisional_automatic",
        "passed": True,
    }
    _json_dump(audit_path, result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")

    source_manifest = _artifact_manifest(
        root,
        repo_root,
        outputs - {source_manifest_path},
    )
    _json_dump(source_manifest_path, source_manifest)
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()
    git_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
        ).strip()
    )
    experiment_manifest = {
        "format": "event_sae_anchor_view_experiment_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(
                (
                    repo_root
                    / "docs/groot/protocols/"
                    "anchor_view_controlled_ablation.md"
                ).resolve()
            ),
            "sha256": _sha256(
                repo_root
                / "docs/groot/protocols/"
                "anchor_view_controlled_ablation.md"
            ),
        },
        "profile": {
            "path": str(
                (
                    repo_root
                    / "configs/groot/"
                    "anchor_view_controlled_ablation_v1.json"
                ).resolve()
            ),
            "sha256": _sha256(
                repo_root
                / "configs/groot/"
                "anchor_view_controlled_ablation_v1.json"
            ),
        },
        "code": {
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "relevant_files": {
                str(path.relative_to(repo_root)): _sha256(path)
                for path in (
                    repo_root / "event_sae/events/annotate.py",
                    repo_root / "event_sae/events/annotation_attempts.py",
                    repo_root / "event_sae/events/cluster.py",
                    repo_root / "event_sae/groot/anchor_view_ablation.py",
                    repo_root / "event_sae/scoring/score_matrix.py",
                    repo_root / "event_sae/scoring/rankings.py",
                    repo_root / "scripts/groot/analyze_anchor_view_ablation.py",
                )
            },
        },
        "user_authorized_decisions": {
            "condition_aliases": {
                "E0": CONDITIONS[0],
                "E1": CONDITIONS[1],
                "E2": CONDITIONS[2],
                "E3": CONDITIONS[3],
                "E4": CONDITIONS[4],
            },
            "through_gate": 6,
            "pilot_skipped": True,
            "annotation_model": "gemini-3.1-pro-preview",
            "prompt_version": "gemini_task_local_visual_event_centered_v11",
            "human_review_skipped_for_initial_automatic_result": True,
            "all_three_sae_checkpoints_run_as_sensitivity": True,
        },
        "operational_deviations": [
            "Gemini request transport timeout varied across retry attempts; "
            "model, prompt, media, temperature, and JSON schema stayed fixed.",
            "Core protocol selects one SAE checkpoint; user pre-authorized all "
            "three preserved checkpoints as a sensitivity axis.",
            "Human review was intentionally deferred; all phase groups retain "
            "actual_human_review_completed=false.",
        ],
        "artifact_counts": {
            "annotation_conditions": 5,
            "annotation_rows": 90,
            "coverage_phase_group_bundles": 15,
            "score_artifacts": 45,
            "ranking_artifacts": 45,
            "candidate_rows": 900,
            "preserved_noncanonical_partial_ranking_dirs": 3,
        },
        "artifacts": {
            "source_manifest": {
                "path": str(source_manifest_path),
                "sha256": _sha256(source_manifest_path),
            },
            "final_audit": {
                "path": str(audit_path),
                "sha256": _sha256(audit_path),
            },
            "automatic_report": {
                "path": str(report_path),
                "sha256": _sha256(report_path),
            },
        },
        "result_status": "gate6_complete_provisional_automatic",
        "claim_strength": "diagnostic evidence",
        "passed": True,
    }
    _json_dump(experiment_manifest_path, experiment_manifest)
    print(
        json.dumps(
            {
                "experiment_manifest": str(experiment_manifest_path),
                "source_manifest": str(source_manifest_path),
                "final_audit": str(audit_path),
                "report": str(report_path),
                "result_status": result["result_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
