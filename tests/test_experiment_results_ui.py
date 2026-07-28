import json
import shutil
from pathlib import Path

import pytest

from event_sae.groot.oracle_results import (
    build_oracle_phase_results_dataset,
)
from event_sae.groot.phase_feature_results import RESULT_RANKINGS
from event_sae.groot.phase_feature_results import (
    DIRECTIONAL_ALIGNMENT_RELATIVE,
    build_directional_alignment_dataset,
)
from event_sae.groot.results_browser import (
    build_experiment_results_service,
    build_experiment_results_dataset,
)
from scripts.review_clusters import (
    DEFAULT_ORACLE_EXPERIMENT_ROOT,
    DEFAULT_RESULTS_EXPERIMENT_ROOT,
    DEFAULT_RESULTS_UI_PATH,
    build_parser,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture_oracle_results(
    tmp_path: Path,
    *,
    with_rankings: bool = False,
) -> Path:
    root = tmp_path / "oracle"
    keyframes = root / "oracle_keyframes"
    trajectory = root / "trajectory"
    multiview = root / "features/multiview"
    cluster_run = root / "clustering/c2_siglip_only"
    cluster_bundle = cluster_run / "phase_state_clusters"
    for directory in (
        keyframes,
        trajectory,
        multiview,
        cluster_bundle,
    ):
        directory.mkdir(parents=True)

    sample_id = "oracle_ep0000_kf000_s0000"
    phase_group_id = "oracle_task_phase_reach"
    state_cluster_id = "oracle_task_phase_reach_cluster_00"
    event = {
        "sample_id": sample_id,
        "episode_num": 0,
        "task_episode_idx": 7,
        "task_id": 5,
        "task_description": "Pick bread.",
        "success": True,
        "phase_scheme": "event_state",
        "phase": "reach-to-object",
        "state_env_step_index": 0,
        "activation_env_step_index": 0,
        "action_token_offset": 0,
        "progress_percent": 0.0,
        "anchor_source": "oracle_phase_entry",
        "event_labels": [],
        "oracle_upper_bound": True,
    }
    _write_jsonl(keyframes / "oracle_phase_events.jsonl", [event])
    _write_jsonl(
        keyframes / "oracle_phase_cluster_assignments.jsonl",
        [
            {
                "sample_id": sample_id,
                "cluster_id": phase_group_id,
            }
        ],
    )
    _write_jsonl(
        keyframes / "oracle_phase_cluster_annotations.jsonl",
        [
            {
                "cluster_id": phase_group_id,
                "task_description": "Pick bread.",
                "phase": "reach-to-object",
                "phrase": "simulator-oracle anchor in reach-to-object",
                "num_members": 1,
                "num_episodes": 1,
                "episode_coverage": 1.0,
            }
        ],
    )
    (keyframes / "oracle_phase_keyframes_manifest.json").write_text(
        json.dumps(
            {
                "format": "event_sae_oracle_phase_keyframes_v1",
                "claim_scope": "simulator-oracle diagnostic upper bound",
                "label_source": "env_step_phases",
                "label_resolution": "environment_state",
                "num_keyframes": 1,
                "num_source_episodes": 1,
                "phase_counts": {"reach-to-object": 1},
                "activation_alignment": {
                    "state_index": "env_step_phases[k] labels state s_k",
                    "phase_action_index": "state s_k maps to action k",
                    "causal_action_index": "transition uses action k-1",
                },
            }
        ),
        encoding="utf-8",
    )
    (trajectory / "trajectory_manifest.json").write_text(
        json.dumps({"inventory_verified": False}),
        encoding="utf-8",
    )

    for view in ("left", "right", "wrist"):
        view_root = root / "media" / view
        frame_root = view_root / "frames" / sample_id
        frame_root.mkdir(parents=True)
        frame_paths = []
        for index in range(5):
            frame_path = frame_root / f"frame_{index:02d}.jpg"
            frame_path.write_bytes(b"jpeg")
            frame_paths.append(str(frame_path.resolve()))
        _write_jsonl(
            view_root / "samples.jsonl",
            [
                {
                    "sample_id": sample_id,
                    "view_name": view,
                    "phase": "reach-to-object",
                    "frame_paths": frame_paths,
                    "anchor_env_step_error": 1,
                    "phase_frame_padding_count": 0,
                    "frames_within_oracle_phase": True,
                }
            ],
        )
        (view_root / "packaging_report.json").write_text(
            json.dumps(
                {
                    "format": "event_sae_oracle_phase_media_v1",
                    "view_name": view,
                    "passed": True,
                    "alignment": (
                        "frame f represents state env_step=1+f*2"
                    ),
                }
            ),
            encoding="utf-8",
        )

    (multiview / "event_features_manifest.json").write_text(
        json.dumps(
            {
                "format": "event_sae_multiview_event_features_v1",
                "passed": True,
                "num_samples": 1,
            }
        ),
        encoding="utf-8",
    )
    cluster = {
        "cluster_id": state_cluster_id,
        "task_id": 5,
        "task_description": "Pick bread.",
        "phase_scheme": "event_state",
        "phase": "reach-to-object",
        "phase_cluster_index": 0,
        "num_members": 1,
        "episode_coverage": 1.0,
        "phase_episode_coverage": 1.0,
        "task_episode_coverage": 1.0,
        "total_phase_episodes": 1,
        "total_task_episodes": 1,
        "meets_min_coverage": True,
        "representative_sample_ids": [sample_id],
        "member_sample_ids": [sample_id],
        "oracle_upper_bound": True,
    }
    _write_jsonl(cluster_bundle / "clusters.jsonl", [cluster])
    _write_jsonl(
        cluster_bundle / "cluster_assignments.jsonl",
        [
            {
                "cluster_id": state_cluster_id,
                "sample_id": sample_id,
            }
        ],
    )
    _write_jsonl(
        cluster_bundle / "cluster_annotations.jsonl",
        [
            {
                "cluster_id": state_cluster_id,
                "phase": "reach-to-object",
                "phrase": "simulator-oracle reach / state cluster 00",
                "oracle_upper_bound": True,
                "actual_human_review_completed": False,
            }
        ],
    )
    clustering = {
        "format": "event_sae_oracle_phase_state_clustering_v1",
        "claim_scope": "simulator-oracle diagnostic upper bound",
        "annotation_mode": "programmatic_oracle_no_vlm",
        "coverage_scope": "episodes_with_oracle_phase_keyframe",
        "distance_threshold": 0.18,
        "vision_weight": 1.0,
        "state_weight": 0.0,
        "progress_weight": 0.0,
        "progress_used_for_fitting": False,
        "success_used_for_fitting_or_selection": False,
        "min_coverage": 0.5,
        "num_events": 1,
        "num_clusters": 1,
        "num_singleton_clusters": 1,
        "singleton_event_fraction": 1.0,
        "median_cluster_size": 1.0,
        "max_cluster_size": 1,
        "num_clusters_meeting_min_coverage": 1,
        "partition_summaries": [
            {
                "phase_scheme": "event_state",
                "phase": "reach-to-object",
                "num_events": 1,
                "num_phase_episodes": 1,
                "num_task_episodes": 1,
                "num_clusters": 1,
                "num_singleton_clusters": 1,
            }
        ],
    }
    (cluster_run / "oracle_phase_clustering_manifest.json").write_text(
        json.dumps(
            {
                "format": "event_sae_oracle_phase_state_clustering_v1",
                "alignment": {
                    "max_state_vector_env_step_lag": 0,
                    "oracle_events_path": str(
                        (
                            keyframes / "oracle_phase_events.jsonl"
                        ).resolve()
                    ),
                },
                "clustering": clustering,
            }
        ),
        encoding="utf-8",
    )

    if with_rankings:
        rankings = root / "scores/checkpoint_a/rankings"
        rankings.mkdir(parents=True)
        (rankings / "ranking_config.json").write_text(
            json.dumps(
                {
                    "scores_pt": str(
                        (
                            root
                            / "scores/c2_siglip_only/event_feature_scores.pt"
                        ).resolve()
                    ),
                    "topk_run_dir": (
                        "/tmp/l15_sae1p2k_exec5_mean4_top96_v1/topk"
                    ),
                    "top_k": 5,
                    "top_n_per_row": 5,
                    "min_coverage": 0.5,
                }
            ),
            encoding="utf-8",
        )
        top_features = [
            {"feature_id": index, "score": 1.0 / (index + 1)}
            for index in range(5)
        ]
        for ranking in ("event_aligned", "window_mean"):
            _write_jsonl(
                rankings / f"{ranking}.jsonl",
                [
                    {
                        "ranking": ranking,
                        "cluster_id": phase_group_id,
                        "task_description": "Pick bread.",
                        "phase": "reach-to-object",
                        "top_features": top_features,
                    }
                ],
            )
        _write_jsonl(
            rankings / "task_mean.jsonl",
            [
                {
                    "ranking": "task_mean",
                    "task_description": "Pick bread.",
                    "top_features": top_features,
                }
            ],
        )
        _write_jsonl(
            rankings / "random_alive.jsonl",
            [
                {
                    "ranking": "random_alive",
                    "feature_id": 10,
                    "rank": 1,
                    "score": 0.0,
                }
            ],
        )
    return root


def _fixture_results(tmp_path: Path) -> Path:
    root = tmp_path / "experiment"
    conditions = [
        "e0_rel_pos_cluster_left_label_left",
        "e1_rel_pos_cluster_left_label_multiview",
    ]
    coverages = [0.3, 0.4, 0.5]
    checkpoints = [
        "l15_sae1p2k_exec5_mean4_top96_v1",
        "l15_sae10k_exec5_mean4_top96_v1",
        "l15_sae10k_bs8192_exec5_mean4_top96_v1",
    ]
    expected_runs = len(conditions) * len(coverages) * len(checkpoints)
    (root / "audits").mkdir(parents=True)
    (root / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "artifact_counts": {
                    "score_artifacts": expected_runs,
                    "ranking_artifacts": expected_runs,
                    "candidate_rows": expected_runs * len(RESULT_RANKINGS),
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "audits/final_audit.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-07-24T00:00:00+00:00",
                "passed": True,
                "result_status": "complete_test",
                "claim_strength": "diagnostic evidence",
                "scope": {
                    "conditions": conditions,
                    "coverage_thresholds": coverages,
                    "sae_sensitivity_checkpoints": checkpoints,
                },
                "mechanical": {
                    "partitions": {
                        "p0_r_pos_left": {
                            "anchor_source_counts": {"position": 12},
                            "block_normalization": "balanced",
                            "distance_threshold": 0.18,
                            "max_cluster_size": 4,
                            "median_cluster_size": 2.0,
                            "num_clusters": 6,
                            "num_clusters_meeting_min_coverage": 2,
                            "num_events": 12,
                            "num_singleton_clusters": 1,
                            "singleton_event_fraction": 1 / 12,
                        }
                    },
                    "pairwise_partition": {
                        "q2_p0_vs_p1_all_r_pos": {
                            "common_anchors": 12,
                            "left_only": 0,
                            "right_only": 0,
                            "pooled": {"ari": 0.8, "nmi": 0.9},
                            "within_task_macro": {"ari": 0.75, "nmi": 0.85},
                            "within_task_weighted": {
                                "ari": 0.75,
                                "nmi": 0.85,
                            },
                            "by_task": {},
                        }
                    },
                    "q3_gripper_only": {
                        "gripper_only_events": 3,
                        "entered_coverage_ge_0p3_cluster": 2,
                    },
                    "q4_anchor_sets": {
                        "common": 10,
                        "relative_only": 1,
                        "absolute_only": 2,
                    },
                },
                "stage4_ranking": {
                    "aggregates": {},
                    "cells": [],
                },
            }
        ),
        encoding="utf-8",
    )
    partition_dir = root / "partitions/p0_r_pos_left"
    partition_dir.mkdir(parents=True)
    _write_jsonl(
        partition_dir / "clusters.jsonl",
        [
            {
                "cluster_id": "drawer_phase_pull",
                "task_description": "Open the drawer.",
                "episode_coverage": 0.6,
            }
        ],
    )

    for condition in conditions:
        for coverage in ("cov0p3", "cov0p4", "cov0p5"):
            coverage_dir = root / "stage4" / condition / coverage
            phase_groups_dir = coverage_dir / "phase_groups"
            phase_groups_dir.mkdir(parents=True)
            _write_jsonl(
                coverage_dir / "filtered_finalized_annotations.jsonl",
                [
                    {
                        "cluster_id": "drawer_phase_pull",
                        "task_description": "Open the drawer.",
                        "phase": "pull",
                        "phrase": "pulling",
                        "actual_human_review_completed": False,
                    }
                ],
            )
            _write_jsonl(
                phase_groups_dir / "phase_groups.jsonl",
                [
                    {
                        "phase_group_id": "open_the_drawer_phase_pull",
                        "task_description": "Open the drawer.",
                        "phase": "pull",
                        "phrase": "pull phase group",
                        "source_phrases": ["pulling"],
                        "source_cluster_ids": ["drawer_phase_pull"],
                        "num_source_clusters": 1,
                        "num_members": 6,
                        "episode_coverage": 0.6,
                        "total_task_episodes": 10,
                        "review_mode": "automatic",
                        "review_verdict": "assumed_approved",
                        "actual_human_review_completed": False,
                    }
                ],
            )
            (phase_groups_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "result_status": "provisional_automatic",
                        "actual_human_review_completed": False,
                        "require_human_review": False,
                        "num_input_clusters": 1,
                        "num_reviewed_clusters": 1,
                        "num_grouped_samples": 6,
                        "num_phase_groups": 1,
                    }
                ),
                encoding="utf-8",
            )
            run_dir = (
                root
                / "stage4"
                / condition
                / coverage
                / checkpoints[0]
            )
            rankings_dir = run_dir / "rankings"
            rankings_dir.mkdir(parents=True)
            (run_dir / "event_feature_scores.pt").write_bytes(b"scores")
            (rankings_dir / "ranking_config.json").write_text(
                json.dumps(
                    {
                        "scores_pt": str(
                            (run_dir / "event_feature_scores.pt").resolve()
                        ),
                        "topk_run_dir": str(
                            (
                                root
                                / "topk_runs"
                                / checkpoints[0]
                                / "topk"
                            ).resolve()
                        ),
                        "top_k": 1,
                        "top_n_per_row": 5,
                    }
                ),
                encoding="utf-8",
            )
            _write_jsonl(
                rankings_dir / "candidates.jsonl",
                [
                    {
                        "ranking": ranking,
                        "rank": 1,
                        "feature_id": index + 10,
                        **(
                            {"score": 1.0 + index}
                            if ranking != "random_alive"
                            else {}
                        ),
                    }
                    for index, ranking in enumerate(RESULT_RANKINGS)
                ],
            )
            for index, ranking in enumerate(RESULT_RANKINGS):
                if ranking == "random_alive":
                    rows = [{"ranking": ranking, "feature_id": 13}]
                else:
                    rows = [
                        {
                            "ranking": ranking,
                            "task_description": "Open the drawer.",
                            **(
                                {
                                    "cluster_id": "open_the_drawer_phase_pull",
                                    "phrase": "pull phase group",
                                    "phase": "pull",
                                }
                                if ranking != "task_mean"
                                else {}
                            ),
                            "top_features": [
                                {
                                    "feature_id": index * 10 + offset,
                                    "score": 2.5 - offset / 10,
                                }
                                for offset in range(5)
                            ],
                        }
                    ]
                _write_jsonl(rankings_dir / f"{ranking}.jsonl", rows)
            for checkpoint in checkpoints[1:]:
                copied_run_dir = run_dir.parent / checkpoint
                shutil.copytree(run_dir, copied_run_dir)
                copied_config_path = (
                    copied_run_dir / "rankings" / "ranking_config.json"
                )
                copied_config = json.loads(
                    copied_config_path.read_text(encoding="utf-8")
                )
                copied_config["scores_pt"] = str(
                    (copied_run_dir / "event_feature_scores.pt").resolve()
                )
                copied_config["topk_run_dir"] = str(
                    (root / "topk_runs" / checkpoint / "topk").resolve()
                )
                copied_config_path.write_text(
                    json.dumps(copied_config),
                    encoding="utf-8",
                )
    return root


def test_results_dataset_indexes_complete_experiment_grid(tmp_path: Path) -> None:
    root = _fixture_results(tmp_path)

    payload = build_experiment_results_dataset(root)

    assert payload["format"] == "event_sae_stage4_results_browser_v1"
    assert payload["meta"]["expected_runs"] == 18
    assert payload["meta"]["available_runs"] == 18
    assert payload["meta"]["candidate_rows"] == 72
    assert len(payload["runs"]) == 18
    assert len(payload["facets"]["checkpoints"]) == 3
    assert {run["coverage_label"] for run in payload["runs"]} == {
        "0.3",
        "0.4",
        "0.5",
    }
    assert {run["condition_code"] for run in payload["runs"]} == {"E0", "E1"}
    assert all(run["score_artifact_available"] for run in payload["runs"])
    assert all(
        set(run["candidates"]) == set(RESULT_RANKINGS)
        for run in payload["runs"]
    )
    assert all(
        run["phase_group_count"] == 1 and run["task_count"] == 1
        for run in payload["runs"]
    )
    assert len(payload["phase_sets"]) == 6
    assert all(
        phase_set["num_phase_groups"] == 1
        and phase_set["num_reviewed_clusters"] == 1
        and not phase_set["actual_human_review_completed"]
        for phase_set in payload["phase_sets"]
    )
    assert (
        payload["phase_sets"][0]["phase_groups"][0]["phase_group_id"]
        == "open_the_drawer_phase_pull"
    )
    assert payload["clustering"]["partitions"][0]["num_clusters"] == 6
    assert payload["clustering"]["comparisons"][0]["common_anchors"] == 12
    assert "scores_pt" not in json.dumps(payload)
    assert str(root) not in json.dumps(payload)


def test_results_dataset_rejects_incomplete_run(tmp_path: Path) -> None:
    root = _fixture_results(tmp_path)
    missing = next(root.glob("stage4/**/event_feature_scores.pt"))
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="Incomplete result run"):
        build_experiment_results_dataset(root)


def test_results_dataset_rejects_phase_ranking_key_mismatch(
    tmp_path: Path,
) -> None:
    root = _fixture_results(tmp_path)
    ranking_path = next(root.glob("stage4/**/rankings/event_aligned.jsonl"))
    rows = [
        {
            **row,
            "cluster_id": "not_a_phase_group",
        }
        for row in (
            json.loads(line)
            for line in ranking_path.read_text(encoding="utf-8").splitlines()
        )
    ]
    _write_jsonl(ranking_path, rows)

    with pytest.raises(ValueError, match="phase keys do not match phase groups"):
        build_experiment_results_dataset(root)


def test_results_dataset_rejects_duplicate_phase_ranking_key(
    tmp_path: Path,
) -> None:
    root = _fixture_results(tmp_path)
    ranking_path = next(root.glob("stage4/**/rankings/window_mean.jsonl"))
    rows = [
        json.loads(line)
        for line in ranking_path.read_text(encoding="utf-8").splitlines()
    ]
    _write_jsonl(ranking_path, [*rows, rows[0]])

    with pytest.raises(ValueError, match="contains duplicate phase keys"):
        build_experiment_results_dataset(root)


def test_results_dataset_rejects_short_phase_feature_list(
    tmp_path: Path,
) -> None:
    root = _fixture_results(tmp_path)
    ranking_path = next(root.glob("stage4/**/rankings/event_aligned.jsonl"))
    rows = [
        json.loads(line)
        for line in ranking_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["top_features"] = rows[0]["top_features"][:4]
    _write_jsonl(ranking_path, rows)

    with pytest.raises(ValueError, match="has fewer than 5 features"):
        build_experiment_results_dataset(root)


def test_results_dataset_rejects_duplicate_phase_group_key(
    tmp_path: Path,
) -> None:
    root = _fixture_results(tmp_path)
    phase_path = next(root.glob("stage4/**/phase_groups/phase_groups.jsonl"))
    rows = [
        json.loads(line)
        for line in phase_path.read_text(encoding="utf-8").splitlines()
    ]
    _write_jsonl(phase_path, [*rows, rows[0]])
    summary_path = phase_path.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["num_phase_groups"] = 2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate task/group keys"):
        build_experiment_results_dataset(root)


def test_results_dataset_rejects_phase_source_count_mismatch(
    tmp_path: Path,
) -> None:
    root = _fixture_results(tmp_path)
    phase_path = next(root.glob("stage4/**/phase_groups/phase_groups.jsonl"))
    rows = [
        json.loads(line)
        for line in phase_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["num_source_clusters"] = 2
    _write_jsonl(phase_path, rows)

    with pytest.raises(ValueError, match="source-cluster count mismatch"):
        build_experiment_results_dataset(root)


def test_results_dataset_rejects_unknown_phase_source_cluster(
    tmp_path: Path,
) -> None:
    root = _fixture_results(tmp_path)
    phase_path = next(root.glob("stage4/**/phase_groups/phase_groups.jsonl"))
    rows = [
        json.loads(line)
        for line in phase_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["source_cluster_ids"] = ["stale_cluster_id"]
    _write_jsonl(phase_path, rows)

    with pytest.raises(ValueError, match="references an unknown raw cluster"):
        build_experiment_results_dataset(root)


def test_results_dataset_rejects_filtered_annotation_mismatch(
    tmp_path: Path,
) -> None:
    root = _fixture_results(tmp_path)
    annotation_path = next(
        root.glob("stage4/**/filtered_finalized_annotations.jsonl")
    )
    rows = [
        json.loads(line)
        for line in annotation_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["phase"] = "reach"
    _write_jsonl(annotation_path, rows)

    with pytest.raises(ValueError, match="source annotation task/phase mismatch"):
        build_experiment_results_dataset(root)


def test_results_dataset_rejects_human_review_status_mismatch(
    tmp_path: Path,
) -> None:
    root = _fixture_results(tmp_path)
    phase_path = next(root.glob("stage4/**/phase_groups/phase_groups.jsonl"))
    rows = [
        json.loads(line)
        for line in phase_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["actual_human_review_completed"] = True
    _write_jsonl(phase_path, rows)

    with pytest.raises(ValueError, match="human-review status"):
        build_experiment_results_dataset(root)


def test_results_dataset_rejects_canonical_phase_metadata_mismatch(
    tmp_path: Path,
) -> None:
    root = _fixture_results(tmp_path)
    ranking_path = next(root.glob("stage4/**/rankings/event_aligned.jsonl"))
    rows = [
        json.loads(line)
        for line in ranking_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["phrase"] = "stale phrase"
    _write_jsonl(ranking_path, rows)

    with pytest.raises(ValueError, match="canonical phase group"):
        build_experiment_results_dataset(root)


def test_results_dataset_rejects_duplicate_top_five_feature(
    tmp_path: Path,
) -> None:
    root = _fixture_results(tmp_path)
    ranking_path = next(root.glob("stage4/**/rankings/window_mean.jsonl"))
    rows = [
        json.loads(line)
        for line in ranking_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["top_features"][1]["feature_id"] = rows[0]["top_features"][0][
        "feature_id"
    ]
    _write_jsonl(ranking_path, rows)

    with pytest.raises(ValueError, match="duplicate IDs in its Top-5"):
        build_experiment_results_dataset(root)


def test_results_dataset_rejects_top_n_below_five(tmp_path: Path) -> None:
    root = _fixture_results(tmp_path)
    config_path = next(root.glob("stage4/**/rankings/ranking_config.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["top_n_per_row"] = 4
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="top_n_per_row >= 5"):
        build_experiment_results_dataset(root)


def test_results_dataset_rejects_checkpoint_provenance_mismatch(
    tmp_path: Path,
) -> None:
    root = _fixture_results(tmp_path)
    config_path = next(root.glob("stage4/**/rankings/ranking_config.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["scores_pt"] = str((root / "stale_scores.pt").resolve())
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="score provenance"):
        build_experiment_results_dataset(root)


def test_results_application_loads_dedicated_read_only_ui(
    tmp_path: Path,
) -> None:
    root = _fixture_results(tmp_path)
    ui_path = tmp_path / "results.html"
    ui_path.write_text("<html>results</html>", encoding="utf-8")

    application = build_experiment_results_service(
        experiment_root=root,
        ui_path=ui_path,
    )

    assert application.ui == b"<html>results</html>"
    assert application.data()["meta"]["available_runs"] == 18
    with pytest.raises(KeyError):
        application.explorer("not-a-condition")


def test_directional_alignment_builds_instruction_family_and_global_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    root = tmp_path / "experiment"
    analysis_root = root / DIRECTIONAL_ALIGNMENT_RELATIVE
    analysis_root.mkdir(parents=True)
    (analysis_root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "directional_phase_view_analysis_v1",
                "claim_strength": "diagnostic_evidence",
                "feature_identity_contract": {
                    "same_checkpoint_sha256": True,
                },
                "held_claims": ["causal effect is not established"],
            }
        ),
        encoding="utf-8",
    )
    identities = [
        (8, "Open the left drawer.", "left", "OpenDrawer"),
        (7, "Open the right drawer.", "right", "OpenDrawer"),
        (15, "Pick beer.", "beer", "PickPlaceCounterToCabinet"),
        (5, "Pick bread.", "bread", "PickPlaceCounterToCabinet"),
        (16, "Pick pizza cutter.", "pizza", "PickPlaceCounterToCabinet"),
    ]
    by_description = {}
    by_task = {}
    for task_id, description, cell_id, family in identities:
        identity = {
            "raw_task_id": task_id,
            "instruction_id": f"instruction-{task_id}",
            "cell_id": cell_id,
            "task_family_id": family,
            "task_family_label": family,
            "task_identity_source": "fixture",
        }
        by_description[description] = identity
        by_task[(task_id, description)] = identity
    registry = {
        "by_description": by_description,
        "by_task": by_task,
        "meta": {
            "available": True,
            "num_source_episodes": 150,
            "num_instruction_cells": 5,
            "num_task_families": 2,
            "task_families": [
                {
                    "task_family_id": "OpenDrawer",
                    "task_family_label": "Drawer",
                    "cell_ids": ["left", "right"],
                    "cell_count": 2,
                },
                {
                    "task_family_id": "PickPlaceCounterToCabinet",
                    "task_family_label": "PnP",
                    "cell_ids": ["beer", "bread", "pizza"],
                    "cell_count": 3,
                },
            ],
        },
    }

    oracle_raw = torch.tensor(
        [[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]] * 5
    )
    e3_raw = torch.tensor(
        [[8.0, 7.0, 6.0, 2.0, 1.0, 5.0, 4.0, 3.0]] * 5
    )

    def write_score(
        source: str,
        view: str,
        *,
        raw: torch.Tensor,
        e3: bool,
    ) -> None:
        row_keys = []
        for task_id, description, _, family in identities:
            phase = (
                "grasp"
                if view == "coarse4_exact_rescore"
                or family == "PickPlaceCounterToCabinet"
                else "grasp-handle"
            )
            row_keys.append(
                {
                    "cluster_id": f"{source}-{task_id}-{phase}",
                    "task_description": description,
                    "task_id": task_id,
                    "phase": phase,
                    "phrase": phase,
                    "episode_coverage": 1.0,
                }
            )
        pulse = torch.zeros_like(raw)
        step_up = torch.zeros_like(raw)
        step_down = torch.zeros_like(raw)
        step_up[:, 0] = 3.0
        step_down[:, 1] = 3.0
        if e3:
            step_up[:, 2] = 3.0
        else:
            pulse[:, 2] = 3.0
        pulse[:, 3:] = 2.0
        path = (
            analysis_root
            / "scores"
            / source
            / view
            / "w5"
            / "event_feature_scores.pt"
        )
        path.parent.mkdir(parents=True)
        torch.save(
            {
                "window_size": 5,
                "row_semantics": "fixture",
                "score_definitions": {"matrix_raw": "fixture score"},
                "row_keys": row_keys,
                "row_results": [],
                "selected_events": [],
                "matrix_raw": raw,
                "matrix_pulse": pulse,
                "matrix_step_up": step_up,
                "matrix_step_down": step_down,
            },
            path,
        )

    for view in ("fine_original", "coarse4_exact_rescore"):
        write_score("oracle_full", view, raw=oracle_raw, e3=False)
        write_score("v12_e3_cov0p3", view, raw=e3_raw, e3=True)

    real_torch_load = torch.load
    loaded_paths: list[Path] = []

    def tracked_torch_load(path: Path, *args, **kwargs):
        loaded_paths.append(Path(path))
        return real_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_torch_load)
    payload = build_directional_alignment_dataset(
        experiment_root=root,
        task_identity_registry=registry,
    )

    assert len(loaded_paths) == 4
    assert len(set(loaded_paths)) == 4
    assert payload["format"] == "event_sae_oracle_e3_alignment_v1"
    assert payload["method"]["ranking"] == "matrix_raw"
    assert payload["method"]["direction_is_rollout_consistency"] is False
    assert [view["id"] for view in payload["views"]] == [
        "fine_original",
        "coarse4_exact_rescore",
    ]
    for view in payload["views"]:
        levels = {level["id"]: level for level in view["levels"]}
        assert levels["instruction"]["summary"] == {
            "comparison_count": 5,
            "id_overlap_count": 15,
            "same_direction_overlap_count": 10,
            "same_direction_fraction": 2 / 3,
        }
        assert levels["family"]["summary"]["comparison_count"] == 2
        assert levels["task_agnostic"]["summary"]["comparison_count"] == 1
        grasp = levels["task_agnostic"]["rows"][0]
        assert grasp["phase"] == "grasp"
        assert [item["feature_id"] for item in grasp["overlap"]] == [0, 1, 2]
        assert [item["same_direction"] for item in grasp["overlap"]] == [
            True,
            True,
            False,
        ]


def test_oracle_results_adapter_supports_partial_clustering_pilot(
    tmp_path: Path,
) -> None:
    root = _fixture_oracle_results(tmp_path)

    payload, media = build_oracle_phase_results_dataset(root)

    assert payload["format"] == "event_sae_oracle_phase_results_browser_v1"
    assert payload["available"] is True
    assert payload["status"] == "clustering_ready"
    assert payload["claim_scope"] == (
        "simulator-oracle diagnostic upper bound"
    )
    assert payload["annotation"] == {
        "label_source": "env_step_phases",
        "mode": "programmatic_oracle_no_vlm",
        "review_status": "not_applicable",
        "oracle_upper_bound": True,
    }
    assert payload["meta"]["inventory_verified"] is False
    assert payload["default_variant_id"] == "c2_siglip_only"
    assert payload["variants"][0]["distance_threshold"] == 0.18
    assert payload["variants"][0]["min_coverage"] == 0.5
    assert payload["variants"][0]["phase_scheme"] == "event_state"
    assert payload["variants"][0]["coverage_schema"] == (
        "explicit_phase_and_task"
    )
    cluster = payload["variants"][0]["clusters"][0]
    assert cluster["phase_episode_coverage"] == 1.0
    assert cluster["task_episode_coverage"] == 1.0
    assert cluster["total_task_episodes"] == 1
    assert payload["phase_groups"][0]["coverage_scope"] == (
        "all_task_episodes"
    )
    assert payload["phase_groups"][0]["source_cluster_ids_by_variant"] == {
        "c2_siglip_only": [
            "oracle_task_phase_reach_cluster_00",
        ]
    }
    assert len(media) == 3
    assert len(media[("oracle_ep0000_kf000_s0000", "left")]) == 5
    assert payload["stages"][-1]["state"] == "waiting"


def test_oracle_results_adapter_loads_completed_phase_ranking(
    tmp_path: Path,
) -> None:
    root = _fixture_oracle_results(tmp_path, with_rankings=True)

    payload, _ = build_oracle_phase_results_dataset(root)

    assert payload["status"] == "feature_rankings_partial"
    assert payload["meta"]["num_ranking_packages"] == 1
    assert payload["meta"]["num_ranking_checkpoints"] == 1
    assert payload["meta"]["num_encoded_checkpoints"] == 0
    assert payload["default_checkpoint_id"] == (
        "l15_sae1p2k_exec5_mean4_top96_v1"
    )
    assert payload["ranking_checkpoints"] == [
        {
            "id": "l15_sae1p2k_exec5_mean4_top96_v1",
            "label": "SAE 1.2k",
            "batch_size": None,
            "training_steps": None,
            "sae_sha256": None,
            "dict_size": None,
            "topk": None,
            "total_rows": None,
            "lossless_topk": None,
            "topk_run_id": "topk",
            "is_reference": False,
            "encoded": False,
            "state": "ready",
            "num_discovered_packages": 1,
            "num_connected_packages": 1,
            "variant_ids": ["c2_siglip_only"],
            "analysis_ids": ["rankings"],
            "window_sizes": [],
            "scopes": ["phase"],
        }
    ]
    package = payload["ranking_packages"][0]
    assert package["scope"] == "phase"
    assert package["source_variant_id"] == "c2_siglip_only"
    assert package["checkpoint_label"] == "SAE 1.2k"
    assert len(
        package["rankings"]["event_aligned"][0]["top_features"]
    ) == 5


def test_oracle_results_adapter_links_versioned_w4_to_w5_cluster(
    tmp_path: Path,
) -> None:
    root = _fixture_oracle_results(tmp_path)
    source_variant = root / "clustering/c2_siglip_only"
    target_variant = (
        root / "clustering/c2_siglip_only_d0p08_causal_w5_v2"
    )
    shutil.copytree(source_variant, target_variant)
    selected_events_dir = root / "selection/causal_full_w5"
    selected_events_dir.mkdir(parents=True)
    selected_events_path = selected_events_dir / "oracle_phase_events.jsonl"
    shutil.copyfile(
        root / "oracle_keyframes/oracle_phase_events.jsonl",
        selected_events_path,
    )
    target_manifest_path = (
        target_variant / "oracle_phase_clustering_manifest.json"
    )
    target_manifest = json.loads(
        target_manifest_path.read_text(encoding="utf-8")
    )
    target_manifest["alignment"]["oracle_events_path"] = str(
        selected_events_path.resolve()
    )
    target_manifest_path.write_text(
        json.dumps(target_manifest),
        encoding="utf-8",
    )

    ranking_dir = root / "rankings/c2_siglip_only_d0p08_causal_w4_v2"
    ranking_dir.mkdir(parents=True)
    (ranking_dir / "ranking_config.json").write_text(
        json.dumps(
            {
                "scores_pt": str(
                    (
                        root
                        / "scores/c2_siglip_only_d0p08_causal_w4_v2"
                        / "event_feature_scores.pt"
                    ).resolve()
                ),
                "topk_run_dir": (
                    "/tmp/l15_sae1p2k_exec5_mean4_top96_v1/topk"
                ),
                "top_k": 5,
                "top_n_per_row": 5,
                "min_coverage": 0.5,
            }
        ),
        encoding="utf-8",
    )
    top_features = [
        {"feature_id": index, "score": 1.0 / (index + 1)}
        for index in range(5)
    ]
    state_cluster_id = "oracle_task_phase_reach_cluster_00"
    for ranking in ("event_aligned", "window_mean"):
        _write_jsonl(
            ranking_dir / f"{ranking}.jsonl",
            [
                {
                    "ranking": ranking,
                    "cluster_id": state_cluster_id,
                    "task_description": "Pick bread.",
                    "phase": "reach-to-object",
                    "top_features": top_features,
                }
            ],
        )
    _write_jsonl(
        ranking_dir / "task_mean.jsonl",
        [
            {
                "ranking": "task_mean",
                "task_description": "Pick bread.",
                "top_features": top_features,
            }
        ],
    )
    _write_jsonl(
        ranking_dir / "random_alive.jsonl",
        [
            {
                "ranking": "random_alive",
                "feature_id": 10,
                "rank": 1,
                "score": 0.0,
            }
        ],
    )

    payload, _ = build_oracle_phase_results_dataset(root)

    package = payload["ranking_packages"][0]
    assert package["score_run_id"] == (
        "c2_siglip_only_d0p08_causal_w4_v2"
    )
    assert package["source_variant_id"] == (
        "c2_siglip_only_d0p08_causal_w5_v2"
    )
    assert package["source_variant_inferred"] is True
    assert package["window_size"] == 4
    assert package["scope"] == "state_cluster"
    assert payload["default_variant_id"] == (
        "c2_siglip_only_d0p08_causal_w5_v2"
    )
    selected = next(
        variant
        for variant in payload["variants"]
        if variant["id"] == payload["default_variant_id"]
    )
    assert selected["analysis_label"] == "Causal W5 · v2"
    assert selected["num_ranking_packages"] == 1


def test_oracle_results_adapter_catalogs_three_sae_checkpoints(
    tmp_path: Path,
) -> None:
    root = _fixture_oracle_results(tmp_path)
    source_variant = root / "clustering/c2_siglip_only"
    target_variant = root / "clustering/phase_only_d0p18_causal_w5_v2"
    shutil.copytree(source_variant, target_variant)
    state_cluster_id = "oracle_task_phase_reach_cluster_00"
    top_features = [
        {"feature_id": index, "score": 1.0 / (index + 1)}
        for index in range(5)
    ]
    checkpoints = (
        (
            "topk96_sae1p2k",
            "bs4096_steps001200_seed0",
            "sae1p2k_v1",
            "1" * 64,
        ),
        (
            "topk96",
            "bs8192_steps005000_rows40960000_seed0",
            None,
            "5" * 64,
        ),
        (
            "topk96_sae10k",
            "bs4096_steps010000_seed0",
            "sae10k_v1",
            "a" * 64,
        ),
    )
    for topk_run_id, checkpoint_id, sae_suffix, sae_sha256 in checkpoints:
        topk_dir = root / "activations" / topk_run_id
        topk_dir.mkdir(parents=True)
        checkpoint_root = root / "checkpoints" / checkpoint_id
        (topk_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "token_topk_sparse_v1",
                    "checkpoint_source_manifest": str(
                        checkpoint_root / "groot_source_manifest.json"
                    ),
                    "sae_path": str(checkpoint_root / "trainer_0/ae.pt"),
                    "sae_sha256": sae_sha256,
                    "dict_size": 1536,
                    "topk": 96,
                    "total_rows": 50048,
                    "encoding_stats": {"lossless_topk": True},
                }
            ),
            encoding="utf-8",
        )
        for window_size in (4, 5):
            if sae_suffix is None:
                score_run_id = (
                    f"phase_only_d0p18_causal_w{window_size}_v2"
                )
            else:
                score_run_id = (
                    f"phase_only_d0p18_causal_w{window_size}_"
                    f"{sae_suffix}"
                )
            ranking_dir = root / "rankings" / score_run_id
            ranking_dir.mkdir(parents=True)
            (ranking_dir / "ranking_config.json").write_text(
                json.dumps(
                    {
                        "scores_pt": str(
                            root
                            / "scores"
                            / score_run_id
                            / "event_feature_scores.pt"
                        ),
                        "topk_run_dir": str(topk_dir),
                        "top_k": 10,
                        "top_n_per_row": 50,
                        "min_coverage": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            for ranking in ("event_aligned", "window_mean"):
                _write_jsonl(
                    ranking_dir / f"{ranking}.jsonl",
                    [
                        {
                            "ranking": ranking,
                            "cluster_id": state_cluster_id,
                            "task_description": "Pick bread.",
                            "phase": "reach-to-object",
                            "top_features": top_features,
                        }
                    ],
                )
            _write_jsonl(
                ranking_dir / "task_mean.jsonl",
                [
                    {
                        "ranking": "task_mean",
                        "task_description": "Pick bread.",
                        "top_features": top_features,
                    }
                ],
            )
            _write_jsonl(
                ranking_dir / "random_alive.jsonl",
                [
                    {
                        "ranking": "random_alive",
                        "feature_id": 10,
                        "rank": 1,
                        "score": 0.0,
                    }
                ],
            )

    payload, _ = build_oracle_phase_results_dataset(root)

    assert payload["status"] == "feature_rankings_ready"
    assert payload["meta"]["num_encoded_checkpoints"] == 3
    assert payload["meta"]["num_ranking_checkpoints"] == 3
    assert payload["meta"]["num_ranking_packages"] == 6
    assert payload["meta"]["num_unmapped_ranking_packages"] == 0
    assert [
        checkpoint["label"]
        for checkpoint in payload["ranking_checkpoints"]
    ] == [
        "SAE 1.2k · bs4096",
        "SAE 5k · bs8192",
        "SAE 10k · bs4096",
    ]
    assert all(
        checkpoint["num_connected_packages"] == 2
        and checkpoint["window_sizes"] == [4, 5]
        and checkpoint["state"] == "ready"
        for checkpoint in payload["ranking_checkpoints"]
    )
    assert payload["default_checkpoint_id"] == (
        "bs8192_steps005000_rows40960000_seed0"
    )
    assert payload["default_variant_id"] == (
        "phase_only_d0p18_causal_w5_v2"
    )
    assert all(
        package["source_variant_id"]
        == "phase_only_d0p18_causal_w5_v2"
        and package["source_mapping_verified"] is True
        and package["window_size"] in {4, 5}
        for package in payload["ranking_packages"]
    )


def test_oracle_unmapped_package_does_not_mark_checkpoint_ready(
    tmp_path: Path,
) -> None:
    root = _fixture_oracle_results(tmp_path, with_rankings=True)
    ranking_dir = root / "scores/checkpoint_a/rankings"
    for ranking in ("event_aligned", "window_mean"):
        rows = [
            {
                "ranking": ranking,
                "cluster_id": "unknown_cluster",
                "task_description": "Pick bread.",
                "phase": "reach-to-object",
                "top_features": [{"feature_id": 1, "score": 1.0}],
            }
        ]
        _write_jsonl(ranking_dir / f"{ranking}.jsonl", rows)

    payload, _ = build_oracle_phase_results_dataset(root)

    assert payload["status"] == "clustering_ready"
    assert payload["meta"]["num_discovered_ranking_packages"] == 1
    assert payload["meta"]["num_unmapped_ranking_packages"] == 1
    assert payload["meta"]["num_ranking_packages"] == 0
    assert payload["meta"]["num_ranking_checkpoints"] == 0
    assert payload["ranking_packages"][0]["source_variant_id"] is None
    assert payload["ranking_checkpoints"][0]["state"] == "discovered"
    assert payload["ranking_checkpoints"][0]["num_connected_packages"] == 0


def test_oracle_results_adapter_returns_waiting_state_for_missing_root(
    tmp_path: Path,
) -> None:
    payload, media = build_oracle_phase_results_dataset(
        tmp_path / "not-created"
    )

    assert payload["available"] is False
    assert payload["status"] == "waiting_for_artifacts"
    assert media == {}


def test_results_application_keeps_legacy_and_oracle_payloads_separate(
    tmp_path: Path,
) -> None:
    root = _fixture_results(tmp_path)
    oracle_root = _fixture_oracle_results(tmp_path)
    ui_path = tmp_path / "results.html"
    ui_path.write_text("<html>results</html>", encoding="utf-8")
    application = build_experiment_results_service(
        experiment_root=root,
        oracle_experiment_root=oracle_root,
        ui_path=ui_path,
    )

    assert application.data()["format"] == "event_sae_stage4_results_browser_v1"
    assert "oracle_phase" not in application.data()
    assert application.oracle()["format"] == (
        "event_sae_oracle_phase_results_browser_v1"
    )
    frame = application.oracle_frame(
        "oracle_ep0000_kf000_s0000",
        "wrist",
        4,
    )
    assert frame.name == "frame_04.jpg"


def test_results_command_defaults_to_controlled_ablation() -> None:
    args = build_parser().parse_args(["results"])

    assert args.experiment_root == DEFAULT_RESULTS_EXPERIMENT_ROOT
    assert args.oracle_experiment_root == DEFAULT_ORACLE_EXPERIMENT_ROOT
    assert args.ui_path == DEFAULT_RESULTS_UI_PATH
    assert args.host == "127.0.0.1"
    assert args.port == 8766


def test_results_ui_contains_experiment_explorer_controls() -> None:
    html = DEFAULT_RESULTS_UI_PATH.read_text(encoding="utf-8")

    for element_id in (
        "conditionFilter",
        "coverageFilter",
        "checkpointFilter",
        "rankingFilter",
        "validationFlow",
        "clusterAuditSummary",
        "phaseVerificationStatus",
        "phaseCheckpointMatrix",
        "phaseMatrixCount",
        "rankingHelp",
        "stage4OverviewCount",
        "stage4OverviewStatus",
        "stage4HeadlineGrid",
        "stage4FamilySupport",
        "stage4CrossFamily",
        "stage4RepresentativeGrid",
        "stage4CompactGrid",
        "stage4AuditNotice",
        "alignmentTab",
        "alignmentPanel",
        "alignmentCount",
        "alignmentSummary",
        "alignmentStatus",
        "alignmentTable",
        "controlledHeatmapMode",
        "controlledHeatmapLimit",
        "controlledHeatmapTask",
        "controlledHeatmapTable",
        "controlledHeatmapInspector",
        "conditionJourney",
        "selectedRunSummary",
        "matrix",
        "runList",
        "runDetail",
        "comparisonGrid",
        "comparisonRanking",
        "partitionGrid",
        "partitionComparisons",
        "clusterSupplement",
        "clusterCanvas",
        "explorerFrame",
        "explorerTimeline",
        "previousRollout",
        "nextRollout",
        "oraclePanel",
        "oracleContent",
        "oracleStatusLive",
        "oracleRefresh",
        "oracleStageGrid",
        "oracleVariant",
        "oracleCoverage",
        "oraclePhaseTabs",
        "oracleClusterList",
        "oracleFrameLeft",
        "oracleFrameRight",
        "oracleFrameWrist",
        "oraclePreviousRollout",
        "oracleNextRollout",
        "oracleCheckpoint",
        "oracleCheckpointSummary",
        "oracleHeatmapWindow",
        "oracleHeatmapMode",
        "oracleHeatmapLimit",
        "oracleHeatmapTable",
        "oracleHeatmapInspector",
        "oracleRankingGrid",
        "oracleAuditGrid",
    ):
        assert f'id="{element_id}"' in html

    assert 'fetch("/api/results")' in html
    assert 'fetch("/api/stage4-overview")' in html
    assert 'fetch("/api/directional-alignment")' in html
    assert 'fetch("/api/oracle")' in html
    assert "/api/explorer?condition_id=" in html
    assert "/api/explorer/frame?" in html
    assert "/api/oracle/frame?" in html
    assert 'data-app-tab="clusters"' in html
    assert 'data-app-tab="features"' in html
    assert 'data-app-tab="alignment"' in html
    assert 'data-app-tab="experiments"' in html
    assert 'data-app-tab="oracle"' in html
    assert "5개 condition." in html
    assert "E0 → E4 변경 순서" in html
    assert "45개 run 중 하나 선택" in html
    assert "한 번에 하나씩 바꾼 결과 비교" in html
    assert "Phase feature 상세 보기 →" in html
    assert "보조 control과 해석 제한" in html
    assert "Action phase × Event-aligned feature" in html
    assert "Oracle ↔ E3 feature alignment" in html
    assert "Fine · 6-phase" in html
    assert "Coarse · 4-phase" in html
    assert "Event aligned only" in html
    assert "instruction → task family → cross-family" in html
    assert "automatic AWE event anchor" in html
    assert "unknown-combined" in html
    assert "추가 실험 없음" in html
    assert 'id="controlledHeatmapRanking"' not in html
    assert 'id="oracleHeatmapRanking"' not in html
    assert "Phase cluster·annotation 확인" in html
    assert "Phase-wise SAE feature 확인" in html
    assert "집중해서 볼 SAE checkpoint" in html
    assert "기존 single-SAE pilot의 기준 checkpoint" in html
    assert "W4 · 민감도 확인" in html
    assert "SAE를 바꾸면 Feature ID도 새로 해석" in html
    assert "같은 raw clustering·annotation을 필터링" in html
    assert "Run-level shortlist" in html
    assert "대조군 추출 · score 없음" in html
    assert "Cluster 구성은 바뀌지 않음" in html
    assert "Feature ID는 같은 SAE checkpoint 안에서만 비교" in html
    assert "기존 45개 실험의 automatic annotation은 human review 전까지 provisional" in html
    assert "Oracle phase 실험" in html
    assert "해당 phase가 실제로 나타난 rollout 중 반복 비율" in html
    assert "전체 task coverage 미기록" in html
    assert "phase 기준과 전체 task 기준 coverage를 함께 표시" in html
    assert "Confounded — 판정 보류" in html
    assert "simulator-oracle diagnostic upper bound" in html
    assert "Phase가 짧으면 마지막 유효 frame을 반복" in html
    assert "[hidden] { display: none !important; }" in html
    assert "<svg" not in html


def _write_phase_heatmap_score_fixture(path: Path) -> list[list[float]]:
    """Write a real score artifact with task-local and singleton-phase cases."""

    import torch

    matrix_raw = torch.tensor(
        [
            [10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [100.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 6.0, 0.0, 0.0, 0.0, 0.0],
            [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    row_keys = [
        {
            "task_id": 11,
            "task_description": "Shared instruction",
            "instruction_id": "instruction-11",
            "cell_id": "cell-a",
            "task_family_id": "family-one",
            "task_family_label": "Family One",
            "phase_scheme": "event_state",
            "cluster_id": "task11_reach",
            "phrase": "task 11 reach",
            "phase": "reach",
            "episode_coverage": 0.8,
            "num_episode_groups": 3,
            "num_events": 6,
        },
        {
            "task_id": 11,
            "task_description": "Shared instruction",
            "instruction_id": "instruction-11",
            "cell_id": "cell-a",
            "task_family_id": "family-one",
            "task_family_label": "Family One",
            "phase_scheme": "event_state",
            "cluster_id": "task11_grasp",
            "phrase": "task 11 grasp",
            "phase": "grasp",
            "episode_coverage": 0.7,
            "num_episode_groups": 2,
            "num_events": 4,
        },
        {
            "task_id": 22,
            "task_description": "Shared instruction",
            "instruction_id": "instruction-22",
            "cell_id": "cell-b",
            "task_family_id": "family-one",
            "task_family_label": "Family One",
            "phase_scheme": "event_state",
            "cluster_id": "task22_open",
            "phrase": "task 22 open",
            "phase": "reach",
            "episode_coverage": 0.9,
            "num_episode_groups": 4,
            "num_events": 8,
        },
        {
            "task_id": 22,
            "task_description": "Shared instruction",
            "instruction_id": "instruction-22",
            "cell_id": "cell-b",
            "task_family_id": "family-one",
            "task_family_label": "Family One",
            "phase_scheme": "event_state",
            "cluster_id": "task22_done",
            "phrase": "task 22 done",
            "phase": "done",
            "episode_coverage": 0.6,
            "num_episode_groups": 2,
            "num_events": 3,
        },
        {
            "task_id": 33,
            "task_description": "Single phase task",
            "instruction_id": "instruction-33",
            "cell_id": "cell-c",
            "task_family_id": "family-two",
            "task_family_label": "Family Two",
            "phase_scheme": "event_state",
            "cluster_id": "task33_terminal",
            "phrase": "task 33 terminal",
            "phase": "terminal",
            "episode_coverage": 1.0,
            "num_episode_groups": 1,
            "num_events": 2,
        },
    ]
    row_results = [
        {
            "task_id": row["task_id"],
            "task_description": row["task_description"],
            "phase_scheme": row["phase_scheme"],
            "cluster_id": row["cluster_id"],
            "phrase": row["phrase"],
            "phase": row["phase"],
            "num_episode_groups": row["num_episode_groups"],
            "num_events": row["num_events"],
            "episode_coverage": row["episode_coverage"],
        }
        for row in row_keys
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "window_size": 5,
            "row_semantics": (
                "(task_id, task_description, phase_scheme, phase, cluster_id)"
            ),
            "score_definitions": {
                "matrix_raw": "exact event-aligned score",
                "matrix_window_mean": "not valid for this heatmap",
            },
            "row_keys": row_keys,
            "row_results": row_results,
            "matrix_raw": matrix_raw,
            "matrix_window_mean": matrix_raw + 1000.0,
            "matrix": matrix_raw + 2000.0,
        },
        path,
    )
    return matrix_raw.tolist()


def test_phase_heatmap_uses_exact_raw_scores_and_task_local_evidence(
    tmp_path: Path,
) -> None:
    from event_sae.groot.phase_feature_results import (
        build_phase_feature_heatmap,
        load_phase_feature_score_matrix,
    )

    score_path = tmp_path / "event_feature_scores.pt"
    expected_matrix = _write_phase_heatmap_score_fixture(score_path)
    score_data = load_phase_feature_score_matrix(
        score_path=score_path,
        ranking="event_aligned",
    )

    assert score_data["matrix_key"] == "matrix_raw"
    assert score_data["matrix"].tolist() == expected_matrix
    payload = build_phase_feature_heatmap(
        score_data=score_data,
        dataset="controlled",
        context={"analysis_id": "fixture-analysis"},
        mode="strength",
        limit=8,
    )

    assert payload["format"] == "event_sae_phase_feature_heatmap_v1"
    assert payload["ranking"] == "event_aligned"
    assert payload["row_semantics"] == (
        "(task_id, task_description, phase_scheme, phase, cluster_id)"
    )
    rows = {row["cluster_id"]: row for row in payload["rows"]}
    assert set(rows) == {
        "task11_reach",
        "task11_grasp",
        "task22_open",
        "task22_done",
        "task33_terminal",
    }
    selected_feature_ids = {
        feature["feature_id"] for feature in payload["features"]
    }
    assert selected_feature_ids == {0, 1, 2, 3}
    for row_index, cluster_id in enumerate(
        (
            "task11_reach",
            "task11_grasp",
            "task22_open",
            "task22_done",
            "task33_terminal",
        )
    ):
        cells = {
            cell["feature_id"]: cell["score"]
            for cell in rows[cluster_id]["cells"]
        }
        assert cells == {
            feature_id: expected_matrix[row_index][feature_id]
            for feature_id in selected_feature_ids
        }

    task11_reach = rows["task11_reach"]
    assert task11_reach["task_id"] == 11
    assert task11_reach["phase_scheme"] == "event_state"
    assert task11_reach["num_episode_groups"] == 3
    assert task11_reach["num_events"] == 6
    task11_feature0 = next(
        cell
        for cell in task11_reach["cells"]
        if cell["feature_id"] == 0
    )
    assert task11_feature0["selectivity_margin"] == pytest.approx(8.0)
    evidence = task11_feature0["evidence_ladder"]
    assert evidence["instruction"] == {
        "status": "top20_candidate",
        "instruction_id": "instruction-11",
        "cell_id": "cell-a",
        "task_description": "Shared instruction",
        "raw_task_id": 11,
        "selectivity_margin": pytest.approx(8.0),
        "selectivity_rank": 1,
    }
    assert evidence["family"] == {
        "status": "repeated_top20",
        "task_family_id": "family-one",
        "task_family_label": "Family One",
        "eligible_cell_count": 2,
        "positive_cell_count": 2,
        "top20_cell_count": 2,
        "supporting_cell_ids": ["cell-a", "cell-b"],
        "comparison_contract": (
            "same canonical task family + exact phase + distinct cells"
        ),
    }
    assert evidence["cross_family"] == {
        "status": "not_assessable",
        "phase_label": "reach",
        "eligible_family_count": 1,
        "positive_family_count": 1,
        "top20_family_count": 1,
        "eligible_task_family_ids": ["family-one"],
        "comparison_contract": "exact phase-label match across families only",
    }
    task22_open = rows["task22_open"]
    task22_feature0 = next(
        cell
        for cell in task22_open["cells"]
        if cell["feature_id"] == 0
    )
    assert task22_feature0["selectivity_margin"] == pytest.approx(99.0)


def test_phase_heatmap_pins_only_event_aligned_candidates(
    tmp_path: Path,
) -> None:
    from event_sae.groot.phase_feature_results import (
        build_phase_feature_heatmap,
        load_phase_feature_score_matrix,
    )

    score_path = tmp_path / "event_feature_scores.pt"
    _write_phase_heatmap_score_fixture(score_path)
    score_data = load_phase_feature_score_matrix(
        score_path=score_path,
        ranking="event_aligned",
    )
    payload = build_phase_feature_heatmap(
        score_data=score_data,
        dataset="controlled",
        context={"analysis_id": "fixture-analysis"},
        mode="selectivity",
        limit=4,
        pinned_feature_id=3,
    )

    assert payload["pinned_feature_id"] == 3
    assert 3 in {feature["feature_id"] for feature in payload["features"]}
    for feature in payload["features"]:
        feature_id = feature["feature_id"]
        assert any(
            row["cells"][column_index]["selectivity_margin"] is not None
            and row["cells"][column_index]["selectivity_margin"] > 0
            and row["cells"][column_index]["selectivity_rank"] <= 20
            for row in payload["rows"]
            for column_index, candidate in enumerate(payload["features"])
            if candidate["feature_id"] == feature_id
        )

    with pytest.raises(ValueError, match="not an event-aligned"):
        build_phase_feature_heatmap(
            score_data=score_data,
            dataset="controlled",
            context={"analysis_id": "fixture-analysis"},
            mode="selectivity",
            limit=4,
            pinned_feature_id=4,
        )


def test_phase_heatmap_single_phase_task_has_null_selectivity(
    tmp_path: Path,
) -> None:
    from event_sae.groot.phase_feature_results import (
        build_phase_feature_heatmap,
        load_phase_feature_score_matrix,
    )

    score_path = tmp_path / "event_feature_scores.pt"
    _write_phase_heatmap_score_fixture(score_path)
    score_data = load_phase_feature_score_matrix(
        score_path=score_path,
        ranking="event_aligned",
    )
    payload = build_phase_feature_heatmap(
        score_data=score_data,
        dataset="controlled",
        context={"analysis_id": "fixture-analysis"},
        mode="strength",
        limit=8,
        task_description="Single phase task",
    )

    assert payload["row_count"] == 1
    assert payload["features"] == []
    assert payload["rows"] == []
    assert payload["candidate_contract"] == (
        "event_aligned Δ>0 row-local Top-20 only; persistent controls excluded"
    )
    assert "비교 가능한 event-aligned" in payload["message"]


def test_phase_heatmap_application_rejects_unknown_or_non_event_analysis(
    tmp_path: Path,
) -> None:
    root = _fixture_results(tmp_path)
    run_dir = (
        root
        / "stage4"
        / "e0_rel_pos_cluster_left_label_left"
        / "cov0p3"
        / "l15_sae1p2k_exec5_mean4_top96_v1"
    )
    expected_matrix = _write_phase_heatmap_score_fixture(
        run_dir / "event_feature_scores.pt"
    )
    ui_path = tmp_path / "results.html"
    ui_path.write_text("<html>results</html>", encoding="utf-8")
    application = build_experiment_results_service(
        experiment_root=root,
        ui_path=ui_path,
    )
    run = next(
        item
        for item in application.data()["runs"]
        if (
            item["condition_id"] == "e0_rel_pos_cluster_left_label_left"
            and item["coverage_id"] == "cov0p3"
            and item["sae_id"] == "l15_sae1p2k_exec5_mean4_top96_v1"
        )
    )

    payload = application.feature_heatmap(
        dataset="controlled",
        analysis_id=run["id"],
        ranking="event_aligned",
        mode="strength",
        limit=8,
    )
    assert payload["context"]["analysis_id"] == run["id"]
    rows = {row["cluster_id"]: row for row in payload["rows"]}
    selected_feature_ids = {
        feature["feature_id"] for feature in payload["features"]
    }
    assert selected_feature_ids == {0, 1, 2, 3}
    assert {
        cell["feature_id"]: cell["score"]
        for cell in rows["task11_reach"]["cells"]
    } == {
        feature_id: expected_matrix[0][feature_id]
        for feature_id in selected_feature_ids
    }

    with pytest.raises(KeyError):
        application.feature_heatmap(
            dataset="controlled",
            analysis_id="unknown-analysis",
            ranking="event_aligned",
            mode="strength",
            limit=8,
        )
    with pytest.raises(ValueError, match="Unsupported phase heatmap ranking"):
        application.feature_heatmap(
            dataset="controlled",
            analysis_id=run["id"],
            ranking="window_mean",
            mode="strength",
            limit=8,
        )
