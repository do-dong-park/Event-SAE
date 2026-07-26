from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import event_sae.scoring.feature_activation_grid as feature_activation_grid
import event_sae.scoring.task_phase_ranking as task_phase_ranking
from event_sae.events.cluster import audit_and_freeze_annotation_bundle
from event_sae.events.review import finalize_reviewed_annotations
from event_sae.groot.activations import (
    _validate_checkpoint_source_manifest,
    encode_activation_cache_to_sparse_topk,
    join_activation_trajectory_inventories,
    reconstruct_action_token_row_metadata,
    save_activation_cache,
)
from event_sae.groot.phase_feature_results import summarize_stage4_grid
from event_sae.scoring.feature_activation_grid import (
    DirectionalPhaseViewAnalysisConfig,
    EventPhaseActivationGridConfig,
    FocusedPhaseViewAnalysisConfig,
    _rank_focused_tasks,
    analyze_directional_phase_views,
    analyze_event_phase_activation_grid,
    analyze_focused_phase_views,
    build_fixed_phase_composition_cells,
    build_fine_phase_cells,
    classify_event_phase_candidates,
    regroup_phase_assignments_to_coarse,
    rank_phase_cells,
    rank_task_balanced_event_features,
)
from event_sae.scoring.phase_selectivity import (
    CheckpointPhaseStabilityConfig,
    PhaseFeatureRun,
    analyze_checkpoint_phase_stability,
    episode_stratified_phase_permutations,
    holm_adjusted_p_values,
    max_t_p_value,
    phase_vs_rest_margin,
)
from event_sae.scoring.rankings import (
    alive_feature_ids,
    event_aligned_suite_top_k,
    event_aligned_top_features_per_row,
    match_mutual_nearest_decoder_features,
    task_mean_suite_top_k,
    task_mean_top_features_per_task,
    window_mean_suite_top_k,
    window_mean_top_features_per_row,
)
from event_sae.scoring.task_phase_ranking import (
    CoarsePhaseCandidateRankingConfig,
    TaskLocalPhaseRankingConfig,
    _coarse_phase_cells,
    _confidence_and_observation_diagnostics,
    _inferential_family_description,
    _rank_coarse_phase_cells,
    _render_task_local_markdown,
    _validate_task_local_score_provenance,
    load_task_local_score_pair,
    rank_coarse_phase_candidates,
    rank_task_local_phase_features,
)
from event_sae.scoring.score_matrix import (
    _merge_episode_task_ids,
    join_cluster_events,
    aggregate_sparse_activations_by_timestep,
    open_sparse_topk_artifact,
    score_cluster_features,
)
from scripts.extract_topk import _validate_row_spans
from scripts.groot.analyze_phase_features import (
    build_parser as build_phase_feature_parser,
)


def test_phase_selectivity_math_contracts() -> None:
    phase_scores = np.asarray(
        [
            [5.0, 1.0],
            [2.0, 4.0],
            [3.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(
        phase_vs_rest_margin(phase_scores),
        np.asarray(
            [
                [2.0, -3.0],
                [-3.0, 3.0],
                [-2.0, -4.0],
            ]
        ),
    )

    null_max = np.asarray([0.1, 0.2, 0.3])
    assert max_t_p_value(null_max, 0.25) == pytest.approx(0.5)
    assert holm_adjusted_p_values(
        {"first": 0.01, "second": 0.04, "third": 0.03}
    ) == pytest.approx(
        {"first": 0.03, "third": 0.06, "second": 0.06}
    )


def _descriptive_grid_task(
    name: str,
    phases: list[str],
    phase_scores: np.ndarray,
) -> SimpleNamespace:
    matrix = np.asarray(phase_scores, dtype=np.float64)
    task_mean = np.zeros_like(matrix)
    return SimpleNamespace(
        task_description=name,
        pair=SimpleNamespace(
            phases=phases,
            phase_w4=matrix,
            phase_w5=matrix,
        ),
        window_mean_w5=matrix,
        task_mean_w4=task_mean,
        task_mean_w5=task_mean,
    )


def test_event_grid_macro_averages_exact_instructions() -> None:
    tasks = {
        "two phases": _descriptive_grid_task(
            "two phases",
            ["reach", "grasp"],
            np.asarray([[9.0, 0.0], [9.0, 0.0]]),
        ),
        "one phase": _descriptive_grid_task(
            "one phase",
            ["place"],
            np.asarray([[0.0, 15.0]]),
        ),
    }

    result = rank_task_balanced_event_features(
        tasks,
        artifact_top_n=2,
        primary_top_n=1,
        comparator_top_n=2,
    )

    # Equal task weighting gives F1=(0+15)/2 > F0=(9+0)/2.
    # Pooling all three phase rows would incorrectly reverse that order.
    assert result["primary_top_feature_ids"] == [1]
    assert result["top_feature_ids"]["w4"] == [1, 0]
    assert result["top_feature_ids"]["mean_w4_w5"] == [1, 0]


def test_event_grid_ranks_fine_phase_mean_other_contrasts() -> None:
    tasks = {
        "task a": _descriptive_grid_task(
            "task a",
            ["reach-to-object", "grasp"],
            np.asarray([[10.0, 1.0], [1.0, 8.0]]),
        ),
        "task b": _descriptive_grid_task(
            "task b",
            ["reach-to-object", "place"],
            np.asarray([[8.0, 1.0], [1.0, 6.0]]),
        ),
    }

    cells, diagnostics = build_fine_phase_cells(tasks)
    ranked = rank_phase_cells(
        cells,
        artifact_top_n=2,
        primary_top_n=1,
    )

    assert diagnostics["num_contrastable_tasks"] == 2
    assert diagnostics["phase_instruction_counts"] == {
        "grasp": 1,
        "place": 1,
        "reach-to-object": 2,
    }
    assert ranked["reach-to-object"]["primary_top_feature_ids"] == [0]
    assert ranked["grasp"]["primary_top_feature_ids"] == [1]
    assert ranked["place"]["primary_top_feature_ids"] == [1]


def test_event_grid_fixes_oracle_phase_vocabulary_without_imputation() -> None:
    tasks = {
        "task a": _descriptive_grid_task(
            "task a",
            ["reach-to-object", "grasp", "transport"],
            np.asarray(
                [
                    [10.0, 1.0],
                    [1.0, 8.0],
                    [100.0, 100.0],
                ]
            ),
        ),
        "task b": _descriptive_grid_task(
            "task b",
            ["place", "insert-settle"],
            np.asarray([[7.0, 1.0], [1.0, 6.0]]),
        ),
    }

    cells, diagnostics = build_fixed_phase_composition_cells(tasks)

    assert diagnostics["suite_composition_complete"] is True
    assert diagnostics["comparison_eligible"] is True
    assert diagnostics["exact_instruction_complete_count"] == 0
    assert diagnostics["missing_phases"] == []
    assert diagnostics["phase_instruction_counts"] == {
        "reach-to-object": 1,
        "grasp": 1,
        "place": 1,
        "insert-settle": 1,
    }
    assert "transport" not in cells
    np.testing.assert_array_equal(
        cells["reach-to-object"]["task a"]["margin_w5"],
        np.asarray([9.0, -7.0]),
    )

    _, incomplete = build_fixed_phase_composition_cells(
        {"task a": tasks["task a"]}
    )
    assert incomplete["comparison_eligible"] is False
    assert incomplete["missing_phases"] == ["place", "insert-settle"]


def test_event_grid_taxonomy_is_mece_with_comparator_overlays() -> None:
    event_result = {
        "primary_top_feature_ids": [1, 2],
        "top_candidates": [
            {
                "feature_id": 1,
                "task_mean_top20": True,
                "window_mean_top20": False,
            },
            {
                "feature_id": 2,
                "task_mean_top20": False,
                "window_mean_top20": False,
            },
        ],
    }
    phase_results = {
        "reach": {
            "top_candidates": [
                {
                    "feature_id": 2,
                    "task_mean_top20": False,
                    "window_mean_same_sign": True,
                },
                {
                    "feature_id": 3,
                    "task_mean_top20": False,
                    "window_mean_same_sign": False,
                },
            ]
        }
    }

    result = classify_event_phase_candidates(
        event_result,
        phase_results,
        phase_primary_top_n=2,
    )

    assert result["event_ranked_only"] == [1]
    assert result["phase_aligned_only"] == [3]
    assert result["dual_ranked"] == [2]
    assert result["task_mean_top20_overlay"] == [1]
    assert result["window_magnitude_or_persistence_overlay"] == [2]


def test_event_grid_refuses_existing_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        analyze_event_phase_activation_grid(
            EventPhaseActivationGridConfig(
                stage4_root=tmp_path / "stage4",
                oracle_summary=tmp_path / "oracle.json",
                output_dir=output_dir,
            )
        )


def test_focused_coarse_regroup_preserves_source_rows_and_merges_exactly() -> None:
    groups = [
        {
            "phase_group_id": "place_group",
            "task_description": "task",
            "phase": "place",
            "total_task_episodes": 2,
            "representative_sample_ids": ["place"],
        },
        {
            "phase_group_id": "settle_group",
            "task_description": "task",
            "phase": "insert-settle",
            "total_task_episodes": 2,
            "representative_sample_ids": ["settle"],
        },
        {
            "phase_group_id": "wrong_group",
            "task_description": "task",
            "phase": "wrong-grasp",
            "total_task_episodes": 2,
            "representative_sample_ids": [],
        },
    ]
    assignments = [
        {
            "sample_id": "place",
            "task_description": "task",
            "episode_num": 0,
            "phase": "place",
            "phase_group_id": "place_group",
            "cluster_id": "place_group",
            "source_cluster_id": "place_source",
            "phase_after": "place",
            "clock_tick": 17,
        },
        {
            "sample_id": "settle",
            "task_description": "task",
            "episode_num": 0,
            "phase": "insert-settle",
            "phase_group_id": "settle_group",
            "cluster_id": "settle_group",
            "source_cluster_id": "settle_source",
            "phase_after": "insert-settle",
            "clock_tick": 18,
        },
        {
            "sample_id": "wrong",
            "task_description": "task",
            "episode_num": 1,
            "phase": "wrong-grasp",
            "phase_group_id": "wrong_group",
            "cluster_id": "wrong_group",
            "source_cluster_id": "wrong_source",
            "clock_tick": 19,
        },
    ]

    derived, derived_groups, diagnostics = (
        regroup_phase_assignments_to_coarse(
            assignments,
            groups,
            source_label="unit_source",
        )
    )

    assert len(derived) == 2
    assert len(derived_groups) == 1
    assert {row["phase"] for row in derived} == {"terminal"}
    assert derived[0]["clock_tick"] == 17
    assert derived[0]["phase_after"] == "place"
    assert derived[0]["source_fine_phase"] == "place"
    assert derived[0]["source_phase_group_id"] == "place_group"
    assert derived[0]["cluster_id"] == derived[1]["cluster_id"]
    assert derived_groups[0]["source_fine_phases"] == [
        "insert-settle",
        "place",
    ]
    assert diagnostics["known_excluded_label_counts"] == {"wrong-grasp": 1}
    assert diagnostics["source_episode_phase_group_count"] == 2
    assert diagnostics["derived_episode_phase_group_count"] == 1
    assert diagnostics["fine_to_coarse_episode_group_reduction"] == 1


def test_focused_coarse_regroup_rejects_unknown_phase_label() -> None:
    assignments = [
        {
            "sample_id": "mystery",
            "task_description": "task",
            "episode_num": 0,
            "phase": "unmapped-phase",
            "phase_group_id": "mystery_group",
            "cluster_id": "mystery_group",
            "source_cluster_id": "mystery_source",
        }
    ]
    groups = [
        {
            "phase_group_id": "mystery_group",
            "task_description": "task",
            "phase": "unmapped-phase",
            "total_task_episodes": 1,
        }
    ]

    with pytest.raises(ValueError, match="unknown fine phase labels"):
        regroup_phase_assignments_to_coarse(
            assignments,
            groups,
            source_label="unit_source",
        )


def test_focused_candidates_use_set_controls_without_numeric_subtraction(
    tmp_path: Path,
) -> None:
    dict_size = 25
    phase_w4 = torch.zeros((2, dict_size), dtype=torch.float32)
    phase_w5 = torch.zeros((2, dict_size), dtype=torch.float32)
    phase_w4[0, 0:2] = torch.tensor([10.0, 8.0])
    phase_w5[0, 0:2] = torch.tensor([9.0, 7.0])
    window_w4 = torch.zeros_like(phase_w4)
    window_w5 = torch.zeros_like(phase_w5)
    window_w4[0, 0] = 9.0
    window_w5[0, 0] = 8.0
    task_mean = torch.zeros_like(phase_w4)
    task_mean[:, 2] = 5.0
    row_keys = [
        {
            "task_description": "task",
            "cluster_id": "reach",
            "phase": "reach",
        },
        {
            "task_description": "task",
            "cluster_id": "grasp",
            "phase": "grasp",
        },
    ]
    episode_group_keys = [
        {"cluster_id": "reach", "episode_num": 0, "num_events": 1},
        {"cluster_id": "grasp", "episode_num": 0, "num_events": 1},
    ]

    def payload(
        window_size: int,
        phase_matrix: torch.Tensor,
        window_matrix: torch.Tensor,
    ) -> dict:
        return {
            "window_size": window_size,
            "row_keys": row_keys,
            "episode_group_keys": episode_group_keys,
            "episode_group_matrix_raw": phase_matrix,
            "matrix_raw": phase_matrix,
            "matrix_window_mean": window_matrix,
            "matrix_task_mean": task_mean,
        }

    score_w4 = tmp_path / "w4.pt"
    score_w5 = tmp_path / "w5.pt"
    torch.save(payload(4, phase_w4, window_w4), score_w4)
    torch.save(payload(5, phase_w5, window_w5), score_w5)
    tasks = load_task_local_score_pair(score_w4, score_w5)

    view, candidates = _rank_focused_tasks(
        tasks,
        source_label="unit_source",
        view="fine_original",
        comparator_top_n=1,
        filtered_top_n=1,
    )

    reach = next(cell for cell in view["cells"] if cell["phase"] == "reach")
    reach_candidates = [
        row for row in candidates if row["phase"] == "reach"
    ]
    assert [row["feature_id"] for row in reach_candidates] == [0, 1]
    assert reach_candidates[0]["control_filter_reasons"] == [
        "window_mean_top20_w4",
        "window_mean_top20_w5",
    ]
    assert reach_candidates[0]["removed_by_control_filter"] is True
    assert reach_candidates[1]["filtered_rank"] == 1
    assert reach_candidates[1]["in_control_filtered_top10"] is True
    assert reach_candidates[1]["numeric_control_subtraction_applied"] is False
    assert reach["control_filtered_top10"][0]["feature_id"] == 1
    assert reach_candidates[1]["robust_event_delta_min_w4_w5"] == pytest.approx(
        7.0
    )


def test_focused_phase_views_refuse_existing_output(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        analyze_focused_phase_views(
            FocusedPhaseViewAnalysisConfig(
                stage4_root=tmp_path / "stage4",
                oracle_summary=tmp_path / "oracle.json",
                output_dir=tmp_path,
            )
        )


def test_focused_phase_view_cli_contract(tmp_path: Path) -> None:
    args = build_phase_feature_parser().parse_args(
        [
            "analyze-focused-phase-views",
            "--stage4-root",
            str(tmp_path / "stage4"),
            "--oracle-summary",
            str(tmp_path / "oracle.json"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert args.handler.__name__ == "_analyze_focused_phase_views"
    assert args.stage4_root == tmp_path / "stage4"
    assert args.oracle_summary == tmp_path / "oracle.json"


def test_phase_permutations_are_shared_and_episode_stratified() -> None:
    group_keys = [
        (1, "reach"),
        (1, "grasp"),
        (2, "reach"),
        (2, "grasp"),
    ]
    first, first_sha = episode_stratified_phase_permutations(
        group_keys,
        ["reach", "grasp"],
        num_permutations=20,
        seed=17,
    )
    second, second_sha = episode_stratified_phase_permutations(
        group_keys,
        ["reach", "grasp"],
        num_permutations=20,
        seed=17,
    )

    np.testing.assert_array_equal(first, second)
    assert first_sha == second_sha
    for row in first:
        assert sorted(row[:2].tolist()) == [0, 1]
        assert sorted(row[2:].tolist()) == [0, 1]


def test_decoder_matching_recovers_permuted_feature_ids() -> None:
    reference = torch.eye(3)
    permuted = reference[:, [2, 0, 1]] * torch.tensor([7.0, 2.0, 0.5])

    matches = match_mutual_nearest_decoder_features(reference, permuted)

    assert matches["left_to_right"].tolist() == [1, 2, 0]
    assert matches["mutual"].tolist() == [True, True, True]
    assert matches["mutual_count"] == 3


def test_task_local_score_loader_allows_phase_names_to_repeat_across_tasks(
    tmp_path: Path,
) -> None:
    row_keys = [
        {
            "task_description": "task a",
            "cluster_id": "a_reach",
            "phase": "reach",
        },
        {
            "task_description": "task a",
            "cluster_id": "a_grasp",
            "phase": "grasp",
        },
        {
            "task_description": "task b",
            "cluster_id": "b_reach",
            "phase": "reach",
        },
        {
            "task_description": "task b",
            "cluster_id": "b_place",
            "phase": "place",
        },
    ]
    episode_group_keys = [
        {"cluster_id": row["cluster_id"], "episode_num": 0, "num_events": 1}
        for row in row_keys
    ]
    group_w4 = torch.tensor(
        [
            [4.0, 1.0],
            [1.0, 3.0],
            [5.0, 0.0],
            [2.0, 2.0],
        ]
    )
    group_w5 = group_w4 + 1.0

    def payload(window_size: int, group_matrix: torch.Tensor) -> dict:
        return {
            "window_size": window_size,
            "row_keys": row_keys,
            "episode_group_keys": episode_group_keys,
            "episode_group_matrix_raw": group_matrix,
            "matrix_raw": group_matrix.clone(),
            "matrix_window_mean": group_matrix.clone(),
            "matrix_task_mean": group_matrix.clone(),
        }

    score_w4 = tmp_path / "w4.pt"
    score_w5 = tmp_path / "w5.pt"
    torch.save(payload(4, group_w4), score_w4)
    torch.save(payload(5, group_w5), score_w5)

    tasks = load_task_local_score_pair(score_w4, score_w5)

    assert list(tasks) == ["task a", "task b"]
    assert tasks["task a"].pair.phases == ["reach", "grasp"]
    assert tasks["task b"].pair.phases == ["reach", "place"]
    assert tasks["task a"].pair.robust_margin.shape == (2, 2)
    assert tasks["task b"].pair.robust_margin.shape == (2, 2)


def test_coarse_phase_cells_merge_fine_groups_within_episode(
    tmp_path: Path,
) -> None:
    row_keys = [
        {
            "task_description": "task",
            "cluster_id": "reach",
            "phase": "reach-to-object",
        },
        {
            "task_description": "task",
            "cluster_id": "place",
            "phase": "place",
        },
        {
            "task_description": "task",
            "cluster_id": "settle",
            "phase": "insert-settle",
        },
    ]
    episode_group_keys = [
        {"cluster_id": "reach", "episode_num": 0, "num_events": 1},
        {"cluster_id": "reach", "episode_num": 1, "num_events": 1},
        {"cluster_id": "place", "episode_num": 0, "num_events": 1},
        {"cluster_id": "settle", "episode_num": 0, "num_events": 3},
    ]
    group_matrix = torch.tensor(
        [
            [4.0, 0.0],
            [6.0, 0.0],
            [0.0, 2.0],
            [0.0, 6.0],
        ]
    )
    matrix_raw = torch.tensor(
        [
            [5.0, 0.0],
            [0.0, 2.0],
            [0.0, 6.0],
        ]
    )

    def payload(window_size: int) -> dict:
        return {
            "window_size": window_size,
            "row_keys": row_keys,
            "episode_group_keys": episode_group_keys,
            "episode_group_matrix_raw": group_matrix,
            "matrix_raw": matrix_raw,
            "matrix_window_mean": matrix_raw,
            "matrix_task_mean": torch.ones_like(matrix_raw),
        }

    score_w4 = tmp_path / "w4.pt"
    score_w5 = tmp_path / "w5.pt"
    torch.save(payload(4), score_w4)
    torch.save(payload(5), score_w5)

    tasks = load_task_local_score_pair(score_w4, score_w5)
    cells, diagnostics = _coarse_phase_cells(tasks)

    assert list(cells) == ["reach", "terminal"]
    np.testing.assert_allclose(
        cells["reach"]["task"]["margin_w5"],
        np.asarray([5.0, -5.0]),
    )
    np.testing.assert_allclose(
        cells["terminal"]["task"]["margin_w5"],
        np.asarray([-5.0, 5.0]),
    )
    assert diagnostics == {
        "num_tasks": 1,
        "num_contrastable_tasks": 1,
        "num_source_episode_groups": 4,
        "num_coarse_episode_groups": 3,
        "num_approximated_episode_groups": 1,
    }


def test_relaxed_coarse_ranking_uses_majority_not_inference_gates() -> None:
    cells = {
        "instruction a": {
            "margin_w4": np.asarray([4.0, 2.0, -1.0]),
            "margin_w5": np.asarray([2.0, -1.0, 4.0]),
            "window_margin_w5": np.asarray([-2.0, 3.0, 1.0]),
            "task_mean_w5": np.asarray([3.0, 1.0, 2.0]),
        },
        "instruction b": {
            "margin_w4": np.asarray([2.0, 2.0, -1.0]),
            "margin_w5": np.asarray([2.0, 3.0, 4.0]),
            "window_margin_w5": np.asarray([-1.0, 2.0, 1.0]),
            "task_mean_w5": np.asarray([3.0, 1.0, 2.0]),
        },
    }

    result, arrays = _rank_coarse_phase_cells(cells, artifact_top_n=3)

    assert result["required_positive_instruction_count"] == 2
    assert result["eligible_feature_count"] == 2
    assert [row["feature_id"] for row in result["top_candidates"]] == [0, 1]
    assert arrays["eligible"].tolist() == [True, True, False]
    assert result["top_candidates"][0]["window_mean_same_sign"] is False
    assert result["top_candidates"][1]["window_mean_same_sign"] is True


def test_phase_selectivity_refuses_existing_output(tmp_path: Path) -> None:
    placeholder = PhaseFeatureRun(
        label="placeholder",
        checkpoint=tmp_path / "checkpoint.pt",
        score_w4=tmp_path / "w4.pt",
        score_w5=tmp_path / "w5.pt",
        topk_dir=tmp_path / "topk",
    )
    config = CheckpointPhaseStabilityConfig(
        runs=(placeholder, placeholder, placeholder),
        reference_label="placeholder",
        output_dir=tmp_path,
        num_permutations=1,
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        analyze_checkpoint_phase_stability(config)


def test_task_local_phase_selectivity_refuses_existing_output(
    tmp_path: Path,
) -> None:
    placeholder = PhaseFeatureRun(
        label="placeholder",
        checkpoint=tmp_path / "checkpoint.pt",
        score_w4=tmp_path / "w4.pt",
        score_w5=tmp_path / "w5.pt",
        topk_dir=tmp_path / "topk",
    )
    config = TaskLocalPhaseRankingConfig(
        runs=(placeholder, placeholder, placeholder),
        reference_label="placeholder",
        condition_id="condition",
        accepted_annotations=tmp_path / "accepted.jsonl",
        phase_groups=tmp_path / "groups.jsonl",
        phase_assignments=tmp_path / "assignments.jsonl",
        output_dir=tmp_path,
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        rank_task_local_phase_features(config)


def test_coarse_phase_ranking_refuses_existing_output(
    tmp_path: Path,
) -> None:
    config = CoarsePhaseCandidateRankingConfig(
        stage4_root=tmp_path / "stage4",
        output_dir=tmp_path,
        primary_condition="primary",
        sensitivity_condition="sensitivity",
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        rank_coarse_phase_candidates(config)


def test_phase_selectivity_rejects_checkpoint_topk_hash_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint bytes")
    group_matrix = torch.tensor(
        [
            [3.0, 0.0],
            [0.0, 2.0],
            [4.0, 0.0],
            [0.0, 3.0],
        ]
    )
    common_payload = {
        "row_keys": [
            {
                "task_description": "task",
                "phase": "reach",
                "cluster_id": "reach",
            },
            {
                "task_description": "task",
                "phase": "grasp",
                "cluster_id": "grasp",
            },
        ],
        "episode_group_keys": [
            {"episode_num": 0, "cluster_id": "reach"},
            {"episode_num": 0, "cluster_id": "grasp"},
            {"episode_num": 1, "cluster_id": "reach"},
            {"episode_num": 1, "cluster_id": "grasp"},
        ],
        "episode_group_matrix_raw": group_matrix,
        "matrix_raw": torch.tensor([[3.5, 0.0], [0.0, 2.5]]),
        "matrix_window_mean": torch.tensor([[3.5, 0.0], [0.0, 2.5]]),
        "matrix_task_mean": torch.tensor([[3.5, 0.0], [0.0, 2.5]]),
    }
    score_w4 = tmp_path / "score_w4.pt"
    score_w5 = tmp_path / "score_w5.pt"
    torch.save({**common_payload, "window_size": 4}, score_w4)
    torch.save({**common_payload, "window_size": 5}, score_w5)

    topk_dir = tmp_path / "topk"
    topk_dir.mkdir()
    torch.save(
        {
            "token_idx": torch.tensor([0]),
            "top_feature_ids": torch.tensor([[0, 1]]),
            "top_feature_vals": torch.tensor([[1.0, 0.0]]),
        },
        topk_dir / "shard.pt",
    )
    (topk_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_action_token_slice": {"start": 0},
                "executed_action_steps": 1,
                "action_horizon": 2,
                "shards": [{"path": "shard.pt"}],
                "encoding_stats": {
                    "lossless_topk": True,
                    "mean_positive_features_per_row": 1.0,
                    "max_positive_features_per_row": 1,
                },
                "sae_sha256": "not-the-checkpoint-hash",
            }
        ),
        encoding="utf-8",
    )
    runs = tuple(
        PhaseFeatureRun(
            label=f"run_{index}",
            checkpoint=checkpoint,
            score_w4=score_w4,
            score_w5=score_w5,
            topk_dir=topk_dir,
        )
        for index in range(3)
    )

    with pytest.raises(ValueError, match="does not match checkpoint hash"):
        analyze_checkpoint_phase_stability(
            CheckpointPhaseStabilityConfig(
                runs=runs,
                reference_label="run_0",
                output_dir=tmp_path / "analysis",
                num_permutations=1,
            )
        )


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_label_diagnostics_are_episode_balanced_and_exact(
    tmp_path: Path,
) -> None:
    accepted = _write_jsonl(
        tmp_path / "accepted.jsonl",
        [
            {
                "cluster_id": "source",
                "task_description": "task",
                "phase": "grasp",
                "confidence_tier": "strong",
            }
        ],
    )
    assignment_rows = [
        {
            "sample_id": "event-1",
            "task_description": "task",
            "phase": "grasp",
            "source_cluster_id": "source",
            "episode_num": 1,
            "progress_percent": 0.1,
            "success": False,
        },
        {
            "sample_id": "event-2",
            "task_description": "task",
            "phase": "grasp",
            "source_cluster_id": "source",
            "episode_num": 1,
            "progress_percent": 0.2,
            "success": False,
        },
        {
            "sample_id": "event-3",
            "task_description": "task",
            "phase": "grasp",
            "source_cluster_id": "source",
            "episode_num": 2,
            "progress_percent": 0.9,
            "success": True,
        },
    ]
    assignments = _write_jsonl(
        tmp_path / "assignments.jsonl",
        assignment_rows,
    )
    groups = _write_jsonl(
        tmp_path / "groups.jsonl",
        [
            {
                "phase_group_id": "task_phase_grasp",
                "task_description": "task",
                "phase": "grasp",
                "source_cluster_ids": ["source"],
                "member_sample_ids": [
                    row["sample_id"] for row in assignment_rows
                ],
                "num_members": 3,
            }
        ],
    )

    diagnostics = _confidence_and_observation_diagnostics(
        accepted_annotations=accepted,
        phase_groups=groups,
        phase_assignments=assignments,
    )

    assert diagnostics[("task", "grasp")]["episode_success_rate"] == 0.5
    assert diagnostics[("task", "grasp")]["median_event_progress"] == 0.2

    _write_jsonl(
        accepted,
        [
            {
                "cluster_id": "source",
                "task_description": "task",
                "phase": "grasp",
                "confidence_tier": "strong",
            },
            {
                "cluster_id": "unused",
                "task_description": "task",
                "phase": "grasp",
                "confidence_tier": "strong",
            },
        ],
    )
    with pytest.raises(ValueError, match="source sets differ"):
        _confidence_and_observation_diagnostics(
            accepted_annotations=accepted,
            phase_groups=groups,
            phase_assignments=assignments,
        )


def test_label_diagnostics_preserve_user_directed_override_provenance(
    tmp_path: Path,
) -> None:
    accepted = _write_jsonl(
        tmp_path / "accepted.jsonl",
        [
            {
                "cluster_id": "model-source",
                "task_description": "task",
                "phase": "grasp",
                "confidence_tier": "strong",
                "status": "consensus-5-of-5",
                "phase_source": "source_annotation",
            },
            {
                "cluster_id": "override-source",
                "task_description": "task",
                "phase": "grasp",
                "confidence_tier": "user-directed",
                "status": "user-directed-phase-override",
                "phase_source": "user_directed_override",
                "phase_override": {
                    "applied": True,
                    "authorized_by": "workspace_owner",
                    "reason": "resolve tied sensitivity label",
                    "formal_blind_review_completed": False,
                    "provider_response_generated": False,
                },
            },
        ],
    )
    assignment_rows = [
        {
            "sample_id": "model-event",
            "task_description": "task",
            "phase": "grasp",
            "source_cluster_id": "model-source",
            "episode_num": 1,
            "progress_percent": 0.2,
            "success": True,
        },
        {
            "sample_id": "override-event",
            "task_description": "task",
            "phase": "grasp",
            "source_cluster_id": "override-source",
            "episode_num": 2,
            "progress_percent": 0.4,
            "success": False,
        },
    ]
    assignments = _write_jsonl(
        tmp_path / "assignments.jsonl",
        assignment_rows,
    )
    groups = _write_jsonl(
        tmp_path / "groups.jsonl",
        [
            {
                "phase_group_id": "task_phase_grasp",
                "task_description": "task",
                "phase": "grasp",
                "source_cluster_ids": [
                    "model-source",
                    "override-source",
                ],
                "member_sample_ids": [
                    row["sample_id"] for row in assignment_rows
                ],
                "num_members": 2,
            }
        ],
    )

    row = _confidence_and_observation_diagnostics(
        accepted_annotations=accepted,
        phase_groups=groups,
        phase_assignments=assignments,
    )[("task", "grasp")]

    assert row["source_cluster_confidence"] == {
        "strong": 1,
        "majority": 0,
        "plurality": 0,
        "user-directed": 1,
    }
    assert row["event_membership_confidence"] == {
        "strong": 1,
        "majority": 0,
        "plurality": 0,
        "user-directed": 1,
    }
    assert row["claim_limit"] == "includes_user_directed_phase_override"
    override = row["source_cluster_label_provenance"]["override-source"]
    assert override["confidence_tier"] == "user-directed"
    assert override["status"] == "user-directed-phase-override"
    assert override["phase_override"] == {
        "applied": True,
        "authorized_by": "workspace_owner",
        "reason": "resolve tied sensitivity label",
        "formal_blind_review_completed": False,
        "provider_response_generated": False,
    }


def test_user_directed_confidence_requires_override_provenance(
    tmp_path: Path,
) -> None:
    accepted = _write_jsonl(
        tmp_path / "accepted.jsonl",
        [
            {
                "cluster_id": "source",
                "task_description": "task",
                "phase": "grasp",
                "confidence_tier": "user-directed",
                "status": "consensus-3-of-5",
            }
        ],
    )
    assignments = _write_jsonl(
        tmp_path / "assignments.jsonl",
        [
            {
                "sample_id": "event",
                "task_description": "task",
                "phase": "grasp",
                "source_cluster_id": "source",
                "episode_num": 1,
                "progress_percent": 0.2,
                "success": True,
            }
        ],
    )
    groups = _write_jsonl(
        tmp_path / "groups.jsonl",
        [
            {
                "phase_group_id": "task_phase_grasp",
                "task_description": "task",
                "phase": "grasp",
                "source_cluster_ids": ["source"],
                "member_sample_ids": ["event"],
                "num_members": 1,
            }
        ],
    )

    with pytest.raises(ValueError, match="lacks explicit phase-override"):
        _confidence_and_observation_diagnostics(
            accepted_annotations=accepted,
            phase_groups=groups,
            phase_assignments=assignments,
        )


def _direct_oracle_annotation(
    *,
    cluster_id: str,
    task_description: str = "task",
    phase: str = "grasp",
) -> dict:
    return {
        "cluster_id": cluster_id,
        "task_description": task_description,
        "phase": phase,
        "confidence_tier": "simulator-oracle",
        "phase_source": "simulator_oracle_env_step_phases",
        "label_source": "env_step_phases",
        "model": "simulator_oracle_labeler",
        "review_mode": "programmatic_oracle",
        "review_verdict": "oracle_generated",
        "actual_human_review_completed": False,
        "oracle_upper_bound": True,
        "oracle_provenance": {
            "source": "trusted_rollout_env_step_phases",
            "label_resolution": "environment_state",
            "generation": "programmatic",
            "upper_bound": True,
        },
    }


def test_label_diagnostics_accept_explicit_direct_oracle_provenance(
    tmp_path: Path,
) -> None:
    accepted = _write_jsonl(
        tmp_path / "accepted.jsonl",
        [_direct_oracle_annotation(cluster_id="oracle-source")],
    )
    assignments = _write_jsonl(
        tmp_path / "assignments.jsonl",
        [
            {
                "sample_id": "event",
                "task_description": "task",
                "phase": "grasp",
                "source_cluster_id": "oracle-source",
                "episode_num": 1,
                "progress_percent": 0.2,
                "success": True,
            }
        ],
    )
    groups = _write_jsonl(
        tmp_path / "groups.jsonl",
        [
            {
                "phase_group_id": "task_phase_grasp",
                "task_description": "task",
                "phase": "grasp",
                "source_cluster_ids": ["oracle-source"],
                "member_sample_ids": ["event"],
                "num_members": 1,
            }
        ],
    )

    row = _confidence_and_observation_diagnostics(
        accepted_annotations=accepted,
        phase_groups=groups,
        phase_assignments=assignments,
    )[("task", "grasp")]

    assert row["source_cluster_confidence"]["simulator-oracle"] == 1
    assert row["event_membership_confidence"]["simulator-oracle"] == 1
    assert row["claim_limit"] == "simulator_oracle_phase_labels"
    provenance = row["source_cluster_label_provenance"]["oracle-source"]
    assert provenance["confidence_tier"] == "simulator-oracle"
    assert provenance["simulator_oracle"]["oracle_provenance"] == {
        "source": "trusted_rollout_env_step_phases",
        "label_resolution": "environment_state",
        "generation": "programmatic",
        "upper_bound": True,
    }

    invalid = _direct_oracle_annotation(cluster_id="oracle-source")
    invalid.pop("oracle_provenance")
    _write_jsonl(accepted, [invalid])
    with pytest.raises(ValueError, match="direct-label provenance"):
        _confidence_and_observation_diagnostics(
            accepted_annotations=accepted,
            phase_groups=groups,
            phase_assignments=assignments,
        )


def test_task_local_report_uses_actual_condition_and_hypothesis_count() -> None:
    condition_id = "e2_rel_pos_cluster_multiview_label_multiview/cov0p4"
    assert _inferential_family_description(
        condition_id=condition_id,
        hypothesis_count=7,
    ) == (
        "7 decoder-matched task-phase best-candidate hypotheses within condition "
        f"{condition_id!r}"
    )
    report = _render_task_local_markdown(
        {
            "scope": {
                "condition_id": condition_id,
                "num_tasks": 5,
                "num_phase_rows": 8,
                "num_contrastable_phase_rows": 7,
                "num_runs": 3,
                "dict_size": 1536,
                "matched_statistically_supported": {
                    "count": 1,
                    "total": 7,
                },
                "cross_instruction_descriptive": {
                    "num_phase_rows": 2,
                },
            },
            "permutation": {"num_permutations": 5_000},
            "decoder_matching": {"reference_label": "sae10k"},
            "matched_task_phase_results": [],
            "cross_instruction_results": [],
            "confound_audit": [],
        }
    )

    assert (
        f"1/7 decoder-matched task-phase hypotheses in `{condition_id}`"
        in report
    )
    assert "E4 task-phase" not in report


def test_task_local_cli_separates_score_and_topk_step_scales() -> None:
    args = build_phase_feature_parser().parse_args(
        [
            "rank-task-local-phases",
            "--run",
            "sae",
            "checkpoint.pt",
            "w4.pt",
            "w5.pt",
            "topk",
            "--reference-label",
            "sae",
            "--condition-id",
            "oracle",
            "--accepted-annotations",
            "accepted.jsonl",
            "--phase-groups",
            "groups.jsonl",
            "--phase-assignments",
            "assignments.jsonl",
            "--score-event-step-scale",
            "1",
            "--topk-event-step-scale",
            "5",
            "--output-dir",
            "output",
        ]
    )

    assert args.score_event_step_scale == 1
    assert args.topk_event_step_scale == 5


def test_task_local_score_provenance_rejects_swapped_topk_input(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"decoder.weight": torch.eye(2)}, checkpoint)
    event_features = tmp_path / "events.jsonl"
    event_features.write_text("{}\n", encoding="utf-8")
    phase_groups = tmp_path / "groups.jsonl"
    phase_groups.write_text("{}\n", encoding="utf-8")
    phase_assignments = tmp_path / "assignments.jsonl"
    phase_assignments.write_text("{}\n", encoding="utf-8")
    expected_topk = tmp_path / "expected_topk"
    expected_topk.mkdir()
    wrong_topk = tmp_path / "wrong_topk"
    wrong_topk.mkdir()
    source = {
        "contract_version": "event_feature_score_source_v2",
        "topk_run_dir": str(wrong_topk),
        "event_features_path": str(event_features),
        "cluster_assignments_path": str(phase_assignments),
        "cluster_annotations_path": str(phase_groups),
        "dict_size": 2,
        "topk": 2,
        "layer": 1,
        "sae_path": str(checkpoint),
        "capture_target": "action_expert",
        "event_step_scale": 5,
    }
    selected_event = {
        "sample_id": "event",
        "task_description": "task",
        "cluster_id": "phase_group",
        "phase": "grasp",
        "episode_num": 1,
        "waypoint_step": 2,
    }
    score_w4 = tmp_path / "w4.pt"
    score_w5 = tmp_path / "w5.pt"
    torch.save(
        {
            "source": source,
            "step_mapping": "action_executed",
            "event_step_scale": 5,
            "selected_events": [selected_event],
        },
        score_w4,
    )
    torch.save(
        {
            "source": source,
            "step_mapping": "action_executed",
            "event_step_scale": 5,
            "selected_events": [selected_event],
        },
        score_w5,
    )
    spec = PhaseFeatureRun(
        label="run",
        checkpoint=checkpoint,
        score_w4=score_w4,
        score_w5=score_w5,
        topk_dir=expected_topk,
    )
    config = TaskLocalPhaseRankingConfig(
        runs=(spec, spec, spec),
        reference_label="run",
        condition_id="condition",
        accepted_annotations=tmp_path / "accepted.jsonl",
        phase_groups=phase_groups,
        phase_assignments=phase_assignments,
        output_dir=tmp_path / "output",
    )
    manifest = {
        "sae_path": str(checkpoint),
        "format": "token_topk_sparse_v1",
        "capture_target": "action_expert",
        "event_step_scale": 5,
        "dict_size": 2,
        "activation_dim": 2,
        "topk": 2,
        "layer": 1,
        "encoding_stats": {"lossless_topk": True},
    }

    with pytest.raises(ValueError, match="topk_run_dir"):
        _validate_task_local_score_provenance(spec, config, manifest)


def test_task_local_score_provenance_accepts_independent_step_scales(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"decoder.weight": torch.eye(2)}, checkpoint)
    event_features = _write_jsonl(tmp_path / "events.jsonl", [{}])
    phase_groups = _write_jsonl(tmp_path / "groups.jsonl", [{}])
    phase_assignments = _write_jsonl(tmp_path / "assignments.jsonl", [{}])
    topk_dir = tmp_path / "topk"
    topk_dir.mkdir()
    manifest = {
        "sae_path": str(checkpoint),
        "sae_sha256": task_phase_ranking.sha256_file(checkpoint),
        "format": "token_topk_sparse_v1",
        "capture_target": "action_expert",
        "event_step_scale": 5,
        "dict_size": 2,
        "activation_dim": 2,
        "topk": 2,
        "layer": 1,
        "activation_source_manifest_sha256": "activation-source",
        "trajectory_manifest_sha256": "trajectory-source",
        "encoding_stats": {"lossless_topk": True},
    }
    (topk_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    source = {
        "contract_version": "event_feature_score_source_v2",
        "topk_run_dir": str(topk_dir),
        "topk_manifest_sha256": task_phase_ranking.sha256_file(
            topk_dir / "manifest.json"
        ),
        "event_features_path": str(event_features),
        "event_features_sha256": task_phase_ranking.sha256_file(event_features),
        "cluster_assignments_path": str(phase_assignments),
        "cluster_assignments_sha256": task_phase_ranking.sha256_file(
            phase_assignments
        ),
        "cluster_annotations_path": str(phase_groups),
        "cluster_annotations_sha256": task_phase_ranking.sha256_file(
            phase_groups
        ),
        "prompt_records_path": None,
        "prompt_records_sha256": None,
        "dict_size": 2,
        "topk": 2,
        "layer": 1,
        "sae_path": str(checkpoint),
        "sae_sha256": task_phase_ranking.sha256_file(checkpoint),
        "capture_target": "action_expert",
        "event_step_scale": 1,
        "activation_source_manifest_sha256": "activation-source",
        "trajectory_manifest_sha256": "trajectory-source",
    }
    selected_event = {
        "sample_id": "event",
        "task_description": "task",
        "cluster_id": "phase_group",
        "phase": "grasp",
        "episode_num": 1,
        "waypoint_step": 2,
    }
    score_w4 = tmp_path / "w4.pt"
    score_w5 = tmp_path / "w5.pt"
    for path, window_size in ((score_w4, 4), (score_w5, 5)):
        torch.save(
            {
                "source": source,
                "window_size": window_size,
                "step_mapping": "action_executed",
                "event_step_scale": 1,
                "selected_events": [selected_event],
            },
            path,
        )
    spec = PhaseFeatureRun(
        label="sae",
        checkpoint=checkpoint,
        score_w4=score_w4,
        score_w5=score_w5,
        topk_dir=topk_dir,
    )
    config = TaskLocalPhaseRankingConfig(
        runs=(spec, spec, spec),
        reference_label="sae",
        condition_id="oracle",
        accepted_annotations=tmp_path / "accepted.jsonl",
        phase_groups=phase_groups,
        phase_assignments=phase_assignments,
        output_dir=tmp_path / "output",
        score_event_step_scale=1,
        topk_event_step_scale=5,
    )

    provenance, selected = _validate_task_local_score_provenance(
        spec,
        config,
        manifest,
    )

    assert selected == [
        ("event", "task", "phase_group", "grasp", 1, 2)
    ]
    assert provenance["score_step_mapping"] == "action_executed"
    assert provenance["score_event_step_scale"] == 1
    assert provenance["topk_event_step_scale"] == 5


def test_task_local_oracle_five_cells_use_one_global_phase_holm_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = ("sae1p2k", "sae10k", "sae10k_bs8192")
    phases = ("reach", "grasp")
    row_keys = []
    episode_group_keys = []
    episode_group_matrix = []
    matrix_raw = []
    accepted_rows = []
    phase_group_rows = []
    assignment_rows = []
    for cell_index in range(5):
        task_description = f"Oracle instruction {cell_index}"
        for phase_index, phase in enumerate(phases):
            source_cluster_id = f"source_{cell_index}_{phase}"
            phase_group_id = f"group_{cell_index}_{phase}"
            accepted_rows.append(
                _direct_oracle_annotation(
                    cluster_id=source_cluster_id,
                    task_description=task_description,
                    phase=phase,
                )
            )
            member_sample_ids = []
            phase_vector = [4.0, 0.0] if phase_index == 0 else [0.0, 4.0]
            for episode_num in range(2):
                sample_id = (
                    f"oracle_cell{cell_index}_{phase}_ep{episode_num}"
                )
                member_sample_ids.append(sample_id)
                assignment_rows.append(
                    {
                        "sample_id": sample_id,
                        "task_description": task_description,
                        "cluster_id": phase_group_id,
                        "phase": phase,
                        "source_cluster_id": source_cluster_id,
                        "episode_num": episode_num,
                        "waypoint_step": 10 + phase_index,
                        "progress_percent": 0.2 + 0.5 * phase_index,
                        "success": True,
                        "anchor_source": "oracle_phase_entry",
                        "anchor_sources": ["phase-entry"],
                        "phase_before": f"before-{phase}",
                        "phase_after": phase,
                        "is_phase_transition": True,
                        "state_env_step_index": 10 + phase_index,
                        "activation_env_step_index": 10 + phase_index,
                        "causal_action_env_step_index": 9 + phase_index,
                        "activation_record_index": 2,
                        "activation_record_phase": phase,
                        "action_token_offset": phase_index,
                        "observation_record_index": (
                            2 if phase_index == 0 else None
                        ),
                        "observation_record_phase": (
                            phase if phase_index == 0 else None
                        ),
                        "n_action_steps": 5,
                        "num_records": 6,
                        "num_steps": 30,
                        "oracle_upper_bound": True,
                    }
                )
                episode_group_keys.append(
                    {
                        "cluster_id": phase_group_id,
                        "episode_num": episode_num,
                        "num_events": 1,
                    }
                )
                episode_group_matrix.append(phase_vector)
            phase_group_rows.append(
                {
                    "phase_group_id": phase_group_id,
                    "task_description": task_description,
                    "phase": phase,
                    "source_cluster_ids": [source_cluster_id],
                    "member_sample_ids": member_sample_ids,
                    "num_members": len(member_sample_ids),
                }
            )
            row_keys.append(
                {
                    "task_description": task_description,
                    "cluster_id": phase_group_id,
                    "phase": phase,
                    "num_events": len(member_sample_ids),
                }
            )
            matrix_raw.append(phase_vector)

    accepted = _write_jsonl(tmp_path / "accepted.jsonl", accepted_rows)
    phase_groups = _write_jsonl(
        tmp_path / "phase_groups.jsonl",
        phase_group_rows,
    )
    phase_assignments = _write_jsonl(
        tmp_path / "phase_assignments.jsonl",
        assignment_rows,
    )
    score_w4 = tmp_path / "scores_w4.pt"
    score_w5 = tmp_path / "scores_w5.pt"
    group_tensor = torch.tensor(episode_group_matrix)
    row_tensor = torch.tensor(matrix_raw)
    for score_path, window_size in ((score_w4, 4), (score_w5, 5)):
        torch.save(
            {
                "window_size": window_size,
                "row_keys": row_keys,
                "episode_group_keys": episode_group_keys,
                "episode_group_matrix_raw": group_tensor,
                "matrix_raw": row_tensor,
                "matrix_window_mean": row_tensor,
                "matrix_task_mean": torch.zeros_like(row_tensor),
            },
            score_path,
        )

    runs = []
    for label in labels:
        checkpoint = tmp_path / f"{label}.pt"
        checkpoint.write_bytes(label.encode("utf-8"))
        topk_dir = tmp_path / f"{label}_topk"
        topk_dir.mkdir()
        (topk_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "sae_sha256": task_phase_ranking.sha256_file(checkpoint),
                    "activation_dim": 2,
                }
            ),
            encoding="utf-8",
        )
        runs.append(
            PhaseFeatureRun(
                label=label,
                checkpoint=checkpoint,
                score_w4=score_w4,
                score_w5=score_w5,
                topk_dir=topk_dir,
            )
        )

    selected_event_contract = [
        task_phase_ranking._selected_event_identity(row)
        for row in assignment_rows
    ]

    def fake_validate_score_provenance(
        spec: PhaseFeatureRun,
        config: TaskLocalPhaseRankingConfig,
        manifest: dict,
    ) -> tuple[dict, list[tuple[str, str, str, str, int, int]]]:
        del spec, config, manifest
        return (
            {
                "event_features_sha256": "events",
                "selected_event_contract_sha256": "selected-events",
                "activation_source_manifest_sha256": "activations",
                "trajectory_manifest_sha256": "trajectories",
                "dict_size": 2,
                "activation_dim": 2,
                "topk": 2,
                "layer": 15,
                "capture_target": "action_expert",
                "score_step_mapping": "action_executed",
                "score_event_step_scale": 1,
                "topk_event_step_scale": 5,
                "lossless_topk": True,
            },
            selected_event_contract,
        )

    def fake_decoder_matches(*args: object, **kwargs: object) -> dict:
        del args, kwargs
        triplets = [
            {
                "feature_ids": {label: feature_id for label in labels},
                "decoder_cosines": {
                    "sae10k__sae1p2k": 1.0,
                    "sae10k__sae10k_bs8192": 1.0,
                    "sae1p2k__sae10k_bs8192": 1.0,
                },
                "min_decoder_cosine": 1.0,
            }
            for feature_id in range(2)
        ]
        return {
            "reference_label": "sae10k",
            "target_labels": ["sae1p2k", "sae10k_bs8192"],
            "method": "fixture exact matches",
            "strict_all_pair_mnn_triplets": 2,
            "triplets": triplets,
        }

    monkeypatch.setattr(
        task_phase_ranking,
        "_validate_task_local_score_provenance",
        fake_validate_score_provenance,
    )
    monkeypatch.setattr(
        task_phase_ranking,
        "match_decoder_features_across_checkpoints",
        fake_decoder_matches,
    )

    summary = rank_task_local_phase_features(
        TaskLocalPhaseRankingConfig(
            runs=tuple(runs),
            reference_label="sae10k",
            condition_id="programmatic_oracle_5cell",
            accepted_annotations=accepted,
            phase_groups=phase_groups,
            phase_assignments=phase_assignments,
            output_dir=tmp_path / "ranking",
            num_permutations=3,
            chunk_size=2,
            top_n=2,
            score_event_step_scale=1,
            topk_event_step_scale=5,
        )
    )

    assert summary["scope"]["num_runs"] == 3
    assert summary["scope"]["num_tasks"] == 5
    assert summary["scope"]["num_phase_rows"] == 10
    assert summary["scope"]["inferential_family"]["cell_count"] == 10
    assert (
        summary["scope"]["inferential_family"][
            "task_phase_hypothesis_count"
        ]
        == 10
    )
    assert len(summary["matched_task_phase_results"]) == 10
    assert len(summary["matched_outer_holm"]) == 10
    assert {
        row["num_instructions"]
        for row in summary["cross_instruction_results"]
    } == {5}
    assert summary["analysis_config"]["score_event_step_scale"] == 1
    assert summary["analysis_config"]["topk_event_step_scale"] == 5
    label_gate = next(
        row
        for row in summary["confound_audit"]
        if row["gate"] == "Label confidence"
    )
    assert label_gate["status"] == "PASS"
    exact_entry_gate = next(
        row
        for row in summary["confound_audit"]
        if row["gate"] == "Exact phase entry"
    )
    assert exact_entry_gate["status"] == "PASS"
    assert summary["scope"]["exact_phase_entry_validation"] == {
        "required_window_half_width": 5,
        "num_assignments": 20,
        "num_invalid_assignments": 0,
        "sample_errors": {},
        "action_token_offset_counts": {"0": 10, "1": 10},
        "observation_boundary_count": 10,
        "activation_record_phase_comparison_count": 20,
        "activation_record_phase_mismatch_count": 0,
    }
    assert summary["scope"]["selected_event_scope"] == {
        "num_events": 20,
        "num_episodes": 10,
        "num_episode_phase_groups": 20,
    }

    forged_assignment_rows = [dict(row) for row in assignment_rows]
    forged_assignment_rows[0]["is_phase_transition"] = False
    forged_phase_assignments = _write_jsonl(
        tmp_path / "forged_phase_assignments.jsonl",
        forged_assignment_rows,
    )
    forged_summary = rank_task_local_phase_features(
        TaskLocalPhaseRankingConfig(
            runs=tuple(runs),
            reference_label="sae10k",
            condition_id="programmatic_oracle_5cell_forged",
            accepted_annotations=accepted,
            phase_groups=phase_groups,
            phase_assignments=forged_phase_assignments,
            output_dir=tmp_path / "ranking_forged",
            num_permutations=3,
            chunk_size=2,
            top_n=2,
            score_event_step_scale=1,
            topk_event_step_scale=5,
        )
    )
    forged_exact_entry_gate = next(
        row
        for row in forged_summary["confound_audit"]
        if row["gate"] == "Exact phase entry"
    )
    assert forged_exact_entry_gate["status"] == "FAIL"
    assert (
        forged_summary["scope"]["exact_phase_entry_validation"][
            "num_invalid_assignments"
        ]
        == 1
    )
    assert forged_summary["scope"]["exact_phase_entry_validation"][
        "sample_errors"
    ][assignment_rows[0]["sample_id"]] == ["not_phase_transition"]


def _write_topk_artifact(root: Path, *, nested: bool) -> Path:
    artifact_dir = (
        root / "sae_activations" / "post_mlp_residual"
        if nested
        else root
    )
    artifact_dir.mkdir(parents=True)
    shard_rows = [
        {
            "episode_num": torch.tensor([0]),
            "step_in_episode": torch.tensor([0]),
            "token_idx": torch.tensor([0]),
            "chunk_start_step": torch.tensor([-1]),
            "executed_chunk_len": torch.tensor([-1]),
            "top_feature_ids": torch.tensor([[1, 2]]),
            "top_feature_vals": torch.tensor([[1.5, 0.0]]),
        },
        {
            "episode_num": torch.tensor([0]),
            "step_in_episode": torch.tensor([1]),
            "token_idx": torch.tensor([0]),
            "chunk_start_step": torch.tensor([-1]),
            "executed_chunk_len": torch.tensor([-1]),
            "top_feature_ids": torch.tensor([[3, 4]]),
            "top_feature_vals": torch.tensor([[2.5, 4.5]]),
        },
    ]
    shards = []
    for index, payload in enumerate(shard_rows):
        name = f"shard_{index:06d}.pt"
        torch.save(payload, artifact_dir / name)
        shards.append({"path": name, "num_rows": 1})
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "token_topk_sparse_v1",
                "dict_size": 5,
                "topk": 2,
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )
    return root


def _event(
    sample_id: str,
    *,
    episode_num: int = 0,
    task_id: int = 1,
    task: str = "task",
) -> dict:
    return {
        "sample_id": sample_id,
        "task_description": task,
        "task_id": task_id,
        "task_episode_idx": episode_num,
        "episode_num": episode_num,
        "waypoint_rank": 0,
        "waypoint_step": 1,
        "progress_percent": 0.5,
        "num_steps": 3,
    }


def _assignment(
    sample_id: str,
    *,
    cluster_id: str = "c0",
    task: str = "task",
) -> dict:
    return {
        "sample_id": sample_id,
        "cluster_id": cluster_id,
        "task_description": task,
    }


def _annotation(
    *,
    cluster_id: str = "c0",
    task: str = "task",
) -> dict:
    return {
        "cluster_id": cluster_id,
        "task_description": task,
        "phrase": "event",
        "phase": "phase",
    }


def test_reconstruct_row_metadata_restores_record_denoise_and_token_axes() -> None:
    meta = reconstruct_action_token_row_metadata(
        num_records=2,
        episode_num=7,
        global_record_start=10,
        denoise_steps=2,
        action_horizon=4,
        executed_action_steps=2,
    )

    assert len(meta["episode_num"]) == 16
    assert meta["record_idx"].tolist() == [0] * 8 + [1] * 8
    assert meta["denoise_step"].tolist() == [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4
    assert meta["token_idx"].tolist() == [0, 1, 2, 3] * 4
    assert meta["chunk_start_step"].tolist() == [0] * 8 + [2] * 8
    assert meta["step_in_episode"].tolist() == [0, 1, 2, 3] * 2 + [2, 3, 4, 5] * 2
    assert meta["global_forward_idx"].tolist() == [10] * 8 + [11] * 8


def test_inventory_join_requires_exact_record_contract() -> None:
    source = {
        "row_order": ["source_file", "record", "denoise_step", "action_token_offset"],
        "source_denoising_steps": 2,
        "action_horizon": 4,
        "num_activation_rows": 16,
        "num_records": 2,
        "source_inventory": [
            {"path": "task/ep.pkl", "num_records": 2, "row_start": 0, "row_stop": 16}
        ],
    }
    trajectory = {
        "episodes": [
            {
                "source_file": "task/ep.pkl",
                "episode_num": 3,
                "task_id": 1,
                "task_description": "task",
                "num_records": 2,
                "n_action_steps": 2,
            }
        ]
    }

    joined = join_activation_trajectory_inventories(
        source_manifest=source,
        trajectory_manifest=trajectory,
        denoise_steps=2,
        action_horizon=4,
        executed_action_steps=2,
    )
    assert joined[0]["episode_num"] == 3
    assert joined[0]["global_record_start"] == 0

    trajectory["episodes"][0]["num_records"] = 1
    with pytest.raises(ValueError, match="record count mismatch"):
        join_activation_trajectory_inventories(
            source_manifest=source,
            trajectory_manifest=trajectory,
            denoise_steps=2,
            action_horizon=4,
            executed_action_steps=2,
        )


@pytest.mark.parametrize(
    ("records", "match"),
    [
        (
            [
                {"row_start": 0, "row_end": 2},
                {"row_start": 3, "row_end": 4},
            ],
            "gap",
        ),
        (
            [
                {"row_start": 0, "row_end": 3},
                {"row_start": 2, "row_end": 4},
            ],
            "overlap",
        ),
        ([{"row_start": 0, "row_end": 3}], "expected"),
    ],
)
def test_extract_topk_requires_exact_dense_shard_partition(
    records: list[dict],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_row_spans(
            records,
            num_rows=4,
            shard_path=Path("dense.pt"),
        )

    _validate_row_spans(
        [
            {"row_start": 0, "row_end": 2},
            {"row_start": 2, "row_end": 4},
        ],
        num_rows=4,
        shard_path=Path("dense.pt"),
    )


def test_checkpoint_source_manifest_is_found_above_nested_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "relocated" / "run"
    checkpoint = (
        run_dir / "trainer_0" / "checkpoints" / "step_100" / "ae.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    manifest_path = run_dir / "groot_source_manifest.json"
    manifest_path.write_text(
        json.dumps({"z": [3, 2, 1], "a": {"value": 7}}, indent=4),
        encoding="utf-8",
    )

    found_path, content_sha256 = _validate_checkpoint_source_manifest(
        checkpoint_path=checkpoint,
        activation_source_manifest={"a": {"value": 7}, "z": [3, 2, 1]},
    )

    assert found_path == manifest_path
    assert len(content_sha256) == 64


def test_checkpoint_source_manifest_rejects_cache_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "trainer_0" / "ae.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (run_dir / "groot_source_manifest.json").write_text(
        json.dumps({"source_files": ["checkpoint.pkl"], "num_records": 1}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not exactly match"):
        _validate_checkpoint_source_manifest(
            checkpoint_path=checkpoint,
            activation_source_manifest={
                "source_files": ["activation-cache.pkl"],
                "num_records": 1,
            },
        )


def test_sparse_encoding_records_cache_and_source_identity_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_manifest = {
        "format": "groot_n15_robocasa_pq3_action_tokens_v2",
        "source_root": "/source",
        "source_files": ["task/ep.pkl"],
        "source_inventory": [
            {
                "path": "task/ep.pkl",
                "num_records": 1,
                "row_start": 0,
                "row_stop": 2,
            }
        ],
        "num_files": 1,
        "num_records": 1,
        "num_activation_rows": 2,
        "feature_kind": "groot_n15_dit_block_residual_full_tokens_denoise",
        "feature_axes": [
            "layer",
            "denoise_step",
            "model_token",
            "feature_dim",
        ],
        "capture_token_mode": "all_token_full",
        "capture_layers": [15],
        "physical_layer": 15,
        "source_denoising_steps": 1,
        "source_model_tokens": 2,
        "token_scope": "action",
        "action_horizon": 2,
        "action_token_slice": {
            "start": 0,
            "stop": 2,
            "semantics": "half_open_model_token_indices",
        },
        "row_order": [
            "source_file",
            "record",
            "denoise_step",
            "action_token_offset",
        ],
        "activation_dim": 3,
        "source_dtype": "float16",
    }
    activation_cache = tmp_path / "activation_cache.pt"
    save_activation_cache(
        activation_cache,
        torch.arange(6, dtype=torch.float16).reshape(2, 3),
        source_manifest,
    )
    trajectory_manifest = tmp_path / "trajectory_manifest.json"
    trajectory_manifest.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "source_file": "task/ep.pkl",
                        "episode_num": 0,
                        "task_id": 1,
                        "task_description": "task",
                        "num_records": 1,
                        "n_action_steps": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "moved_run"
    checkpoint = run_dir / "trainer_0" / "ae.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (run_dir / "groot_source_manifest.json").write_text(
        json.dumps(source_manifest, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    class _FakeSAE:
        def encode(self, batch: torch.Tensor) -> torch.Tensor:
            return torch.tensor(
                [[0.0, 1.0, 2.0, 3.0]] * len(batch),
                dtype=torch.float32,
                device=batch.device,
            )

    monkeypatch.setattr(
        "event_sae.groot.activations.load_batch_topk_sae",
        lambda path, device: (
            _FakeSAE(),
            {"trainer": {"dict_size": 4, "activation_dim": 3}},
        ),
    )
    output_dir = tmp_path / "topk"
    encode_activation_cache_to_sparse_topk(
        SimpleNamespace(
            activation_cache=activation_cache,
            trajectory_manifest=trajectory_manifest,
            sae_checkpoint=checkpoint,
            output_dir=output_dir,
            layer=15,
            activation_dim=3,
            denoise_steps=1,
            action_horizon=2,
            executed_action_steps=1,
            topk=2,
            batch_size=2,
            device="cpu",
            expected_files=1,
            expected_records=1,
            expected_rows=2,
            max_sources=0,
            allow_partial=False,
            require_lossless_topk=False,
            progress_every=0,
        )
    )

    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["activation_cache_sha256"] == hashlib.sha256(
        activation_cache.read_bytes()
    ).hexdigest()
    assert manifest["activation_source_manifest_sha256"] == (
        manifest["checkpoint_source_manifest_sha256"]
    )
    assert manifest["checkpoint_source_manifest"] == str(
        run_dir / "groot_source_manifest.json"
    )
    assert manifest["source_manifest_comparison"] == "canonical_json_exact"


def test_finalize_reviewed_annotations_records_assumed_review_without_faking_human_review(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "annotations.jsonl"
    output = tmp_path / "assumed.jsonl"
    _write_jsonl(
        annotations,
        [
            {
                "cluster_id": "c0",
                "task_description": "task",
                "phrase": "approaching",
                "phase": "reach-to-object",
                "allowed_phase_labels": ["reach-to-object", "grasp"],
                "api_error": None,
                "parse_error": None,
            }
        ],
    )

    rows = finalize_reviewed_annotations(
        annotations_path=annotations,
        output_path=output,
        expected_clusters=1,
        reviews_path=None,
        assume_approved=True,
    )
    assert rows[0]["review_verdict"] == "assumed_approved"
    assert rows[0]["actual_human_review_completed"] is False
    assert rows[0]["review_mode"] == "user_authorized_assumed_review"
    with pytest.raises(FileExistsError):
        finalize_reviewed_annotations(
            annotations_path=annotations,
            output_path=output,
            expected_clusters=1,
            reviews_path=None,
            assume_approved=True,
        )


def test_alive_features_use_only_positive_executed_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "topk"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "token_topk_sparse_v1",
                "capture_target": "action_expert",
                "shards": [{"path": "shard.pt"}],
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "step_in_episode": torch.tensor([0, 2, 1]),
            "token_idx": torch.tensor([0, 2, 1]),
            "chunk_start_step": torch.tensor([0, 0, 0]),
            "executed_chunk_len": torch.tensor([2, 2, 2]),
            "top_feature_ids": torch.tensor([[1, 2], [3, 4], [5, 6]]),
            "top_feature_vals": torch.tensor([[1.0, 0.0], [9.0, 9.0], [0.0, 2.0]]),
        },
        run_dir / "shard.pt",
    )

    assert alive_feature_ids(run_dir) == {1, 6}


def test_score_matrix_scales_record_events_to_environment_steps(tmp_path: Path) -> None:
    topk = tmp_path / "topk"
    topk.mkdir()
    (topk / "manifest.json").write_text(
        json.dumps(
            {
                "format": "token_topk_sparse_v1",
                "capture_target": "action_expert",
                "dict_size": 2,
                "topk": 2,
                "layer": 15,
                "sae_path": "ae.pt",
                "shards": [{"path": "shard.pt"}],
            }
        ),
        encoding="utf-8",
    )
    values = torch.zeros((10, 2), dtype=torch.float32)
    values[:, 1] = 1.0
    values[4, 0] = 5.0
    torch.save(
        {
            "episode_num": torch.zeros(10, dtype=torch.int64),
            "step_in_episode": torch.arange(10),
            "token_idx": torch.zeros(10, dtype=torch.int64),
            "chunk_start_step": torch.arange(10),
            "executed_chunk_len": torch.ones(10, dtype=torch.int64),
            "top_feature_ids": torch.tensor([[0, 1]] * 10),
            "top_feature_vals": values,
        },
        topk / "shard.pt",
    )

    events = tmp_path / "events.jsonl"
    assignments = tmp_path / "assignments.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    output = tmp_path / "scores.pt"
    _write_jsonl(
        events,
        [
            {
                "sample_id": "s0",
                "task_description": "task",
                "task_id": 0,
                "task_episode_idx": 0,
                "episode_num": 0,
                "waypoint_rank": 0,
                "waypoint_step": 2,
                "progress_percent": 40.0,
                "num_steps": 5,
            }
        ],
    )
    _write_jsonl(
        assignments,
        [{"sample_id": "s0", "cluster_id": "c0", "task_description": "task"}],
    )
    _write_jsonl(
        annotations,
        [
            {
                "cluster_id": "c0",
                "task_description": "task",
                "phrase": "event",
                "phase": "phase",
                "episode_coverage": 1.0,
                "review_mode": "user_authorized_assumed_review",
                "review_verdict": "assumed_approved",
                "actual_human_review_completed": False,
            }
        ],
    )
    _write_jsonl(prompts, [{"episode_num": 0, "task_id": 0}])

    summary = score_cluster_features(
        topk_run_dir=topk,
        event_features_path=events,
        cluster_assignments_path=assignments,
        cluster_annotations_path=annotations,
        output_path=output,
        window_size=1,
        top_n=2,
        step_mapping="action_executed",
        event_step_scale=2,
        prompt_records_path=prompts,
    )
    payload = torch.load(output, map_location="cpu")
    assert summary["event_step_scale"] == 2
    assert tuple(payload["matrix_raw"].shape) == (1, 2)
    assert payload["selected_events"][0]["waypoint_step"] == 2
    assert payload["selected_events"][0]["event_center_step"] == 4
    assert payload["selected_events"][0]["window_steps"] == [3, 4, 5]
    assert payload["row_keys"][0]["review_verdict"] == "assumed_approved"
    assert (
        "average event projections separately"
        in payload["score_definitions"]["combined_score"]
    )
    assert (
        "episode-balanced per-cluster mean"
        in payload["score_definitions"]["matrix_raw"]
    )
    with pytest.raises(FileExistsError):
        score_cluster_features(
            topk_run_dir=topk,
            event_features_path=events,
            cluster_assignments_path=assignments,
            cluster_annotations_path=annotations,
            output_path=output,
        )


@pytest.mark.parametrize("nested", [False, True])
def test_sparse_topk_reader_preserves_manifest_row_and_value_order(
    tmp_path: Path,
    nested: bool,
) -> None:
    run_dir = _write_topk_artifact(tmp_path / "run", nested=nested)
    artifact = open_sparse_topk_artifact(run_dir)

    payloads = [
        payload for _metadata, payload in artifact.iter_shards()
    ]
    assert [
        int(payload["step_in_episode"][0]) for payload in payloads
    ] == [0, 1]
    assert [
        payload["top_feature_vals"].tolist() for payload in payloads
    ] == [
        [[1.5, 0.0]],
        [[2.5, 4.5]],
    ]

    (
        timestep_vectors,
        task_means,
        task_counts,
        manifest,
        counters,
    ) = aggregate_sparse_activations_by_timestep(
        run_dir,
        step_mapping="inference_step",
        episode_to_task_id={0: 7},
        task_id_set={7},
        dict_size=5,
    )
    assert list(timestep_vectors) == [(0, 0), (0, 1)]
    assert timestep_vectors[(0, 0)].tolist() == pytest.approx(
        [0.0, 1.5, 0.0, 0.0, 0.0]
    )
    assert timestep_vectors[(0, 1)].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 2.5, 4.5]
    )
    assert task_means[7].tolist() == pytest.approx(
        [0.0, 0.75, 0.0, 1.25, 2.25]
    )
    assert task_counts == {7: 2}
    assert manifest["dict_size"] == 5
    assert counters["shards_loaded"] == 2
    assert alive_feature_ids(run_dir) == {1, 3, 4}


def test_sparse_topk_reader_resolves_rebased_online_openpi_paths(
    tmp_path: Path,
) -> None:
    run_dir = _write_topk_artifact(tmp_path / "run", nested=True)
    nested_dir = run_dir / "sae_activations" / "post_mlp_residual"
    manifest = json.loads(
        (nested_dir / "manifest.json").read_text(encoding="utf-8")
    )
    for shard in manifest["shards"]:
        shard["path"] = (
            f"sae_activations/post_mlp_residual/{shard['path']}"
        )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    artifact = open_sparse_topk_artifact(run_dir)
    assert artifact.manifest_path == (
        run_dir / "manifest.json"
    ).resolve()
    assert [
        int(payload["step_in_episode"][0])
        for _metadata, payload in artifact.iter_shards()
    ] == [0, 1]
    assert alive_feature_ids(run_dir) == {1, 3, 4}


def test_join_rejects_duplicate_assignment_sample_id() -> None:
    with pytest.raises(
        ValueError,
        match="Duplicate sample_id in cluster assignments",
    ):
        join_cluster_events(
            event_features=[_event("s0")],
            cluster_assignments=[
                _assignment("s0"),
                _assignment("s0", cluster_id="c1"),
            ],
            cluster_annotations=[
                _annotation(),
                _annotation(cluster_id="c1"),
            ],
        )


def test_join_rejects_duplicate_annotation_cluster_even_when_invalid() -> None:
    with pytest.raises(
        ValueError,
        match="Duplicate cluster_id in cluster annotations",
    ):
        join_cluster_events(
            event_features=[],
            cluster_assignments=[],
            cluster_annotations=[
                {
                    "cluster_id": "c0",
                    "api_error": "first failure",
                },
                {
                    "cluster_id": "c0",
                    "parse_error": "second failure",
                },
            ],
        )


@pytest.mark.parametrize(
    ("assignment_task", "annotation_task"),
    [("other", "task"), ("task", "other")],
)
def test_join_rejects_task_mismatch_across_all_three_inputs(
    assignment_task: str,
    annotation_task: str,
) -> None:
    with pytest.raises(ValueError, match="Task description mismatch"):
        join_cluster_events(
            event_features=[_event("s0")],
            cluster_assignments=[
                _assignment("s0", task=assignment_task)
            ],
            cluster_annotations=[
                _annotation(task=annotation_task)
            ],
        )


def test_join_rejects_episode_to_task_id_collision() -> None:
    with pytest.raises(
        ValueError,
        match="episode_num=0 maps to multiple task_id",
    ):
        join_cluster_events(
            event_features=[
                _event("s0", task_id=1),
                _event("s1", task_id=2),
            ],
            cluster_assignments=[],
            cluster_annotations=[],
        )


def test_join_rejects_cluster_to_task_id_collision() -> None:
    with pytest.raises(
        ValueError,
        match="cluster_id=c0 maps to multiple task_id",
    ):
        join_cluster_events(
            event_features=[
                _event("s0", episode_num=0, task_id=1),
                _event("s1", episode_num=1, task_id=2),
            ],
            cluster_assignments=[
                _assignment("s0"),
                _assignment("s1"),
            ],
            cluster_annotations=[_annotation()],
        )


@pytest.mark.parametrize(
    ("initial", "records"),
    [
        ({0: 1}, [{"episode_num": 0, "task_id": 2}]),
        (
            {},
            [
                {"episode_num": 0, "task_id": 1},
                {"episode_num": 0, "task_id": 2},
            ],
        ),
    ],
)
def test_prompt_episode_task_merge_rejects_conflicts(
    initial: dict[int, int],
    records: list[dict],
) -> None:
    with pytest.raises(
        ValueError,
        match="while merging prompt_records",
    ):
        _merge_episode_task_ids(
            dict(initial),
            records,
            source_name="prompt_records",
        )


def test_legacy_matrix_alias_is_event_aligned_only(
    tmp_path: Path,
) -> None:
    scores_path = tmp_path / "legacy_scores.pt"
    torch.save(
        {
            "matrix": torch.tensor([[1.0, 3.0]]),
            "row_keys": [
                {
                    "task_description": "task",
                    "task_id": 1,
                    "cluster_id": "c0",
                    "phrase": "event",
                    "phase": "phase",
                    "episode_coverage": 1.0,
                    "num_events": 1,
                }
            ],
        },
        scores_path,
    )

    assert event_aligned_top_features_per_row(
        scores_path,
        1,
    )[0]["top_features"][0]["feature_id"] == 1
    assert event_aligned_suite_top_k(
        scores_path,
        1,
    )[0]["feature_id"] == 1

    with pytest.raises(ValueError, match="event_aligned-only"):
        window_mean_top_features_per_row(
            scores_pt_path=scores_path,
            top_n=1,
        )
    with pytest.raises(ValueError, match="event_aligned-only"):
        window_mean_suite_top_k(
            scores_pt_path=scores_path,
            top_k=1,
        )
    with pytest.raises(ValueError, match="event_aligned-only"):
        task_mean_top_features_per_task(
            scores_pt_path=scores_path,
            top_n=1,
        )
    with pytest.raises(ValueError, match="event_aligned-only"):
        task_mean_suite_top_k(
            scores_pt_path=scores_path,
            top_k=1,
        )


def test_annotation_audit_exact_joins_and_freezes_attempt(
    tmp_path: Path,
) -> None:
    features = _write_jsonl(
        tmp_path / "features.jsonl",
        [{"sample_id": "s1"}, {"sample_id": "s2"}],
    )
    assignments = _write_jsonl(
        tmp_path / "assignments.jsonl",
        [
            {
                "sample_id": "s1",
                "cluster_id": "c1",
                "anchor_source": "position",
            },
            {
                "sample_id": "s2",
                "cluster_id": "c2",
                "anchor_source": "gripper_close",
            },
        ],
    )
    clusters = _write_jsonl(
        tmp_path / "clusters.jsonl",
        [
            {
                "cluster_id": "c1",
                "task_description": "Open the drawer.",
                "episode_coverage": 0.5,
                "member_sample_ids": ["s1"],
                "representative_sample_ids": ["s1"],
            },
            {
                "cluster_id": "c2",
                "task_description": "Open the drawer.",
                "episode_coverage": 0.1,
                "member_sample_ids": ["s2"],
                "representative_sample_ids": ["s2"],
            },
        ],
    )
    media = _write_jsonl(
        tmp_path / "media.jsonl",
        [
            {
                "cluster_id": "c1",
                "representative_sample_ids": ["s1"],
                "representative_frame_paths": [["a", "b"]],
            }
        ],
    )
    annotations = _write_jsonl(
        tmp_path / "annotations.jsonl",
        [
            {
                "cluster_id": "c1",
                "task_description": "Open the drawer.",
                "representative_sample_ids": ["s1"],
                "allowed_phase_labels": ["pull"],
                "phase": "pull",
                "phrase": "pulling the drawer",
                "prompt_version": "test_prompt_v1",
                "annotation_media_layout": "triptych",
                "api_error": None,
                "parse_error": None,
            }
        ],
    )
    frozen = tmp_path / "frozen.jsonl"
    audit = tmp_path / "audit.json"

    report = audit_and_freeze_annotation_bundle(
        event_features_path=features,
        assignments_path=assignments,
        clusters_path=clusters,
        media_clusters_path=media,
        annotations_path=annotations,
        output_annotations_path=frozen,
        audit_path=audit,
        expected_events=2,
        expected_annotation_clusters=1,
        min_episode_coverage=0.3,
    )

    assert report["passed"] is True
    assert report["counts"]["selected_cluster_events"] == 1
    assert report["source_counts_all_events"] == {
        "gripper_close": 1,
        "position": 1,
    }
    assert report["phase_anchor_source_event_counts"] == {
        "pull": {"position": 1}
    }
    assert frozen.read_bytes() == annotations.read_bytes()


def _write_task_local_grid_summary(
    stage4_root: Path,
    *,
    condition_id: str,
    coverage_id: str,
    shared_task_description: str = "Shared instruction",
) -> Path:
    labels = ["sae1p2k", "sae10k", "sae10k_bs8192"]
    task_phase_keys = [
        (shared_task_description, "reach-to-object"),
        (f"{condition_id} instruction", "grasp"),
    ]
    runs = {}
    for label_index, label in enumerate(labels):
        task_phase_results: dict[str, dict[str, dict]] = {}
        for task_description, phase in task_phase_keys:
            is_focus_common = (
                label == "sae10k"
                and task_description == shared_task_description
            )
            raw_p = 0.0001 if is_focus_common else 0.8
            task_phase_results.setdefault(task_description, {})[phase] = {
                "positive_feature_count": 1,
                "top_candidates": [
                    {
                        # Equal integers are intentional: the aggregator must
                        # keep checkpoint identity in the feature key.
                        "feature_id": 7,
                        "robust_margin": 10.0 - label_index,
                        "max_t_p": raw_p,
                    }
                ],
                "best_holm_p_across_all_run_task_phase_cells": (
                    0.01 if is_focus_common else 1.0
                ),
                "statistically_supported": False,
                "inference_scope": "descriptive_only",
            }
        runs[label] = {
            "checkpoint": f"/fixture/{label}/ae.pt",
            "checkpoint_sha256": f"{label}-checkpoint-sha",
            "task_phase_results": task_phase_results,
        }

    matched_results = []
    for task_description, phase in task_phase_keys:
        is_common = task_description == shared_task_description
        matched_results.append(
            {
                "task_description": task_description,
                "phase": phase,
                "top_candidates": [
                    {
                        "feature_ids": {label: 7 for label in labels},
                        "p_all3_conjunction": 0.0001 if is_common else 0.8,
                    }
                ],
                "best_holm_p": 0.01 if is_common else 1.0,
                "statistically_supported": is_common,
            }
        )
    payload = {
        "schema_version": "task_local_phase_feature_ranking_v2",
        "scope": {
            "condition_id": condition_id,
            "num_runs": 3,
            "alpha": 0.05,
            "matched_statistically_supported": {
                "count": 1,
                "total": 2,
            },
        },
        "analysis_config": {
            "run_labels": labels,
            "reference_label": "sae10k",
        },
        "runs": runs,
        "matched_task_phase_results": matched_results,
        "confound_audit": [
            {"gate": "Length", "status": "FAIL", "evidence": "fixture"},
            {
                "gate": "Task identity",
                "status": "PASS",
                "evidence": "fixture",
            },
            {
                "gate": "Instruction balance",
                "status": "N/A",
                "evidence": "fixture",
            },
            {
                "gate": "In-sample rescue",
                "status": "N/A",
                "evidence": "fixture",
            },
            {
                "gate": "Rollout pooling",
                "status": "PASS",
                "evidence": "fixture",
            },
            {
                "gate": "Phase / dwell",
                "status": "FAIL",
                "evidence": "fixture",
            },
            {
                "gate": "Label confidence",
                "status": "FAIL",
                "evidence": "fixture",
            },
            {
                "gate": "Checkpoint independence",
                "status": "FAIL",
                "evidence": "fixture",
            },
            {
                "gate": "Observation != causation",
                "status": "PASS",
                "evidence": "fixture",
            },
            {
                "gate": "Scene-local != general",
                "status": "FAIL",
                "evidence": "fixture",
            },
            {
                "gate": "Exact phase entry",
                "status": "FAIL",
                "evidence": "fixture",
            },
        ],
    }
    path = (
        stage4_root
        / condition_id
        / coverage_id
        / "analysis"
        / "task_local_phase_feature_ranking"
        / "summary.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_stage4_grid_summary_corrects_full_grid_and_keeps_feature_scope(
    tmp_path: Path,
) -> None:
    stage4_root = tmp_path / "stage4_v12"
    for condition_id in ("e0", "e1"):
        for coverage_id in ("cov0p3", "cov0p5"):
            _write_task_local_grid_summary(
                stage4_root,
                condition_id=condition_id,
                coverage_id=coverage_id,
            )

    output_dir = tmp_path / "grid_summary"
    summary = summarize_stage4_grid(
        stage4_root=stage4_root,
        output_dir=output_dir,
        expected_conditions=2,
        expected_coverages=2,
        focus_checkpoint_label="sae10k",
    )

    assert summary["schema_version"] == "event_sae_stage4_grid_summary_v1"
    assert summary["scope"]["analysis_cell_count"] == 4
    assert summary["scope"]["checkpoint_test_count"] == 24
    assert summary["scope"]["matched_test_count"] == 8
    assert summary["matched_support"]["grid_corrected_support_count"] == 4
    assert [
        (row["task_description"], row["phase"])
        for row in summary["common_exact_task_phases"]
    ] == [("Shared instruction", "reach-to-object")]
    comparison = summary["focus_checkpoint_comparison"]
    assert comparison["corrected_support_counts"] == {
        "sae1p2k": 0,
        "sae10k": 4,
        "sae10k_bs8192": 0,
    }
    assert comparison["focus_is_unique_count_leader"] is True
    assert comparison["superiority_established"] is False
    assert comparison["status"] == "focus_is_unique_corrected_count_leader"
    assert summary["feature_identity_contract"][
        "integer_ids_comparable_across_checkpoints"
    ] is False

    common_feature_rows = [
        row
        for row in summary["evidence"]["checkpoint_tests"]
        if row["condition_id"] == "e0"
        and row["coverage_id"] == "cov0p3"
        and row["task_description"] == "Shared instruction"
    ]
    assert {row["feature_identity"]["feature_id"] for row in common_feature_rows} == {
        7
    }
    assert {
        (
            row["feature_identity"]["checkpoint_label"],
            row["feature_identity"]["checkpoint_sha256"],
            row["feature_identity"]["feature_id"],
        )
        for row in common_feature_rows
    } == {
        ("sae1p2k", "sae1p2k-checkpoint-sha", 7),
        ("sae10k", "sae10k-checkpoint-sha", 7),
        ("sae10k_bs8192", "sae10k_bs8192-checkpoint-sha", 7),
    }
    assert summary["verdict"] == "confounded — 판정 보류"
    assert (output_dir / "summary.json").is_file()
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "4 (2 conditions × 2 coverages)" in report
    assert "confounded — 판정 보류" in report
    assert "Feature ID는 반드시" in report

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        summarize_stage4_grid(
            stage4_root=stage4_root,
            output_dir=output_dir,
            expected_conditions=2,
            expected_coverages=2,
        )


def test_stage4_grid_summary_requires_complete_rectangular_grid(
    tmp_path: Path,
) -> None:
    stage4_root = tmp_path / "incomplete_stage4"
    for condition_id, coverage_id in (
        ("e0", "cov0p3"),
        ("e0", "cov0p5"),
        ("e1", "cov0p3"),
    ):
        _write_task_local_grid_summary(
            stage4_root,
            condition_id=condition_id,
            coverage_id=coverage_id,
        )

    with pytest.raises(ValueError, match="expected 2 coverages"):
        summarize_stage4_grid(
            stage4_root=stage4_root,
            output_dir=tmp_path / "unused",
            expected_conditions=2,
            expected_coverages=2,
        )


def test_stage4_grid_summary_handles_no_common_task_phase(
    tmp_path: Path,
) -> None:
    stage4_root = tmp_path / "no_common_stage4"
    for condition_id in ("e0", "e1"):
        _write_task_local_grid_summary(
            stage4_root,
            condition_id=condition_id,
            coverage_id="cov0p3",
            shared_task_description=f"{condition_id} primary instruction",
        )

    output_dir = tmp_path / "no_common_summary"
    summary = summarize_stage4_grid(
        stage4_root=stage4_root,
        output_dir=output_dir,
        expected_conditions=2,
        expected_coverages=1,
    )

    assert summary["common_exact_task_phases"] == []
    comparison = summary["focus_checkpoint_comparison"]
    assert comparison["tests_per_checkpoint"] == 0
    assert comparison["count_leaders"] == []
    assert comparison["status"] == "not_assessable_no_common_task_phase"
    assert "2개 cell 모두에 공통인" in (
        output_dir / "report.md"
    ).read_text(encoding="utf-8")


def _directional_candidate_fixture(
    feature_id: int,
    *,
    margin: float,
    coverage: float,
) -> dict:
    return {
        "feature_id": feature_id,
        "w5": {
            "margin": margin,
            "margin_percentile": 0.9,
        },
        "w4_sensitivity": {
            "positive_margin": True,
        },
        "controls": {
            "w5": {
                "overlap": False,
            }
        },
        "episode_support": {
            "w5": {
                "positive": 2,
                "comparable": 3,
            }
        },
        "phase_coverage": coverage,
    }


def test_directional_phase_table_keeps_pool_and_distinguishes_sentinels(
) -> None:
    candidates = [
        _directional_candidate_fixture(
            feature_id,
            margin=float(20 - feature_id),
            coverage=0.2,
        )
        for feature_id in range(12)
    ]
    missing_templates = {
        template: {
            "status": "missing",
            "display": "N/A",
            "num_candidates": None,
            "candidates": None,
        }
        for template in task_phase_ranking.DIRECTIONAL_TEMPLATE_NAMES
    }
    phase_rankings = {
        "phase_order": ["reach", "grasp"],
        "tasks": {
            "Open the drawer.": {
                "observed_phases": ["reach"],
                "phases": {
                    "reach": {
                        "status": "available",
                        "display": None,
                        "phase_coverage": 0.2,
                        "templates": {
                            "pulse": {
                                "status": "available",
                                "display": None,
                                "num_candidates": 12,
                                "candidates": candidates,
                            },
                            "step_up": {
                                "status": "no_candidate",
                                "display": "—",
                                "num_candidates": 0,
                                "candidates": [],
                            },
                            "step_down": {
                                "status": "no_candidate",
                                "display": "—",
                                "num_candidates": 0,
                                "candidates": [],
                            },
                        },
                    },
                    "grasp": {
                        "status": "missing",
                        "display": "N/A",
                        "reason": "phase_not_observed",
                        "phase_coverage": None,
                        "templates": missing_templates,
                    },
                },
            }
        },
    }
    recurrence_template = {
        "status": "no_candidate",
        "display": "—",
        "available_task_count": 1,
        "expected_task_count": 5,
        "num_features": 0,
        "features": [],
    }
    phase_recurrence = {
        "task_family_contract": {},
        "global_contract": {},
        "phases": {
            phase: {
                "templates": {
                    template: dict(recurrence_template)
                    for template in (
                        task_phase_ranking.DIRECTIONAL_TEMPLATE_NAMES
                    )
                }
            }
            for phase in ("reach", "grasp")
        },
    }

    table, records = (
        feature_activation_grid._directional_phase_tables_and_records(
            source_label="oracle_full",
            source_kind="simulator_oracle",
            view="coarse4_exact_rescore",
            phase_rankings=phase_rankings,
            phase_recurrence=phase_recurrence,
            display_top_n=10,
            low_coverage_threshold=0.3,
        )
    )

    phase_records = [
        record
        for record in records
        if record["record_type"] == "phase_feature"
    ]
    assert len(phase_records) == 12
    assert sum(record["in_display_top10"] for record in phase_records) == 10
    reach = table["tasks"][0]["phase_cells"][0]
    grasp = table["tasks"][0]["phase_cells"][1]
    assert reach["coverage_marker"] == "†"
    assert len(reach["templates"]["pulse"]["top10"]) == 10
    assert reach["templates"]["step_up"]["display"] == "—"
    assert grasp["templates"]["pulse"]["display"] == "N/A"
    assert {
        record["persistence"]["status"] for record in phase_records
    } == {"not-measured"}
    assert all(
        record["persistence"]["ranking_support"] is False
        for record in phase_records
    )
    assert all(
        record["persistence"]["boundary_evidence"]["status"]
        == "boundary-only"
        for record in phase_records
    )


def _probe_phase_record(
    *,
    source: str,
    feature_id: int,
    phase: str = "reach",
    template: str = "step_up",
    coverage: float = 0.5,
    rank: int = 1,
) -> dict:
    return {
        "record_type": "phase_feature",
        "source": source,
        "view": "coarse4_exact_rescore",
        "task_description": "Open the drawer.",
        "phase": phase,
        "template": template,
        "feature_id": feature_id,
        "display_rank": rank,
        "low_coverage": coverage < 0.3,
        "persistence": {"status": "not-measured"},
        "candidate": {"phase_coverage": coverage},
    }


def test_directional_probe_matches_oracle_direction_without_score_pooling(
) -> None:
    rows = []
    for feature_id in (1, 2, 3, 4):
        rows.extend(
            [
                _probe_phase_record(
                    source="v12_e3_cov0p3",
                    feature_id=feature_id,
                    rank=feature_id,
                ),
                _probe_phase_record(
                    source="v12_e4_cov0p3",
                    feature_id=feature_id,
                    rank=feature_id,
                ),
            ]
        )
    rows.extend(
        [
            _probe_phase_record(
                source="oracle_full",
                feature_id=1,
                coverage=0.5,
            ),
            _probe_phase_record(
                source="oracle_full",
                feature_id=2,
                coverage=0.2,
            ),
            _probe_phase_record(
                source="oracle_full",
                feature_id=3,
                template="step_down",
            ),
            _probe_phase_record(
                source="v12_e3_cov0p3",
                feature_id=5,
                rank=1,
            ),
            _probe_phase_record(
                source="oracle_full",
                feature_id=5,
            ),
        ]
    )
    missing_cell_candidate = _probe_phase_record(
        source="v12_e3_cov0p3",
        feature_id=6,
    )
    missing_cell_candidate["task_description"] = (
        "Open the unavailable drawer."
    )
    rows.append(missing_cell_candidate)

    def availability_view(source: str) -> dict:
        return {
            "source": source,
            "view": "coarse4_exact_rescore",
            "phase_table": {
                "tasks": [
                    {
                        "task_description": "Open the drawer.",
                        "phase_cells": [
                            {
                                "phase": "reach",
                                "templates": {
                                    "step_up": {"status": "available"},
                                    "step_down": {"status": "available"},
                                },
                            }
                        ],
                    }
                ]
            },
            "state_pairs": {"status": "not_applicable"},
        }

    table, probes = feature_activation_grid._directional_probe_ranking(
        rows,
        tables={
            "oracle": [availability_view("oracle_full")],
            "v12_e3_e4": [
                availability_view("v12_e3_cov0p3"),
                availability_view("v12_e4_cov0p3"),
            ],
        },
        display_top_n=10,
    )

    cell = next(
        cell
        for cell in table["cells"]
        if cell["phase"] == "reach"
        and cell["template"] == "step_up"
    )
    by_feature = {
        row["feature_id"]: row
        for row in probes
        if row["record_type"] == "phase_probe"
        and row["phase"] == "reach"
        and row["template"] == "step_up"
    }
    assert by_feature[1]["match_status"] == "✓"
    assert by_feature[2]["match_status"] == "✓†"
    assert by_feature[3]["match_status"] == "partial"
    assert by_feature[4]["match_status"] == "—"
    assert by_feature[5]["match_status"] == "✓"
    assert by_feature[5]["e3_e4_same_feature_direction"] is False
    assert cell["top10"][0]["feature_id"] == 1
    assert all(
        row["raw_scores_aggregated_across_sources"] is False
        for row in by_feature.values()
    )
    assert all(
        row["persistence_support_source_count"] == 0
        for row in by_feature.values()
    )
    assert "not_measured_neutral" in by_feature[1][
        "persistence_ranking_status"
    ]
    missing_probe = next(
        row
        for row in probes
        if row["feature_id"] == 6
        and row["task_description"] == "Open the unavailable drawer."
    )
    assert missing_probe["e3_e4_match_status"] == "N/A"
    assert missing_probe["match_status"] == "N/A"


def test_directional_legacy_raw_reproduction_is_explicit(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy.pt"
    exact_path = tmp_path / "exact.pt"
    tolerant_path = tmp_path / "tolerant.pt"
    mismatch_path = tmp_path / "mismatch.pt"
    base = {
        "row_keys": [{"cluster_id": "reach"}],
        "episode_group_keys": [
            {"cluster_id": "reach", "episode_num": 1}
        ],
        "matrix_raw": torch.tensor([[0.0, 1.0]]),
    }
    torch.save(base, legacy_path)
    torch.save(base, exact_path)
    torch.save(
        {
            **base,
            "matrix_raw": torch.tensor([[5e-8, 1.0]]),
        },
        tolerant_path,
    )
    torch.save(
        {
            **base,
            "matrix_raw": torch.tensor([[0.01, 1.0]]),
        },
        mismatch_path,
    )

    exact = feature_activation_grid._directional_legacy_reproduction(
        legacy_path=legacy_path,
        directional_path=exact_path,
    )
    tolerant = feature_activation_grid._directional_legacy_reproduction(
        legacy_path=legacy_path,
        directional_path=tolerant_path,
    )

    assert exact["matrix_raw_exact_torch_equal"] is True
    assert exact["reproduced"] is True
    assert tolerant["matrix_raw_exact_torch_equal"] is False
    assert tolerant["matrix_raw_tolerance_equal"] is True
    assert tolerant["matrix_raw_comparison"]["max_abs"] == pytest.approx(
        5e-8
    )
    with pytest.raises(
        ValueError,
        match="failed immutable legacy reproduction",
    ):
        feature_activation_grid._directional_legacy_reproduction(
            legacy_path=legacy_path,
            directional_path=mismatch_path,
        )


def test_directional_transition_boundary_evidence_reads_flat_pair_support(
) -> None:
    component = _directional_candidate_fixture(
        7,
        margin=1.0,
        coverage=0.5,
    )
    evidence = (
        feature_activation_grid._directional_transition_boundary_evidence(
            {
                "on": component,
                "off": component,
                "episode_pair_support": {
                    "positive": 2,
                    "comparable": 3,
                    "fraction": 2 / 3,
                    "display": "2/3",
                },
            }
        )
    )

    assert evidence["status"] == "boundary-only"
    assert evidence["same_episode_pair_support"] == {
        "positive": 2,
        "comparable": 3,
        "fraction": pytest.approx(2 / 3),
        "display": "2/3",
    }
    assert evidence["dense_recollection_required"] is False
    assert evidence["absence_is_censored_by_sparse_topk"] is None


def test_directional_lossless_trace_classifies_repeated_patterns(
) -> None:
    task_description = "Open the drawer."
    selected_events = []
    for episode_num in (1, 2):
        selected_events.extend(
            [
                {
                    "task_description": task_description,
                    "episode_num": episode_num,
                    "phase": "reach",
                    "event_center_step": 7,
                },
                {
                    "task_description": task_description,
                    "episode_num": episode_num,
                    "phase": "reach",
                    "event_center_step": 5,
                },
                {
                    "task_description": task_description,
                    "episode_num": episode_num,
                    "phase": "grasp",
                    "event_center_step": 4,
                },
                {
                    "task_description": task_description,
                    "episode_num": episode_num,
                    "phase": "grasp",
                    "event_center_step": 15,
                },
            ]
        )

    timestep_vectors = {}
    for episode_num in (1, 2):
        for step in range(3, 17):
            within_interval = 5 <= step < 15
            vector = torch.tensor(
                [
                    4.0 if within_interval else 0.0,
                    (
                        4.0
                        if within_interval
                        or (episode_num == 2 and step >= 15)
                        else 0.0
                    ),
                    1.0,
                ]
            )
            timestep_vectors[(episode_num, step)] = vector
    transition_rankings = {
        "ordered_transition_pairs": ["reach->grasp"],
        "tasks": {
            task_description: {
                "transitions": {
                    "reach->grasp": {
                        "candidates": [
                            {"feature_id": feature_id}
                            for feature_id in range(4)
                        ]
                    }
                }
            }
        },
    }
    config = (
        feature_activation_grid.DirectionalTracePersistenceConfig(
            local_window_size=2,
            minimum_interval_steps=3,
            minimum_comparable_episodes=2,
            confirmed_minimum_full_repeat_ratio=0.75,
            partial_minimum_full_repeat_ratio=0.25,
            partial_minimum_component_repeat_ratio=0.5,
        )
    )

    results, audit = (
        feature_activation_grid._directional_evaluate_trace_persistence(
            selected_events=selected_events,
            timestep_vectors=timestep_vectors,
            transition_rankings=transition_rankings,
            display_top_n=10,
            config=config,
        )
    )

    by_feature = {
        feature_id: results[
            (task_description, "reach->grasp", feature_id)
        ]
        for feature_id in range(4)
    }
    assert {
        feature_id: row["status"]
        for feature_id, row in by_feature.items()
    } == {
        0: "confirmed",
        1: "partial",
        2: "boundary-only",
        3: "insufficient support",
    }
    assert by_feature[0]["full_pattern_repeat_ratio"] == 1.0
    assert by_feature[1]["full_pattern_repeat_ratio"] == 0.5
    assert by_feature[1]["two_of_three_repeat_ratio"] == 1.0
    assert by_feature[3]["comparable_episode_count"] == 0
    assert {
        (
            row["on_anchor_step"],
            row["off_anchor_step"],
        )
        for row in by_feature[0]["episode_patterns"]
    } == {(5, 15)}
    assert by_feature[0]["ranking_gate"] is False
    assert by_feature[0]["absence_is_censored_by_sparse_topk"] is False
    assert audit["status_counts"] == {
        "confirmed": 1,
        "partial": 1,
        "boundary-only": 1,
        "insufficient support": 1,
    }


def test_directional_trace_rejects_lossy_topk_manifest() -> None:
    manifest = {
        "topk": 4,
        "encoding_stats": {
            "lossless_topk": True,
            "max_positive_features_per_row": 4,
            "rows_with_more_positive_features_than_topk": 0,
        },
    }
    contract = (
        feature_activation_grid
        ._validate_directional_lossless_topk_manifest(manifest)
    )
    assert contract["absence_is_exact_zero"] is True
    lossy = {
        **manifest,
        "encoding_stats": {
            **manifest["encoding_stats"],
            "lossless_topk": False,
            "rows_with_more_positive_features_than_topk": 1,
        },
    }
    with pytest.raises(ValueError, match="requires a lossless Top-K"):
        feature_activation_grid._validate_directional_lossless_topk_manifest(
            lossy
        )


def _directional_task_score_fixture(
    task_description: str,
    phases: list[str],
    phase_w5: dict[str, np.ndarray],
    *,
    phase_w4: dict[str, np.ndarray] | None = None,
    group_keys: list[tuple[int, str]] | None = None,
    group_w5: dict[str, np.ndarray] | None = None,
    group_w4: dict[str, np.ndarray] | None = None,
    window_mean_w5: np.ndarray | None = None,
    task_mean_w5: np.ndarray | None = None,
) -> task_phase_ranking.DirectionalTemplateTaskScores:
    phase_matrices_w5 = {
        template: np.asarray(phase_w5[template], dtype=np.float64)
        for template in task_phase_ranking.DIRECTIONAL_TEMPLATE_NAMES
    }
    num_phases, num_features = phase_matrices_w5["pulse"].shape
    assert num_phases == len(phases)
    if group_keys is None:
        group_keys = [(0, phase) for phase in phases]
    group_matrices_w5 = (
        {
            template: phase_matrices_w5[template].copy()
            for template in task_phase_ranking.DIRECTIONAL_TEMPLATE_NAMES
        }
        if group_w5 is None
        else {
            template: np.asarray(
                group_w5[template],
                dtype=np.float64,
            )
            for template in task_phase_ranking.DIRECTIONAL_TEMPLATE_NAMES
        }
    )
    phase_matrices_w4 = (
        {
            template: np.asarray(
                phase_w4[template],
                dtype=np.float64,
            )
            for template in task_phase_ranking.DIRECTIONAL_TEMPLATE_NAMES
        }
        if phase_w4 is not None
        else None
    )
    if phase_matrices_w4 is None:
        group_matrices_w4 = None
    elif group_w4 is None:
        group_matrices_w4 = {
            template: phase_matrices_w4[template].copy()
            for template in task_phase_ranking.DIRECTIONAL_TEMPLATE_NAMES
        }
    else:
        group_matrices_w4 = {
            template: np.asarray(
                group_w4[template],
                dtype=np.float64,
            )
            for template in task_phase_ranking.DIRECTIONAL_TEMPLATE_NAMES
        }
    window_w5 = (
        np.zeros((num_phases, num_features), dtype=np.float64)
        if window_mean_w5 is None
        else np.asarray(window_mean_w5, dtype=np.float64)
    )
    task_w5 = (
        np.zeros((num_phases, num_features), dtype=np.float64)
        if task_mean_w5 is None
        else np.asarray(task_mean_w5, dtype=np.float64)
    )
    return task_phase_ranking.DirectionalTemplateTaskScores(
        task_description=task_description,
        row_keys=[
            {
                "task_description": task_description,
                "cluster_id": f"{task_description}:{phase}",
                "phase": phase,
                "episode_coverage": 0.5,
            }
            for phase in phases
        ],
        phases=list(phases),
        group_keys=list(group_keys),
        group_event_counts=[1] * len(group_keys),
        phase_coverage={phase: 0.5 for phase in phases},
        phase_template_scores_w5=phase_matrices_w5,
        group_template_scores_w5=group_matrices_w5,
        window_mean_w5=window_w5,
        task_mean_w5=task_w5,
        phase_template_scores_w4=phase_matrices_w4,
        group_template_scores_w4=group_matrices_w4,
        window_mean_w4=(
            window_w5.copy() if phase_matrices_w4 is not None else None
        ),
        task_mean_w4=(
            task_w5.copy() if phase_matrices_w4 is not None else None
        ),
    )


def _directional_feature(
    candidates: list[dict],
    feature_id: int,
) -> dict:
    return next(
        candidate
        for candidate in candidates
        if int(candidate["feature_id"]) == feature_id
    )


def test_directional_w4_and_control_overlap_never_gate_w5_candidates(
) -> None:
    zeros = np.zeros((2, 3), dtype=np.float64)
    task = _directional_task_score_fixture(
        "Open the drawer.",
        ["reach", "grasp"],
        {
            "pulse": zeros,
            "step_up": np.asarray(
                [[5.0, 3.0, 0.0], [0.0, 0.0, 0.0]]
            ),
            "step_down": zeros,
        },
        phase_w4={
            "pulse": zeros,
            "step_up": np.asarray(
                [[0.0, 0.0, 0.0], [4.0, 1.0, 0.0]]
            ),
            "step_down": zeros,
        },
        window_mean_w5=np.asarray(
            [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
        ),
        task_mean_w5=np.asarray(
            [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
        ),
    )
    ranked = task_phase_ranking.rank_directional_phase_candidates(
        {task.task_description: task},
        config=task_phase_ranking.DirectionalDiscoveryConfig(
            control_top_n=1,
            phase_order=("reach", "grasp"),
        ),
    )

    step_up = ranked["tasks"][task.task_description]["phases"]["reach"][
        "templates"
    ]["step_up"]
    assert [
        candidate["feature_id"] for candidate in step_up["candidates"]
    ] == [0, 1]
    primary = step_up["candidates"][0]
    assert primary["controls"]["w5"]["overlap"] is True
    assert primary["w4_sensitivity"]["margin"] == pytest.approx(-4.0)
    assert primary["w4_sensitivity"]["positive_margin"] is False
    assert primary["eligibility"] == {
        "eligible": True,
        "basis": "positive_w5_phase_vs_strongest_rest_margin",
        "w4_is_gate": False,
        "controls_are_gate": False,
        "reasons": [],
    }
    assert step_up["num_eligible"] == step_up["num_candidates"] == 2


def test_directional_state_pair_uses_direction_and_same_episode_support(
) -> None:
    phase_w5 = {
        "pulse": np.asarray(
            [[0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        ),
        "step_up": np.asarray(
            [[5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]]
        ),
        "step_down": np.asarray(
            [[1.0, 0.0, 4.0, 0.0], [2.0, 0.0, 0.0, 0.0]]
        ),
    }
    group_w5 = {
        "pulse": np.asarray(
            [
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "step_up": np.asarray(
            [
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
            ]
        ),
        "step_down": np.asarray(
            [
                [2.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [4.0, 0.0, 0.0, 0.0],
            ]
        ),
    }
    task = _directional_task_score_fixture(
        "Pick the object and place it.",
        ["grasp", "terminal"],
        phase_w5,
        group_keys=[
            (10, "grasp"),
            (10, "terminal"),
            (11, "grasp"),
            (11, "terminal"),
        ],
        group_w5=group_w5,
    )
    config = task_phase_ranking.DirectionalDiscoveryConfig(
        control_top_n=1,
        phase_order=("grasp", "terminal"),
    )
    phase_rankings = task_phase_ranking.rank_directional_phase_candidates(
        {task.task_description: task},
        config=config,
    )
    transition_rankings = (
        task_phase_ranking.rank_directional_transition_candidates(
            phase_rankings,
            config=config,
        )
    )

    phases = phase_rankings["tasks"][task.task_description]["phases"]
    assert _directional_feature(
        phases["grasp"]["templates"]["pulse"]["candidates"],
        1,
    )
    assert _directional_feature(
        phases["terminal"]["templates"]["step_up"]["candidates"],
        2,
    )
    assert _directional_feature(
        phases["grasp"]["templates"]["step_down"]["candidates"],
        2,
    )
    pair_candidates = transition_rankings["tasks"][
        task.task_description
    ]["transitions"]["grasp->terminal"]["candidates"]
    assert [candidate["feature_id"] for candidate in pair_candidates] == [0]
    paired = pair_candidates[0]["episode_pair_support"]
    assert paired["positive"] == 1
    assert paired["comparable"] == 2
    assert paired["fraction"] == pytest.approx(0.5)
    assert paired["positive_episode_ids"] == [11]
    assert paired["comparable_episode_ids"] == [10, 11]
    assert paired["ranking_gate"] is False


def test_directional_recurrence_separates_support_and_availability(
) -> None:
    task_names = [
        "Open the drawer.",
        "Close the drawer.",
        "Pick object A and place it.",
        "Pick object B and place it.",
        "Pick object C and place it.",
    ]
    tasks = {}
    for task_index, task_name in enumerate(task_names):
        phases = (
            ["reach", "grasp", "terminal"]
            if task_index < 4
            else ["reach", "terminal"]
        )
        template_scores = {
            template: np.zeros((len(phases), 5), dtype=np.float64)
            for template in (
                task_phase_ranking.DIRECTIONAL_TEMPLATE_NAMES
            )
        }
        phase_index = {phase: index for index, phase in enumerate(phases)}
        template_scores["step_up"][phase_index["reach"], 2] = 5.0
        if "grasp" in phase_index:
            template_scores["step_up"][phase_index["grasp"], 0] = 4.0
        template_scores["step_down"][
            phase_index["terminal"], 0
        ] = 4.0
        template_scores["step_down"][
            phase_index["terminal"], 2
        ] = 5.0
        template_scores["pulse"][phase_index["terminal"], 3] = 4.0
        tasks[task_name] = _directional_task_score_fixture(
            task_name,
            phases,
            template_scores,
        )
    config = task_phase_ranking.DirectionalDiscoveryConfig(
        control_top_n=1
    )
    phase_rankings = task_phase_ranking.rank_directional_phase_candidates(
        tasks,
        config=config,
    )
    phase_recurrence = (
        task_phase_ranking.summarize_directional_phase_recurrence(
            phase_rankings,
            config=config,
        )
    )
    transition_rankings = (
        task_phase_ranking.rank_directional_transition_candidates(
            phase_rankings,
            config=config,
        )
    )
    transition_recurrence = (
        task_phase_ranking.summarize_directional_transition_recurrence(
            transition_rankings,
            config=config,
        )
    )

    grasp_feature = _directional_feature(
        phase_recurrence["phases"]["grasp"]["templates"]["step_up"][
            "features"
        ],
        0,
    )
    drawer = grasp_feature["families"]["drawer"]
    object_family = grasp_feature["families"]["object"]
    assert drawer["support_over_eligible"]["display"] == "2/2"
    assert drawer["eligible_over_expected"]["display"] == "2/2"
    assert drawer["strict"] is True
    assert object_family["support_over_eligible"]["display"] == "2/2"
    assert object_family["eligible_over_expected"]["display"] == "2/3"
    assert object_family["strict"] is False
    assert object_family["relaxed"] is True
    assert grasp_feature["support_over_eligible"]["display"] == "4/4"
    assert grasp_feature["eligible_over_expected"]["display"] == "4/5"
    assert grasp_feature["global_strict"] is False
    assert grasp_feature["global_relaxed"] is True

    terminal_pulse = _directional_feature(
        phase_recurrence["phases"]["terminal"]["templates"]["pulse"][
            "features"
        ],
        3,
    )
    assert terminal_pulse["global_strict"] is True
    assert phase_recurrence["phases"]["transport"]["display"] == "N/A"
    assert phase_recurrence["phases"]["grasp"]["templates"]["pulse"][
        "display"
    ] == "—"

    grasp_transition = _directional_feature(
        transition_recurrence["transitions"]["grasp->terminal"][
            "features"
        ],
        0,
    )
    assert grasp_transition["support_over_eligible"]["display"] == "4/4"
    assert grasp_transition["eligible_over_expected"]["display"] == "4/5"
    assert grasp_transition["global_strict"] is False
    assert grasp_transition["global_relaxed"] is True
    reach_transition = _directional_feature(
        transition_recurrence["transitions"]["reach->terminal"][
            "features"
        ],
        2,
    )
    assert reach_transition["global_strict"] is True
    assert transition_recurrence["transitions"]["reach->grasp"][
        "display"
    ] == "—"


def test_directional_analyzer_writes_exactly_twelve_new_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    focused_summary = tmp_path / "focused_summary.json"
    focused_summary.write_text("{}\n", encoding="utf-8")
    dummy_input = tmp_path / "input.jsonl"
    dummy_input.write_text("{}\n", encoding="utf-8")
    source_inventory = {}
    for source_label in feature_activation_grid.DIRECTIONAL_SOURCE_ORDER:
        source_inventory[source_label] = {
            "source_label": source_label,
            "source_kind": (
                "simulator_oracle"
                if source_label == "oracle_full"
                else "v12_annotation"
            ),
            "coverage_filter": (
                None if source_label == "oracle_full" else "cov0p3"
            ),
            "expected": {
                "selected_events": 1,
                "event_step_scale": 1,
                "w4_shifted_windows": 0,
                "w5_shifted_windows": 0,
            },
            "checkpoint": dummy_input,
            "topk_dir": tmp_path,
            "event_features": dummy_input,
            "prompt_records": dummy_input,
            "step_mapping": "action_executed",
            "event_step_scale": 1,
            "selected_sample_ids": {"sample"},
            "views": {
                view: {
                    "phase_assignments": dummy_input,
                    "phase_groups": dummy_input,
                    "expected_rows": 1,
                    "expected_episode_groups": 1,
                    "legacy_score_w4": dummy_input,
                    "legacy_score_w5": dummy_input,
                }
                for view in (
                    feature_activation_grid.DIRECTIONAL_VIEW_ORDER
                )
            },
        }
    focused_spec = feature_activation_grid._focused_file_spec(
        focused_summary
    )
    monkeypatch.setattr(
        feature_activation_grid,
        "_directional_focused_inventory",
        lambda _path: (
            focused_spec,
            source_inventory,
            {"focused_summary": focused_spec},
        ),
    )
    score_calls = []

    def fake_score_cluster_features(**kwargs: object) -> dict:
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"directional-score")
        score_calls.append(dict(kwargs))
        return {"output_path": str(output_path)}

    monkeypatch.setattr(
        feature_activation_grid,
        "score_cluster_features",
        fake_score_cluster_features,
    )
    monkeypatch.setattr(
        feature_activation_grid,
        "_validate_score_pair_inventory",
        lambda **_kwargs: {
            "num_rows": 1,
            "num_episode_groups": 1,
            "num_selected_events": 1,
            "w4_shifted_windows": 0,
            "w5_shifted_windows": 0,
        },
    )
    monkeypatch.setattr(
        feature_activation_grid,
        "_directional_score_compatibility",
        lambda path: {
            "window_size": (
                4 if path.parent.parent.name == "w4" else 5
            ),
            "legacy_raw_relation": {"num_diff": 0, "max_abs": 0.0},
            "episode_group_raw_equals_template_max": True,
            "row_raw_equals_episode_group_raw_mean": True,
            "combined_role": "compatibility_only",
        },
    )
    monkeypatch.setattr(
        feature_activation_grid,
        "_directional_legacy_reproduction",
        lambda **_kwargs: {
            "reproduced": True,
            "matrix_raw_exact_torch_equal": True,
            "matrix_raw_tolerance_equal": True,
        },
    )
    fake_tasks = {
        f"Task {index}": SimpleNamespace(phases=["fine"])
        for index in range(5)
    }
    monkeypatch.setattr(
        feature_activation_grid,
        "load_directional_template_scores",
        lambda **_kwargs: fake_tasks,
    )
    monkeypatch.setattr(
        feature_activation_grid,
        "rank_directional_phase_candidates",
        lambda _tasks, *, config: {
            "phase_order": list(config.phase_order),
            "tasks": {},
        },
    )
    monkeypatch.setattr(
        feature_activation_grid,
        "summarize_directional_phase_recurrence",
        lambda _ranking, *, config: {
            "phase_order": list(config.phase_order),
        },
    )
    six_pairs = [
        "reach->grasp",
        "reach->transport",
        "reach->terminal",
        "grasp->transport",
        "grasp->terminal",
        "transport->terminal",
    ]
    monkeypatch.setattr(
        feature_activation_grid,
        "discover_directional_phase_features",
        lambda **_kwargs: {
            "contract": {"candidate_membership": "positive_w5"},
            "phase_rankings": {},
            "phase_recurrence": {},
            "transition_rankings": {
                "ordered_transition_pairs": six_pairs,
            },
            "transition_recurrence": {},
        },
    )
    empty_phase_table = {
        "phase_vocabulary": [],
        "phase_vocabulary_role": "fixture",
        "tasks": [],
        "recurrence": {
            "task_family_contract": {},
            "global_contract": {},
            "rows": [],
        },
    }
    monkeypatch.setattr(
        feature_activation_grid,
        "_directional_phase_tables_and_records",
        lambda **_kwargs: (dict(empty_phase_table), []),
    )
    empty_state_pairs = {
        "status": "available",
        "scope": "coarse4_only",
        "phase_order": list(task_phase_ranking.COARSE_PHASE_ORDER),
        "pair_count": 6,
        "pair_definition": "fixture",
        "tasks": [],
        "recurrence": {
            "task_family_contract": {},
            "global_contract": {},
            "rows": [],
        },
    }
    monkeypatch.setattr(
        feature_activation_grid,
        "_directional_transition_tables_and_records",
        lambda **_kwargs: (dict(empty_state_pairs), []),
    )
    trace_calls = []

    def fake_trace_persistence(**kwargs: object) -> tuple[dict, dict]:
        trace_calls.append(dict(kwargs))
        return {}, {
            "source": kwargs["source_label"],
            "scan_count": 1,
            "measured_candidate_count": 0,
            "status_counts": {},
        }

    monkeypatch.setattr(
        feature_activation_grid,
        "_directional_source_trace_persistence",
        fake_trace_persistence,
    )

    output_dir = tmp_path / "directional"
    summary = analyze_directional_phase_views(
        DirectionalPhaseViewAnalysisConfig(
            focused_summary=focused_summary,
            output_dir=output_dir,
        )
    )

    assert len(score_calls) == 12
    assert sorted(
        int(call["window_size"]) for call in score_calls
    ) == [4] * 6 + [5] * 6
    assert all(
        Path(call["output_path"]).is_relative_to(output_dir)
        for call in score_calls
    )
    assert summary["scope"]["score_artifact_count"] == 12
    assert len(trace_calls) == 3
    assert {
        call["source_label"] for call in trace_calls
    } == set(feature_activation_grid.DIRECTIONAL_SOURCE_ORDER)
    assert summary["scope"]["trace_scan_count"] == 3
    assert len(summary["tables"]["oracle"]) == 2
    assert len(summary["tables"]["v12_e3_e4"]) == 4
    assert summary["inputs"]["before_after_hash_verification"] == {
        "input_files_unchanged": True,
        "implementation_files_unchanged": True,
    }
    assert feature_activation_grid.sha256_file(focused_summary) == (
        focused_spec["sha256"]
    )
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "report.md").is_file()
    assert (output_dir / "candidates.jsonl").is_file()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        analyze_directional_phase_views(
            DirectionalPhaseViewAnalysisConfig(
                focused_summary=focused_summary,
                output_dir=output_dir,
            )
        )


def test_directional_cli_subcommand_contract() -> None:
    args = build_phase_feature_parser().parse_args(
        [
            "analyze-directional-phase-views",
            "--focused-summary",
            "focused.json",
            "--output-dir",
            "directional-output",
        ]
    )

    assert args.handler.__name__ == "_analyze_directional_phase_views"
    assert args.focused_summary == Path("focused.json")
    assert args.output_dir == Path("directional-output")
