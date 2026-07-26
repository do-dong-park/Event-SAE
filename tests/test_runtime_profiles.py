from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from event_sae import (
    DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_EXPERIMENT_ROOT,
    DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_PROFILE,
    DEFAULT_GROOT_LOG_ROOT,
    LEGACY_GROOT_ARTIFACT_RELOCATIONS,
    REPO_ROOT,
    load_pipeline_profile,
    resolve_groot_artifact_path,
)
from event_sae.openpi.eval.config import (
    validate_intervention_layer,
    validate_policy_override_pair,
    validate_sae_intervention_checkpoint,
)


def test_default_groot_profile_is_consumable() -> None:
    profile = load_pipeline_profile()

    assert (
        profile.path
        == DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_PROFILE.resolve()
    )
    assert DEFAULT_GROOT_LOG_ROOT == REPO_ROOT / "logs/groot_n15"
    assert DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_EXPERIMENT_ROOT == (
        DEFAULT_GROOT_LOG_ROOT
        / "experiments/v9_abs_position_gripper_3view_action_phase_v1"
    )
    assert profile.profile_id == "v9_abs_gripper_3view_actionphase_v1"
    assert profile.require("media", "view_order") == [
        "left",
        "right",
        "wrist",
    ]
    assert profile.require(
        "clustering",
        "episode_coverage_sweep",
    ) == [0.3, 0.4, 0.5]
    assert (
        profile.require(
            "clustering",
            "annotation_min_episode_coverage",
        )
        == 0.3
    )
    assert profile.path_value(
        "source",
        "trajectory_records_path",
    ) == (
        REPO_ROOT
        / "logs/groot_n15/stage2_waypoints/absolute_position/"
        "trajectory_records.jsonl"
    )


def test_anchor_view_ablation_profile_matches_canonical_artifacts() -> None:
    profile = load_pipeline_profile(
        REPO_ROOT / "configs/groot/anchor_view_controlled_ablation_v1.json"
    )

    assert profile.profile_id == "anchor_view_controlled_ablation_v1"
    assert profile.require("clustering", "distance_threshold") == 0.18
    assert profile.require("clustering", "distance_sweep_enabled") is False
    assert profile.require("annotation", "temperature") == 0.0
    assert profile.require("stage4", "event_step_scale") == 5
    assert profile.require("stage4", "ranking_min_coverage") == 0.0
    assert profile.require("stage4", "selected_topk_run_dir") is None
    assert len(profile.require("conditions")) == 5

    source_paths = (
        ("source", "relative_trajectory_records_path"),
        ("source", "absolute_trajectory_records_path"),
        ("reuse_sources", "relative_position_waypoints"),
        ("reuse_sources", "relative_position_features"),
        ("reuse_sources", "absolute_position_gripper_run"),
        ("reuse_sources", "absolute_position_gripper_partition"),
    )
    for keys in source_paths:
        assert profile.path_value(*keys).exists()

    for raw_path in profile.require(
        "stage4",
        "candidate_topk_run_dirs",
    ):
        assert (REPO_ROOT / raw_path).is_dir()


def test_profile_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps({"format": "unknown", "profile_id": "test"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported profile format"):
        load_pipeline_profile(path)


@pytest.mark.parametrize(
    ("legacy_name", "canonical"),
    tuple(LEGACY_GROOT_ARTIFACT_RELOCATIONS.items()),
)
def test_legacy_groot_artifact_paths_relocate_without_symlinks(
    legacy_name: str,
    canonical: Path,
) -> None:
    expected = DEFAULT_GROOT_LOG_ROOT / canonical

    assert resolve_groot_artifact_path(
        Path("logs/groot_n15") / legacy_name
    ) == expected
    assert resolve_groot_artifact_path(
        REPO_ROOT / "logs/groot_n15" / legacy_name
    ) == expected
    assert resolve_groot_artifact_path(Path(legacy_name)) == expected


def test_legacy_groot_artifact_suffix_is_preserved() -> None:
    path = Path(
        "logs/groot_n15/pq3_stage4_event_sae/"
        "l15_sae1p2k_exec5_mean4_top96_v1/topk/manifest.json"
    )

    assert resolve_groot_artifact_path(path) == (
        DEFAULT_GROOT_LOG_ROOT
        / "stage4_feature_ranking/"
        "l15_sae1p2k_exec5_mean4_top96_v1/topk/manifest.json"
    )


def test_nonlegacy_artifact_path_is_unchanged() -> None:
    path = Path("logs/groot_n15/stage2_waypoints/absolute_position")

    assert resolve_groot_artifact_path(path) == path


def test_profile_rejects_unsorted_coverage_sweep(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "format": "event_sae_pipeline_profile_v1",
                "profile_id": "test",
                "clustering": {
                    "episode_coverage_sweep": [0.5, 0.3],
                    "annotation_min_episode_coverage": 0.3,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique, sorted"):
        load_pipeline_profile(path)


def test_intervention_layer_accepts_effective_targets() -> None:
    validate_intervention_layer("action_expert", 17, 18)
    validate_intervention_layer("paligemma", 16, 18)


def test_policy_override_pair_accepts_both_or_neither() -> None:
    validate_policy_override_pair(None, None)
    validate_policy_override_pair("custom_config", "/checkpoints/custom")


@pytest.mark.parametrize(
    ("config_name", "checkpoint_dir"),
    (("custom_config", None), (None, "/checkpoints/custom")),
)
def test_policy_override_pair_rejects_partial_contract(
    config_name: str | None,
    checkpoint_dir: str | None,
) -> None:
    with pytest.raises(ValueError, match="provide both or neither"):
        validate_policy_override_pair(config_name, checkpoint_dir)


@pytest.mark.parametrize("layer_idx", [-1, 18])
def test_intervention_layer_rejects_out_of_range_indices(
    layer_idx: int,
) -> None:
    with pytest.raises(ValueError, match="outside model depth"):
        validate_intervention_layer(
            "action_expert",
            layer_idx,
            18,
        )


def test_intervention_layer_rejects_final_paligemma_layer() -> None:
    with pytest.raises(ValueError, match="final PaliGemma layer"):
        validate_intervention_layer("paligemma", 17, 18)


def _write_sae_checkpoint_contract(
    root: Path,
    *,
    layer: int = 17,
    activation_dim: int = 3,
    dict_size: int = 6,
    submodule_name: str = "post_mlp_residual",
) -> tuple[Path, dict[str, np.ndarray]]:
    checkpoint_path = root / "ae.pt"
    checkpoint_path.write_bytes(b"test checkpoint placeholder")
    (root / "config.json").write_text(
        json.dumps(
            {
                "trainer": {
                    "dict_class": "BatchTopKSAE",
                    "layer": layer,
                    "activation_dim": activation_dim,
                    "dict_size": dict_size,
                    "submodule_name": submodule_name,
                }
            }
        ),
        encoding="utf-8",
    )
    state_dict = {
        "encoder.weight": np.zeros((dict_size, activation_dim)),
        "encoder.bias": np.zeros((dict_size,)),
        "decoder.weight": np.zeros((activation_dim, dict_size)),
        "b_dec": np.zeros((activation_dim,)),
        "k": np.array(2),
        "threshold": np.array(0.0),
    }
    return checkpoint_path, state_dict


def test_sae_intervention_checkpoint_contract_accepts_matching_metadata(
    tmp_path: Path,
) -> None:
    checkpoint_path, state_dict = _write_sae_checkpoint_contract(tmp_path)

    contract = validate_sae_intervention_checkpoint(
        checkpoint_path,
        capture_target="action_expert",
        layer_idx=17,
        activation_dim=3,
        state_dict=state_dict,
        feature_indices=(0, 5),
    )

    assert contract["dict_size"] == 6
    assert contract["submodule_name"] == "post_mlp_residual"


def test_sae_intervention_checkpoint_contract_rejects_wrong_layer(
    tmp_path: Path,
) -> None:
    checkpoint_path, state_dict = _write_sae_checkpoint_contract(
        tmp_path,
        layer=5,
    )

    with pytest.raises(ValueError, match="SAE layer mismatch"):
        validate_sae_intervention_checkpoint(
            checkpoint_path,
            capture_target="action_expert",
            layer_idx=17,
            activation_dim=3,
            state_dict=state_dict,
        )


def test_sae_intervention_checkpoint_contract_rejects_wrong_target(
    tmp_path: Path,
) -> None:
    checkpoint_path, state_dict = _write_sae_checkpoint_contract(tmp_path)

    with pytest.raises(ValueError, match="SAE capture-target mismatch"):
        validate_sae_intervention_checkpoint(
            checkpoint_path,
            capture_target="paligemma",
            layer_idx=17,
            activation_dim=3,
            state_dict=state_dict,
        )


def test_sae_intervention_checkpoint_contract_rejects_feature_out_of_range(
    tmp_path: Path,
) -> None:
    checkpoint_path, state_dict = _write_sae_checkpoint_contract(tmp_path)

    with pytest.raises(ValueError, match=r"outside \[0, 6\)"):
        validate_sae_intervention_checkpoint(
            checkpoint_path,
            capture_target="action_expert",
            layer_idx=17,
            activation_dim=3,
            state_dict=state_dict,
            feature_indices=(6,),
        )
