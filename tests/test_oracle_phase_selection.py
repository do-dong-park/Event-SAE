from __future__ import annotations

import json
from pathlib import Path

import pytest

from event_sae.groot.oracle_phase_selection import (
    select_causal_oracle_phase_entries,
)
from event_sae.groot.oracle_phase_keyframes import (
    oracle_phase_entry_invariant_errors,
)
from event_sae.scoring.task_phase_ranking import (
    _confidence_and_observation_diagnostics,
)
from scripts.groot.oracle_phase_pipeline import build_parser


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _phase_entry_event(
    *,
    sample_id: str,
    state_step: int,
    num_steps: int = 20,
    episode_num: int = 0,
    task_episode_idx: int = 7,
    phase: str = "phase",
    waypoint_rank: int = 0,
    success: bool = True,
) -> dict:
    activation_step = min(state_step, num_steps - 1)
    n_action_steps = 5
    return {
        "sample_id": sample_id,
        "task_id": 5,
        "task_description": "Pick the bread.",
        "task_episode_idx": task_episode_idx,
        "episode_num": episode_num,
        "cell_id": "bread",
        "success": success,
        "anchor_source": "oracle_phase_entry",
        "anchor_sources": ["phase-entry"],
        "phase_scheme": "event_state",
        "phase": phase,
        "phase_before": None if state_step == 0 else "previous-phase",
        "phase_after": phase,
        "is_phase_transition": True,
        "oracle_upper_bound": True,
        "waypoint_rank": waypoint_rank,
        "waypoint_step": activation_step,
        "progress_percent": activation_step / max(num_steps - 1, 1),
        "state_env_step_index": state_step,
        "activation_env_step_index": activation_step,
        "activation_record_index": activation_step // n_action_steps,
        "activation_record_phase": phase,
        "action_token_offset": activation_step % n_action_steps,
        "observation_record_index": (
            state_step // n_action_steps
            if state_step % n_action_steps == 0
            and state_step // n_action_steps
            < (num_steps + n_action_steps - 1) // n_action_steps
            else None
        ),
        "observation_record_phase": (
            phase
            if state_step % n_action_steps == 0
            and state_step // n_action_steps
            < (num_steps + n_action_steps - 1) // n_action_steps
            else None
        ),
        "n_action_steps": n_action_steps,
        "num_records": (num_steps + n_action_steps - 1) // n_action_steps,
        "causal_action_env_step_index": (
            None if state_step == 0 else state_step - 1
        ),
        "num_steps": num_steps,
    }


def test_selection_excludes_initial_terminal_and_shifted_windows(
    tmp_path: Path,
) -> None:
    events = []
    features = []
    for index, state_step in enumerate((0, 4, 5, 10, 15, 16, 20)):
        sample_id = f"s{state_step}"
        activation_step = min(state_step, 19)
        record_index = activation_step // 5
        events.append(
            _phase_entry_event(
                sample_id=sample_id,
                state_step=state_step,
                waypoint_rank=index,
            )
        )
        features.append(
            {
                "sample_id": sample_id,
                "waypoint_step": record_index,
            }
        )
    events_path = tmp_path / "events.jsonl"
    features_path = tmp_path / "features.jsonl"
    _write_jsonl(events_path, events)
    _write_jsonl(features_path, features)

    result = select_causal_oracle_phase_entries(
        oracle_events_path=events_path,
        event_features_path=features_path,
        output_dir=tmp_path / "output",
        window_size=5,
    )

    assert result["num_selected_events"] == 2
    selected = [
        json.loads(line)
        for line in (tmp_path / "output" / "oracle_phase_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["state_env_step_index"] for row in selected] == [5, 10]
    assert result["exclusion_reason_counts"] == {
        "initial_state_no_causal_action": 1,
        "left_window_out_of_range": 2,
        "right_window_out_of_range": 3,
        "terminal_state_no_outgoing_action": 1,
    }
    assert result["event_features_mode"] == "provided_event_features"
    assert result["waypoint_clock"] == "policy_inference_record"
    assert result["recommended_event_step_scale"] == 5
    assert result["scoring_inputs"]["cluster_annotations_path"].endswith(
        "phase_groups/phase_groups.jsonl"
    )
    assert result["ranking_inputs"]["accepted_annotations_path"].endswith(
        "source_clusters/accepted_annotations.jsonl"
    )

    source_annotations = [
        json.loads(line)
        for line in (
            tmp_path
            / "output"
            / "source_clusters"
            / "accepted_annotations.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert source_annotations[0]["confidence_tier"] == "simulator-oracle"
    assert source_annotations[0]["member_sample_ids"] == ["s5", "s10"]
    assert source_annotations[0]["phase_group_id"]

    phase_groups = [
        json.loads(line)
        for line in (
            tmp_path / "output" / "phase_groups" / "phase_groups.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert phase_groups[0]["phase_group_id"]
    assert len(phase_groups[0]["source_cluster_ids"]) == 1
    assert phase_groups[0]["oracle_upper_bound"] is True

    phase_assignments = [
        json.loads(line)
        for line in (
            tmp_path
            / "output"
            / "phase_groups"
            / "phase_group_assignments.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["sample_id"] for row in phase_assignments} == {"s5", "s10"}
    assert all(row["source_cluster_id"] for row in phase_assignments)
    assert all(row["phase_group_id"] for row in phase_assignments)
    assert all(
        not oracle_phase_entry_invariant_errors(
            row,
            required_window_size=5,
        )
        for row in phase_assignments
    )

    diagnostics = _confidence_and_observation_diagnostics(
        accepted_annotations=(
            tmp_path
            / "output"
            / "source_clusters"
            / "accepted_annotations.jsonl"
        ),
        phase_groups=(
            tmp_path / "output" / "phase_groups" / "phase_groups.jsonl"
        ),
        phase_assignments=(
            tmp_path
            / "output"
            / "phase_groups"
            / "phase_group_assignments.jsonl"
        ),
    )
    assert diagnostics[("Pick the bread.", "phase")]["claim_limit"] == (
        "simulator_oracle_phase_labels"
    )

    source_assignments = [
        json.loads(line)
        for line in (
            tmp_path
            / "output"
            / "source_clusters"
            / "cluster_assignments.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["phase_group_id"] for row in source_assignments)


def test_selection_generates_env_step_scorer_features_without_media(
    tmp_path: Path,
) -> None:
    events = []
    for index, state_step in enumerate((5, 10)):
        events.append(
            _phase_entry_event(
                sample_id=f"s{state_step}",
                state_step=state_step,
                task_episode_idx=index,
                episode_num=index,
                phase="grasp",
                success=bool(index),
            )
        )
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(events_path, events)

    result = select_causal_oracle_phase_entries(
        oracle_events_path=events_path,
        output_dir=tmp_path / "output",
        window_size=5,
    )

    features = [
        json.loads(line)
        for line in (tmp_path / "output" / "event_features.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["waypoint_step"] for row in features] == [5, 10]
    assert all(
        row["waypoint_clock"] == "environment_action_step"
        for row in features
    )
    assert result["event_features_mode"] == "programmatic_oracle_metadata"
    assert result["recommended_event_step_scale"] == 1
    assert result["exact_phase_entry_contract_validated"] is True


def test_selection_rejects_anchor_source_without_transition_contract(
    tmp_path: Path,
) -> None:
    forged = _phase_entry_event(sample_id="forged", state_step=10)
    forged["is_phase_transition"] = False
    forged["phase_before"] = forged["phase"]
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(events_path, [forged])

    with pytest.raises(
        ValueError,
        match="invalid Oracle phase-entry contract",
    ):
        select_causal_oracle_phase_entries(
            oracle_events_path=events_path,
            output_dir=tmp_path / "output",
            window_size=5,
        )


def test_selection_requires_exact_feature_coverage(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    features_path = tmp_path / "features.jsonl"
    _write_jsonl(events_path, [{"sample_id": "event"}])
    _write_jsonl(features_path, [{"sample_id": "other"}])

    with pytest.raises(ValueError, match="do not exactly match"):
        select_causal_oracle_phase_entries(
            oracle_events_path=events_path,
            event_features_path=features_path,
            output_dir=tmp_path / "output",
        )


def test_cli_only_makes_event_features_optional_for_direct_selection() -> None:
    parser = build_parser()
    selection = parser.parse_args(
        [
            "select-scoring-events",
            "--oracle-events",
            "events.jsonl",
            "--output-dir",
            "selection",
        ]
    )
    assert selection.event_features is None

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "cluster-features",
                "--oracle-events",
                "events.jsonl",
                "--output-dir",
                "clusters",
            ]
        )


def test_cli_sidecar_mode_does_not_require_pickle_trust() -> None:
    args = build_parser().parse_args(
        [
            "extract-keyframes",
            "--trajectory-manifest",
            "trajectory_manifest.json",
            "--rollout-metadata-dir",
            "metadata",
            "--output-dir",
            "oracle",
        ]
    )
    assert args.rollout_metadata_dir == Path("metadata")
    assert args.raw_rollouts_dir is None
    assert args.trust_pkl is False
