from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from event_sae.groot.oracle_phase_keyframes import (
    ORACLE_ANNOTATIONS_NAME,
    ORACLE_ASSIGNMENTS_NAME,
    ORACLE_CLUSTERS_NAME,
    ORACLE_EVENTS_NAME,
    extract_oracle_phase_keyframes,
    oracle_phase_entry_invariant_errors,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    raw_root = tmp_path / "raw_rollouts"
    source_file = Path(
        "PickPlaceCounterToCabinet/cell/task5--ep0--succ1.pkl"
    )
    pkl_path = raw_root / source_file
    pkl_path.parent.mkdir(parents=True)
    payload = {
        "task_id": 5,
        "episode_idx": 0,
        "episode_success": 1,
        "cell_id": "cell",
        "phase_scheme": "event_state",
        "n_action_steps": 2,
        "env_step_n_action_steps": 2,
        "model_action_horizon": 4,
        "feature_phases": ["reach", "grasp", "transport"],
        "env_step_phases": [
            "reach",
            "reach",
            "grasp",
            "transport",
            "transport",
            "reach",
            "grasp",
        ],
        "env_step_event_steps": {
            "grasp:obj": 2,
            "place:obj": 4,
            "release:obj": 6,
        },
        "env_step_grasp_steps": [2, 6],
        "env_step_drop_steps": [5],
        "env_step_wrong_grasp_steps": [3],
        "hidden_states": [None, None, None],
    }
    with pkl_path.open("wb") as handle:
        pickle.dump(payload, handle)

    manifest_path = tmp_path / "trajectory_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "groot_n15_pq3_trajectory_manifest_v1",
                "source_root": str(raw_root),
                "episodes": [
                    {
                        "episode_num": 0,
                        "task_id": 5,
                        "task_episode_idx": 0,
                        "task_description": "Pick the bread.",
                        "prompt_task_description": "Pick the bread.",
                        "cell_id": "cell",
                        "success": True,
                        "source_file": source_file.as_posix(),
                        "num_records": 3,
                        "n_action_steps": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, raw_root


def test_phase_entry_keyframes_preserve_state_and_action_alignment(
    tmp_path: Path,
) -> None:
    manifest_path, raw_root = _write_fixture(tmp_path)
    output_dir = tmp_path / "output"

    result = extract_oracle_phase_keyframes(
        trajectory_manifest_path=manifest_path,
        raw_rollouts_dir=raw_root,
        output_dir=output_dir,
        trust_pkl=True,
        anchor_mode="phase-entry",
        progress_every=0,
    )

    events = _read_jsonl(output_dir / ORACLE_EVENTS_NAME)
    assert [event["state_env_step_index"] for event in events] == [0, 2, 3, 5, 6]
    assert [event["activation_env_step_index"] for event in events] == [0, 2, 3, 5, 5]
    assert [event["causal_action_env_step_index"] for event in events] == [
        None, 1, 2, 4, 5
    ]
    assert [event["activation_record_index"] for event in events] == [0, 1, 1, 2, 2]
    assert [event["action_token_offset"] for event in events] == [0, 0, 1, 1, 1]
    assert [event["observation_record_index"] for event in events] == [
        0,
        1,
        None,
        None,
        None,
    ]
    assert [event["phase"] for event in events] == [
        "reach",
        "grasp",
        "transport",
        "reach",
        "grasp",
    ]
    assert result["num_keyframes"] == 5
    assert result["claim_scope"] == "simulator-oracle diagnostic upper bound"

    assignments = _read_jsonl(output_dir / ORACLE_ASSIGNMENTS_NAME)
    clusters = _read_jsonl(output_dir / ORACLE_CLUSTERS_NAME)
    annotations = _read_jsonl(output_dir / ORACLE_ANNOTATIONS_NAME)
    assert len(assignments) == len(events)
    assert {row["phase"] for row in annotations} == {
        "reach",
        "grasp",
        "transport",
    }
    grasp_annotation = next(row for row in annotations if row["phase"] == "grasp")
    assert grasp_annotation["num_members"] == 2
    assert grasp_annotation["episode_coverage"] == 1.0
    assert grasp_annotation["model"] == "simulator_oracle_labeler"
    assert grasp_annotation["confidence_tier"] == "simulator-oracle"
    assert grasp_annotation["source_cluster_ids"] == [
        grasp_annotation["cluster_id"]
    ]
    assert len(grasp_annotation["member_sample_ids"]) == 2
    assert all(
        assignment["source_cluster_id"] == assignment["cluster_id"]
        for assignment in assignments
    )
    for assignment, event in zip(assignments, events, strict=True):
        for field in (
            "is_phase_transition",
            "phase_before",
            "phase_after",
            "state_env_step_index",
            "activation_env_step_index",
            "causal_action_env_step_index",
            "activation_record_index",
            "activation_record_phase",
            "action_token_offset",
            "observation_record_index",
            "observation_record_phase",
            "n_action_steps",
            "num_records",
        ):
            assert assignment[field] == event[field]
        assert not oracle_phase_entry_invariant_errors(assignment)
    assert all(cluster["oracle_upper_bound"] is True for cluster in clusters)


def test_labeler_event_mode_deduplicates_sources_at_same_step(
    tmp_path: Path,
) -> None:
    manifest_path, raw_root = _write_fixture(tmp_path)
    output_dir = tmp_path / "output"

    extract_oracle_phase_keyframes(
        trajectory_manifest_path=manifest_path,
        raw_rollouts_dir=raw_root,
        output_dir=output_dir,
        trust_pkl=True,
        anchor_mode="labeler-event",
        progress_every=0,
    )

    events = _read_jsonl(output_dir / ORACLE_EVENTS_NAME)
    assert [event["state_env_step_index"] for event in events] == [2, 3, 4, 5, 6]
    step_two = events[0]
    assert step_two["event_labels"] == ["grasp", "grasp:obj"]
    assert step_two["anchor_sources"] == ["labeler-event", "phase-entry"]


def test_extractor_requires_explicit_pickle_trust(tmp_path: Path) -> None:
    manifest_path, raw_root = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="trust_pkl"):
        extract_oracle_phase_keyframes(
            trajectory_manifest_path=manifest_path,
            raw_rollouts_dir=raw_root,
            output_dir=tmp_path / "output",
            trust_pkl=False,
        )


def test_extractor_refuses_to_overwrite_output(tmp_path: Path) -> None:
    manifest_path, raw_root = _write_fixture(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        extract_oracle_phase_keyframes(
            trajectory_manifest_path=manifest_path,
            raw_rollouts_dir=raw_root,
            output_dir=output_dir,
            trust_pkl=True,
        )


def test_extractor_reads_cell_stem_json_sidecar_without_pickle_trust(
    tmp_path: Path,
) -> None:
    manifest_path, _raw_root = _write_fixture(tmp_path)
    metadata_root = tmp_path / "metadata"
    sidecar_path = (
        metadata_root
        / "cell"
        / "task5--ep0--succ1.json"
    )
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_text(
        json.dumps(
            {
                "phase_scheme": "event_state",
                "feature_phases": ["reach", "grasp", "transport"],
                "env_step_phases": [
                    "reach",
                    "reach",
                    "grasp",
                    "transport",
                    "transport",
                    "reach",
                    "grasp",
                ],
            }
        ),
        encoding="utf-8",
    )

    result = extract_oracle_phase_keyframes(
        trajectory_manifest_path=manifest_path,
        rollout_metadata_dir=metadata_root,
        output_dir=tmp_path / "output",
        anchor_mode="phase-entry",
        progress_every=0,
    )

    assert result["source_mode"] == "json_sidecar"
    assert result["source_raw_rollouts_root"] is None
    assert result["source_rollout_metadata_root"] == str(
        metadata_root.resolve()
    )
    assert result["source_payload_content_hashes_recorded"] is True
    assert result["source_payloads"][0]["path_resolution"] == (
        "cell_stem_fallback"
    )
    assert result["source_payloads"][0]["sha256"]


def test_extractor_uses_safe_source_manifest_sidecar_mapping(
    tmp_path: Path,
) -> None:
    manifest_path, _raw_root = _write_fixture(tmp_path)
    metadata_root = tmp_path / "metadata"
    mapped_path = metadata_root / "portable" / "episode.json"
    mapped_path.parent.mkdir(parents=True)
    mapped_path.write_text(
        json.dumps(
            {
                "feature_phases": ["reach", "grasp", "transport"],
                "env_step_phases": [
                    "reach",
                    "reach",
                    "grasp",
                    "transport",
                    "transport",
                    "reach",
                    "grasp",
                ],
            }
        ),
        encoding="utf-8",
    )
    (metadata_root / "source_manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    (
                        "PickPlaceCounterToCabinet/cell/"
                        "task5--ep0--succ1.pkl"
                    ): "portable/episode.json"
                }
            }
        ),
        encoding="utf-8",
    )

    result = extract_oracle_phase_keyframes(
        trajectory_manifest_path=manifest_path,
        rollout_metadata_dir=metadata_root,
        output_dir=tmp_path / "output",
        progress_every=0,
    )

    assert result["source_payloads"][0]["path_resolution"] == (
        "source_manifest"
    )
    assert result["source_metadata_manifest_sha256"]


def test_extractor_rejects_unsafe_sidecar_manifest_path(
    tmp_path: Path,
) -> None:
    manifest_path, _raw_root = _write_fixture(tmp_path)
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    (metadata_root / "source_manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    (
                        "PickPlaceCounterToCabinet/cell/"
                        "task5--ep0--succ1.pkl"
                    ): "../episode.json"
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="safe relative path"):
        extract_oracle_phase_keyframes(
            trajectory_manifest_path=manifest_path,
            rollout_metadata_dir=metadata_root,
            output_dir=tmp_path / "output",
            progress_every=0,
        )
