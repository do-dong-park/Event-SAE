from __future__ import annotations

import json
from pathlib import Path

import pytest

from event_sae.groot.oracle_phase_clustering import (
    ALIGNED_FEATURES_NAME,
    align_and_cluster_oracle_phase_features,
    align_oracle_event_features,
)
from event_sae.scoring.score_matrix import join_cluster_events


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fixture_rows() -> tuple[list[dict], list[dict]]:
    features: list[dict] = []
    events: list[dict] = []
    phases = ("reach", "grasp")
    for phase_index, phase in enumerate(phases):
        for local_index, state_value in enumerate((-4.0, -3.8, 3.8, 4.0)):
            episode_num = phase_index * 10 + local_index
            record_index = phase_index + 1
            action_token_offset = local_index % 2
            env_step = record_index * 5 + action_token_offset
            sample_id = f"{phase}_{local_index}"
            common = {
                "sample_id": sample_id,
                "task_id": 5,
                "task_description": "Pick the bread.",
                "task_episode_idx": episode_num,
                "episode_num": episode_num,
                "waypoint_rank": phase_index,
            }
            features.append(
                {
                    **common,
                    "waypoint_step": record_index,
                    "progress_percent": record_index / 20.0,
                    "num_steps": 20,
                    "vision_embedding": [1.0, 0.0],
                    "state_vector": [state_value],
                    "clip_path": f"{sample_id}.mp4",
                    "frame_paths": [[f"{sample_id}.jpg"]],
                    "selected_frame_paths": [f"{sample_id}.jpg"],
                }
            )
            events.append(
                {
                    **common,
                    "phase": phase,
                    "phase_scheme": "event_state",
                    "oracle_upper_bound": True,
                    "state_env_step_index": env_step,
                    "activation_env_step_index": env_step,
                    "causal_action_env_step_index": env_step - 1,
                    "activation_record_index": record_index,
                    "action_token_offset": action_token_offset,
                    "observation_record_index": (
                        record_index if action_token_offset == 0 else None
                    ),
                    "n_action_steps": 5,
                    "progress_percent": env_step / 99.0,
                    "num_steps": 100,
                }
            )
    return features, events


def test_oracle_phase_hard_partition_and_state_subclusters(tmp_path: Path) -> None:
    features, events = _fixture_rows()
    features_path = tmp_path / "features.jsonl"
    events_path = tmp_path / "events.jsonl"
    output_dir = tmp_path / "output"
    _write_jsonl(features_path, features)
    _write_jsonl(events_path, events)

    result = align_and_cluster_oracle_phase_features(
        event_features_path=features_path,
        oracle_events_path=events_path,
        output_dir=output_dir,
        distance_threshold=0.18,
        expected_samples=8,
    )

    cluster_dir = output_dir / "phase_state_clusters"
    assignments = _read_jsonl(cluster_dir / "cluster_assignments.jsonl")
    clusters = _read_jsonl(cluster_dir / "clusters.jsonl")
    annotations = _read_jsonl(cluster_dir / "cluster_annotations.jsonl")
    assert result["clustering"]["num_phase_partitions"] == 2
    assert result["clustering"]["num_clusters"] == 4
    assert {row["phase"] for row in clusters} == {"reach", "grasp"}
    assert {row["phase_scheme"] for row in clusters} == {"event_state"}
    assert len({row["cluster_id"] for row in clusters}) == 4
    assert all(row["total_task_episodes"] == 8 for row in clusters)
    assert all(row["total_phase_episodes"] == 4 for row in clusters)

    cluster_phases = {
        row["cluster_id"]: row["phase"] for row in annotations
    }
    assert all(
        cluster_phases[row["cluster_id"]] == row["phase"]
        for row in assignments
    )
    assert all(row["model"] == "simulator_oracle_labeler" for row in annotations)
    assert all(row["actual_human_review_completed"] is False for row in annotations)
    assert all(row["phase_scheme"] == "event_state" for row in assignments)
    assert all("activation_record_index" in row for row in assignments)

    aligned = _read_jsonl(output_dir / ALIGNED_FEATURES_NAME)
    joined = join_cluster_events(
        event_features=aligned,
        cluster_assignments=assignments,
        cluster_annotations=annotations,
    )
    assert len(joined.selected_events) == 8
    assert joined.counts["joined_events"] == 8


def test_alignment_restores_env_step_and_preserves_record_clock(
    tmp_path: Path,
) -> None:
    features, events = _fixture_rows()
    features_path = tmp_path / "features.jsonl"
    events_path = tmp_path / "events.jsonl"
    output_path = tmp_path / "aligned.jsonl"
    _write_jsonl(features_path, features)
    _write_jsonl(events_path, events)

    align_oracle_event_features(
        event_features_path=features_path,
        oracle_events_path=events_path,
        output_path=output_path,
    )
    aligned = _read_jsonl(output_path)
    row = next(item for item in aligned if item["sample_id"] == "reach_1")
    assert row["descriptor_record_index"] == 1
    assert row["waypoint_step"] == 6
    assert row["state_vector_record_index"] == 1
    assert row["state_vector_env_step_index"] == 5
    assert row["state_vector_env_step_lag"] == 1
    assert row["oracle_phase"] == "reach"


def test_alignment_fails_closed_on_record_env_clock_mismatch(
    tmp_path: Path,
) -> None:
    features, events = _fixture_rows()
    features[0]["waypoint_step"] = 99
    features_path = tmp_path / "features.jsonl"
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(features_path, features)
    _write_jsonl(events_path, events)

    with pytest.raises(ValueError, match="activation_record_index"):
        align_oracle_event_features(
            event_features_path=features_path,
            oracle_events_path=events_path,
            output_path=tmp_path / "aligned.jsonl",
        )


def test_alignment_requires_exact_sample_id_set(tmp_path: Path) -> None:
    features, events = _fixture_rows()
    features_path = tmp_path / "features.jsonl"
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(features_path, features[:-1])
    _write_jsonl(events_path, events)

    with pytest.raises(ValueError, match="do not exactly match"):
        align_oracle_event_features(
            event_features_path=features_path,
            oracle_events_path=events_path,
            output_path=tmp_path / "aligned.jsonl",
        )


def test_phase_coverage_counts_unique_phase_episodes(tmp_path: Path) -> None:
    features, events = _fixture_rows()
    duplicate_feature = dict(features[0])
    duplicate_event = dict(events[0])
    duplicate_feature.update(
        {
            "sample_id": "reach_reentry",
            "waypoint_rank": 9,
            "waypoint_step": 2,
        }
    )
    duplicate_event.update(
        {
            "sample_id": "reach_reentry",
            "waypoint_rank": 9,
            "state_env_step_index": 10,
            "activation_env_step_index": 10,
            "causal_action_env_step_index": 9,
            "activation_record_index": 2,
            "action_token_offset": 0,
            "observation_record_index": 2,
        }
    )
    features.append(duplicate_feature)
    events.append(duplicate_event)
    features_path = tmp_path / "features.jsonl"
    events_path = tmp_path / "events.jsonl"
    output_dir = tmp_path / "output"
    _write_jsonl(features_path, features)
    _write_jsonl(events_path, events)

    align_and_cluster_oracle_phase_features(
        event_features_path=features_path,
        oracle_events_path=events_path,
        output_dir=output_dir,
    )
    reach_clusters = [
        row
        for row in _read_jsonl(
            output_dir / "phase_state_clusters" / "clusters.jsonl"
        )
        if row["phase"] == "reach"
    ]
    assert all(row["total_phase_episodes"] == 4 for row in reach_clusters)
    assert all(row["total_task_episodes"] == 8 for row in reach_clusters)
    assert all(
        row["phase_episode_coverage"] == row["episode_coverage"]
        for row in reach_clusters
    )
    assert all(
        row["coverage_scope"] == "episodes_with_oracle_phase_keyframe"
        for row in reach_clusters
    )


def test_pipeline_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    features, events = _fixture_rows()
    features_path = tmp_path / "features.jsonl"
    events_path = tmp_path / "events.jsonl"
    output_dir = tmp_path / "output"
    _write_jsonl(features_path, features)
    _write_jsonl(events_path, events)
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        align_and_cluster_oracle_phase_features(
            event_features_path=features_path,
            oracle_events_path=events_path,
            output_dir=output_dir,
        )
