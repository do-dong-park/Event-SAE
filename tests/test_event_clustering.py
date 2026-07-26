import json
from pathlib import Path

import numpy as np
import pytest

from event_sae.events.cluster import (
    build_phase_groups,
    build_task_vectors,
    cluster_events,
    run_clustering_sweep,
)
from event_sae.events.io import load_jsonl
from event_sae.scoring.score_matrix import join_cluster_events


def _write_event_features(path: Path) -> None:
    records = []
    sample_index = 0
    for task_index, task_description in enumerate(("task a", "task b")):
        for local_index in range(4):
            vision = [1.0, 0.0] if local_index < 2 else [0.0, 1.0]
            records.append(
                {
                    "sample_id": f"sample_{sample_index:02d}",
                    "task_id": task_index,
                    "task_description": task_description,
                    "task_episode_idx": local_index,
                    "episode_num": sample_index,
                    "waypoint_rank": 0,
                    "waypoint_step": local_index,
                    "anchor_source": "position" if local_index < 2 else "gripper_close",
                    "clip_path": "",
                    "frame_paths": [f"frame_{sample_index}.jpg"],
                    "vision_embedding": vision,
                    "state_vector": [float(local_index), 0.0, 0.0],
                    "progress_percent": local_index / 3.0,
                    "num_steps": 4,
                    "cell_id": f"cell_{task_index}",
                    "success": bool(local_index % 2),
                    "boundary_shift_category": "interior",
                }
            )
            sample_index += 1
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _block_normalization_records(
    *,
    duplicate_state: bool,
) -> list[dict]:
    rows = []
    for vision, state, progress in (
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([0.8, 0.2], [1.0, 3.0], 0.4),
        ([0.0, 1.0], [3.0, 2.0], 1.0),
    ):
        if duplicate_state:
            state = [*state, *state]
        rows.append(
            {
                "vision_embedding": vision,
                "state_vector": state,
                "progress_percent": progress,
            }
        )
    return rows


def _write_phase_group_artifacts(
    root: Path,
    *,
    reviewed: bool = True,
) -> tuple[Path, Path, Path]:
    task = "Open the drawer."
    clusters = []
    assignments = []
    annotations = []
    phases = ("pull", "pull", "open-done")
    for index, phase in enumerate(phases):
        cluster_id = f"drawer_cluster_{index:02d}"
        sample_id = f"sample_{index}"
        clusters.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": index,
                "task_description": task,
                "num_members": 1,
                "total_task_episodes": 3,
                "episode_coverage": 1 / 3,
                "member_sample_ids": [sample_id],
                "member_episode_nums": [index],
                "representative_sample_ids": [sample_id],
            }
        )
        assignments.append(
            {
                "sample_id": sample_id,
                "cluster_id": cluster_id,
                "task_description": task,
                "task_id": 1,
                "task_episode_idx": index,
                "episode_num": index,
                "waypoint_rank": 0,
                "waypoint_step": 2,
                "progress_percent": 0.5,
                "num_steps": 5,
            }
        )
        annotations.append(
            {
                "cluster_id": cluster_id,
                "task_description": task,
                "phrase": f"phrase {index}",
                "phase": phase,
                "allowed_phase_labels": ["pull", "open-done"],
                "phase_scheme": "robocasa_action",
                "prompt_version": "test_prompt_v1",
                "model": "gemini",
                "episode_coverage": 1 / 3,
                "representative_sample_ids": [sample_id],
                "representative_clip_paths": [f"clip_{index}.mp4"],
                "representative_frame_paths": [
                    [f"frame_{index}.jpg"]
                ],
                "representative_progress_percents": [0.5],
                "review_verdict": (
                    "approved" if reviewed else "assumed_approved"
                ),
                "actual_human_review_completed": reviewed,
                "api_error": None,
                "parse_error": None,
            }
        )
    clusters_path = root / "clusters.jsonl"
    assignments_path = root / "assignments.jsonl"
    annotations_path = root / "annotations.jsonl"
    _write_jsonl(clusters_path, clusters)
    _write_jsonl(assignments_path, assignments)
    _write_jsonl(annotations_path, annotations)
    return clusters_path, assignments_path, annotations_path


def test_cluster_events_audits_exact_coverage_and_refuses_overwrite(tmp_path: Path) -> None:
    features_path = tmp_path / "event_features.jsonl"
    _write_event_features(features_path)
    output_dir = tmp_path / "clusters"

    summary = cluster_events(
        event_features_path=features_path,
        output_dir=output_dir,
        vision_weight=1.0,
        state_weight=0.0,
        progress_weight=0.0,
        distance_threshold=0.2,
        expected_samples=8,
    )

    assignments = [
        json.loads(line)
        for line in (output_dir / "cluster_assignments.jsonl").read_text().splitlines()
    ]
    assert summary["passed"] is True
    assert summary["num_events"] == 8
    assert summary["num_clusters"] == 4
    assert summary["num_singleton_clusters"] == 0
    assert len({record["sample_id"] for record in assignments}) == 8
    assert all("success" in record for record in assignments)
    assert {record["anchor_source"] for record in assignments} == {
        "position",
        "gripper_close",
    }

    with pytest.raises(FileExistsError, match="non-empty cluster directory"):
        cluster_events(
            event_features_path=features_path,
            output_dir=output_dir,
            expected_samples=8,
        )


def test_sweep_runs_configs_and_records_adjacent_ari(tmp_path: Path) -> None:
    features_path = tmp_path / "event_features.jsonl"
    _write_event_features(features_path)

    report = run_clustering_sweep(
        event_features_path=features_path,
        output_root=tmp_path / "sweep",
        config_ids=["c0", "c2"],
        thresholds=[0.15, 0.20],
        expected_samples=8,
    )

    assert report["passed"] is True
    assert report["success_used_for_fitting_or_selection"] is False
    assert len(report["runs"]) == 4
    for config_id in ("c0", "c2"):
        runs = [run for run in report["runs"] if run["config_id"] == config_id]
        assert runs[0]["adjacent_threshold_ari"] is None
        assert 0.0 <= runs[1]["adjacent_threshold_ari"] <= 1.0
    assert (tmp_path / "sweep" / "sweep_summary.json").is_file()


def test_build_task_vectors_matches_documented_zscore_formula() -> None:
    records = [
        {
            "vision_embedding": [3.0, 4.0],
            "state_vector": [0.0, 1.0, 2.0],
            "progress_percent": 0.0,
        },
        {
            "vision_embedding": [0.0, 2.0],
            "state_vector": [2.0, 4.0, 8.0],
            "progress_percent": 0.5,
        },
        {
            "vision_embedding": [1.0, 0.0],
            "state_vector": [4.0, 2.0, 5.0],
            "progress_percent": 1.0,
        },
    ]
    actual = build_task_vectors(
        records,
        vision_weight=1.0,
        state_weight=0.5,
        progress_weight=0.4,
    )

    vision = np.asarray([record["vision_embedding"] for record in records], dtype=np.float32)
    vision = vision / np.linalg.norm(vision, axis=1, keepdims=True)
    state = np.asarray([record["state_vector"] for record in records], dtype=np.float32)
    state = (state - state.mean(axis=0, keepdims=True)) / state.std(axis=0, keepdims=True)
    state = state / np.linalg.norm(state, axis=1, keepdims=True)
    progress = np.asarray(
        [[record["progress_percent"]] for record in records],
        dtype=np.float32,
    )
    progress = (progress - progress.mean(axis=0, keepdims=True)) / progress.std(
        axis=0,
        keepdims=True,
    )
    expected = np.concatenate([vision, 0.5 * state, 0.4 * progress], axis=1)
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)

    assert np.allclose(actual, expected)


def test_balanced_state_block_is_invariant_to_duplicate_dimensions() -> None:
    base = build_task_vectors(
        _block_normalization_records(duplicate_state=False),
        block_normalization="balanced",
    )
    duplicated = build_task_vectors(
        _block_normalization_records(duplicate_state=True),
        block_normalization="balanced",
    )

    assert np.allclose(base @ base.T, duplicated @ duplicated.T)


def test_unknown_block_normalization_fails_closed() -> None:
    with pytest.raises(ValueError, match="block_normalization"):
        build_task_vectors(
            _block_normalization_records(duplicate_state=False),
            block_normalization="unknown",
        )


def test_phase_groups_union_many_clusters_and_remain_scoring_compatible(
    tmp_path: Path,
) -> None:
    (
        clusters_path,
        assignments_path,
        annotations_path,
    ) = _write_phase_group_artifacts(tmp_path)
    output_dir = tmp_path / "phase_groups"

    summary = build_phase_groups(
        clusters_path=clusters_path,
        assignments_path=assignments_path,
        finalized_annotations_path=annotations_path,
        output_dir=output_dir,
    )

    groups = load_jsonl(output_dir / "phase_groups.jsonl")
    group_assignments = load_jsonl(
        output_dir / "phase_group_assignments.jsonl"
    )
    pull = next(row for row in groups if row["phase"] == "pull")
    assert pull["num_source_clusters"] == 2
    assert pull["num_members"] == 2
    assert pull["episode_coverage"] == pytest.approx(2 / 3)
    assert summary["num_phase_groups"] == 2
    assert len({row["cluster_id"] for row in group_assignments}) == 2

    event_features = [
        {
            "sample_id": f"sample_{index}",
            "task_description": "Open the drawer.",
            "task_id": 1,
            "task_episode_idx": index,
            "episode_num": index,
            "waypoint_rank": 0,
            "waypoint_step": 2,
            "progress_percent": 0.5,
            "num_steps": 5,
        }
        for index in range(3)
    ]
    joined = join_cluster_events(
        event_features=event_features,
        cluster_assignments=group_assignments,
        cluster_annotations=groups,
    )
    assert joined.counts["valid_clusters"] == 2
    assert joined.counts["joined_events"] == 3
    assert {event["phase"] for event in joined.selected_events} == {
        "pull",
        "open-done",
    }


def test_phase_groups_require_actual_human_review_by_default(
    tmp_path: Path,
) -> None:
    (
        clusters_path,
        assignments_path,
        annotations_path,
    ) = _write_phase_group_artifacts(
        tmp_path,
        reviewed=False,
    )

    with pytest.raises(ValueError, match="human review"):
        build_phase_groups(
            clusters_path=clusters_path,
            assignments_path=assignments_path,
            finalized_annotations_path=annotations_path,
            output_dir=tmp_path / "phase_groups",
        )


def test_phase_groups_preserve_assumed_review_provenance(
    tmp_path: Path,
) -> None:
    (
        clusters_path,
        assignments_path,
        annotations_path,
    ) = _write_phase_group_artifacts(
        tmp_path,
        reviewed=False,
    )

    summary = build_phase_groups(
        clusters_path=clusters_path,
        assignments_path=assignments_path,
        finalized_annotations_path=annotations_path,
        output_dir=tmp_path / "phase_groups",
        require_human_review=False,
    )

    groups = load_jsonl(tmp_path / "phase_groups/phase_groups.jsonl")
    assert groups
    assert all(
        row["review_mode"]
        == "phase_group_of_user_authorized_assumed_review_clusters"
        for row in groups
    )
    assert all(
        row["actual_human_review_completed"] is False for row in groups
    )
    assert all(row["review_verdict"] == "assumed_approved" for row in groups)
    assert summary["actual_human_review_completed"] is False
    assert summary["result_status"] == "provisional_automatic"
