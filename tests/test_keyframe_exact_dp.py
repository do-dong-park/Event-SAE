import json
from pathlib import Path

import numpy as np
import pytest

from event_sae.events.extract_media import _anchor_sources_for_episode
from event_sae.keyframes import (
    EpisodeTrajectory,
    extract_waypoints_dp,
    extract_waypoints_exact_pos_only,
    gripper_closing_indices,
    load_episode_trajectories,
    merge_waypoint_anchors,
    require_gripper_qpos_inputs,
    select_waypoint_anchors,
)
from event_sae.physical_state import (
    build_task_local_gripper_states,
    gripper_aperture_from_qpos,
    gripper_closing_peak_indices,
    normalize_gripper_aperture,
)


def _episode(positions: list[list[float]]) -> EpisodeTrajectory:
    num_steps = len(positions)
    return EpisodeTrajectory(
        episode_num=0,
        task_id=0,
        task_episode_idx=0,
        task_description="test",
        prompt_task_description="test",
        success=True,
        step_indices=list(range(num_steps)),
        positions=np.asarray(positions, dtype=np.float32),
    )


def test_exact_pos_only_dp_uses_one_segment_for_straight_trajectory() -> None:
    episode = _episode([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])

    assert extract_waypoints_exact_pos_only(episode, err_threshold=0.01) == [3]


def test_exact_pos_only_dp_adds_corner_and_satisfies_threshold() -> None:
    episode = _episode(
        [[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 0, 0], [4, 0, 0]]
    )

    waypoints = extract_waypoints_exact_pos_only(episode, err_threshold=0.1)

    assert waypoints == [1, 2, 3, 4]


def test_exact_pos_only_is_explicit_wrapper_option() -> None:
    episode = _episode([[0, 0, 0], [1, 0, 0], [2, 0, 0]])

    assert extract_waypoints_dp(
        episode,
        waypoint_mode="pos_only",
        err_threshold=0.05,
        dp_implementation="exact_pos_only",
    ) == [2]


def test_exact_pos_only_rejects_nonpositive_threshold() -> None:
    episode = _episode([[0, 0, 0], [1, 0, 0]])

    with pytest.raises(ValueError, match="finite and positive"):
        extract_waypoints_exact_pos_only(episode, err_threshold=0.0)


def test_trajectory_loader_selects_rel_or_abs_eef_positions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trajectory_records.jsonl"
    records = [
        {
            "episode_num": 0,
            "task_id": 8,
            "task_episode_idx": 0,
            "task_description": "Open the left drawer.",
            "step_in_episode": step,
            "eef_pos": [float(step), 0.0, 0.0],
            "eef_pos_rel": [float(step), 0.0, 0.0],
            "eef_pos_abs": [10.0 + float(step), 1.0, 0.0],
            "done": step == 2,
        }
        for step in range(3)
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    relative = load_episode_trajectories(
        path,
        eef_position_frame="rel",
    )[0]
    absolute = load_episode_trajectories(
        path,
        eef_position_frame="abs",
    )[0]

    assert relative.position_frame == "rel"
    assert relative.positions.tolist() == [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]
    assert absolute.position_frame == "abs"
    assert absolute.positions.tolist() == [
        [10.0, 1.0, 0.0],
        [11.0, 1.0, 0.0],
        [12.0, 1.0, 0.0],
    ]

    legacy_records = [
        {
            key: value
            for key, value in record.items()
            if key not in {"eef_pos_rel", "eef_pos_abs"}
        }
        for record in records
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in legacy_records),
        encoding="utf-8",
    )

    legacy_relative = load_episode_trajectories(path)[0]
    assert legacy_relative.positions.tolist() == relative.positions.tolist()
    with pytest.raises(ValueError, match="re-export"):
        load_episode_trajectories(path, eef_position_frame="abs")


def test_anchor_sources_exactly_cover_waypoints() -> None:
    episode = {
        "waypoint_indices": [0, 4, 8],
        "waypoint_anchors": [
            {"waypoint_index": 0, "anchor_source": "position"},
            {"waypoint_index": 4, "anchor_source": "both"},
            {"waypoint_index": 8, "anchor_source": "gripper_close"},
        ],
    }

    assert _anchor_sources_for_episode(episode) == {
        0: "position",
        4: "both",
        8: "gripper_close",
    }


def test_anchor_sources_reject_partial_provenance() -> None:
    episode = {
        "waypoint_indices": [0, 4],
        "waypoint_anchors": [
            {"waypoint_index": 0, "anchor_source": "position"},
        ],
    }

    with pytest.raises(ValueError, match="exactly cover"):
        _anchor_sources_for_episode(episode)


def test_legacy_waypoint_summary_without_anchor_sources_is_supported() -> None:
    assert _anchor_sources_for_episode({"waypoint_indices": [0, 4]}) == {}


def test_waypoint_merge_marks_nearest_position_anchor_and_keeps_new_peak() -> None:
    anchors = merge_waypoint_anchors(
        [3, 7],
        [5, 10],
        dedup_distance=2,
    )

    assert [(anchor.index, anchor.source) for anchor in anchors] == [
        (3, "both"),
        (7, "position"),
        (10, "gripper_close"),
    ]


def test_composite_waypoint_mode_unions_exact_abs_position_and_closing_peak() -> None:
    episode = EpisodeTrajectory(
        episode_num=0,
        task_id=0,
        task_episode_idx=0,
        task_description="task",
        prompt_task_description="task",
        success=True,
        step_indices=list(range(6)),
        positions=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        position_frame="abs",
    )
    episode.gripper_state = normalize_gripper_aperture(
        np.asarray([1.0, 1.0, 0.2, 0.2, 0.2, 0.2]),
        normalization_min=0.0,
        normalization_max=1.0,
    )

    selection = select_waypoint_anchors(
        episode,
        waypoint_mode="pos_gripper_close",
        dp_implementation="exact_pos_only",
        err_threshold=0.01,
        gripper_peak_height=0.1,
        gripper_peak_prominence=0.05,
        waypoint_dedup_distance=0,
    )

    assert selection.indices == [2, 5]
    assert selection.position_indices == (5,)
    assert selection.gripper_close_indices == (2,)
    assert [(anchor.index, anchor.source) for anchor in selection.anchors] == [
        (2, "gripper_close"),
        (5, "position"),
    ]
    assert extract_waypoints_dp(
        episode,
        waypoint_mode="pos_gripper_close",
        dp_implementation="exact_pos_only",
        err_threshold=0.01,
        gripper_peak_height=0.1,
        gripper_peak_prominence=0.05,
        waypoint_dedup_distance=0,
    ) == selection.indices


def test_loader_uses_all_source_episodes_for_task_local_gripper_bounds(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trajectory_records.jsonl"
    records = []
    for episode_num, qpos_values in enumerate(((0.0, 0.25), (0.5, 0.25))):
        for step, qpos in enumerate(qpos_values):
            records.append(
                {
                    "episode_num": episode_num,
                    "task_id": 1,
                    "task_episode_idx": episode_num,
                    "task_description": "Pick and place.",
                    "step_in_episode": step,
                    "eef_pos_abs": [float(step), 0.0, 0.0],
                    "gripper_qpos": [qpos, -qpos],
                    "done": step == 1,
                }
            )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    episodes = load_episode_trajectories(path, eef_position_frame="abs")

    assert episodes[0].gripper_state is not None
    assert episodes[0].gripper_state.normalized_aperture.tolist() == pytest.approx(
        [0.0, 0.5]
    )
    assert episodes[1].gripper_state.normalized_aperture.tolist() == pytest.approx(
        [1.0, 0.5]
    )
    assert gripper_closing_indices(
        episodes[1],
        min_height=0.1,
        min_prominence=0.0,
    ) == []


def test_composite_mode_rejects_missing_qpos() -> None:
    episode = EpisodeTrajectory(
        episode_num=3,
        task_id=0,
        task_episode_idx=0,
        task_description="task",
        prompt_task_description="task",
        success=False,
        step_indices=[0, 1],
        positions=np.zeros((2, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Missing episodes"):
        require_gripper_qpos_inputs([episode])


def test_task_local_gripper_state_uses_shared_bounds_and_backward_delta() -> None:
    states, bounds = build_task_local_gripper_states(
        {
            0: np.asarray([[0.0, 0.0], [0.25, -0.25]], dtype=np.float32),
            1: np.asarray([[0.5, -0.5], [0.25, -0.25]], dtype=np.float32),
        },
        {0: "task", 1: "task"},
    )

    assert bounds == {"task": (0.0, 1.0)}
    assert states[0].normalized_aperture.tolist() == pytest.approx([0.0, 0.5])
    assert states[1].normalized_aperture.tolist() == pytest.approx([1.0, 0.5])
    assert states[1].aperture_delta.tolist() == pytest.approx([0.0, -0.5])


def test_constant_aperture_is_finite_zero_state() -> None:
    aperture = gripper_aperture_from_qpos(
        np.asarray([[0.25, -0.25], [0.25, -0.25]], dtype=np.float32)
    )
    state = normalize_gripper_aperture(
        aperture,
        normalization_min=0.5,
        normalization_max=0.5,
    )

    assert state.normalized_aperture.tolist() == [0.0, 0.0]
    assert state.aperture_delta.tolist() == [0.0, 0.0]


def test_closing_peak_detection_is_record_aligned_and_deterministic() -> None:
    normalized = np.asarray([1.0, 0.98, 0.4, 0.39, 0.1, 0.1])

    first = gripper_closing_peak_indices(
        normalized,
        min_height=0.1,
        min_prominence=0.05,
        min_distance=2,
    )
    second = gripper_closing_peak_indices(
        normalized,
        min_height=0.1,
        min_prominence=0.05,
        min_distance=2,
    )

    assert first == second == [2, 4]


def test_gripper_qpos_rejects_non_finite_or_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\[T,D\]"):
        gripper_aperture_from_qpos(np.asarray([0.1, 0.2]))
    with pytest.raises(ValueError, match="non-finite"):
        gripper_aperture_from_qpos(np.asarray([[0.1, np.nan]]))
