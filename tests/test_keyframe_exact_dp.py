import numpy as np
import pytest

from event_sae.keyframes import (
    EpisodeTrajectory,
    extract_waypoints_dp,
    extract_waypoints_exact_pos_only,
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
