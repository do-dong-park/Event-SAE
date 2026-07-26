from __future__ import annotations

import pytest

from event_sae.groot.oracle_phase_media import select_phase_frame_indices


def test_phase_frames_start_after_odd_transition_and_stay_in_phase() -> None:
    indices, env_steps, phase_match = select_phase_frame_indices(
        state_env_step_index=3,
        phase_segment_stop=10,
        steps_per_render=2,
        video_num_frames=20,
        frames_per_sample=3,
    )
    assert indices == [1, 2, 3]
    assert env_steps == [3, 5, 7]
    assert phase_match is True


def test_short_phase_pads_last_valid_frame() -> None:
    indices, env_steps, phase_match = select_phase_frame_indices(
        state_env_step_index=20,
        phase_segment_stop=22,
        steps_per_render=2,
        video_num_frames=20,
        frames_per_sample=5,
    )
    assert indices == [10, 10, 10, 10, 10]
    assert env_steps == [21, 21, 21, 21, 21]
    assert phase_match is True


def test_terminal_state_has_an_exact_final_render() -> None:
    indices, env_steps, phase_match = select_phase_frame_indices(
        state_env_step_index=215,
        phase_segment_stop=216,
        steps_per_render=2,
        video_num_frames=108,
        frames_per_sample=2,
    )
    assert indices == [107, 107]
    assert env_steps == [215, 215]
    assert phase_match is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("state_env_step_index", -1),
        ("phase_segment_stop", 0),
        ("steps_per_render", 0),
        ("video_num_frames", 0),
        ("frames_per_sample", 0),
    ),
)
def test_phase_frame_selection_rejects_invalid_inputs(
    field: str,
    value: int,
) -> None:
    kwargs = {
        "state_env_step_index": 0,
        "phase_segment_stop": 1,
        "steps_per_render": 1,
        "video_num_frames": 1,
        "frames_per_sample": 1,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        select_phase_frame_indices(**kwargs)
