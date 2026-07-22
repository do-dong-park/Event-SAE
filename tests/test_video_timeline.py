import importlib.util
import sys
from pathlib import Path

import pytest


def _load_timeline_class():
    path = (
        Path(__file__).resolve().parents[1]
        / "event_sae"
        / "events"
        / "video_timeline.py"
    )
    spec = importlib.util.spec_from_file_location("standalone_video_timeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.VideoTimeline


VideoTimeline = _load_timeline_class()


@pytest.mark.parametrize(
    ("num_records", "expected_frames"),
    [(144, 360), (35, 88)],
)
def test_groot_expected_frame_count(num_records: int, expected_frames: int) -> None:
    timeline = VideoTimeline(
        num_records=num_records,
        n_action_steps=5,
        steps_per_render=2,
    )
    assert timeline.expected_num_frames == expected_frames


def test_groot_record_frame_bounds_and_anchors() -> None:
    timeline = VideoTimeline(
        num_records=144,
        n_action_steps=5,
        steps_per_render=2,
    )

    assert timeline.frame_bounds(0) == (0, 2)
    assert timeline.record_to_frame(0, anchor="first") == 0
    assert timeline.record_to_frame(0, anchor="center") == 1
    assert timeline.record_to_frame(0, anchor="last") == 2
    assert timeline.frame_bounds(143) == (358, 359)


def test_legacy_identity_mapping_is_preserved() -> None:
    timeline = VideoTimeline(num_records=7)
    assert timeline.expected_num_frames == 7
    assert timeline.is_identity is True
    assert [timeline.record_to_frame(index) for index in range(7)] == list(range(7))


def test_invalid_timeline_and_index_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        VideoTimeline(num_records=1, n_action_steps=0)
    timeline = VideoTimeline(num_records=1)
    with pytest.raises(IndexError, match="outside"):
        timeline.record_to_frame(1)
    with pytest.raises(ValueError, match="Unsupported"):
        timeline.record_to_frame(0, anchor="middle")
