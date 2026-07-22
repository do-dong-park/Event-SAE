"""Map policy-record indices to frames in rollout videos.

OpenVLA/OpenPI artifacts historically use one video frame per trajectory
record. GR00T RoboCasa rollouts instead execute ``n_action_steps`` simulator
steps per policy record and render every ``steps_per_render`` simulator steps.
This module keeps that timing contract explicit and independent of video I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


@dataclass(frozen=True)
class VideoTimeline:
    """Integer mapping between policy records and rendered video frames."""

    num_records: int
    n_action_steps: int = 1
    steps_per_render: int = 1

    def __post_init__(self) -> None:
        if self.num_records < 0:
            raise ValueError("num_records must be non-negative")
        if self.n_action_steps <= 0:
            raise ValueError("n_action_steps must be positive")
        if self.steps_per_render <= 0:
            raise ValueError("steps_per_render must be positive")
        if self.n_action_steps < self.steps_per_render:
            raise ValueError(
                "n_action_steps must be >= steps_per_render so every record has a frame"
            )

    @property
    def expected_num_frames(self) -> int:
        """Expected frames for a complete episode video."""
        return _ceil_div(
            self.num_records * self.n_action_steps,
            self.steps_per_render,
        )

    @property
    def is_identity(self) -> bool:
        return self.n_action_steps == self.steps_per_render

    def frame_bounds(self, record_index: int) -> tuple[int, int]:
        """Inclusive frame interval associated with one policy record."""
        if not 0 <= record_index < self.num_records:
            raise IndexError(
                f"record_index={record_index} outside [0, {self.num_records})"
            )
        start = _ceil_div(
            record_index * self.n_action_steps,
            self.steps_per_render,
        )
        stop = (
            _ceil_div(
                (record_index + 1) * self.n_action_steps,
                self.steps_per_render,
            )
            - 1
        )
        return start, stop

    def record_to_frame(self, record_index: int, anchor: str = "first") -> int:
        """Choose a representative frame from a record's frame interval."""
        start, stop = self.frame_bounds(record_index)
        if anchor == "first":
            return start
        if anchor == "center":
            return (start + stop) // 2
        if anchor == "last":
            return stop
        raise ValueError(f"Unsupported frame anchor: {anchor!r}")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "mapping": "policy_record_to_rendered_frame_v1",
            "num_records": self.num_records,
            "n_action_steps": self.n_action_steps,
            "steps_per_render": self.steps_per_render,
            "expected_num_frames": self.expected_num_frames,
        }

    @classmethod
    def from_episode_manifest(cls, episode: dict) -> "VideoTimeline":
        return cls(
            num_records=int(episode["num_records"]),
            n_action_steps=int(episode.get("n_action_steps", 1)),
            steps_per_render=int(episode.get("steps_per_render", 1)),
        )
