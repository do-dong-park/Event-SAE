"""AWE keyframe extraction from LIBERO `trajectory_records.jsonl`.

Backbone-agnostic — operates on per-step `eef_pos` / `eef_quat` /
`gripper_action` records produced by any rollout (openVLA, openpi, etc.).

The dynamic-programming core call is `waypoint_extraction.dp_waypoint_selection`
(see `external/awe/`).
"""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from waypoint_extraction import dp_waypoint_selection

from event_sae.physical_state import (
    GripperStateSeries,
    build_task_local_gripper_states,
    gripper_closing_peak_indices,
)


@dataclass
class EpisodeTrajectory:
    episode_num: int
    task_id: int
    task_episode_idx: int
    task_description: str
    prompt_task_description: str
    success: bool
    step_indices: list[int]
    positions: np.ndarray
    quaternions: np.ndarray | None = None
    gripper_actions: np.ndarray | None = None
    gripper_qpos: np.ndarray | None = None
    gripper_state: GripperStateSeries | None = None
    position_frame: str = "rel"


@dataclass(frozen=True)
class WaypointAnchor:
    index: int
    source: str


@dataclass(frozen=True)
class WaypointSelection:
    """Waypoint indices plus provenance used by artifact-producing callers."""

    anchors: tuple[WaypointAnchor, ...]
    position_indices: tuple[int, ...] = ()
    gripper_close_indices: tuple[int, ...] = ()

    @property
    def indices(self) -> list[int]:
        return [anchor.index for anchor in self.anchors]


def _load_optional_vector_field(records: list[dict], key: str) -> np.ndarray | None:
    if not all(key in record for record in records):
        return None
    return np.asarray([record[key] for record in records], dtype=np.float32)


def _load_optional_scalar_field(records: list[dict], key: str) -> np.ndarray | None:
    if not all(key in record for record in records):
        return None
    return np.asarray([float(record[key]) for record in records], dtype=np.float32)


def _eef_position_key(records: list[dict], eef_position_frame: str) -> str:
    if eef_position_frame == "rel":
        candidates = ("eef_pos_rel", "eef_pos")
    elif eef_position_frame == "abs":
        candidates = ("eef_pos_abs",)
    else:
        raise ValueError(
            f"Unsupported eef_position_frame={eef_position_frame!r}; "
            "expected 'rel' or 'abs'"
        )
    for key in candidates:
        if all(key in record for record in records):
            return key
    if eef_position_frame == "abs":
        raise ValueError(
            "eef_position_frame='abs' requires eef_pos_abs in every trajectory "
            "record; re-export the PQ3 PKLs with the dual-frame exporter"
        )
    raise ValueError(
        "eef_position_frame='rel' requires eef_pos_rel or the legacy eef_pos "
        "field in every trajectory record"
    )


def load_episode_trajectories(
    path: Path,
    *,
    eef_position_frame: str = "rel",
) -> list[EpisodeTrajectory]:
    """Load and group `trajectory_records.jsonl` into per-episode `EpisodeTrajectory`."""
    grouped: dict[tuple[int, int, int], list[dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            key = (
                int(record["episode_num"]),
                int(record["task_id"]),
                int(record["task_episode_idx"]),
            )
            grouped.setdefault(key, []).append(record)

    episodes: list[EpisodeTrajectory] = []
    for (episode_num, task_id, task_episode_idx), records in sorted(grouped.items()):
        records.sort(key=lambda item: int(item["step_in_episode"]))
        first = records[0]
        position_key = _eef_position_key(records, eef_position_frame)
        positions = np.asarray(
            [record[position_key] for record in records], dtype=np.float32
        )
        quaternions = _load_optional_vector_field(records, "eef_quat")
        gripper_actions = _load_optional_scalar_field(records, "gripper_action")
        gripper_qpos = _load_optional_vector_field(records, "gripper_qpos")
        step_indices = [int(record["step_in_episode"]) for record in records]
        episodes.append(
            EpisodeTrajectory(
                episode_num=episode_num,
                task_id=task_id,
                task_episode_idx=task_episode_idx,
                task_description=str(first.get("task_description", "")),
                prompt_task_description=str(
                    first.get(
                        "prompt_task_description",
                        first.get("task_description", ""),
                    )
                ),
                success=bool(records[-1]["done"]),
                step_indices=step_indices,
                positions=positions,
                quaternions=quaternions,
                gripper_actions=gripper_actions,
                gripper_qpos=gripper_qpos,
                position_frame=eef_position_frame,
            )
        )
    episode_keys = {
        (
            episode.episode_num,
            episode.task_id,
            episode.task_episode_idx,
        ): episode
        for episode in episodes
    }
    qpos_by_key = {
        key: episode.gripper_qpos
        for key, episode in episode_keys.items()
        if episode.gripper_qpos is not None
    }
    if qpos_by_key:
        task_by_key = {
            key: episode.task_description
            for key, episode in episode_keys.items()
            if episode.gripper_qpos is not None
        }
        gripper_states, _ = build_task_local_gripper_states(
            qpos_by_key,
            task_by_key,
        )
        for key, state in gripper_states.items():
            episode_keys[key].gripper_state = state

    return episodes


def filter_episodes(
    episodes: list[EpisodeTrajectory],
    *,
    success_filter: str = "all",
    task_description: str | None = None,
    max_episodes: int | None = None,
) -> list[EpisodeTrajectory]:
    """Filter episodes by final-done flag, task description, and count."""
    filtered = episodes
    if success_filter == "success":
        filtered = [episode for episode in filtered if episode.success]
    elif success_filter == "failure":
        filtered = [episode for episode in filtered if not episode.success]
    if task_description is not None:
        filtered = [episode for episode in filtered if episode.task_description == task_description]
    if max_episodes is not None:
        filtered = filtered[:max_episodes]
    return filtered


def require_geometric_gripper_inputs(episodes: list[EpisodeTrajectory]) -> None:
    missing_quat = [e.episode_num for e in episodes if e.quaternions is None]
    missing_gripper = [e.episode_num for e in episodes if e.gripper_actions is None]
    if missing_quat or missing_gripper:
        raise ValueError(
            "waypoint_mode='geometric_gripper' requires every selected episode to have "
            f"eef_quat and gripper_action. Missing eef_quat episodes: {missing_quat[:10]}; "
            f"missing gripper_action episodes: {missing_gripper[:10]}"
        )


def require_gripper_qpos_inputs(episodes: list[EpisodeTrajectory]) -> None:
    missing_qpos = [e.episode_num for e in episodes if e.gripper_state is None]
    if missing_qpos:
        raise ValueError(
            "waypoint_mode='pos_gripper_close' requires gripper_qpos in every "
            f"selected episode. Missing episodes: {missing_qpos[:10]}"
        )
def gripper_toggle_indices(episode: EpisodeTrajectory) -> list[int]:
    """Step indices where the gripper command flipped sign."""
    if episode.gripper_actions is None:
        return []
    return [
        int(episode.step_indices[idx])
        for idx in range(len(episode.gripper_actions) - 1)
        if episode.gripper_actions[idx] != episode.gripper_actions[idx + 1]
    ]


def gripper_closing_indices(
    episode: EpisodeTrajectory,
    *,
    min_height: float = 0.08,
    min_prominence: float = 0.04,
    min_distance: int = 3,
) -> list[int]:
    """Return local trajectory indices for task-normalized closing peaks."""

    if episode.gripper_state is None:
        raise ValueError(
            f"episode {episode.episode_num} has no task-normalized gripper_qpos state"
        )
    return gripper_closing_peak_indices(
        episode.gripper_state.normalized_aperture,
        min_height=min_height,
        min_prominence=min_prominence,
        min_distance=min_distance,
    )


def merge_waypoint_anchors(
    position_waypoints: list[int],
    gripper_close_waypoints: list[int],
    *,
    dedup_distance: int = 2,
) -> list[WaypointAnchor]:
    """Merge position and closing anchors with deterministic nearest dedup."""

    if dedup_distance < 0:
        raise ValueError("dedup_distance must be non-negative")
    position = sorted(set(int(index) for index in position_waypoints))
    closing = sorted(set(int(index) for index in gripper_close_waypoints))
    if any(index < 0 for index in [*position, *closing]):
        raise ValueError("waypoint indices must be non-negative")

    sources: dict[int, set[str]] = {index: {"position"} for index in position}
    for close_index in closing:
        nearby_position = [
            index for index in position if abs(index - close_index) <= dedup_distance
        ]
        if nearby_position:
            matched = min(
                nearby_position,
                key=lambda index: (abs(index - close_index), index),
            )
            sources[matched].add("gripper_close")
        else:
            sources.setdefault(close_index, set()).add("gripper_close")

    source_name = {
        frozenset({"position"}): "position",
        frozenset({"gripper_close"}): "gripper_close",
        frozenset({"position", "gripper_close"}): "both",
    }
    return [
        WaypointAnchor(index=index, source=source_name[frozenset(sources[index])])
        for index in sorted(sources)
    ]


def _pos_segment_max_error(
    positions: np.ndarray,
    start: int,
    stop: int,
) -> float:
    """Maximum AWE point-to-line-segment error on the half-open interval."""
    points = positions[start:stop]
    line_start = positions[start]
    line_vector = positions[stop] - line_start
    denominator = float(np.dot(line_vector, line_vector))
    if denominator == 0.0:
        return float(np.max(np.linalg.norm(points - line_start, axis=1)))
    fractions = (points - line_start) @ line_vector / denominator
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = line_start + fractions[:, np.newaxis] * line_vector
    return float(np.max(np.linalg.norm(points - projections, axis=1)))


def extract_waypoints_exact_pos_only(
    episode: EpisodeTrajectory,
    *,
    err_threshold: float = 0.05,
) -> list[int]:
    """Select the fewest position waypoints under AWE's geometric error.

    This is a shortest-path dynamic program over trajectory indices. An edge
    from start to stop is feasible only when every point in that contiguous
    interval lies within err_threshold of the endpoint line segment.
    Index 0 is an implicit anchor, matching AWE's reconstruction convention;
    the returned list contains subsequent anchors including the final record.
    """
    if not np.isfinite(err_threshold) or err_threshold <= 0:
        raise ValueError("err_threshold must be finite and positive")
    positions = np.asarray(episode.positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape [T,3], got {positions.shape}")
    if len(positions) < 2:
        return []

    num_steps = len(positions)
    unreachable = np.iinfo(np.int32).max
    waypoint_count = np.full(num_steps, unreachable, dtype=np.int64)
    predecessor = np.full(num_steps, -1, dtype=np.int64)
    waypoint_count[0] = 0

    for stop in range(1, num_steps):
        for start in range(stop):
            if waypoint_count[start] == unreachable:
                continue
            if _pos_segment_max_error(positions, start, stop) >= err_threshold:
                continue
            candidate_count = waypoint_count[start] + 1
            if candidate_count < waypoint_count[stop]:
                waypoint_count[stop] = candidate_count
                predecessor[stop] = start

    if predecessor[-1] < 0:
        raise RuntimeError("No feasible waypoint path; adjacent records should be feasible")
    waypoints: list[int] = []
    current = num_steps - 1
    while current > 0:
        waypoints.append(int(current))
        current = int(predecessor[current])
    waypoints.reverse()
    return waypoints


def _extract_base_waypoints_dp(
    episode: EpisodeTrajectory,
    *,
    waypoint_mode: str = "pos_only",
    err_threshold: float = 0.05,
    show_awe_logs: bool = False,
    dp_implementation: str = "awe",
) -> list[int]:
    """Select waypoint indices with upstream AWE or the corrected exact DP.

    Position-only mode uses end-effector position. Geometric-gripper mode also
    consumes quaternion and gripper input. The awe implementation option
    preserves the upstream call for compatibility.
    """
    if dp_implementation == "exact_pos_only":
        if waypoint_mode != "pos_only":
            raise ValueError(
                "exact_pos_only implementation requires waypoint_mode='pos_only'"
            )
        return extract_waypoints_exact_pos_only(
            episode,
            err_threshold=err_threshold,
        )
    if dp_implementation != "awe":
        raise ValueError(f"Unsupported DP implementation: {dp_implementation}")

    if waypoint_mode == "pos_only":
        call_kwargs = {
            "env": None,
            "actions": episode.positions,
            "gt_states": episode.positions,
            "err_threshold": err_threshold,
            "pos_only": True,
        }
    elif waypoint_mode == "geometric_gripper":
        if episode.quaternions is None or episode.gripper_actions is None:
            raise ValueError(
                "geometric_gripper mode requires both eef_quat and gripper_action per step."
            )
        call_kwargs = {
            "env": None,
            "actions": np.concatenate(
                [episode.positions, episode.gripper_actions[:, np.newaxis]],
                axis=1,
            ),
            "gt_states": [
                {
                    "robot0_eef_pos": episode.positions[idx],
                    "robot0_eef_quat": episode.quaternions[idx],
                }
                for idx in range(len(episode.positions))
            ],
            "err_threshold": err_threshold,
            "pos_only": False,
        }
    else:
        raise ValueError(f"Unsupported waypoint mode: {waypoint_mode}")

    if show_awe_logs:
        waypoints = dp_waypoint_selection(**call_kwargs)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            waypoints = dp_waypoint_selection(**call_kwargs)
    return [int(idx) for idx in waypoints]


def select_waypoint_anchors(
    episode: EpisodeTrajectory,
    *,
    waypoint_mode: str = "pos_only",
    err_threshold: float = 0.05,
    show_awe_logs: bool = False,
    dp_implementation: str = "awe",
    gripper_peak_height: float = 0.08,
    gripper_peak_prominence: float = 0.04,
    gripper_min_peak_distance: int = 3,
    waypoint_dedup_distance: int = 2,
) -> WaypointSelection:
    """Select waypoints once and retain their source provenance."""
    if waypoint_mode != "pos_gripper_close":
        waypoints = _extract_base_waypoints_dp(
            episode,
            waypoint_mode=waypoint_mode,
            err_threshold=err_threshold,
            show_awe_logs=show_awe_logs,
            dp_implementation=dp_implementation,
        )
        source = (
            "position" if waypoint_mode == "pos_only" else "geometric_gripper"
        )
        return WaypointSelection(
            anchors=tuple(
                WaypointAnchor(index=index, source=source)
                for index in waypoints
            ),
            position_indices=(
                tuple(waypoints) if waypoint_mode == "pos_only" else ()
            ),
        )

    position_waypoints = _extract_base_waypoints_dp(
        episode,
        waypoint_mode="pos_only",
        err_threshold=err_threshold,
        show_awe_logs=show_awe_logs,
        dp_implementation=dp_implementation,
    )
    closing_waypoints = gripper_closing_indices(
        episode,
        min_height=gripper_peak_height,
        min_prominence=gripper_peak_prominence,
        min_distance=gripper_min_peak_distance,
    )
    return WaypointSelection(
        anchors=tuple(
            merge_waypoint_anchors(
                position_waypoints,
                closing_waypoints,
                dedup_distance=waypoint_dedup_distance,
            )
        ),
        position_indices=tuple(position_waypoints),
        gripper_close_indices=tuple(closing_waypoints),
    )


def extract_waypoints_dp(
    episode: EpisodeTrajectory,
    *,
    waypoint_mode: str = "pos_only",
    err_threshold: float = 0.05,
    show_awe_logs: bool = False,
    dp_implementation: str = "awe",
    gripper_peak_height: float = 0.08,
    gripper_peak_prominence: float = 0.04,
    gripper_min_peak_distance: int = 3,
    waypoint_dedup_distance: int = 2,
) -> list[int]:
    """Backward-compatible index-only wrapper around waypoint selection."""

    return select_waypoint_anchors(
        episode,
        waypoint_mode=waypoint_mode,
        err_threshold=err_threshold,
        show_awe_logs=show_awe_logs,
        dp_implementation=dp_implementation,
        gripper_peak_height=gripper_peak_height,
        gripper_peak_prominence=gripper_peak_prominence,
        gripper_min_peak_distance=gripper_min_peak_distance,
        waypoint_dedup_distance=waypoint_dedup_distance,
    ).indices
