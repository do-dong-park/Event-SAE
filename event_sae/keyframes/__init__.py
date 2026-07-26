"""Kinematic waypoint extraction with lazy public exports."""

__all__ = [
    "EpisodeTrajectory",
    "WaypointAnchor",
    "WaypointSelection",
    "extract_waypoints_dp",
    "extract_waypoints_exact_pos_only",
    "filter_episodes",
    "gripper_closing_indices",
    "gripper_toggle_indices",
    "load_episode_trajectories",
    "merge_waypoint_anchors",
    "require_geometric_gripper_inputs",
    "require_gripper_qpos_inputs",
    "select_waypoint_anchors",
]


def __getattr__(name: str):
    if name in __all__:
        from event_sae.keyframes import extract as _extract

        return getattr(_extract, name)
    raise AttributeError(f"module 'event_sae.keyframes' has no attribute {name!r}")
