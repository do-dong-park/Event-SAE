"""CLI: extract AWE keyframes from a `trajectory_records.jsonl` file.

Writes one summary JSON (`waypoint_summary.json`) under `--output-dir`,
listing per-episode waypoint indices.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sae.keyframes import (
    filter_episodes,
    gripper_toggle_indices,
    load_episode_trajectories,
    require_geometric_gripper_inputs,
    require_gripper_qpos_inputs,
    select_waypoint_anchors,
)


def _default_output_dir(
    records_path: Path,
    err_threshold: float,
    waypoint_mode: str,
    eef_position_frame: str,
) -> Path:
    threshold_tag = f"{err_threshold:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    # Pick the backend bucket from the source path so OpenPI runs do not
    # land under logs/openvla/. Falls back to "openvla" for legacy layouts.
    parts = records_path.resolve().parts
    backend = next((p for p in parts if p in {"openvla", "openpi"}), None)
    if backend is None:
        backend = "groot" if any(p.startswith("groot") for p in parts) else "openvla"
    frame_tag = "" if eef_position_frame == "rel" else f"_{eef_position_frame}"
    return (
        Path("logs")
        / backend
        / "keyframes"
        / records_path.parent.name
        / f"dp_{waypoint_mode}{frame_tag}_err{threshold_tag}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract AWE keyframes from trajectory_records.jsonl.")
    parser.add_argument("--trajectory-records-path", required=True, help="Path to trajectory_records.jsonl")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: derived from records path)")
    parser.add_argument(
        "--waypoint-mode",
        choices=("pos_only", "pos_gripper_close", "geometric_gripper"),
        default="pos_only",
        help=(
            "'pos_gripper_close' unions position waypoints with qpos closing peaks; "
            "'geometric_gripper' requires eef_quat + gripper_action"
        ),
    )
    parser.add_argument(
        "--eef-position-frame",
        choices=("rel", "abs"),
        default="rel",
        help=(
            "EEF position coordinates used by waypoint selection. rel preserves "
            "the existing base-relative behavior; abs uses reconstructed world "
            "positions and requires a dual-frame PQ3 trajectory export."
        ),
    )

    parser.add_argument("--err-threshold", type=float, default=0.05, help="AWE error threshold")
    parser.add_argument(
        "--dp-implementation",
        choices=("awe", "exact_pos_only"),
        default="awe",
        help=(
            "DP implementation. exact_pos_only enforces AWE's geometric threshold "
            "on every contiguous segment; awe preserves the upstream implementation."
        ),
    )
    parser.add_argument(
        "--gripper-peak-height",
        type=float,
        default=0.08,
        help="Minimum task-normalized closing velocity peak height",
    )
    parser.add_argument(
        "--gripper-peak-prominence",
        type=float,
        default=0.04,
        help="Minimum task-normalized closing peak prominence",
    )
    parser.add_argument(
        "--gripper-min-peak-distance",
        type=int,
        default=3,
        help="Minimum distance between closing peaks in policy records",
    )
    parser.add_argument(
        "--waypoint-dedup-distance",
        type=int,
        default=2,
        help="Merge closing peaks within this many records of a position waypoint",
    )
    parser.add_argument(
        "--success-filter", choices=("all", "success", "failure"), default="all"
    )
    parser.add_argument("--task-description", default=None, help="Exact task_description filter")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--show-awe-logs", action="store_true", help="Don't suppress AWE internal prints")
    args = parser.parse_args()

    records_path = Path(args.trajectory_records_path).resolve()
    if not records_path.is_file():
        raise FileNotFoundError(f"trajectory_records.jsonl not found: {records_path}")

    episodes = load_episode_trajectories(
        records_path,
        eef_position_frame=args.eef_position_frame,
    )
    selected = filter_episodes(
        episodes,
        success_filter=args.success_filter,
        task_description=args.task_description,
        max_episodes=args.max_episodes,
    )
    if not selected:
        raise ValueError("No episodes matched the requested filters.")
    if args.waypoint_mode == "geometric_gripper":
        require_geometric_gripper_inputs(selected)
    elif args.waypoint_mode == "pos_gripper_close":
        require_gripper_qpos_inputs(selected)
    if args.dp_implementation == "exact_pos_only" and args.waypoint_mode == "geometric_gripper":
        raise ValueError(
            "--dp-implementation exact_pos_only requires a position-only base mode"
        )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else _default_output_dir(
            records_path,
            args.err_threshold,
            args.waypoint_mode,
            args.eef_position_frame,
        ).resolve()
    )
    summary_path = output_dir / "waypoint_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite waypoint summary: {summary_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "format": "event_sae_waypoint_summary_v3",
        "source_trajectory_records_path": str(records_path),
        "output_dir": str(output_dir),
        "waypoint_mode": args.waypoint_mode,
        "eef_position_frame": args.eef_position_frame,
        "dp_implementation": args.dp_implementation,
        "err_threshold": float(args.err_threshold),
        "gripper_peak_height": float(args.gripper_peak_height),
        "gripper_peak_prominence": float(args.gripper_peak_prominence),
        "gripper_min_peak_distance": int(args.gripper_min_peak_distance),
        "waypoint_dedup_distance": int(args.waypoint_dedup_distance),
        "gripper_aperture_definition": "sum(abs(gripper_qpos))",
        "gripper_normalization_scope": "task_description_minmax_all_source_episodes",
        "success_filter": args.success_filter,
        "task_description_filter": args.task_description,
        "num_available_episodes": len(episodes),
        "num_selected_episodes": len(selected),
        "episodes": [],
    }

    extraction_times: list[float] = []
    wall_start = time.perf_counter()
    for idx, episode in enumerate(selected):
        if len(episode.positions) < 2:
            continue
        ep_start = time.perf_counter()
        selection = select_waypoint_anchors(
            episode,
            waypoint_mode=args.waypoint_mode,
            err_threshold=args.err_threshold,
            show_awe_logs=args.show_awe_logs,
            dp_implementation=args.dp_implementation,
            gripper_peak_height=args.gripper_peak_height,
            gripper_peak_prominence=args.gripper_peak_prominence,
            gripper_min_peak_distance=args.gripper_min_peak_distance,
            waypoint_dedup_distance=args.waypoint_dedup_distance,
        )
        waypoints = selection.indices
        anchors = selection.anchors
        ep_seconds = time.perf_counter() - ep_start
        extraction_times.append(ep_seconds)
        summary["episodes"].append(
            {
                "episode_num": episode.episode_num,
                "task_id": episode.task_id,
                "task_episode_idx": episode.task_episode_idx,
                "task_description": episode.task_description,
                "prompt_task_description": episode.prompt_task_description,
                "success": episode.success,
                "num_steps": int(len(episode.positions)),
                "waypoint_indices": waypoints,
                "position_waypoint_indices": list(selection.position_indices),
                "gripper_close_indices": list(selection.gripper_close_indices),
                "waypoint_anchors": [
                    {"waypoint_index": anchor.index, "anchor_source": anchor.source}
                    for anchor in anchors
                ],
                "waypoint_gripper_state": (
                    []
                    if episode.gripper_state is None
                    else [
                        {
                            "waypoint_index": anchor.index,
                            "aperture": float(episode.gripper_state.aperture[anchor.index]),
                            "normalized_aperture": float(
                                episode.gripper_state.normalized_aperture[anchor.index]
                            ),
                            "aperture_delta": float(
                                episode.gripper_state.aperture_delta[anchor.index]
                            ),
                        }
                        for anchor in anchors
                    ]
                ),

                "num_waypoints": len(waypoints),
                "waypoint_positions": episode.positions[waypoints].tolist(),
                "waypoint_mode": args.waypoint_mode,
                "eef_position_frame": episode.position_frame,
                "dp_implementation": args.dp_implementation,
                "has_eef_quat": episode.quaternions is not None,
                "has_gripper_action": episode.gripper_actions is not None,
                "has_gripper_qpos": episode.gripper_qpos is not None,
                "gripper_normalization": (
                    None
                    if episode.gripper_state is None
                    else {
                        "scope": "task_description_minmax_all_source_episodes",
                        "min": episode.gripper_state.normalization_min,
                        "max": episode.gripper_state.normalization_max,
                    }
                ),
                "gripper_toggle_indices": gripper_toggle_indices(episode),
                "extraction_seconds": ep_seconds,
            }
        )
        mean_seconds = float(np.mean(extraction_times))
        eta = mean_seconds * (len(selected) - (idx + 1))
        print(
            f"[{idx + 1}/{len(selected)}] episode={episode.episode_num} "
            f"task={episode.task_id} steps={len(episode.positions)} "
            f"waypoints={len(waypoints)} time={ep_seconds:.2f}s eta={eta:.1f}s"
        )

    total = time.perf_counter() - wall_start
    summary["total_extraction_seconds"] = total
    summary["mean_episode_extraction_seconds"] = (
        float(np.mean(extraction_times)) if extraction_times else 0.0
    )

    with summary_path.open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {summary_path}")
    print(f"Total time: {total:.2f}s; mean per episode: {summary['mean_episode_extraction_seconds']:.2f}s")


if __name__ == "__main__":
    main()
