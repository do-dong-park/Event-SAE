"""Audit GR00T PQ3 AWE waypoint summaries against exported trajectories."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


AUDIT_FORMAT = "groot_n15_pq3_waypoint_audit_v1"


def _load_records(path: Path) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if "episode_num" not in record or "eef_pos" not in record:
                raise ValueError(f"{path}: line {line_number} missing required fields")
            grouped[int(record["episode_num"])].append(record)

    episodes: dict[int, dict[str, Any]] = {}
    for episode_num, records in grouped.items():
        records.sort(key=lambda item: int(item["step_in_episode"]))
        steps = [int(item["step_in_episode"]) for item in records]
        if steps != list(range(len(records))):
            raise ValueError(f"episode {episode_num}: non-contiguous steps")
        positions = np.asarray(
            [item["eef_pos"] for item in records],
            dtype=np.float64,
        )
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"episode {episode_num}: eef_pos shape={positions.shape}, expected [T,3]"
            )
        if not np.isfinite(positions).all():
            raise ValueError(f"episode {episode_num}: non-finite eef_pos")
        first = records[0]
        episodes[episode_num] = {
            "positions": positions,
            "task_id": int(first["task_id"]),
            "task_episode_idx": int(first["task_episode_idx"]),
            "task_description": str(first.get("task_description", "")),
            "cell_id": str(first.get("cell_id", "")),
            "success": bool(records[-1]["done"]),
        }
    if not episodes:
        raise ValueError(f"No trajectory records in {path}")
    return episodes


def _load_manifest_events(path: Path | None) -> dict[int, list[int]]:
    if path is None:
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    events: dict[int, list[int]] = {}
    for episode in manifest.get("episodes", []):
        values: list[int] = []
        event_steps = episode.get("event_steps", {})
        if isinstance(event_steps, dict):
            values.extend(int(value) for value in event_steps.values())
        elif isinstance(event_steps, list):
            values.extend(int(value) for value in event_steps)
        for key in ("grasp_steps", "drop_steps"):
            raw = episode.get(key, [])
            if isinstance(raw, list):
                values.extend(int(value) for value in raw)
        events[int(episode["episode_num"])] = sorted(set(values))
    return events


def _load_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if "episodes" not in summary or "err_threshold" not in summary:
        raise ValueError(f"Unexpected waypoint summary format: {path}")
    return summary


def _discover_summaries(
    waypoint_root: Path | None,
    explicit_paths: list[Path] | None,
) -> list[Path]:
    paths: list[Path] = []
    if explicit_paths:
        paths.extend(path.resolve() for path in explicit_paths)
    if waypoint_root is not None:
        paths.extend(
            path.resolve()
            for path in waypoint_root.resolve().glob(
                "dp_pos_only_err*/waypoint_summary.json"
            )
        )
    unique = sorted(set(paths))
    if not unique:
        raise FileNotFoundError("No waypoint_summary.json inputs found")
    for path in unique:
        if not path.is_file():
            raise FileNotFoundError(path)
    return unique


def _reconstruct_positions(
    positions: np.ndarray,
    waypoint_indices: list[int],
) -> tuple[np.ndarray, list[int]]:
    num_steps = len(positions)
    anchors = sorted(set([0, num_steps - 1, *waypoint_indices]))
    reconstruction = np.empty_like(positions)
    for start, stop in zip(anchors[:-1], anchors[1:], strict=True):
        span = stop - start
        if span <= 0:
            raise ValueError(f"Invalid reconstruction segment: {start}, {stop}")
        alpha = np.arange(span + 1, dtype=np.float64)[:, None] / span
        reconstruction[start : stop + 1] = (
            (1.0 - alpha) * positions[start] + alpha * positions[stop]
        )
    if num_steps == 1:
        reconstruction[0] = positions[0]
    return reconstruction, anchors


def _awe_geometric_errors(
    positions: np.ndarray,
    waypoint_indices: list[int],
) -> np.ndarray:
    """Replicate AWE pos_only point-to-line-segment error per record."""
    anchors = sorted(set([0, len(positions) - 1, *waypoint_indices]))
    errors: list[np.ndarray] = []
    for start, stop in zip(anchors[:-1], anchors[1:], strict=True):
        points = positions[start:stop]
        line_start = positions[start]
        line_vector = positions[stop] - line_start
        denominator = float(np.dot(line_vector, line_vector))
        if denominator == 0.0:
            segment_errors = np.linalg.norm(points - line_start, axis=1)
        else:
            projection_fraction = (
                (points - line_start) @ line_vector / denominator
            )
            projection_fraction = np.clip(projection_fraction, 0.0, 1.0)
            projections = (
                line_start
                + projection_fraction[:, np.newaxis] * line_vector
            )
            segment_errors = np.linalg.norm(points - projections, axis=1)
        errors.append(segment_errors)
    if not errors:
        return np.zeros((1,), dtype=np.float64)
    return np.concatenate(errors)


def _episode_metrics(
    episode_num: int,
    source: dict[str, Any],
    summary_episode: dict[str, Any],
    reference_events: list[int],
    event_tolerance: int,
    err_threshold: float,
) -> dict[str, Any]:
    positions = source["positions"]
    num_steps = len(positions)
    waypoint_indices = [
        int(index) for index in summary_episode.get("waypoint_indices", [])
    ]
    if waypoint_indices != sorted(waypoint_indices):
        raise ValueError(f"episode {episode_num}: waypoint indices are not ascending")
    if len(waypoint_indices) != len(set(waypoint_indices)):
        raise ValueError(f"episode {episode_num}: duplicate waypoint indices")
    if any(index < 0 or index >= num_steps for index in waypoint_indices):
        raise ValueError(f"episode {episode_num}: waypoint index out of range")
    if int(summary_episode["num_steps"]) != num_steps:
        raise ValueError(f"episode {episode_num}: num_steps mismatch")
    if int(summary_episode["num_waypoints"]) != len(waypoint_indices):
        raise ValueError(f"episode {episode_num}: num_waypoints mismatch")

    reported_positions = np.asarray(
        summary_episode.get("waypoint_positions", []),
        dtype=np.float64,
    )
    if not waypoint_indices and reported_positions.size == 0:
        reported_positions = reported_positions.reshape(0, 3)
    expected_positions = positions[waypoint_indices]
    if reported_positions.shape != expected_positions.shape or not np.allclose(
        reported_positions,
        expected_positions,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError(f"episode {episode_num}: waypoint_positions mismatch")

    reconstructed, interpolation_anchors = _reconstruct_positions(
        positions,
        waypoint_indices,
    )
    per_step_error = np.linalg.norm(positions - reconstructed, axis=1)
    awe_geometric_error = _awe_geometric_errors(
        positions,
        waypoint_indices,
    )
    event_matches = [
        min((abs(event - waypoint) for waypoint in waypoint_indices), default=None)
        for event in reference_events
    ]
    matched_events = sum(
        distance is not None and distance <= event_tolerance
        for distance in event_matches
    )
    num_waypoints = len(waypoint_indices)
    return {
        "episode_num": episode_num,
        "task_id": source["task_id"],
        "task_episode_idx": source["task_episode_idx"],
        "task_description": source["task_description"],
        "cell_id": source["cell_id"],
        "success": source["success"],
        "num_steps": num_steps,
        "num_waypoints": num_waypoints,
        "waypoint_density": num_waypoints / num_steps,
        "compression_factor": (
            num_steps / num_waypoints if num_waypoints > 0 else None
        ),
        "rmse": float(np.sqrt(np.mean(np.square(per_step_error)))),
        "max_error": float(np.max(per_step_error)),
        "awe_geometric_mean_error": float(np.mean(awe_geometric_error)),
        "awe_geometric_max_error": float(np.max(awe_geometric_error)),
        "awe_threshold_satisfied": bool(
            np.max(awe_geometric_error) < err_threshold + 1e-9
        ),
        "starts_at_first_record": bool(
            waypoint_indices and waypoint_indices[0] == 0
        ),
        "ends_at_last_record": bool(
            waypoint_indices and waypoint_indices[-1] == num_steps - 1
        ),
        "interpolation_anchors": interpolation_anchors,
        "implicit_boundary_count": len(
            set(interpolation_anchors) - set(waypoint_indices)
        ),
        "num_reference_events": len(reference_events),
        "num_matched_events": matched_events,
        "event_recall": (
            matched_events / len(reference_events) if reference_events else None
        ),
    }


def _aggregate(run_episodes: list[dict[str, Any]]) -> dict[str, Any]:
    densities = np.asarray(
        [episode["waypoint_density"] for episode in run_episodes],
        dtype=np.float64,
    )
    rmses = np.asarray(
        [episode["rmse"] for episode in run_episodes],
        dtype=np.float64,
    )
    max_errors = np.asarray(
        [episode["max_error"] for episode in run_episodes],
        dtype=np.float64,
    )
    waypoint_counts = np.asarray(
        [episode["num_waypoints"] for episode in run_episodes],
        dtype=np.float64,
    )
    compression_factors = np.asarray(
        [episode["compression_factor"] for episode in run_episodes],
        dtype=np.float64,
    )
    awe_max_errors = np.asarray(
        [episode["awe_geometric_max_error"] for episode in run_episodes],
        dtype=np.float64,
    )
    total_events = sum(episode["num_reference_events"] for episode in run_episodes)
    total_matches = sum(episode["num_matched_events"] for episode in run_episodes)
    return {
        "num_episodes": len(run_episodes),
        "total_waypoints": int(np.sum(waypoint_counts)),
        "mean_waypoints": float(np.mean(waypoint_counts)),
        "median_waypoints": float(np.median(waypoint_counts)),
        "mean_waypoint_density": float(np.mean(densities)),
        "median_waypoint_density": float(np.median(densities)),
        "mean_compression_factor": float(np.mean(compression_factors)),
        "median_compression_factor": float(np.median(compression_factors)),
        "mean_rmse": float(np.mean(rmses)),
        "max_episode_rmse": float(np.max(rmses)),
        "mean_max_error": float(np.mean(max_errors)),
        "global_max_error": float(np.max(max_errors)),
        "mean_awe_geometric_max_error": float(np.mean(awe_max_errors)),
        "global_awe_geometric_max_error": float(np.max(awe_max_errors)),
        "num_awe_threshold_violations": sum(
            not episode["awe_threshold_satisfied"]
            for episode in run_episodes
        ),
        "num_reference_events": total_events,
        "num_matched_events": total_matches,
        "micro_event_recall": (
            total_matches / total_events if total_events > 0 else None
        ),
    }


def audit_waypoints(args: argparse.Namespace) -> dict[str, Any]:
    records_path = args.trajectory_records_path.resolve()
    manifest_path = (
        args.trajectory_manifest.resolve()
        if args.trajectory_manifest is not None
        else None
    )
    source_episodes = _load_records(records_path)
    reference_events = _load_manifest_events(manifest_path)
    summary_paths = _discover_summaries(
        args.waypoint_root,
        args.waypoint_summary,
    )

    runs: list[dict[str, Any]] = []
    counts_by_run: list[tuple[float, dict[int, int]]] = []
    for summary_path in summary_paths:
        summary = _load_summary(summary_path)
        summary_episodes: dict[int, dict[str, Any]] = {}
        for episode in summary["episodes"]:
            episode_num = int(episode["episode_num"])
            if episode_num in summary_episodes:
                raise ValueError(
                    f"{summary_path}: duplicate episode_num={episode_num}"
                )
            summary_episodes[episode_num] = episode
        if set(summary_episodes) != set(source_episodes):
            missing = sorted(set(source_episodes) - set(summary_episodes))
            extra = sorted(set(summary_episodes) - set(source_episodes))
            raise ValueError(
                f"{summary_path}: episode inventory mismatch; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        threshold = float(summary["err_threshold"])
        run_episodes = [
            _episode_metrics(
                episode_num,
                source_episodes[episode_num],
                summary_episodes[episode_num],
                reference_events.get(episode_num, []),
                args.event_tolerance,
                threshold,
            )
            for episode_num in sorted(source_episodes)
        ]
        counts_by_run.append(
            (
                threshold,
                {
                    episode["episode_num"]: episode["num_waypoints"]
                    for episode in run_episodes
                },
            )
        )
        runs.append(
            {
                "waypoint_summary_path": str(summary_path),
                "waypoint_mode": str(summary.get("waypoint_mode", "")),
                "dp_implementation": str(summary.get("dp_implementation", "awe")),
                "err_threshold": threshold,
                "aggregate": _aggregate(run_episodes),
                "episodes": run_episodes,
            }
        )

    runs.sort(key=lambda run: run["err_threshold"])
    counts_by_run.sort(key=lambda item: item[0])
    monotonic_violations: list[dict[str, Any]] = []
    for episode_num in sorted(source_episodes):
        counts = [run_counts[episode_num] for _, run_counts in counts_by_run]
        if any(next_count > count for count, next_count in zip(counts, counts[1:])):
            monotonic_violations.append(
                {
                    "episode_num": episode_num,
                    "thresholds": [
                        threshold for threshold, _ in counts_by_run
                    ],
                    "waypoint_counts": counts,
                }
            )

    total_threshold_violations = sum(
        run["aggregate"]["num_awe_threshold_violations"] for run in runs
    )
    report = {
        "format": AUDIT_FORMAT,
        "trajectory_records_path": str(records_path),
        "trajectory_manifest_path": (
            str(manifest_path) if manifest_path is not None else None
        ),
        "event_tolerance_records": int(args.event_tolerance),
        "num_source_episodes": len(source_episodes),
        "runs": runs,
        "threshold_monotonicity": {
            "definition": (
                "waypoint count must be non-increasing as err_threshold increases"
            ),
            "passed": not monotonic_violations,
            "num_violations": len(monotonic_violations),
            "violations": monotonic_violations,
        },
        "integrity_audit_passed": True,
        "threshold_contract": {
            "definition": "every episode must have awe_geometric_max_error < err_threshold",
            "passed": total_threshold_violations == 0,
            "num_violations": total_threshold_violations,
        },
        "reconstruction_note": (
            "RMSE/max_error use time-indexed linear interpolation. AWE threshold "
            "checks use point-to-line-segment distance and are reported as "
            "awe_geometric_* metrics. Both prepend the implicit first record "
            "anchor when AWE does not report it."
        ),
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[audit] episodes={len(source_episodes)} runs={len(runs)} "
        f"monotonic_violations={len(monotonic_violations)}",
        flush=True,
    )
    print(f"[audit] report={output_path}", flush=True)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-records-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--trajectory-manifest",
        type=Path,
        default=None,
        help="Optional trajectory manifest containing diagnostic event steps",
    )
    parser.add_argument(
        "--waypoint-root",
        type=Path,
        default=None,
        help="Root containing dp_pos_only_err*/waypoint_summary.json",
    )
    parser.add_argument(
        "--waypoint-summary",
        type=Path,
        action="append",
        default=None,
        help="Explicit waypoint summary path; may be repeated",
    )
    parser.add_argument("--event-tolerance", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.event_tolerance < 0:
        raise ValueError("--event-tolerance must be non-negative")
    if args.waypoint_root is None and not args.waypoint_summary:
        raise ValueError("Pass --waypoint-root or at least one --waypoint-summary")
    audit_waypoints(args)


if __name__ == "__main__":
    main()
