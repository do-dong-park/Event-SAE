"""Render paper keyframe media or assemble a composite GR00T media view.

The historical paper invocation remains the default when no operation is
provided. GR00T anchor supplementation and media assembly are explicit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sae import (
    DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_PROFILE,
    load_pipeline_profile,
)
from event_sae.events.extract_media import (
    assemble_composite_anchor_media_view,
    build_gripper_supplement_summary,
    extract_keyframe_media,
)


_DEFAULT_OPERATION = "extract"
_OPERATIONS = {
    _DEFAULT_OPERATION,
    "build-gripper-supplement",
    "assemble-media-view",
}


def _default_output_dir(waypoint_summary_path: Path) -> Path:
    run_name = waypoint_summary_path.parent.parent.name
    # Pick the backend bucket from the source path so OpenPI runs land
    # under logs/openpi/events/ rather than logs/openvla/.
    parts = waypoint_summary_path.resolve().parts
    backend = next(
        (part for part in parts if part in {"openvla", "openpi"}),
        None,
    )
    if backend is None:
        backend = (
            "groot"
            if any(part.startswith("groot") for part in parts)
            else "openvla"
        )
    return (
        Path("logs")
        / backend
        / "events"
        / run_name
        / "samples_5frames_stride2"
    )


def _add_extract_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        _DEFAULT_OPERATION,
        help="Render the paper 5-frame keyframe bundle",
        description="Render 5-frame bundles around each AWE waypoint.",
    )
    parser.add_argument(
        "--waypoint-summary-path",
        required=True,
        help="Path to waypoint_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: derived from summary path)",
    )
    parser.add_argument(
        "--frame-offsets",
        type=int,
        nargs="+",
        default=[-4, -2, 0, 2, 4],
        help=(
            "Trajectory-record offsets around each waypoint. "
            "Default: -4 -2 0 2 4."
        ),
    )
    parser.add_argument(
        "--trajectory-manifest-path",
        default=None,
        help="Optional trajectory_manifest.json (auto-detected beside JSONL)",
    )
    parser.add_argument(
        "--video-root",
        default=None,
        help="Root containing manifest-relative GR00T MP4 paths",
    )
    parser.add_argument(
        "--frame-anchor",
        choices=("first", "center", "last"),
        default="first",
        help="Representative frame within each policy record video interval",
    )
    parser.add_argument(
        "--require-complete-videos",
        action="store_true",
        help="Fail before packaging when any exact episode MP4 is missing",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on the number of extracted samples",
    )
    parser.set_defaults(handler=_run_extract)


def _add_gripper_supplement_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "build-gripper-supplement",
        help="Select gripper-only anchors for supplemental media packaging",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_PROFILE,
    )
    parser.add_argument(
        "--waypoint-summary-path",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--expected-waypoints", type=int, default=None)
    parser.set_defaults(handler=_run_gripper_supplement)


def _add_assemble_media_view_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "assemble-media-view",
        help="Join reusable abs media and gripper supplement for one view",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_PROFILE,
    )
    parser.add_argument(
        "--waypoint-summary-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reusable-samples-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--supplement-samples-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--trajectory-records-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--view",
        choices=("left", "right", "wrist"),
        required=True,
    )
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.set_defaults(handler=_run_assemble_media_view)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    _add_extract_parser(subparsers)
    _add_gripper_supplement_parser(subparsers)
    _add_assemble_media_view_parser(subparsers)
    return parser


def _run_extract(args: argparse.Namespace) -> None:
    waypoint_summary_path = Path(args.waypoint_summary_path).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else _default_output_dir(waypoint_summary_path).resolve()
    )

    report = extract_keyframe_media(
        waypoint_summary_path=waypoint_summary_path,
        output_dir=output_dir,
        frame_offsets=list(args.frame_offsets),
        max_samples=args.max_samples,
        trajectory_manifest_path=(
            Path(args.trajectory_manifest_path)
            if args.trajectory_manifest_path is not None
            else None
        ),
        video_root=(
            Path(args.video_root)
            if args.video_root is not None
            else None
        ),
        frame_anchor=args.frame_anchor,
        require_complete_videos=args.require_complete_videos,
    )
    print(f"Waypoints source: {report['waypoint_summary_path']}")
    print(f"Output dir: {report['output_dir']}")
    print(f"Saved samples: {report['num_samples']}")
    print(f"Skipped samples: {report['num_skipped']}")
    print(f"Samples manifest: {report['samples_path']}")


def _run_gripper_supplement(args: argparse.Namespace) -> None:
    load_pipeline_profile(args.profile)
    result = build_gripper_supplement_summary(
        waypoint_summary_path=args.waypoint_summary_path,
        output_path=args.output_path,
        expected_waypoints=args.expected_waypoints,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _run_assemble_media_view(args: argparse.Namespace) -> None:
    profile = load_pipeline_profile(args.profile)
    view_order = list(profile.require("media", "view_order"))
    if args.view not in view_order:
        raise ValueError(
            f"view {args.view!r} is not enabled by profile: {view_order}"
        )
    trajectory_records_path = (
        args.trajectory_records_path
        or profile.path_value("source", "trajectory_records_path")
    )
    result = assemble_composite_anchor_media_view(
        waypoint_summary_path=args.waypoint_summary_path,
        reusable_samples_path=args.reusable_samples_path,
        supplement_samples_path=args.supplement_samples_path,
        trajectory_records_path=trajectory_records_path,
        output_path=args.output_path,
        view=args.view,
        expected_samples=args.expected_samples,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _with_default_operation(
    argv: Sequence[str] | None,
) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] not in _OPERATIONS:
        values.insert(0, _DEFAULT_OPERATION)
    return values


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(_with_default_operation(argv))
    args.handler(args)


if __name__ == "__main__":
    main()
