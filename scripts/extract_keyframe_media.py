"""CLI: render 5-frame bundle (PNGs + clip MP4) around each AWE waypoint.

Reads `waypoint_summary.json` from `scripts/extract_keyframes.py` and the
corresponding LIBERO rollout MP4s, writes `samples.jsonl` ready for
`scripts/build_event_features.py`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sae.events.extract_media import extract_keyframe_media


def _default_output_dir(waypoint_summary_path: Path) -> Path:
    run_name = waypoint_summary_path.parent.parent.name
    # Pick the backend bucket from the source path so OpenPI runs land
    # under logs/openpi/events/ rather than logs/openvla/.
    parts = waypoint_summary_path.resolve().parts
    backend = next((p for p in parts if p in {"openvla", "openpi"}), None)
    if backend is None:
        backend = "groot" if any(p.startswith("groot") for p in parts) else "openvla"
    return Path("logs") / backend / "events" / run_name / "samples_5frames_stride2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render 5-frame bundles around each AWE waypoint."
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
        help="Trajectory-record offsets around each waypoint. Default: -4 -2 0 2 4.",
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
    args = parser.parse_args()

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
        video_root=Path(args.video_root) if args.video_root is not None else None,
        frame_anchor=args.frame_anchor,
        require_complete_videos=args.require_complete_videos,
    )
    print(f"Waypoints source: {report['waypoint_summary_path']}")
    print(f"Output dir: {report['output_dir']}")
    print(f"Saved samples: {report['num_samples']}")
    print(f"Skipped samples: {report['num_skipped']}")
    print(f"Samples manifest: {report['samples_path']}")


if __name__ == "__main__":
    main()
