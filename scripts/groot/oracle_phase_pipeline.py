#!/usr/bin/env python3
"""Prepare simulator-oracle phase artifacts for feature analysis.

The subcommands keep the four artifact contracts separate while providing one
stable command-line entry point:

``extract-keyframes``
    Extract phase-entry or labeler-event keyframes from trusted rollouts.
``materialize-media``
    Decode phase-contained image frames for extracted keyframes.
``cluster-features``
    Cluster SigLIP and state descriptors inside oracle phase partitions.
``select-scoring-events``
    Build direct Oracle scoring and phase-group inputs with a complete causal
    SAE scoring window.  Media and SigLIP features are optional.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from event_sae.groot.oracle_phase_clustering import (  # noqa: E402
    align_and_cluster_oracle_phase_features,
)
from event_sae.groot.oracle_phase_keyframes import (  # noqa: E402
    ANCHOR_MODES,
    extract_oracle_phase_keyframes,
)
from event_sae.groot.oracle_phase_media import (  # noqa: E402
    materialize_oracle_phase_media,
)
from event_sae.groot.oracle_phase_selection import (  # noqa: E402
    select_causal_oracle_phase_entries,
)


def _extract_keyframes(args: argparse.Namespace) -> dict:
    return extract_oracle_phase_keyframes(
        trajectory_manifest_path=args.trajectory_manifest,
        raw_rollouts_dir=args.raw_rollouts_dir,
        rollout_metadata_dir=args.rollout_metadata_dir,
        output_dir=args.output_dir,
        trust_pkl=args.trust_pkl,
        anchor_mode=args.anchor_mode,
        progress_every=args.progress_every,
    )


def _materialize_media(args: argparse.Namespace) -> dict:
    return materialize_oracle_phase_media(
        oracle_events_path=args.oracle_events,
        trajectory_manifest_path=args.trajectory_manifest,
        output_dir=args.output_dir,
        video_root=args.video_root,
        trajectory_records_path=args.trajectory_records,
        frames_per_sample=args.frames_per_sample,
        view_name=args.view_name,
        scene_height_pixels=args.scene_height_pixels,
        view_width_pixels=args.view_width_pixels,
        jpeg_quality=args.jpeg_quality,
        expected_samples=args.expected_samples,
        progress_every=args.progress_every,
    )


def _cluster_features(args: argparse.Namespace) -> dict:
    return align_and_cluster_oracle_phase_features(
        event_features_path=args.event_features,
        oracle_events_path=args.oracle_events,
        output_dir=args.output_dir,
        vision_weight=args.vision_weight,
        state_weight=args.state_weight,
        progress_weight=args.progress_weight,
        block_normalization=args.block_normalization,
        distance_threshold=args.distance_threshold,
        min_coverage=args.min_coverage,
        num_exemplars=args.num_exemplars,
        expected_samples=args.expected_samples,
    )


def _select_scoring_events(args: argparse.Namespace) -> dict:
    return select_causal_oracle_phase_entries(
        oracle_events_path=args.oracle_events,
        event_features_path=args.event_features,
        output_dir=args.output_dir,
        window_size=args.window_size,
    )


def _add_extract_keyframes_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "extract-keyframes",
        help="Extract simulator-oracle phase keyframes",
    )
    parser.add_argument(
        "--trajectory-manifest",
        type=Path,
        required=True,
        help="GR00T manifest defining stable episode_num values",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--raw-rollouts-dir",
        type=Path,
        default=None,
        help="Defaults to source_root in the trajectory manifest",
    )
    source.add_argument(
        "--rollout-metadata-dir",
        type=Path,
        default=None,
        help=(
            "Read safe JSON sidecars instead of PKLs; accepts an optional "
            "source_manifest.json files mapping and <cell>/<stem>.json fallback"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--anchor-mode",
        choices=ANCHOR_MODES,
        default="phase-entry",
    )
    parser.add_argument(
        "--trust-pkl",
        action="store_true",
        help=(
            "Required acknowledgement only when reading rollout PKLs; "
            "not needed with --rollout-metadata-dir"
        ),
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.set_defaults(handler=_extract_keyframes)


def _add_materialize_media_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "materialize-media",
        help="Decode phase-contained frames for oracle keyframes",
    )
    parser.add_argument("--oracle-events", type=Path, required=True)
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="Defaults to source_root in the trajectory manifest",
    )
    parser.add_argument(
        "--trajectory-records",
        type=Path,
        default=None,
        help="Defaults to the manifest's trajectory_records_file",
    )
    parser.add_argument("--frames-per-sample", type=int, default=5)
    parser.add_argument(
        "--view-name",
        choices=("left", "right", "wrist"),
        default="left",
    )
    parser.add_argument("--scene-height-pixels", type=int, default=256)
    parser.add_argument("--view-width-pixels", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.set_defaults(handler=_materialize_media)


def _add_cluster_features_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "cluster-features",
        help="Cluster descriptors within oracle phase partitions",
    )
    parser.add_argument("--event-features", type=Path, required=True)
    parser.add_argument(
        "--oracle-events",
        type=Path,
        required=True,
        help="oracle_phase_events.jsonl from the extract-keyframes subcommand",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vision-weight", type=float, default=1.0)
    parser.add_argument("--state-weight", type=float, default=0.5)
    parser.add_argument(
        "--progress-weight",
        type=float,
        default=0.0,
        help="Zero by default because oracle phase is the time partition",
    )
    parser.add_argument(
        "--block-normalization",
        choices=("legacy", "balanced"),
        default="balanced",
    )
    parser.add_argument("--distance-threshold", type=float, default=0.18)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--num-exemplars", type=int, default=5)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.set_defaults(handler=_cluster_features)


def _add_scoring_selection_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "select-scoring-events",
        help="Select phase entries with a complete causal scoring window",
    )
    parser.add_argument("--oracle-events", type=Path, required=True)
    parser.add_argument(
        "--event-features",
        type=Path,
        default=None,
        help=(
            "Optional legacy/media feature rows. When omitted, scorer-ready "
            "environment-step metadata rows are generated directly."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=5)
    parser.set_defaults(handler=_select_scoring_events)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_extract_keyframes_parser(subparsers)
    _add_materialize_media_parser(subparsers)
    _add_cluster_features_parser(subparsers)
    _add_scoring_selection_parser(subparsers)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
