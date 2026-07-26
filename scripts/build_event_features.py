"""Build paper event features or reuse validated vision embeddings.

The historical paper invocation remains the default when no operation is
provided. Exact vision reuse and multiview fusion are explicit operations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sae import (
    DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_PROFILE,
    load_pipeline_profile,
)
from event_sae.events.build_features import (
    EventFeatureArtifactSpec,
    ExactReuseVisionEmbeddingProvider,
    build_event_features,
    combine_multiview_event_features,
)


_DEFAULT_OPERATION = "build"
_OPERATIONS = {
    _DEFAULT_OPERATION,
    "reuse-vision",
    "combine-multiview",
}


def _add_build_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        _DEFAULT_OPERATION,
        help="Build the paper event feature artifact",
        description=(
            "Build vision embeddings + state vectors per sample."
        ),
    )
    parser.add_argument(
        "--samples-path",
        required=True,
        help="Path to samples.jsonl",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Output JSONL "
            "(default: event_features.jsonl next to samples.jsonl)"
        ),
    )
    parser.add_argument(
        "--vision-model-name-or-path",
        default="google/siglip-base-patch16-224",
        help="Frozen vision encoder",
    )
    parser.add_argument(
        "--vision-model-revision",
        default=None,
        help="Optional Hugging Face model revision or commit",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available)",
    )
    parser.add_argument(
        "--trajectory-records-path",
        default=None,
        help=(
            "Stage 2 trajectory_records.jsonl "
            "(required for stage3 media v4)"
        ),
    )
    parser.add_argument(
        "--state-position-frame",
        choices=("rel", "abs"),
        default="rel",
        help="EEF coordinates used in the clustering state vector",
    )
    parser.add_argument(
        "--gripper-state-mode",
        choices=("legacy", "qpos_aperture_delta"),
        default="legacy",
        help=(
            "Use historical gripper_action state or shared qpos aperture "
            "+ delta"
        ),
    )
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument(
        "--frame-positions",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help=(
            "Frame indices (into sample.frame_paths) used for the "
            "vision embedding"
        ),
    )
    parser.set_defaults(handler=_run_build)


def _add_reuse_vision_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "reuse-vision",
        help="Build event features by exactly reusing vision embeddings",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_PROFILE,
    )
    parser.add_argument("--samples-path", type=Path, required=True)
    parser.add_argument(
        "--reusable-vision-features-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--trajectory-records-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--vision-model-name-or-path", default=None)
    parser.add_argument("--vision-model-revision", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument(
        "--frame-positions",
        type=int,
        nargs="+",
        default=None,
    )
    parser.set_defaults(handler=_run_reuse_vision)


def _add_combine_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "combine-multiview",
        help="Fuse aligned left/right/wrist event features",
    )
    parser.add_argument(
        "--left-features-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--right-features-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--wrist-features-path",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.set_defaults(handler=_run_combine_multiview)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    _add_build_parser(subparsers)
    _add_reuse_vision_parser(subparsers)
    _add_combine_parser(subparsers)
    return parser


def _run_build(args: argparse.Namespace) -> None:
    samples_path = Path(args.samples_path).resolve()
    output_path = (
        Path(args.output_path).resolve()
        if args.output_path is not None
        else samples_path.with_name("event_features.jsonl").resolve()
    )

    build_event_features(
        samples_path=samples_path,
        output_path=output_path,
        vision_model_name_or_path=args.vision_model_name_or_path,
        vision_model_revision=args.vision_model_revision,
        device=args.device,
        frame_positions=list(args.frame_positions),
        trajectory_records_path=(
            Path(args.trajectory_records_path)
            if args.trajectory_records_path
            else None
        ),
        expected_samples=args.expected_samples,
        state_position_frame=args.state_position_frame,
        gripper_state_mode=args.gripper_state_mode,
    )


def _run_reuse_vision(args: argparse.Namespace) -> None:
    profile = load_pipeline_profile(args.profile)
    trajectory_records_path = (
        args.trajectory_records_path
        or profile.path_value("source", "trajectory_records_path")
    )
    vision_model_name_or_path = (
        args.vision_model_name_or_path
        or str(profile.require("vision", "model_name_or_path"))
    )
    frame_positions = args.frame_positions or list(
        range(int(profile.require("media", "frames_per_view")))
    )
    device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    provider = ExactReuseVisionEmbeddingProvider(
        reusable_features_path=args.reusable_vision_features_path,
        vision_model_name_or_path=vision_model_name_or_path,
        vision_model_revision=args.vision_model_revision,
        frame_positions=list(frame_positions),
        device=device,
    )
    result = build_event_features(
        samples_path=args.samples_path,
        output_path=args.output_path,
        vision_model_name_or_path=vision_model_name_or_path,
        vision_model_revision=args.vision_model_revision,
        device=device,
        frame_positions=list(frame_positions),
        trajectory_records_path=trajectory_records_path,
        expected_samples=args.expected_samples,
        state_position_frame="abs",
        gripper_state_mode="qpos_aperture_delta",
        vision_embedding_provider=provider,
        artifact_spec=EventFeatureArtifactSpec(
            manifest_format="event_sae_v9_event_features_v1",
            record_source_format="event_sae_v9_virtual_media_v1",
            include_trajectory_record_sources=False,
            sort_manifest_keys=True,
            manifest_trailing_newline=True,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _run_combine_multiview(args: argparse.Namespace) -> None:
    manifest = combine_multiview_event_features(
        left_features_path=args.left_features_path,
        right_features_path=args.right_features_path,
        wrist_features_path=args.wrist_features_path,
        output_path=args.output_path,
        expected_samples=args.expected_samples,
    )
    print(json.dumps(manifest, indent=2))


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
