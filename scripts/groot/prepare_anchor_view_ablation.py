#!/usr/bin/env python3
"""Prepare audited media and vision-reuse inputs for the anchor/view ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from event_sae.groot.anchor_view_ablation import (
    build_multisource_reusable_features,
    materialize_position_virtual_media,
    materialize_prompt_records_from_event_features,
    materialize_rpg_virtual_media,
)


def _position(args: argparse.Namespace) -> dict:
    return materialize_position_virtual_media(
        waypoint_summary_path=args.waypoint_summary_path,
        source_samples_path=args.source_samples_path,
        trajectory_records_path=args.trajectory_records_path,
        output_path=args.output_path,
        view=args.view,
        expected_samples=args.expected_samples,
    )


def _rpg(args: argparse.Namespace) -> dict:
    return materialize_rpg_virtual_media(
        waypoint_summary_path=args.waypoint_summary_path,
        primary_samples_path=args.primary_samples_path,
        secondary_samples_path=args.secondary_samples_path,
        trajectory_records_path=args.trajectory_records_path,
        output_path=args.output_path,
        view=args.view,
        video_roots=args.video_root,
        expected_samples=args.expected_samples,
        expected_primary_reuse=args.expected_primary_reuse,
        expected_secondary_reuse=args.expected_secondary_reuse,
        expected_computed=args.expected_computed,
    )


def _features(args: argparse.Namespace) -> dict:
    return build_multisource_reusable_features(
        virtual_samples_path=args.virtual_samples_path,
        primary_features_path=args.primary_features_path,
        secondary_features_path=args.secondary_features_path,
        output_path=args.output_path,
        expected_primary=args.expected_primary,
        expected_secondary=args.expected_secondary,
    )


def _prompt_records(args: argparse.Namespace) -> dict:
    return materialize_prompt_records_from_event_features(
        event_features_paths=args.event_features_path,
        output_path=args.output_path,
        expected_episodes=args.expected_episodes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    position = subparsers.add_parser("position-media")
    position.add_argument("--waypoint-summary-path", type=Path, required=True)
    position.add_argument("--source-samples-path", type=Path, required=True)
    position.add_argument("--trajectory-records-path", type=Path, required=True)
    position.add_argument("--output-path", type=Path, required=True)
    position.add_argument(
        "--view",
        choices=("left", "right", "wrist"),
        required=True,
    )
    position.add_argument("--expected-samples", type=int, required=True)
    position.set_defaults(handler=_position)

    rpg = subparsers.add_parser("rpg-media")
    rpg.add_argument("--waypoint-summary-path", type=Path, required=True)
    rpg.add_argument("--primary-samples-path", type=Path, required=True)
    rpg.add_argument("--secondary-samples-path", type=Path, required=True)
    rpg.add_argument("--trajectory-records-path", type=Path, required=True)
    rpg.add_argument("--output-path", type=Path, required=True)
    rpg.add_argument(
        "--view",
        choices=("left", "right", "wrist"),
        required=True,
    )
    rpg.add_argument(
        "--video-root",
        type=Path,
        action="append",
        required=True,
    )
    rpg.add_argument("--expected-samples", type=int, required=True)
    rpg.add_argument("--expected-primary-reuse", type=int, required=True)
    rpg.add_argument("--expected-secondary-reuse", type=int, required=True)
    rpg.add_argument("--expected-computed", type=int, required=True)
    rpg.set_defaults(handler=_rpg)

    features = subparsers.add_parser("feature-reuse-subset")
    features.add_argument("--virtual-samples-path", type=Path, required=True)
    features.add_argument("--primary-features-path", type=Path, required=True)
    features.add_argument("--secondary-features-path", type=Path, required=True)
    features.add_argument("--output-path", type=Path, required=True)
    features.add_argument("--expected-primary", type=int, required=True)
    features.add_argument("--expected-secondary", type=int, required=True)
    features.set_defaults(handler=_features)

    prompt_records = subparsers.add_parser("prompt-records")
    prompt_records.add_argument(
        "--event-features-path",
        type=Path,
        action="append",
        required=True,
    )
    prompt_records.add_argument("--output-path", type=Path, required=True)
    prompt_records.add_argument(
        "--expected-episodes",
        type=int,
        required=True,
    )
    prompt_records.set_defaults(handler=_prompt_records)

    args = parser.parse_args()
    print(json.dumps(args.handler(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
