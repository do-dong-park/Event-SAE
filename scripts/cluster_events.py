"""CLI for paper event clustering and explicit clustering sweeps.

The historical paper invocation remains the default when no operation is
provided. The fixed Stage 3 sensitivity sweep is an explicit operation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sae.events.cluster import cluster_events, run_clustering_sweep


_DEFAULT_OPERATION = "cluster"
_OPERATIONS = {_DEFAULT_OPERATION, "sweep"}


def _add_cluster_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        _DEFAULT_OPERATION,
        help="Run one paper-compatible clustering configuration",
        description=(
            "Task-local agglomerative clustering of event features."
        ),
    )
    parser.add_argument(
        "--event-features-path",
        required=True,
        help="Path to event_features.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory "
            "(default: 'clusters/' next to event_features.jsonl)"
        ),
    )
    parser.add_argument("--vision-weight", type=float, default=1.0)
    parser.add_argument("--state-weight", type=float, default=0.5)
    parser.add_argument("--progress-weight", type=float, default=0.4)
    parser.add_argument(
        "--block-normalization",
        choices=("legacy", "balanced"),
        default="legacy",
        help=(
            "balanced keeps vision/state/progress block scales "
            "dimension-independent"
        ),
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.18,
    )
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--num-exemplars", type=int, default=5)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.set_defaults(handler=_run_cluster)


def _add_sweep_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "sweep",
        help="Run the fixed Stage 3 C0/C1/C2 clustering sweep",
    )
    parser.add_argument("--event-features-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["c0", "c1", "c2"],
    )
    parser.add_argument(
        "--distance-thresholds",
        type=float,
        nargs="+",
        default=[0.12, 0.15, 0.18, 0.21, 0.24],
    )
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--num-exemplars", type=int, default=5)
    parser.add_argument(
        "--block-normalization",
        choices=("legacy", "balanced"),
        default="legacy",
        help=(
            "balanced keeps state dimension changes from changing "
            "block weight"
        ),
    )
    parser.set_defaults(handler=_run_sweep)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    _add_cluster_parser(subparsers)
    _add_sweep_parser(subparsers)
    return parser


def _run_cluster(args: argparse.Namespace) -> None:
    event_features_path = Path(args.event_features_path).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (
            event_features_path.with_suffix("")
            .with_name("clusters")
            .resolve()
        )
    )

    summary = cluster_events(
        event_features_path=event_features_path,
        output_dir=output_dir,
        vision_weight=args.vision_weight,
        state_weight=args.state_weight,
        progress_weight=args.progress_weight,
        block_normalization=args.block_normalization,
        distance_threshold=args.distance_threshold,
        min_coverage=args.min_coverage,
        num_exemplars=args.num_exemplars,
        expected_samples=args.expected_samples,
    )
    print(json.dumps(summary, indent=2))


def _run_sweep(args: argparse.Namespace) -> None:
    report = run_clustering_sweep(
        event_features_path=Path(args.event_features_path),
        output_root=Path(args.output_root),
        config_ids=list(args.configs),
        thresholds=list(args.distance_thresholds),
        expected_samples=args.expected_samples,
        min_coverage=args.min_coverage,
        num_exemplars=args.num_exemplars,
        block_normalization=args.block_normalization,
    )
    print(json.dumps(report, indent=2))


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
