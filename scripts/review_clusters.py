#!/usr/bin/env python3
"""Run cluster review, annotation audit, and phase-group workflows."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sae import (
    DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_EXPERIMENT_ROOT,
    DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_PROFILE,
    DEFAULT_GROOT_LOG_ROOT,
    PipelineProfile,
    load_pipeline_profile,
)
from event_sae.events.cluster import (
    audit_and_freeze_annotation_bundle,
    build_phase_groups,
)
from event_sae.events.multiview_triptychs import (
    build_multiview_annotation_triptychs,
)
from event_sae.events.review import (
    build_cluster_review_service,
    finalize_reviewed_annotations,
    make_cluster_review_http_handler,
    render_contact_sheets,
)
from event_sae.groot.results_browser import (
    build_experiment_results_service,
    make_experiment_results_http_handler,
)


DEFAULT_CLUSTER_REVIEW_ROOT = (
    DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_EXPERIMENT_ROOT
)
DEFAULT_EVENT_FEATURES_PATH = (
    DEFAULT_CLUSTER_REVIEW_ROOT
    / "stage3_features/event_features_multiview_equal_concat_v9.jsonl"
)
DEFAULT_REVIEW_CLUSTER_RUN_DIR = (
    DEFAULT_CLUSTER_REVIEW_ROOT
    / "stage3_clusters/c0_balanced_v9/c0_d0p18"
)
DEFAULT_CLUSTERS_PATH = DEFAULT_REVIEW_CLUSTER_RUN_DIR / "clusters.jsonl"
DEFAULT_ASSIGNMENTS_PATH = (
    DEFAULT_REVIEW_CLUSTER_RUN_DIR / "cluster_assignments.jsonl"
)
DEFAULT_MEDIA_CLUSTERS_PATH = (
    DEFAULT_REVIEW_CLUSTER_RUN_DIR
    / "multiview_v9_cov0p3/clusters_multiview.jsonl"
)
DEFAULT_ANNOTATIONS_PATH = (
    DEFAULT_CLUSTER_REVIEW_ROOT
    / "stage3_annotations/"
    "gemini_3_1_pro_preview_v9_multiview_cov0p3_merged_annotations.jsonl"
)
DEFAULT_REVIEWS_PATH = (
    DEFAULT_REVIEW_CLUSTER_RUN_DIR
    / "human_cluster_reviews_multiview_v9.json"
)
DEFAULT_REVIEW_UI_PATH = (
    Path(__file__).resolve().parents[1]
    / "event_sae/events/templates/cluster_review.html"
)
DEFAULT_RESULTS_EXPERIMENT_ROOT = (
    DEFAULT_GROOT_LOG_ROOT
    / "experiments/anchor_view_controlled_ablation_v1"
)
DEFAULT_ORACLE_EXPERIMENT_ROOT = (
    DEFAULT_GROOT_LOG_ROOT
    / "oracle_phase_five_cell_v1"
)
DEFAULT_RESULTS_UI_PATH = (
    Path(__file__).resolve().parents[1]
    / "event_sae/groot/templates/results_browser.html"
)


def _annotation_min_episode_coverage(
    args: argparse.Namespace,
    profile: PipelineProfile,
) -> float:
    if args.min_episode_coverage is not None:
        return float(args.min_episode_coverage)
    return float(
        profile.require("clustering", "annotation_min_episode_coverage")
    )


def _serve(args: argparse.Namespace) -> None:
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        args.command_parser.error("Review server must bind to loopback")
    condition_metadata = {
        key: value
        for key, value in {
            "awe_anchor": args.awe_anchor,
            "clustering_view": args.clustering_view,
            "annotation_view": args.annotation_view,
        }.items()
        if value
    }
    application = build_cluster_review_service(
        event_features_path=args.event_features_path,
        clusters_path=args.clusters_path,
        assignments_path=args.assignments_path,
        annotations_path=args.annotations_path,
        reviews_path=args.reviews_path,
        media_clusters_path=(
            None if args.use_feature_media else args.media_clusters_path
        ),
        ui_path=args.ui_path,
        projection_method=args.projection,
        condition_id=args.condition_id,
        condition_label=args.condition_label,
        media_layout=args.media_layout,
        block_normalization=args.block_normalization,
        condition_metadata=condition_metadata,
    )
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_cluster_review_http_handler(application),
    )
    url = f"http://{args.host}:{server.server_address[1]}"
    metadata = application.payload["meta"]
    print(
        f"{metadata['condition']['id']} cluster review app: {url}\n"
        f"Playable samples: {metadata['num_playable_samples']}  "
        f"Annotated clusters: {metadata['num_annotated_clusters']}  "
        f"Reviews: {application.review_store.path}"
    )
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review app.")
    finally:
        server.server_close()


def _serve_results(args: argparse.Namespace) -> None:
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        args.command_parser.error("Results server must bind to loopback")
    application = build_experiment_results_service(
        experiment_root=args.experiment_root,
        ui_path=args.ui_path,
        oracle_experiment_root=args.oracle_experiment_root,
    )
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_experiment_results_http_handler(application),
    )
    url = f"http://{args.host}:{server.server_address[1]}"
    metadata = application.payload["meta"]
    oracle_metadata = application.oracle()
    print(
        f"Stage 4 experiment results: {url}\n"
        f"Runs: {metadata['available_runs']} / {metadata['expected_runs']}  "
        f"Status: {metadata['result_status']}\n"
        f"Oracle phase: {oracle_metadata['status']}"
    )
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping results app.")
    finally:
        server.server_close()


def _triptychs(args: argparse.Namespace) -> None:
    profile = load_pipeline_profile(args.profile)
    view_order = list(profile.require("media", "view_order"))
    if view_order != ["left", "right", "wrist"]:
        args.command_parser.error(
            "profile media.view_order must be left/right/wrist, "
            f"got {view_order}"
        )
    min_episode_coverage = _annotation_min_episode_coverage(args, profile)
    manifest = build_multiview_annotation_triptychs(
        clusters_path=args.clusters_path,
        left_samples_path=args.left_samples_path,
        right_samples_path=args.right_samples_path,
        wrist_samples_path=args.wrist_samples_path,
        output_dir=args.output_dir,
        min_episode_coverage=min_episode_coverage,
    )
    print(json.dumps(manifest, indent=2))


def _audit(args: argparse.Namespace) -> None:
    profile = load_pipeline_profile(args.profile)
    min_episode_coverage = _annotation_min_episode_coverage(args, profile)
    report = audit_and_freeze_annotation_bundle(
        event_features_path=args.event_features_path,
        assignments_path=args.assignments_path,
        clusters_path=args.clusters_path,
        media_clusters_path=args.media_clusters_path,
        annotations_path=args.annotations_path,
        output_annotations_path=args.output_annotations_path,
        audit_path=args.audit_path,
        expected_events=args.expected_events,
        expected_annotation_clusters=args.expected_annotation_clusters,
        min_episode_coverage=min_episode_coverage,
    )
    print(json.dumps(report, indent=2))


def _finalize(args: argparse.Namespace) -> None:
    rows = finalize_reviewed_annotations(
        annotations_path=args.annotations_path,
        output_path=args.output_path,
        expected_clusters=args.expected_clusters,
        reviews_path=args.reviews_path,
        assume_approved=args.assume_approved,
    )
    print(
        json.dumps(
            {
                "output_path": str(args.output_path.resolve()),
                "clusters": len(rows),
                "review_mode": rows[0]["review_mode"],
                "actual_human_review_completed": rows[0][
                    "actual_human_review_completed"
                ],
            },
            indent=2,
        )
    )


def _phase_groups(args: argparse.Namespace) -> None:
    summary = build_phase_groups(
        clusters_path=args.clusters_path,
        assignments_path=args.assignments_path,
        finalized_annotations_path=args.finalized_annotations_path,
        output_dir=args.output_dir,
        require_human_review=not args.allow_assumed_review,
    )
    print(json.dumps(summary, indent=2))


def _contact_sheets(args: argparse.Namespace) -> None:
    paths = render_contact_sheets(
        args.clusters_path,
        args.output_dir,
        canonical_only=not args.all_clusters,
        tile_size=args.tile_size,
    )
    print(
        json.dumps(
            {
                "num_sheets": len(paths),
                "sheets": [str(path) for path in paths],
            },
            indent=2,
        )
    )


def _filter_annotation_rows(args: argparse.Namespace) -> None:
    from event_sae.events.annotation_attempts import (
        filter_cluster_rows_by_coverage,
    )

    report = filter_cluster_rows_by_coverage(
        clusters_path=args.clusters_path,
        rows_path=args.rows_path,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
        min_episode_coverage=args.min_episode_coverage,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


def _merge_annotation_attempts(args: argparse.Namespace) -> None:
    from event_sae.events.annotation_attempts import merge_annotation_attempts

    report = merge_annotation_attempts(
        clusters_path=args.clusters_path,
        attempt_paths=args.attempt_path,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
        min_episode_coverage=args.min_episode_coverage,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


def _analyze_annotation_consistency(args: argparse.Namespace) -> None:
    from event_sae.groot.annotation_consistency import (
        analyze_annotation_consistency,
    )

    report = analyze_annotation_consistency(args.run_manifest)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


def _command(
    subparsers: argparse._SubParsersAction,
    name: str,
    *,
    help_text: str,
    description: str,
    handler: object,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        name,
        help=help_text,
        description=description,
    )
    parser.set_defaults(handler=handler, command_parser=parser)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="COMMAND",
    )

    serve = _command(
        subparsers,
        "serve",
        help_text="serve one condition's blind human cluster-review UI",
        description="Serve a condition-scoped blind cluster-review UI.",
        handler=_serve,
    )
    serve.add_argument(
        "--event-features-path",
        type=Path,
        default=DEFAULT_EVENT_FEATURES_PATH,
    )
    serve.add_argument(
        "--clusters-path",
        type=Path,
        default=DEFAULT_CLUSTERS_PATH,
    )
    serve.add_argument(
        "--assignments-path",
        type=Path,
        default=DEFAULT_ASSIGNMENTS_PATH,
    )
    serve.add_argument(
        "--annotations-path",
        type=Path,
        default=DEFAULT_ANNOTATIONS_PATH,
    )
    serve.add_argument(
        "--reviews-path",
        type=Path,
        default=DEFAULT_REVIEWS_PATH,
    )
    serve.add_argument(
        "--media-clusters-path",
        type=Path,
        default=DEFAULT_MEDIA_CLUSTERS_PATH,
    )
    serve.add_argument(
        "--use-feature-media",
        action="store_true",
        help=(
            "Use each feature row's native frame_paths instead of "
            "representative media"
        ),
    )
    serve.add_argument("--condition-id", default="v9")
    serve.add_argument(
        "--condition-label",
        default="3-view clusters · ABS + gripper (v9)",
    )
    serve.add_argument(
        "--media-layout",
        default="synchronized LEFT | RIGHT | WRIST triptych",
        help="Human-review reference media layout",
    )
    serve.add_argument(
        "--awe-anchor",
        default="abs position + gripper",
    )
    serve.add_argument("--clustering-view", default="3-view")
    serve.add_argument("--annotation-view", default="3-view")
    serve.add_argument(
        "--block-normalization",
        choices=("balanced", "legacy"),
        default="balanced",
    )
    serve.add_argument(
        "--ui-path",
        type=Path,
        default=DEFAULT_REVIEW_UI_PATH,
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument(
        "--projection",
        choices=("tsne", "pca"),
        default="tsne",
    )
    serve.add_argument("--open-browser", action="store_true")

    results = _command(
        subparsers,
        "results",
        help_text="serve the read-only Stage 4 experiment results explorer",
        description=(
            "Serve all controlled-ablation score and ranking results without "
            "exposing them inside the blind review workflow."
        ),
        handler=_serve_results,
    )
    results.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_RESULTS_EXPERIMENT_ROOT,
    )
    results.add_argument(
        "--ui-path",
        type=Path,
        default=DEFAULT_RESULTS_UI_PATH,
    )
    results.add_argument(
        "--oracle-experiment-root",
        type=Path,
        default=DEFAULT_ORACLE_EXPERIMENT_ROOT,
        help=(
            "Optional in-progress simulator-oracle experiment root. "
            "Missing artifacts are shown as a waiting state."
        ),
    )
    results.add_argument("--host", default="127.0.0.1")
    results.add_argument("--port", type=int, default=8766)
    results.add_argument("--open-browser", action="store_true")

    triptychs = _command(
        subparsers,
        "triptychs",
        help_text="build synchronized left/right/wrist annotation triptychs",
        description="Build annotation triptychs from aligned multiview media.",
        handler=_triptychs,
    )
    triptychs.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_PROFILE,
    )
    triptychs.add_argument("--clusters-path", type=Path, required=True)
    triptychs.add_argument("--left-samples-path", type=Path, required=True)
    triptychs.add_argument("--right-samples-path", type=Path, required=True)
    triptychs.add_argument("--wrist-samples-path", type=Path, required=True)
    triptychs.add_argument("--output-dir", type=Path, required=True)
    triptychs.add_argument("--min-episode-coverage", type=float, default=None)

    audit = _command(
        subparsers,
        "audit",
        help_text="audit and freeze a cluster-annotation bundle",
        description="Audit and freeze an unreviewed cluster-annotation bundle.",
        handler=_audit,
    )
    audit.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_PROFILE,
    )
    audit.add_argument("--event-features-path", type=Path, required=True)
    audit.add_argument("--assignments-path", type=Path, required=True)
    audit.add_argument("--clusters-path", type=Path, required=True)
    audit.add_argument("--media-clusters-path", type=Path, required=True)
    audit.add_argument("--annotations-path", type=Path, required=True)
    audit.add_argument(
        "--output-annotations-path",
        type=Path,
        required=True,
    )
    audit.add_argument("--audit-path", type=Path, required=True)
    audit.add_argument("--expected-events", type=int, default=1278)
    audit.add_argument(
        "--expected-annotation-clusters",
        type=int,
        default=20,
    )
    audit.add_argument("--min-episode-coverage", type=float, default=None)

    finalize = _command(
        subparsers,
        "finalize",
        help_text="materialize finalized reviewed annotations",
        description="Finalize Stage 3 annotations without hiding review provenance.",
        handler=_finalize,
    )
    finalize.add_argument("--annotations-path", type=Path, required=True)
    finalize.add_argument("--output-path", type=Path, required=True)
    finalize.add_argument(
        "--expected-clusters",
        type=int,
        default=None,
        help=(
            "Optional strict annotation-cluster count; by default the validated "
            "annotation artifact defines the count"
        ),
    )
    review_source = finalize.add_mutually_exclusive_group(required=True)
    review_source.add_argument("--reviews-path", type=Path)
    review_source.add_argument("--assume-approved", action="store_true")

    phase_groups = _command(
        subparsers,
        "phase-groups",
        help_text="create reviewed phase unions for Stage 4",
        description="Create instruction-local reviewed phase unions for scoring.",
        handler=_phase_groups,
    )
    phase_groups.add_argument("--clusters-path", type=Path, required=True)
    phase_groups.add_argument("--assignments-path", type=Path, required=True)
    phase_groups.add_argument(
        "--finalized-annotations-path",
        type=Path,
        required=True,
        help="Human-reviewed finalized cluster annotations",
    )
    phase_groups.add_argument("--output-dir", type=Path, required=True)
    phase_groups.add_argument(
        "--allow-assumed-review",
        action="store_true",
        help="Diagnostic only: permit non-human finalized annotations",
    )

    contact_sheets = _command(
        subparsers,
        "contact-sheets",
        help_text="render task-wise cluster contact sheets",
        description="Render representative cluster frames for visual auditing.",
        handler=_contact_sheets,
    )
    contact_sheets.add_argument("--clusters-path", type=Path, required=True)
    contact_sheets.add_argument("--output-dir", type=Path, required=True)
    contact_sheets.add_argument("--all-clusters", action="store_true")
    contact_sheets.add_argument("--tile-size", type=int, default=128)

    filter_rows = _command(
        subparsers,
        "filter-annotation-rows",
        help_text="filter cluster-keyed annotation rows by episode coverage",
        description=(
            "Filter cluster-keyed JSONL rows by raw-cluster episode coverage."
        ),
        handler=_filter_annotation_rows,
    )
    filter_rows.add_argument("--clusters-path", type=Path, required=True)
    filter_rows.add_argument("--rows-path", type=Path, required=True)
    filter_rows.add_argument("--output-path", type=Path, required=True)
    filter_rows.add_argument("--manifest-path", type=Path, required=True)
    filter_rows.add_argument(
        "--min-episode-coverage",
        type=float,
        required=True,
    )

    merge_attempts = _command(
        subparsers,
        "merge-annotation-attempts",
        help_text="freeze non-overlapping successful annotation attempts",
        description=(
            "Freeze non-overlapping successful rows from annotation attempts."
        ),
        handler=_merge_annotation_attempts,
    )
    merge_attempts.add_argument("--clusters-path", type=Path, required=True)
    merge_attempts.add_argument(
        "--attempt-path",
        type=Path,
        action="append",
        required=True,
        help="Annotation attempt JSONL; repeat in chronological order",
    )
    merge_attempts.add_argument("--output-path", type=Path, required=True)
    merge_attempts.add_argument("--manifest-path", type=Path, required=True)
    merge_attempts.add_argument(
        "--min-episode-coverage",
        type=float,
        default=0.3,
    )

    consistency = _command(
        subparsers,
        "annotation-consistency",
        help_text="print a read-only representative-label consistency report",
        description=(
            "Analyze stored representative-label consistency without "
            "measuring annotation accuracy."
        ),
        handler=_analyze_annotation_consistency,
    )
    consistency.add_argument(
        "--run-manifest",
        type=Path,
        required=True,
        help="Historical representative-annotation run manifest to inspect.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
