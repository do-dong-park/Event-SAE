"""Encode and audit GR00T PQ3 phase-feature analysis artifacts.

The subcommands cover sparse encoding, score audit, and checkpoint analysis:

``encode-topk``
    Encode an activation cache into provenance-aware sparse SAE shards.
``encode-transfer-topk``
    Encode explicitly distinct transfer-rollout activation shards.
``audit-scores``
    Independently audit phase-feature score and ranking artifacts.
``compare-ranking-sets``
    Compare phase-feature ranking sets across SAE checkpoints.
``analyze-checkpoint-stability``
    Test W4/W5 phase selectivity and decoder-matched stability across three
    SAE checkpoints.
``rank-task-local-phases``
    Rank W4/W5-robust phase features within exact instructions.
``rank-coarse-phase-candidates``
    Build relaxed suite-level reach/grasp/transport/terminal candidates.
``summarize-stage4-grid``
    Aggregate corrected support over a complete condition×coverage grid.
``analyze-event-phase-grid``
    Exhaustively rank event and fine/coarse phase candidates for V12 and
    Oracle score pairs.
``analyze-focused-phase-views``
    Compare fixed 10k V12 E3/E4 and full Oracle fine/exact-coarse views.
``analyze-directional-phase-views``
    Rescore those six views with named directional templates and report
    W5-primary candidates with W4/control sensitivity metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from event_sae import resolve_groot_artifact_path
from event_sae.groot.activations import (
    encode_activation_cache_to_sparse_topk,
)
from event_sae.groot.transfer_topk import (
    encode_transfer_activation_shards,
)
from event_sae.groot.phase_feature_results import summarize_stage4_grid
from event_sae.scoring.feature_activation_grid import (
    DirectionalPhaseViewAnalysisConfig,
    EventPhaseActivationGridConfig,
    FocusedPhaseViewAnalysisConfig,
    analyze_directional_phase_views,
    analyze_event_phase_activation_grid,
    analyze_focused_phase_views,
)
from event_sae.scoring.phase_selectivity import (
    CheckpointPhaseStabilityConfig,
    PhaseFeatureRun,
    analyze_checkpoint_phase_stability,
)
from event_sae.scoring.rankings import (
    compare_checkpoint_ranking_sets,
)
from event_sae.scoring.task_phase_ranking import (
    CoarsePhaseCandidateRankingConfig,
    TaskLocalPhaseRankingConfig,
    rank_coarse_phase_candidates,
    rank_task_local_phase_features,
)
from event_sae.scoring.score_matrix import (
    audit_groot_phase_feature_scores,
)


def _add_sparse_encoding_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "encode-topk",
        help="Encode the GR00T activation cache into sparse SAE shards",
    )
    parser.add_argument("--activation-cache", type=Path, required=True)
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--sae-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--activation-dim", type=int, default=1536)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--executed-action-steps", type=int, default=5)
    parser.add_argument("--topk", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--expected-files", type=int, default=150)
    parser.add_argument("--expected-records", type=int, default=12041)
    parser.add_argument("--expected-rows", type=int, default=770624)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-lossless-topk", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.set_defaults(handler=encode_activation_cache_to_sparse_topk)


def _add_transfer_encoding_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "encode-transfer-topk",
        help="Encode explicitly distinct transfer-rollout activation shards",
    )
    parser.add_argument("--activation-shard-dir", type=Path, required=True)
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--sae-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--activation-dim", type=int, default=1536)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--executed-action-steps", type=int, default=5)
    parser.add_argument("--topk", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--expected-sources", type=int, default=None)
    parser.add_argument("--require-lossless-topk", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.set_defaults(handler=_encode_transfer_topk)


def _encode_transfer_topk(args: argparse.Namespace) -> dict:
    result = encode_transfer_activation_shards(
        activation_shard_dir=args.activation_shard_dir,
        trajectory_manifest_path=args.trajectory_manifest,
        sae_checkpoint=args.sae_checkpoint,
        output_dir=args.output_dir,
        layer=args.layer,
        activation_dim=args.activation_dim,
        denoise_steps=args.denoise_steps,
        action_horizon=args.action_horizon,
        executed_action_steps=args.executed_action_steps,
        topk=args.topk,
        batch_size=args.batch_size,
        device=args.device,
        expected_sources=args.expected_sources,
        require_lossless_topk=args.require_lossless_topk,
        progress_every=args.progress_every,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _add_score_audit_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "audit-scores",
        help="Audit phase-feature score and ranking artifacts",
    )
    parser.add_argument("--scores-w5", type=Path, required=True)
    parser.add_argument("--scores-w4", type=Path, required=True)
    parser.add_argument("--rankings-w5", type=Path, required=True)
    parser.add_argument("--rankings-w4", type=Path, required=True)
    parser.add_argument("--topk-run-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-records-path",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dict-size", type=int, default=1536)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--expected-files", type=int, default=150)
    parser.add_argument("--expected-rows", type=int, default=770624)
    parser.add_argument(
        "--expected-executed-rows",
        type=int,
        default=240820,
    )
    parser.add_argument(
        "--expected-environment-steps",
        type=int,
        default=60205,
    )
    parser.add_argument("--expected-clusters", type=int, default=20)
    parser.add_argument("--expected-events", type=int, default=484)
    parser.add_argument("--expected-w5-shifts", type=int, default=100)
    parser.add_argument("--expected-w4-shifts", type=int, default=0)
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=1000,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heatmap-features", type=int, default=20)
    parser.set_defaults(handler=audit_groot_phase_feature_scores)


def _add_checkpoint_comparison_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "compare-ranking-sets",
        help="Compare phase-feature ranking sets across SAE checkpoints",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=RUN_DIR",
        help=(
            "Checkpoint label and phase-feature run directory; repeat at least "
            "twice."
        ),
    )
    parser.add_argument("--ranking-id", required=True)
    parser.add_argument("--audit-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.set_defaults(handler=compare_checkpoint_ranking_sets)


def _add_checkpoint_stability_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "analyze-checkpoint-stability",
        help=(
            "Analyze W4/W5 phase selectivity across three decoder-matched "
            "SAE checkpoints"
        ),
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=5,
        metavar=(
            "LABEL",
            "CHECKPOINT",
            "SCORE_W4",
            "SCORE_W5",
            "TOPK_DIR",
        ),
        required=True,
        help="Repeat exactly three times, once per SAE checkpoint.",
    )
    parser.add_argument(
        "--reference-label",
        required=True,
        help="Run label whose decoder features anchor the matched triplets.",
    )
    parser.add_argument("--num-permutations", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20_260_724)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(handler=_analyze_checkpoint_stability)


def _analyze_checkpoint_stability(
    args: argparse.Namespace,
) -> dict:
    return analyze_checkpoint_phase_stability(
        CheckpointPhaseStabilityConfig(
            runs=_phase_feature_runs(args.run),
            reference_label=args.reference_label,
            output_dir=args.output_dir,
            num_permutations=args.num_permutations,
            seed=args.seed,
            chunk_size=args.chunk_size,
        )
    )


def _add_task_local_ranking_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "rank-task-local-phases",
        help=(
            "Rank W4/W5-robust phase features within exact instructions and "
            "match them across three SAE checkpoints"
        ),
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=5,
        metavar=(
            "LABEL",
            "CHECKPOINT",
            "SCORE_W4",
            "SCORE_W5",
            "TOPK_DIR",
        ),
        required=True,
        help="Repeat exactly three times, once per SAE checkpoint.",
    )
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--accepted-annotations", type=Path, required=True)
    parser.add_argument("--phase-groups", type=Path, required=True)
    parser.add_argument("--phase-assignments", type=Path, required=True)
    parser.add_argument("--num-permutations", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20_260_725)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--score-event-step-scale",
        type=int,
        default=5,
        help=(
            "Multiplier that maps annotation waypoint steps onto score rows; "
            "use 1 for Oracle annotations already expressed in environment "
            "steps."
        ),
    )
    parser.add_argument(
        "--topk-event-step-scale",
        type=int,
        default=5,
        help=(
            "Event-step scale recorded by the reused Top-K manifest; this is "
            "validated independently from the score scale."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(handler=_rank_task_local_phases)


def _rank_task_local_phases(args: argparse.Namespace) -> dict:
    return rank_task_local_phase_features(
        TaskLocalPhaseRankingConfig(
            runs=_phase_feature_runs(args.run),
            reference_label=args.reference_label,
            condition_id=args.condition_id,
            accepted_annotations=args.accepted_annotations.resolve(),
            phase_groups=args.phase_groups.resolve(),
            phase_assignments=args.phase_assignments.resolve(),
            output_dir=args.output_dir.resolve(),
            entrypoint=Path(__file__).resolve(),
            num_permutations=args.num_permutations,
            seed=args.seed,
            chunk_size=args.chunk_size,
            top_n=args.top_n,
            alpha=args.alpha,
            score_event_step_scale=args.score_event_step_scale,
            topk_event_step_scale=args.topk_event_step_scale,
        )
    )


def _add_coarse_phase_ranking_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "rank-coarse-phase-candidates",
        help=(
            "Relax task-local rankings into suite-level "
            "reach/grasp/transport/terminal candidates"
        ),
    )
    parser.add_argument("--stage4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-condition", required=True)
    parser.add_argument("--sensitivity-condition", required=True)
    parser.add_argument("--discovery-coverage", default="cov0p3")
    parser.add_argument("--stability-coverage", default="cov0p4")
    parser.add_argument("--artifact-top-n", type=int, default=10)
    parser.add_argument("--candidate-pool-n", type=int, default=5)
    parser.add_argument("--shortlist-n", type=int, default=3)
    parser.add_argument("--expected-conditions", type=int, default=5)
    parser.set_defaults(handler=_rank_coarse_phase_candidates)


def _rank_coarse_phase_candidates(args: argparse.Namespace) -> dict:
    return rank_coarse_phase_candidates(
        CoarsePhaseCandidateRankingConfig(
            stage4_root=args.stage4_root.resolve(),
            output_dir=args.output_dir.resolve(),
            primary_condition=args.primary_condition,
            sensitivity_condition=args.sensitivity_condition,
            discovery_coverage=args.discovery_coverage,
            stability_coverage=args.stability_coverage,
            artifact_top_n=args.artifact_top_n,
            candidate_pool_n=args.candidate_pool_n,
            shortlist_n=args.shortlist_n,
            expected_conditions=args.expected_conditions,
            entrypoint=Path(__file__).resolve(),
        )
    )


def _add_stage4_grid_summary_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "summarize-stage4-grid",
        help=(
            "Aggregate grid-wide corrected task-local phase support by "
            "condition, coverage, and SAE checkpoint"
        ),
    )
    parser.add_argument("--stage4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-conditions", type=int, default=5)
    parser.add_argument("--expected-coverages", type=int, default=3)
    parser.add_argument(
        "--focus-checkpoint-label",
        default="sae10k",
        help="Checkpoint label to audit for descriptive count leadership.",
    )
    parser.set_defaults(handler=_summarize_stage4_grid)


def _summarize_stage4_grid(args: argparse.Namespace) -> dict:
    summary = summarize_stage4_grid(
        stage4_root=resolve_groot_artifact_path(args.stage4_root).resolve(),
        output_dir=args.output_dir.resolve(),
        expected_conditions=args.expected_conditions,
        expected_coverages=args.expected_coverages,
        focus_checkpoint_label=args.focus_checkpoint_label,
    )
    result = {
        "summary": summary["outputs"]["summary"],
        "report": summary["outputs"]["report"],
        "analysis_cell_count": summary["scope"]["analysis_cell_count"],
        "matched_grid_corrected_support": summary["matched_support"][
            "grid_corrected_support_count"
        ],
        "focus_checkpoint_comparison": summary[
            "focus_checkpoint_comparison"
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return summary


def _add_event_phase_grid_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "analyze-event-phase-grid",
        help=(
            "Rank every V12 condition/coverage/SAE score pair and keep "
            "Oracle as a separate simulator-labeled reference"
        ),
    )
    parser.add_argument("--stage4-root", type=Path, required=True)
    parser.add_argument("--oracle-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-conditions", type=int, default=5)
    parser.add_argument("--event-artifact-top-n", type=int, default=10)
    parser.add_argument("--event-primary-top-n", type=int, default=5)
    parser.add_argument("--phase-artifact-top-n", type=int, default=10)
    parser.add_argument("--phase-primary-top-n", type=int, default=3)
    parser.set_defaults(handler=_analyze_event_phase_grid)


def _analyze_event_phase_grid(args: argparse.Namespace) -> dict:
    return analyze_event_phase_activation_grid(
        EventPhaseActivationGridConfig(
            stage4_root=resolve_groot_artifact_path(
                args.stage4_root
            ).resolve(),
            oracle_summary=resolve_groot_artifact_path(
                args.oracle_summary
            ).resolve(),
            output_dir=args.output_dir.resolve(),
            expected_conditions=args.expected_conditions,
            event_artifact_top_n=args.event_artifact_top_n,
            event_primary_top_n=args.event_primary_top_n,
            phase_artifact_top_n=args.phase_artifact_top_n,
            phase_primary_top_n=args.phase_primary_top_n,
            entrypoint=Path(__file__).resolve(),
        )
    )


def _add_focused_phase_view_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "analyze-focused-phase-views",
        help=(
            "Analyze fixed 10k V12 E3/E4 cov0p3 and full Oracle with "
            "fine-original plus exact-rescored coarse4 views"
        ),
    )
    parser.add_argument("--stage4-root", type=Path, required=True)
    parser.add_argument("--oracle-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(handler=_analyze_focused_phase_views)


def _analyze_focused_phase_views(args: argparse.Namespace) -> dict:
    return analyze_focused_phase_views(
        FocusedPhaseViewAnalysisConfig(
            stage4_root=resolve_groot_artifact_path(
                args.stage4_root
            ).resolve(),
            oracle_summary=resolve_groot_artifact_path(
                args.oracle_summary
            ).resolve(),
            output_dir=args.output_dir.resolve(),
            entrypoint=Path(__file__).resolve(),
        )
    )


def _add_directional_phase_view_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "analyze-directional-phase-views",
        help=(
            "Rescore the immutable focused Oracle/E3/E4 fine and coarse "
            "views with W4/W5 directional templates"
        ),
    )
    parser.add_argument("--focused-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(handler=_analyze_directional_phase_views)


def _analyze_directional_phase_views(args: argparse.Namespace) -> dict:
    return analyze_directional_phase_views(
        DirectionalPhaseViewAnalysisConfig(
            focused_summary=resolve_groot_artifact_path(
                args.focused_summary
            ).resolve(),
            output_dir=args.output_dir.resolve(),
            entrypoint=Path(__file__).resolve(),
        )
    )


def _phase_feature_runs(values: list[list[str]]) -> tuple[PhaseFeatureRun, ...]:
    """Resolve repeated CLI run arguments into one shared typed contract."""

    return tuple(
        PhaseFeatureRun(
            label=str(run[0]),
            checkpoint=resolve_groot_artifact_path(run[1]).resolve(),
            score_w4=resolve_groot_artifact_path(run[2]).resolve(),
            score_w5=resolve_groot_artifact_path(run[3]).resolve(),
            topk_dir=resolve_groot_artifact_path(run[4]).resolve(),
        )
        for run in values
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_sparse_encoding_parser(subparsers)
    _add_transfer_encoding_parser(subparsers)
    _add_score_audit_parser(subparsers)
    _add_checkpoint_comparison_parser(subparsers)
    _add_checkpoint_stability_parser(subparsers)
    _add_task_local_ranking_parser(subparsers)
    _add_coarse_phase_ranking_parser(subparsers)
    _add_stage4_grid_summary_parser(subparsers)
    _add_event_phase_grid_parser(subparsers)
    _add_focused_phase_view_parser(subparsers)
    _add_directional_phase_view_parser(subparsers)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
