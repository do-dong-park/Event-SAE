"""Exhaustive descriptive event- and phase-feature activation analysis.

The grid deliberately keeps two non-exclusive ranking axes:

* ``event_ranked``: W5 event scores averaged equally over exact instructions.
* ``phase_aligned``: within-instruction phase-minus-mean-other contrasts that
  agree in sign for W4 and W5 and are positive in a strict instruction
  majority.

Fine and coarse phase labels are alternative ontology views of the same score
pairs, not independent replications.  Oracle scores are analyzed with the same
descriptive math but remain a separate simulator-labeled reference panel.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from event_sae import resolve_groot_artifact_path, sha256_file
from event_sae.groot.oracle_phase_keyframes import (
    oracle_phase_entry_invariant_errors,
)
from event_sae.scoring.phase_selectivity import feature_ranks_descending
from event_sae.scoring.score_matrix import (
    _merge_episode_task_ids,
    aggregate_sparse_activations_by_timestep,
    open_sparse_topk_artifact,
    score_cluster_features,
)
from event_sae.scoring.task_phase_ranking import (
    COARSE_PHASE_BY_ANNOTATION,
    COARSE_PHASE_ORDER,
    DIRECTIONAL_TEMPLATE_NAMES,
    DirectionalDiscoveryConfig,
    TaskLocalPhaseScores,
    _coarse_phase_cells,
    _mean_other_phase_margin,
    _rank_coarse_phase_cells,
    discover_directional_phase_features,
    load_directional_template_scores,
    load_task_local_score_pair,
    rank_directional_phase_candidates,
    summarize_directional_phase_recurrence,
)


SCHEMA_VERSION = "event_phase_activation_grid_v1"
DEFAULT_COVERAGES = ("cov0p3", "cov0p4", "cov0p5")
DEFAULT_SAE_LABELS = ("sae1p2k", "sae10k", "sae10k_bs8192")
ORACLE_FINE_PHASES = (
    "reach-to-object",
    "grasp",
    "place",
    "insert-settle",
)
ORACLE_RUN_ALIASES = {
    "bs4096_1p2k": "sae1p2k",
    "bs4096_10k": "sae10k",
    "bs8192_5k": "sae10k_bs8192",
}

FOCUSED_SCHEMA_VERSION = "focused_phase_view_analysis_v1"
DIRECTIONAL_PHASE_VIEW_SCHEMA_VERSION = (
    "directional_phase_view_analysis_v1"
)
DIRECTIONAL_SOURCE_ORDER = (
    "oracle_full",
    "v12_e3_cov0p3",
    "v12_e4_cov0p3",
)
DIRECTIONAL_VIEW_ORDER = (
    "fine_original",
    "coarse4_exact_rescore",
)
FOCUSED_CHECKPOINT_SHA256 = (
    "0f0f35503340d4e619fed269a2a7ab49d49db9d9e07c3317c57070f503dae96b"
)
FOCUSED_CHECKPOINT_PATH = Path(
    "logs/groot_n15/stage1_sae/checkpoints/step_sweep/"
    "bs4096_steps010000_seed0/trainer_0/ae.pt"
)
FOCUSED_V12_CONDITIONS = {
    "v12_e3_cov0p3": (
        "e3_rel_pos_gripper_cluster_multiview_label_multiview"
    ),
    "v12_e4_cov0p3": (
        "e4_abs_pos_gripper_cluster_multiview_label_multiview"
    ),
}
FOCUSED_SOURCE_CONTRACT = {
    "v12_e3_cov0p3": {
        "condition_id": FOCUSED_V12_CONDITIONS["v12_e3_cov0p3"],
        "event_step_scale": 5,
        "fine_rows": 14,
        "fine_episode_groups": 242,
        "coarse_rows": 13,
        "coarse_episode_groups": 234,
        "selected_events": 486,
        "contrastable_tasks": 5,
        "w4_shifted_windows": 0,
        "w5_shifted_windows": 70,
    },
    "v12_e4_cov0p3": {
        "condition_id": FOCUSED_V12_CONDITIONS["v12_e4_cov0p3"],
        "event_step_scale": 5,
        "fine_rows": 12,
        "fine_episode_groups": 210,
        "coarse_rows": 12,
        "coarse_episode_groups": 210,
        "selected_events": 484,
        "contrastable_tasks": 4,
        "w4_shifted_windows": 0,
        "w5_shifted_windows": 70,
    },
    "oracle_full": {
        "condition_id": "oracle_phase_entry",
        "event_step_scale": 1,
        "fine_rows": 25,
        "fine_episode_groups": 433,
        "coarse_rows": 20,
        "coarse_episode_groups": 342,
        "selected_events": 1050,
        "contrastable_tasks": 5,
        "w4_shifted_windows": 0,
        "w5_shifted_windows": 0,
    },
}
KNOWN_NON_PHASE_LABELS = frozenset({"mixed", "wrong-grasp"})
FOCUSED_SCORE_DEFINITIONS = {
    "pulse": (
        "positive projection onto a symmetric local-peak template "
        "(event-centered) after time-centering"
    ),
    "step_up": (
        "positive projection onto a low-to-high step template "
        "(event-centered) after time-centering"
    ),
    "step_down": (
        "positive projection onto a high-to-low step template "
        "(event-centered) after time-centering"
    ),
    "combined_score": (
        "within each (cluster, episode), average event projections separately "
        "for pulse, step_up, and step_down, then take the feature-wise maximum "
        "of those three template means"
    ),
    "matrix_raw": (
        "episode-balanced per-cluster mean of combined_score "
        "(== event_aligned ranking)"
    ),
    "matrix_window_mean": "per-cluster mean of window-mean activation",
    "matrix_task_mean": (
        "per-cluster, broadcast the per-task mean activation over all cached "
        "timesteps"
    ),
}


@dataclass(frozen=True)
class EventPhaseActivationGridConfig:
    """Inputs and shortlist sizes for one immutable exhaustive grid."""

    stage4_root: Path
    oracle_summary: Path
    output_dir: Path
    expected_conditions: int = 5
    coverages: tuple[str, ...] = DEFAULT_COVERAGES
    sae_labels: tuple[str, ...] = DEFAULT_SAE_LABELS
    event_artifact_top_n: int = 10
    event_primary_top_n: int = 5
    phase_artifact_top_n: int = 10
    phase_primary_top_n: int = 3
    comparator_top_n: int = 20
    entrypoint: Path | None = None


@dataclass(frozen=True)
class FocusedPhaseViewAnalysisConfig:
    """Immutable E3/E4/Oracle analysis at one fixed 10k SAE coordinate."""

    stage4_root: Path
    oracle_summary: Path
    output_dir: Path
    comparator_top_n: int = 20
    filtered_top_n: int = 10
    entrypoint: Path | None = None


@dataclass(frozen=True)
class DirectionalTracePersistenceConfig:
    """Diagnostic full-trace checks for displayed coarse state pairs."""

    local_window_size: int = 5
    minimum_interval_steps: int = 5
    activation_epsilon: float = 0.0
    on_minimum_absolute_increase: float = 1e-6
    on_minimum_ratio: float = 1.25
    interval_minimum_on_post_fraction: float = 0.5
    interval_minimum_activation_prevalence: float = 0.5
    off_maximum_interval_fraction: float = 0.75
    off_minimum_absolute_decrease: float = 1e-6
    minimum_comparable_episodes: int = 2
    confirmed_minimum_full_repeat_ratio: float = 0.5
    partial_minimum_full_repeat_ratio: float = 0.25
    partial_minimum_component_repeat_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.local_window_size <= 0:
            raise ValueError("local_window_size must be positive.")
        if self.minimum_interval_steps <= 0:
            raise ValueError("minimum_interval_steps must be positive.")
        if self.minimum_comparable_episodes <= 0:
            raise ValueError(
                "minimum_comparable_episodes must be positive."
            )
        if self.on_minimum_absolute_increase < 0:
            raise ValueError(
                "on_minimum_absolute_increase must be nonnegative."
            )
        if self.off_minimum_absolute_decrease < 0:
            raise ValueError(
                "off_minimum_absolute_decrease must be nonnegative."
            )
        if self.activation_epsilon < 0:
            raise ValueError("activation_epsilon must be nonnegative.")
        if self.on_minimum_ratio < 1:
            raise ValueError("on_minimum_ratio must be at least one.")
        for name in (
            "interval_minimum_on_post_fraction",
            "interval_minimum_activation_prevalence",
            "off_maximum_interval_fraction",
            "confirmed_minimum_full_repeat_ratio",
            "partial_minimum_full_repeat_ratio",
            "partial_minimum_component_repeat_ratio",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one.")


@dataclass(frozen=True)
class DirectionalPhaseViewAnalysisConfig:
    """Immutable directional W5 discovery over the focused six views.

    W4 scores and window/task-mean controls are sensitivity annotations only.
    They never affect candidate membership, ordering, or recurrence support.
    """

    focused_summary: Path
    output_dir: Path
    display_top_n: int = 10
    control_top_n: int = 20
    low_coverage_threshold: float = 0.3
    trace_persistence: DirectionalTracePersistenceConfig = (
        DirectionalTracePersistenceConfig()
    )
    entrypoint: Path | None = None


def _canonical_path(path: str | Path) -> Path:
    return resolve_groot_artifact_path(path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _validated_task_mean(task: TaskLocalPhaseScores, *, window: int) -> np.ndarray:
    matrix = task.task_mean_w4 if window == 4 else task.task_mean_w5
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError(f"{task.task_description!r}: task mean is empty.")
    if not np.allclose(matrix, matrix[0][None, :], rtol=1e-5, atol=1e-5):
        raise ValueError(
            f"{task.task_description!r}: task-mean rows disagree for W{window}."
        )
    return matrix[0]


def rank_task_balanced_event_features(
    tasks: dict[str, TaskLocalPhaseScores],
    *,
    artifact_top_n: int,
    primary_top_n: int,
    comparator_top_n: int,
) -> dict[str, Any]:
    """Rank event activation after equal weighting of exact instructions.

    W5 is the paper-like primary score.  W4 and the equal W4/W5 mean remain
    sensitivity views, so the output contains all three rankings.
    """

    if not tasks:
        raise ValueError("At least one exact instruction is required.")
    if not artifact_top_n >= primary_top_n > 0:
        raise ValueError("Require artifact_top_n >= primary_top_n > 0.")
    instructions = sorted(tasks)
    event_w4_by_task = np.stack(
        [tasks[name].pair.phase_w4.mean(axis=0) for name in instructions]
    )
    event_w5_by_task = np.stack(
        [tasks[name].pair.phase_w5.mean(axis=0) for name in instructions]
    )
    window_w5_by_task = np.stack(
        [tasks[name].window_mean_w5.mean(axis=0) for name in instructions]
    )
    task_mean_w5_by_task = np.stack(
        [_validated_task_mean(tasks[name], window=5) for name in instructions]
    )
    event_w4 = event_w4_by_task.mean(axis=0)
    event_w5 = event_w5_by_task.mean(axis=0)
    event_mean = (event_w4 + event_w5) / 2.0
    window_w5 = window_w5_by_task.mean(axis=0)
    task_mean_w5 = task_mean_w5_by_task.mean(axis=0)

    rank_w4 = feature_ranks_descending(event_w4)
    rank_w5 = feature_ranks_descending(event_w5)
    rank_mean = feature_ranks_descending(event_mean)
    window_rank_w5 = feature_ranks_descending(window_w5)
    task_mean_rank_w5 = feature_ranks_descending(task_mean_w5)
    task_event_ranks_w5 = np.stack(
        [feature_ranks_descending(row) for row in event_w5_by_task]
    )
    instruction_top_count = np.count_nonzero(
        task_event_ranks_w5 <= comparator_top_n,
        axis=0,
    )
    order_w5 = np.argsort(-event_w5, kind="stable")
    order_w4 = np.argsort(-event_w4, kind="stable")
    order_mean = np.argsort(-event_mean, kind="stable")

    candidates = []
    for rank, feature_id in enumerate(order_w5[:artifact_top_n], start=1):
        feature_idx = int(feature_id)
        candidates.append(
            {
                "rank": rank,
                "feature_id": feature_idx,
                "event_score_w5": float(event_w5[feature_idx]),
                "event_score_w4": float(event_w4[feature_idx]),
                "event_score_mean_w4_w5": float(event_mean[feature_idx]),
                "rank_w4": int(rank_w4[feature_idx]),
                "rank_mean_w4_w5": int(rank_mean[feature_idx]),
                "w4_in_artifact_top_n": bool(
                    rank_w4[feature_idx] <= artifact_top_n
                ),
                "instruction_top20_count": int(
                    instruction_top_count[feature_idx]
                ),
                "instruction_count": len(instructions),
                "instruction_top20_fraction": float(
                    instruction_top_count[feature_idx] / len(instructions)
                ),
                "window_mean_rank_w5": int(
                    window_rank_w5[feature_idx]
                ),
                "window_mean_top20": bool(
                    window_rank_w5[feature_idx] <= comparator_top_n
                ),
                "task_mean_rank_w5": int(
                    task_mean_rank_w5[feature_idx]
                ),
                "task_mean_top20": bool(
                    task_mean_rank_w5[feature_idx] <= comparator_top_n
                ),
            }
        )
    return {
        "primary_score": "w5",
        "instructions": instructions,
        "instruction_count": len(instructions),
        "top_feature_ids": {
            "w5": [int(value) for value in order_w5[:artifact_top_n]],
            "w4": [int(value) for value in order_w4[:artifact_top_n]],
            "mean_w4_w5": [
                int(value) for value in order_mean[:artifact_top_n]
            ],
        },
        "primary_top_feature_ids": [
            int(value) for value in order_w5[:primary_top_n]
        ],
        "top_candidates": candidates,
    }


def build_fine_phase_cells(
    tasks: dict[str, TaskLocalPhaseScores],
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, Any]]:
    """Build exact-label phase contrasts without pooling instructions."""

    cells: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    diagnostics: dict[str, Any] = {
        "num_tasks": len(tasks),
        "num_contrastable_tasks": 0,
        "num_phase_rows": sum(len(task.pair.phases) for task in tasks.values()),
        "num_contrastable_phase_rows": 0,
        "phase_instruction_counts": {},
    }
    for task_description, task in sorted(tasks.items()):
        if len(task.pair.phases) < 2:
            continue
        diagnostics["num_contrastable_tasks"] += 1
        diagnostics["num_contrastable_phase_rows"] += len(task.pair.phases)
        margin_w4 = _mean_other_phase_margin(task.pair.phase_w4)
        margin_w5 = _mean_other_phase_margin(task.pair.phase_w5)
        window_margin_w5 = _mean_other_phase_margin(task.window_mean_w5)
        task_mean_w5 = _validated_task_mean(task, window=5)
        for phase_idx, phase in enumerate(task.pair.phases):
            cells.setdefault(phase, {})[task_description] = {
                "margin_w4": margin_w4[phase_idx],
                "margin_w5": margin_w5[phase_idx],
                "window_margin_w5": window_margin_w5[phase_idx],
                "task_mean_w5": task_mean_w5,
            }
    diagnostics["phase_instruction_counts"] = {
        phase: len(rows) for phase, rows in sorted(cells.items())
    }
    return cells, diagnostics


def build_fixed_phase_composition_cells(
    tasks: dict[str, TaskLocalPhaseScores],
    *,
    target_phases: tuple[str, ...] = ORACLE_FINE_PHASES,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, Any]]:
    """Restrict task-local contrasts to one fixed phase vocabulary.

    A suite-level comparison is eligible only when every target phase is
    observed somewhere in the score pair. Missing phases are never imputed.
    Exact-instruction completeness is reported separately because aggregating
    phases from different tasks does not reproduce Oracle's single-task
    four-phase composition.
    """

    if len(target_phases) < 2 or len(target_phases) != len(set(target_phases)):
        raise ValueError("Fixed phase composition requires unique phase names.")
    target_set = set(target_phases)
    cells: dict[str, dict[str, dict[str, np.ndarray]]] = {
        phase: {} for phase in target_phases
    }
    observed_phases: set[str] = set()
    exact_instruction_complete_count = 0
    contrastable_tasks = 0
    task_phase_composition: dict[str, list[str]] = {}
    for task_description, task in sorted(tasks.items()):
        phase_indices = [
            index
            for index, phase in enumerate(task.pair.phases)
            if phase in target_set
        ]
        present_phases = [
            task.pair.phases[index] for index in phase_indices
        ]
        task_phase_composition[task_description] = present_phases
        observed_phases.update(present_phases)
        if set(present_phases) == target_set:
            exact_instruction_complete_count += 1
        if len(phase_indices) < 2:
            continue
        contrastable_tasks += 1
        margin_w4 = _mean_other_phase_margin(
            task.pair.phase_w4[phase_indices]
        )
        margin_w5 = _mean_other_phase_margin(
            task.pair.phase_w5[phase_indices]
        )
        window_margin_w5 = _mean_other_phase_margin(
            task.window_mean_w5[phase_indices]
        )
        task_mean_w5 = _validated_task_mean(task, window=5)
        for local_index, phase in enumerate(present_phases):
            cells[phase][task_description] = {
                "margin_w4": margin_w4[local_index],
                "margin_w5": margin_w5[local_index],
                "window_margin_w5": window_margin_w5[local_index],
                "task_mean_w5": task_mean_w5,
            }
    missing_phases = [
        phase for phase in target_phases if phase not in observed_phases
    ]
    uncontrastable_phases = [
        phase for phase in target_phases if not cells[phase]
    ]
    diagnostics = {
        "target_phases": list(target_phases),
        "observed_phases": [
            phase for phase in target_phases if phase in observed_phases
        ],
        "missing_phases": missing_phases,
        "suite_composition_complete": not missing_phases,
        "uncontrastable_phases": uncontrastable_phases,
        "comparison_eligible": (
            not missing_phases and not uncontrastable_phases
        ),
        "exact_instruction_complete_count": (
            exact_instruction_complete_count
        ),
        "num_tasks": len(tasks),
        "num_contrastable_tasks": contrastable_tasks,
        "phase_instruction_counts": {
            phase: len(cells[phase]) for phase in target_phases
        },
        "task_phase_composition": task_phase_composition,
        "missing_phase_policy": "exclude comparison cell; never impute",
    }
    return cells, diagnostics


def rank_phase_cells(
    cells: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    artifact_top_n: int,
    primary_top_n: int,
) -> dict[str, Any]:
    """Apply the shared W4/W5 sign-and-majority descriptive phase ranker."""

    phases: dict[str, Any] = {}
    for phase, instruction_cells in sorted(cells.items()):
        ranked, _ = _rank_coarse_phase_cells(
            instruction_cells,
            artifact_top_n=artifact_top_n,
        )
        ranked["primary_top_feature_ids"] = [
            int(candidate["feature_id"])
            for candidate in ranked["top_candidates"][:primary_top_n]
        ]
        phases[phase] = ranked
    return phases


def classify_event_phase_candidates(
    event_result: dict[str, Any],
    phase_results: dict[str, Any],
    *,
    phase_primary_top_n: int,
) -> dict[str, Any]:
    """Create non-exclusive event/phase memberships and comparator overlays."""

    event_ids = {
        int(feature_id)
        for feature_id in event_result["primary_top_feature_ids"]
    }
    phase_affiliations: dict[int, list[str]] = {}
    phase_candidate_by_id: dict[int, list[dict[str, Any]]] = {}
    for phase, result in phase_results.items():
        for candidate in result["top_candidates"][:phase_primary_top_n]:
            feature_id = int(candidate["feature_id"])
            phase_affiliations.setdefault(feature_id, []).append(phase)
            phase_candidate_by_id.setdefault(feature_id, []).append(candidate)
    phase_ids = set(phase_affiliations)
    event_candidate_by_id = {
        int(candidate["feature_id"]): candidate
        for candidate in event_result["top_candidates"]
    }
    union = sorted(event_ids | phase_ids)
    task_mean_overlay = []
    window_overlay = []
    for feature_id in union:
        event_candidate = event_candidate_by_id.get(feature_id, {})
        phase_candidates = phase_candidate_by_id.get(feature_id, [])
        if bool(event_candidate.get("task_mean_top20")) or any(
            bool(candidate.get("task_mean_top20"))
            for candidate in phase_candidates
        ):
            task_mean_overlay.append(feature_id)
        if bool(event_candidate.get("window_mean_top20")) or any(
            bool(candidate.get("window_mean_same_sign"))
            for candidate in phase_candidates
        ):
            window_overlay.append(feature_id)
    return {
        "event_ranked_only": sorted(event_ids - phase_ids),
        "phase_aligned_only": sorted(phase_ids - event_ids),
        "dual_ranked": sorted(event_ids & phase_ids),
        "phase_affiliations": {
            str(feature_id): sorted(phases)
            for feature_id, phases in sorted(phase_affiliations.items())
        },
        "task_mean_top20_overlay": task_mean_overlay,
        "window_magnitude_or_persistence_overlay": window_overlay,
        "contract": (
            "The three membership lists are MECE for the primary event Top-5 "
            "and union of per-phase Top-3. Comparator lists are overlays, not "
            "exclusion gates."
        ),
    }


def _analyze_score_pair(
    score_w4: Path,
    score_w5: Path,
    *,
    config: EventPhaseActivationGridConfig,
) -> dict[str, Any]:
    tasks = load_task_local_score_pair(score_w4, score_w5)
    event_result = rank_task_balanced_event_features(
        tasks,
        artifact_top_n=config.event_artifact_top_n,
        primary_top_n=config.event_primary_top_n,
        comparator_top_n=config.comparator_top_n,
    )
    fine_cells, fine_diagnostics = build_fine_phase_cells(tasks)
    coarse_cells, coarse_diagnostics = _coarse_phase_cells(tasks)
    fixed_cells, fixed_diagnostics = build_fixed_phase_composition_cells(
        tasks
    )
    fine_phases = rank_phase_cells(
        fine_cells,
        artifact_top_n=config.phase_artifact_top_n,
        primary_top_n=config.phase_primary_top_n,
    )
    coarse_phases = rank_phase_cells(
        coarse_cells,
        artifact_top_n=config.phase_artifact_top_n,
        primary_top_n=config.phase_primary_top_n,
    )
    fixed_composition_phases = (
        rank_phase_cells(
            fixed_cells,
            artifact_top_n=config.phase_artifact_top_n,
            primary_top_n=config.phase_primary_top_n,
        )
        if fixed_diagnostics["comparison_eligible"]
        else {}
    )
    return {
        "scope": {
            "num_tasks": len(tasks),
            "num_phase_rows": sum(
                len(task.pair.phases) for task in tasks.values()
            ),
            "num_episode_groups": sum(
                len(task.pair.group_keys) for task in tasks.values()
            ),
            "dict_size": int(
                next(iter(tasks.values())).pair.phase_w5.shape[1]
            ),
        },
        "oracle_phase_composition": {
            "comparison_eligible": fixed_diagnostics["comparison_eligible"],
            "diagnostics": fixed_diagnostics,
            "phases": fixed_composition_phases,
            "taxonomy": (
                classify_event_phase_candidates(
                    event_result,
                    fixed_composition_phases,
                    phase_primary_top_n=config.phase_primary_top_n,
                )
                if fixed_composition_phases
                else None
            ),
        },
        "event_ranked": event_result,
        "phase_aligned": {
            "fine": {
                "diagnostics": fine_diagnostics,
                "phases": fine_phases,
                "taxonomy": classify_event_phase_candidates(
                    event_result,
                    fine_phases,
                    phase_primary_top_n=config.phase_primary_top_n,
                ),
            },
            "coarse": {
                "diagnostics": coarse_diagnostics,
                "phases": coarse_phases,
                "taxonomy": classify_event_phase_candidates(
                    event_result,
                    coarse_phases,
                    phase_primary_top_n=config.phase_primary_top_n,
                ),
            },
        },
    }


def _score_spec(
    *,
    score_w4: str | Path,
    score_w5: str | Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    path_w4 = _canonical_path(score_w4)
    path_w5 = _canonical_path(score_w5)
    if not path_w4.is_file() or not path_w5.is_file():
        missing = [str(path) for path in (path_w4, path_w5) if not path.is_file()]
        raise FileNotFoundError(", ".join(missing))
    return {
        "score_w4": str(path_w4),
        "score_w4_sha256": sha256_file(path_w4),
        "score_w5": str(path_w5),
        "score_w5_sha256": sha256_file(path_w5),
        "checkpoint_sha256": str(checkpoint_sha256),
    }


def _v12_inventory(
    config: EventPhaseActivationGridConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    stage4_root = _canonical_path(config.stage4_root)
    paths = sorted(
        stage4_root.glob(
            "e*/cov*/analysis/task_local_phase_feature_ranking/summary.json"
        )
    )
    expected_count = config.expected_conditions * len(config.coverages)
    if len(paths) != expected_count:
        raise ValueError(
            f"Expected {expected_count} V12 summaries, found {len(paths)}."
        )
    cells = []
    inputs = []
    checkpoint_hashes: dict[str, set[str]] = {
        label: set() for label in config.sae_labels
    }
    condition_ids: set[str] = set()
    coverage_ids: set[str] = set()
    for path in paths:
        source = _load_json(path)
        condition_id = str(source.get("scope", {}).get("condition_id", ""))
        coverage = path.parents[2].name
        if not condition_id or condition_id != path.parents[3].name:
            raise ValueError(f"Condition identity mismatch: {path}")
        if coverage not in config.coverages:
            raise ValueError(f"Unexpected coverage {coverage!r}: {path}")
        condition_ids.add(condition_id)
        coverage_ids.add(coverage)
        runs = source.get("runs")
        if not isinstance(runs, dict) or set(runs) != set(config.sae_labels):
            raise ValueError(f"Unexpected SAE runs in {path}.")
        strict_scope = source.get("scope", {}).get(
            "matched_statistically_supported",
            {},
        )
        inputs.append(
            {
                "condition_id": condition_id,
                "condition_code": condition_id.split("_", 1)[0],
                "coverage": coverage,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "source_statistical_gate": {
                    "supported_count": int(strict_scope.get("count", 0)),
                    "family_cell_count": int(strict_scope.get("total", 0)),
                    "scope": "three-SAE matched, condition-local outer Holm",
                },
            }
        )
        for sae_label in config.sae_labels:
            run = runs[sae_label]
            spec = _score_spec(
                score_w4=run["score_w4"],
                score_w5=run["score_w5"],
                checkpoint_sha256=run["checkpoint_sha256"],
            )
            for key in ("score_w4_sha256", "score_w5_sha256"):
                expected_hash = str(run.get(key, ""))
                if expected_hash and spec[key] != expected_hash:
                    raise ValueError(f"{path}: {sae_label} {key} changed.")
            checkpoint_hashes[sae_label].add(spec["checkpoint_sha256"])
            cells.append(
                {
                    "cell_id": (
                        f"v12/{condition_id.split('_', 1)[0]}/"
                        f"{coverage}/{sae_label}"
                    ),
                    "source": "v12",
                    "condition_id": condition_id,
                    "condition_code": condition_id.split("_", 1)[0],
                    "coverage": coverage,
                    "sae_label": sae_label,
                    "score_pair": spec,
                }
            )
    if len(condition_ids) != config.expected_conditions:
        raise ValueError(
            f"Expected {config.expected_conditions} conditions, "
            f"found {len(condition_ids)}."
        )
    if coverage_ids != set(config.coverages):
        raise ValueError(
            f"Coverage set differs: {sorted(coverage_ids)}."
        )
    stable_hashes = {}
    for label, values in checkpoint_hashes.items():
        if len(values) != 1:
            raise ValueError(f"{label}: checkpoint SHA changes across V12.")
        stable_hashes[label] = next(iter(values))
    return cells, inputs, stable_hashes


def _oracle_inventory(
    config: EventPhaseActivationGridConfig,
    *,
    v12_checkpoint_hashes: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_path = _canonical_path(config.oracle_summary)
    source = _load_json(summary_path)
    runs = source.get("runs")
    if not isinstance(runs, dict):
        raise ValueError(f"Oracle summary has no runs: {summary_path}")
    cells = []
    mapped_labels: set[str] = set()
    for oracle_label, sae_label in ORACLE_RUN_ALIASES.items():
        if oracle_label not in runs or sae_label not in config.sae_labels:
            raise ValueError(f"Missing Oracle run mapping {oracle_label}.")
        run = runs[oracle_label]
        checkpoint_hash = str(
            run.get("checkpoint_sha256") or run.get("sae_sha256") or ""
        )
        if checkpoint_hash != v12_checkpoint_hashes[sae_label]:
            raise ValueError(
                f"{sae_label}: Oracle and V12 checkpoint SHA differ."
            )
        spec = _score_spec(
            score_w4=run["score_w4"],
            score_w5=run["score_w5"],
            checkpoint_sha256=checkpoint_hash,
        )
        cells.append(
            {
                "cell_id": f"oracle/{sae_label}",
                "source": "oracle",
                "condition_id": "simulator_phase_reference",
                "condition_code": "oracle",
                "coverage": None,
                "sae_label": sae_label,
                "oracle_run_label": oracle_label,
                "score_pair": spec,
            }
        )
        mapped_labels.add(sae_label)
    if mapped_labels != set(config.sae_labels):
        raise ValueError("Oracle SAE aliases do not cover the configured labels.")
    statistical_cells = []
    for oracle_label, sae_label in ORACLE_RUN_ALIASES.items():
        phases = runs[oracle_label].get("phase_results", {})
        for phase, result in sorted(phases.items()):
            statistical_cells.append(
                {
                    "sae_label": sae_label,
                    "phase": phase,
                    "best_feature": result.get("best_feature"),
                    "holm_p_within_four_phases": result.get(
                        "best_holm_p_across_four_phases"
                    ),
                    "holm_p_across_all_run_phase_cells": result.get(
                        "best_holm_p_across_all_run_phase_cells"
                    ),
                }
            )
    return cells, {
        "path": str(summary_path),
        "sha256": sha256_file(summary_path),
        "scope": source.get("scope"),
        "source_statistical_gate": statistical_cells,
    }


def _occurrence_rows(
    memberships: Iterable[tuple[int, str]],
    *,
    eligible_cell_ids: Iterable[str],
) -> list[dict[str, Any]]:
    eligible = sorted(set(eligible_cell_ids))
    member_cells: dict[int, set[str]] = {}
    for feature_id, cell_id in memberships:
        member_cells.setdefault(int(feature_id), set()).add(str(cell_id))
    rows = []
    for feature_id, cell_ids in member_cells.items():
        rows.append(
            {
                "feature_id": feature_id,
                "occurrence_count": len(cell_ids),
                "eligible_cell_count": len(eligible),
                "occurrence_fraction": (
                    float(len(cell_ids) / len(eligible)) if eligible else 0.0
                ),
                "cell_ids": sorted(cell_ids),
            }
        )
    return sorted(
        rows,
        key=lambda row: (-row["occurrence_count"], row["feature_id"]),
    )


def _recurrence_by_sae(v12_cells: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    sae_labels = sorted({cell["sae_label"] for cell in v12_cells})
    for sae_label in sae_labels:
        cells = [
            cell for cell in v12_cells if cell["sae_label"] == sae_label
        ]
        cell_ids = [cell["cell_id"] for cell in cells]
        event_memberships = [
            (feature_id, cell["cell_id"])
            for cell in cells
            for feature_id in cell["analysis"]["event_ranked"][
                "primary_top_feature_ids"
            ]
        ]
        phase_views: dict[str, Any] = {}
        for view in ("fine", "coarse"):
            phase_names = sorted(
                {
                    phase
                    for cell in cells
                    for phase in cell["analysis"]["phase_aligned"][view][
                        "phases"
                    ]
                }
            )
            phase_views[view] = {}
            for phase in phase_names:
                eligible_cells = [
                    cell
                    for cell in cells
                    if phase
                    in cell["analysis"]["phase_aligned"][view]["phases"]
                ]
                memberships = [
                    (feature_id, cell["cell_id"])
                    for cell in eligible_cells
                    for feature_id in cell["analysis"]["phase_aligned"][view][
                        "phases"
                    ][phase]["primary_top_feature_ids"]
                ]
                phase_views[view][phase] = _occurrence_rows(
                    memberships,
                    eligible_cell_ids=[
                        cell["cell_id"] for cell in eligible_cells
                    ],
                )
        output[sae_label] = {
            "event_ranked_top5": _occurrence_rows(
                event_memberships,
                eligible_cell_ids=cell_ids,
            ),
            "phase_aligned_top3": phase_views,
        }
    return output


def _jaccard(sets: list[set[int]]) -> dict[str, Any]:
    if not sets:
        return {
            "view_count": 0,
            "intersection_size": 0,
            "union_size": 0,
            "jaccard": None,
            "intersection_feature_ids": [],
        }
    intersection = set.intersection(*sets)
    union = set.union(*sets)
    return {
        "view_count": len(sets),
        "intersection_size": len(intersection),
        "union_size": len(union),
        "jaccard": float(len(intersection) / len(union)) if union else 1.0,
        "intersection_feature_ids": sorted(intersection),
    }


def _pairwise_overlap(sets: list[set[int]]) -> dict[str, Any]:
    pairs = list(combinations(sets, 2))
    intersections = [len(left & right) for left, right in pairs]
    return {
        "pair_count": len(pairs),
        "mean_intersection_size": (
            float(np.mean(intersections)) if intersections else None
        ),
        "min_intersection_size": min(intersections) if intersections else None,
        "max_intersection_size": max(intersections) if intersections else None,
    }


def _cell_primary_set(
    cell: dict[str, Any],
    *,
    view: str,
    phase: str | None = None,
) -> set[int]:
    if view == "event":
        return set(
            cell["analysis"]["event_ranked"]["primary_top_feature_ids"]
        )
    result = cell["analysis"]["phase_aligned"][view]["phases"].get(phase)
    return set(result["primary_top_feature_ids"]) if result else set()


def _nested_view_stability(
    v12_cells: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize nested coverage and shared-rollout condition sensitivity."""

    output: dict[str, Any] = {
        "coverage_within_condition": {},
        "condition_within_coverage": {},
    }
    sae_labels = sorted({cell["sae_label"] for cell in v12_cells})
    condition_codes = sorted({cell["condition_code"] for cell in v12_cells})
    coverages = sorted({cell["coverage"] for cell in v12_cells})
    by_key = {
        (cell["condition_code"], cell["coverage"], cell["sae_label"]): cell
        for cell in v12_cells
    }
    for sae_label in sae_labels:
        output["coverage_within_condition"][sae_label] = {}
        for condition in condition_codes:
            cells = [
                by_key[(condition, coverage, sae_label)]
                for coverage in coverages
            ]
            record: dict[str, Any] = {
                "event": {
                    **_jaccard(
                        [_cell_primary_set(cell, view="event") for cell in cells]
                    ),
                    **_pairwise_overlap(
                        [_cell_primary_set(cell, view="event") for cell in cells]
                    ),
                },
                "fine": {},
                "coarse": {},
            }
            for view in ("fine", "coarse"):
                phases = sorted(
                    {
                        phase
                        for cell in cells
                        for phase in cell["analysis"]["phase_aligned"][view][
                            "phases"
                        ]
                    }
                )
                for phase in phases:
                    eligible_sets = [
                        _cell_primary_set(cell, view=view, phase=phase)
                        for cell in cells
                        if phase
                        in cell["analysis"]["phase_aligned"][view]["phases"]
                    ]
                    record[view][phase] = {
                        **_jaccard(eligible_sets),
                        **_pairwise_overlap(eligible_sets),
                    }
            output["coverage_within_condition"][sae_label][condition] = record

        output["condition_within_coverage"][sae_label] = {}
        for coverage in coverages:
            cells = [
                by_key[(condition, coverage, sae_label)]
                for condition in condition_codes
            ]
            record = {
                "event": {
                    **_jaccard(
                        [_cell_primary_set(cell, view="event") for cell in cells]
                    ),
                    **_pairwise_overlap(
                        [_cell_primary_set(cell, view="event") for cell in cells]
                    ),
                },
                "fine": {},
                "coarse": {},
            }
            for view in ("fine", "coarse"):
                phases = sorted(
                    {
                        phase
                        for cell in cells
                        for phase in cell["analysis"]["phase_aligned"][view][
                            "phases"
                        ]
                    }
                )
                for phase in phases:
                    eligible_sets = [
                        _cell_primary_set(cell, view=view, phase=phase)
                        for cell in cells
                        if phase
                        in cell["analysis"]["phase_aligned"][view]["phases"]
                    ]
                    record[view][phase] = {
                        **_jaccard(eligible_sets),
                        **_pairwise_overlap(eligible_sets),
                    }
            output["condition_within_coverage"][sae_label][coverage] = record
    return output


def _mean(values: list[int]) -> float | None:
    return float(np.mean(values)) if values else None


def _stability_headlines(
    v12_cells: list[dict[str, Any]],
    *,
    event_primary_top_n: int,
) -> dict[str, Any]:
    """Return compact W4/W5 and pairwise coverage overlap summaries."""

    output: dict[str, Any] = {}
    sae_labels = sorted({cell["sae_label"] for cell in v12_cells})
    coverages = sorted({cell["coverage"] for cell in v12_cells})
    coverage_pairs = list(combinations(coverages, 2))
    for sae_label in sae_labels:
        cells = [
            cell for cell in v12_cells if cell["sae_label"] == sae_label
        ]
        event_w4_w5 = []
        for cell in cells:
            rankings = cell["analysis"]["event_ranked"]["top_feature_ids"]
            event_w4_w5.append(
                len(
                    set(rankings["w4"][:event_primary_top_n])
                    & set(rankings["w5"][:event_primary_top_n])
                )
            )
        by_key = {
            (cell["condition_code"], cell["coverage"]): cell
            for cell in cells
        }
        conditions = sorted({cell["condition_code"] for cell in cells})
        coverage_results: dict[str, Any] = {}
        all_pair_values = {
            "event": [],
            "fine": [],
            "coarse": [],
        }
        for left_coverage, right_coverage in coverage_pairs:
            event_overlap: list[int] = []
            phase_overlap: dict[str, list[int]] = {
                "fine": [],
                "coarse": [],
            }
            for condition in conditions:
                left = by_key[(condition, left_coverage)]
                right = by_key[(condition, right_coverage)]
                event_overlap.append(
                    len(
                        _cell_primary_set(left, view="event")
                        & _cell_primary_set(right, view="event")
                    )
                )
                for view in ("fine", "coarse"):
                    left_phases = left["analysis"]["phase_aligned"][view][
                        "phases"
                    ]
                    right_phases = right["analysis"]["phase_aligned"][view][
                        "phases"
                    ]
                    for phase in sorted(set(left_phases) & set(right_phases)):
                        phase_overlap[view].append(
                            len(
                                _cell_primary_set(
                                    left,
                                    view=view,
                                    phase=phase,
                                )
                                & _cell_primary_set(
                                    right,
                                    view=view,
                                    phase=phase,
                                )
                            )
                        )
            key = f"{left_coverage}__{right_coverage}"
            coverage_results[key] = {
                "event_top5_mean_intersection": _mean(event_overlap),
                "event_comparison_count": len(event_overlap),
                "fine_phase_top3_mean_intersection": _mean(
                    phase_overlap["fine"]
                ),
                "fine_phase_comparison_count": len(phase_overlap["fine"]),
                "coarse_phase_top3_mean_intersection": _mean(
                    phase_overlap["coarse"]
                ),
                "coarse_phase_comparison_count": len(
                    phase_overlap["coarse"]
                ),
            }
            all_pair_values["event"].extend(event_overlap)
            all_pair_values["fine"].extend(phase_overlap["fine"])
            all_pair_values["coarse"].extend(phase_overlap["coarse"])
        output[sae_label] = {
            "event_w4_w5_top5_mean_intersection": _mean(event_w4_w5),
            "event_w4_w5_cell_count": len(event_w4_w5),
            "coverage_pairs": coverage_results,
            "all_coverage_pairs": {
                "event_top5_mean_intersection": _mean(
                    all_pair_values["event"]
                ),
                "fine_phase_top3_mean_intersection": _mean(
                    all_pair_values["fine"]
                ),
                "coarse_phase_top3_mean_intersection": _mean(
                    all_pair_values["coarse"]
                ),
            },
        }
    return output


def _oracle_reference_comparison(
    oracle_cells: list[dict[str, Any]],
    recurrence: dict[str, Any],
) -> dict[str, Any]:
    output = {}
    for cell in oracle_cells:
        sae_label = cell["sae_label"]
        event_recurrence = {
            row["feature_id"]: row
            for row in recurrence[sae_label]["event_ranked_top5"]
        }
        event_rows = []
        for feature_id in cell["analysis"]["event_ranked"][
            "primary_top_feature_ids"
        ]:
            row = event_recurrence.get(int(feature_id))
            event_rows.append(
                {
                    "feature_id": int(feature_id),
                    "v12_occurrence_count": (
                        int(row["occurrence_count"]) if row else 0
                    ),
                    "v12_eligible_cell_count": 15,
                }
            )
        event_overlap_mean = float(
            sum(row["v12_occurrence_count"] for row in event_rows) / 15
        )
        views = {}
        view_overlap = {}
        for view in ("fine", "coarse"):
            views[view] = {}
            v12_phases = recurrence[sae_label]["phase_aligned_top3"][view]
            for phase, result in cell["analysis"]["phase_aligned"][view][
                "phases"
            ].items():
                lookup = {
                    row["feature_id"]: row
                    for row in v12_phases.get(phase, [])
                }
                views[view][phase] = [
                    {
                        "feature_id": int(feature_id),
                        "v12_same_phase_occurrence_count": int(
                            lookup.get(int(feature_id), {}).get(
                                "occurrence_count",
                                0,
                            )
                        ),
                        "v12_same_phase_eligible_cell_count": (
                            int(
                                next(iter(lookup.values()))[
                                    "eligible_cell_count"
                                ]
                            )
                            if lookup
                            else 0
                        ),
                    }
                    for feature_id in result["primary_top_feature_ids"]
                ]
            overlap_sum = sum(
                row["v12_same_phase_occurrence_count"]
                for rows in views[view].values()
                for row in rows
            )
            eligible_sum = sum(
                rows[0]["v12_same_phase_eligible_cell_count"]
                for rows in views[view].values()
                if rows and rows[0]["v12_same_phase_eligible_cell_count"] > 0
            )
            view_overlap[view] = {
                "mean_oracle_top3_features_present_per_eligible_v12_cell": (
                    float(overlap_sum / eligible_sum)
                    if eligible_sum
                    else None
                ),
                "overlap_membership_count": overlap_sum,
                "eligible_v12_cell_count": eligible_sum,
            }
        output[sae_label] = {
            "cell_id": cell["cell_id"],
            "event_ranked_top5": event_rows,
            "event_overlap": {
                "mean_oracle_top5_features_present_per_v12_cell": (
                    event_overlap_mean
                ),
                "overlap_membership_count": sum(
                    row["v12_occurrence_count"] for row in event_rows
                ),
                "v12_cell_count": 15,
            },
            "phase_aligned_top3": views,
            "phase_overlap": view_overlap,
        }
    return output


def _oracle_fixed_composition_comparison(
    v12_cells: list[dict[str, Any]],
    oracle_cells: list[dict[str, Any]],
    *,
    phase_primary_top_n: int,
) -> dict[str, Any]:
    """Compare only suite-complete Oracle four-phase panels.

    The score contrast remains exact-instruction local. A V12 cell is excluded
    when any Oracle phase is absent or cannot form a task-local contrast.
    """

    oracle_by_sae = {cell["sae_label"]: cell for cell in oracle_cells}
    by_sae: dict[str, Any] = {}
    fixed_slot_count = len(ORACLE_FINE_PHASES) * phase_primary_top_n
    for sae_label in sorted(oracle_by_sae):
        oracle_cell = oracle_by_sae[sae_label]
        oracle_panel = oracle_cell["analysis"]["oracle_phase_composition"]
        if not oracle_panel["comparison_eligible"]:
            raise ValueError(
                f"{sae_label}: Oracle reference lacks its fixed composition."
            )
        oracle_phases = oracle_panel["phases"]
        source_cells = sorted(
            [
                cell
                for cell in v12_cells
                if cell["sae_label"] == sae_label
            ],
            key=lambda cell: (cell["condition_code"], cell["coverage"]),
        )
        eligible_cells = []
        excluded_cells = []
        condition_rows: dict[str, dict[str, Any]] = {}
        for cell in source_cells:
            panel = cell["analysis"]["oracle_phase_composition"]
            diagnostics = panel["diagnostics"]
            if not panel["comparison_eligible"]:
                excluded_cells.append(
                    {
                        "cell_id": cell["cell_id"],
                        "condition_code": cell["condition_code"],
                        "coverage": cell["coverage"],
                        "missing_phases": diagnostics["missing_phases"],
                        "uncontrastable_phases": diagnostics[
                            "uncontrastable_phases"
                        ],
                    }
                )
                continue
            phase_rows = {}
            total_overlap = 0
            total_candidate_count = 0
            for phase in ORACLE_FINE_PHASES:
                v12_result = panel["phases"][phase]
                oracle_result = oracle_phases[phase]
                v12_ids = [
                    int(feature_id)
                    for feature_id in v12_result[
                        "primary_top_feature_ids"
                    ]
                ]
                oracle_ids = [
                    int(feature_id)
                    for feature_id in oracle_result[
                        "primary_top_feature_ids"
                    ]
                ]
                overlap = sorted(set(v12_ids) & set(oracle_ids))
                total_overlap += len(overlap)
                total_candidate_count += len(v12_ids)
                phase_rows[phase] = {
                    "v12_top_feature_ids": v12_ids,
                    "oracle_top_feature_ids": oracle_ids,
                    "overlap_feature_ids": overlap,
                    "overlap_count": len(overlap),
                    "supporting_instruction_count": int(
                        v12_result["instruction_count"]
                    ),
                }
            cell_row = {
                "cell_id": cell["cell_id"],
                "condition_code": cell["condition_code"],
                "coverage": cell["coverage"],
                "exact_instruction_complete_count": diagnostics[
                    "exact_instruction_complete_count"
                ],
                "contrastable_task_count": diagnostics[
                    "num_contrastable_tasks"
                ],
                "candidate_membership_count": total_candidate_count,
                "fixed_candidate_slot_count": fixed_slot_count,
                "overlap_membership_count": total_overlap,
                "overlap_per_fixed_slot": float(
                    total_overlap / fixed_slot_count
                ),
                "phases": phase_rows,
            }
            eligible_cells.append(cell_row)
            condition = condition_rows.setdefault(
                cell["condition_code"],
                {
                    "eligible_cell_count": 0,
                    "overlap_membership_count": 0,
                    "fixed_candidate_slot_count": 0,
                    "cell_ids": [],
                },
            )
            condition["eligible_cell_count"] += 1
            condition["overlap_membership_count"] += total_overlap
            condition["fixed_candidate_slot_count"] += fixed_slot_count
            condition["cell_ids"].append(cell["cell_id"])
        for condition_code in sorted(
            {cell["condition_code"] for cell in source_cells}
        ):
            condition = condition_rows.setdefault(
                condition_code,
                {
                    "eligible_cell_count": 0,
                    "overlap_membership_count": 0,
                    "fixed_candidate_slot_count": 0,
                    "cell_ids": [],
                },
            )
            condition["excluded_cell_count"] = sum(
                row["condition_code"] == condition_code
                for row in excluded_cells
            )
            eligible_count = int(condition["eligible_cell_count"])
            condition["mean_overlap_memberships_per_eligible_cell"] = (
                float(
                    condition["overlap_membership_count"]
                    / eligible_count
                )
                if eligible_count
                else None
            )
            slots = int(condition["fixed_candidate_slot_count"])
            condition["overlap_per_fixed_slot"] = (
                float(condition["overlap_membership_count"] / slots)
                if slots
                else None
            )
        by_sae[sae_label] = {
            "oracle_cell_id": oracle_cell["cell_id"],
            "eligible_v12_cell_count": len(eligible_cells),
            "excluded_v12_cell_count": len(excluded_cells),
            "eligible_cells": eligible_cells,
            "excluded_cells": excluded_cells,
            "conditions": dict(sorted(condition_rows.items())),
        }

    unique_v12_panels: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in v12_cells:
        key = (cell["condition_code"], cell["coverage"])
        panel = cell["analysis"]["oracle_phase_composition"]
        diagnostics = panel["diagnostics"]
        unique_v12_panels.setdefault(
            key,
            {
                "condition_code": cell["condition_code"],
                "coverage": cell["coverage"],
                "comparison_eligible": panel["comparison_eligible"],
                "missing_phases": diagnostics["missing_phases"],
                "uncontrastable_phases": diagnostics[
                    "uncontrastable_phases"
                ],
                "exact_instruction_complete_count": diagnostics[
                    "exact_instruction_complete_count"
                ],
            },
        )
    return {
        "target_phases": list(ORACLE_FINE_PHASES),
        "comparison_unit": (
            "suite-complete phase vocabulary with exact-instruction-local "
            "contrasts"
        ),
        "missing_phase_policy": "exclude comparison cell; never impute",
        "fixed_candidate_slots_per_cell": fixed_slot_count,
        "strict_exact_instruction_match": {
            "v12_condition_coverage_count": sum(
                row["exact_instruction_complete_count"] > 0
                for row in unique_v12_panels.values()
            ),
            "v12_condition_coverage_total": len(unique_v12_panels),
            "oracle_score_pair_count": sum(
                cell["analysis"]["oracle_phase_composition"]["diagnostics"][
                    "exact_instruction_complete_count"
                ]
                > 0
                for cell in oracle_cells
            ),
            "status": "unavailable_for_v12",
            "reason": (
                "No V12 exact instruction observes all four Oracle phases."
            ),
        },
        "v12_condition_coverage_panels": [
            unique_v12_panels[key] for key in sorted(unique_v12_panels)
        ],
        "by_sae": by_sae,
    }


def _canonical_ranking_fingerprint(
    cells: list[dict[str, Any]],
) -> str:
    """Hash only canonical primary feature-ID mappings for regression checks."""

    mapping = {}
    for cell in sorted(cells, key=lambda row: row["cell_id"]):
        analysis = cell["analysis"]
        key = (
            f"{cell['condition_id']}/{cell['coverage']}/{cell['sae_label']}"
            if cell["source"] == "v12"
            else str(cell["sae_label"])
        )
        mapping[key] = {
            "event": analysis["event_ranked"]["primary_top_feature_ids"],
            "fine": {
                phase: result["primary_top_feature_ids"]
                for phase, result in sorted(
                    analysis["phase_aligned"]["fine"]["phases"].items()
                )
            },
            "coarse": {
                phase: result["primary_top_feature_ids"]
                for phase, result in sorted(
                    analysis["phase_aligned"]["coarse"]["phases"].items()
                )
            },
        }
    encoded = json.dumps(
        mapping,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _audit_rows() -> list[dict[str, str]]:
    return [
        {
            "gate": "Length",
            "status": "FAIL",
            "evidence": (
                "score row는 episode-balanced지만 upstream recurring-cluster "
                "선정에서 긴 episode가 더 많은 선택 기회를 가진다."
            ),
        },
        {
            "gate": "Task identity",
            "status": "PASS-local / FAIL-general",
            "evidence": (
                "phase contrast는 exact instruction 내부이고 event score는 "
                "instruction macro 평균이다. 다만 Oracle은 단일 task이며 "
                "여러 V12 phase label도 supporting instruction이 하나다."
            ),
        },
        {
            "gate": "Instruction balance",
            "status": "PASS / N/A-Oracle",
            "evidence": (
                "V12 exact instruction은 동일한 macro weight를 가진다. "
                "Oracle은 instruction이 하나라 balance를 평가할 수 없다."
            ),
        },
        {
            "gate": "In-sample rescue",
            "status": "FAIL",
            "evidence": (
                "동일 observation에서 후보를 골랐고 별도 held-out Hooked-SR "
                "intervention 결과는 아직 사용하지 않았다."
            ),
        },
        {
            "gate": "Rollout pooling",
            "status": "PASS",
            "evidence": (
                "source score는 selected event를 episode group에서 먼저 "
                "평균한 뒤 phase와 exact-instruction macro 집계를 한다."
            ),
        },
        {
            "gate": "Phase / dwell",
            "status": "FAIL",
            "evidence": (
                "dwell, retry, progress, success, phase prevalence가 matched "
                "되지 않았고 mean-other comparator도 관측 phase 구성에 따라 "
                "바뀐다."
            ),
        },
        {
            "gate": "Oracle phase composition",
            "status": "FAIL-strict / PASS-suite-filter",
            "evidence": (
                "V12 exact instruction 중 Oracle 4-phase를 모두 가진 경우는 "
                "없다. suite에서 네 phase가 모두 관측되고 task-local "
                "contrast가 가능한 셀만 남기며 missing phase는 보간하지 않는다."
            ),
        },
        {
            "gate": "Observation != causation",
            "status": "PASS-boundary",
            "evidence": (
                "산출물을 descriptive candidate로만 명명하고 causal 또는 "
                "success-rate 효과를 주장하지 않는다."
            ),
        },
        {
            "gate": "Scene-local != general",
            "status": "FAIL",
            "evidence": (
                "V12는 두 task family의 5개 cell이고 Oracle은 8 episode의 "
                "단일 simulator-labeled cell이다."
            ),
        },
        {
            "gate": "Oracle/V12 score compatibility",
            "status": "FAIL",
            "evidence": (
                "checkpoint coordinate는 같지만 phase ontology, event-step "
                "scale, anchor, dwell, template 집계가 다르다. score를 pooling "
                "하거나 magnitude로 직접 비교하지 않았다."
            ),
        },
    ]


def _format_recurrence(rows: list[dict[str, Any]], limit: int = 6) -> str:
    if not rows:
        return "없음"
    return ", ".join(
        f"F{row['feature_id']} {row['occurrence_count']}/"
        f"{row['eligible_cell_count']}"
        for row in rows[:limit]
    )


def _render_report(summary: dict[str, Any]) -> str:
    scope = summary["scope"]
    recurrence = summary["v12_recurrence"]
    statistical_gate = summary["statistical_gate_overlay"]
    v12_gate = statistical_gate["v12_three_sae_matched_outer_holm"]
    oracle_within_gate = statistical_gate[
        "oracle_within_run_four_phase_holm"
    ]
    oracle_all_gate = statistical_gate["oracle_all_run_phase_cell_holm"]
    stability = summary["stability_headlines"]
    fixed_comparison = summary["oracle_fixed_phase_comparison"]
    fixed_10k = fixed_comparison["by_sae"]["sae10k"]
    strict_phase_match = fixed_comparison["strict_exact_instruction_match"]
    primary_cell = next(
        cell
        for cell in summary["v12_cells"]
        if cell["cell_id"] == "v12/e3/cov0p3/sae10k"
    )
    lines = [
        "# Event / phase feature 전수 조사",
        "",
        "## 1. 결과 수치와 범위",
        "",
        (
            f"- V12: **{scope['v12_score_pair_count']} score pairs** "
            f"(E0–E4 × coverage .3/.4/.5 × SAE 3종)"
        ),
        (
            f"- Oracle: **{scope['oracle_score_pair_count']} score pairs** "
            "(coverage 축 없음, 별도 reference panel)"
        ),
        (
            f"- 총 **{scope['score_pair_count']} base cells**, "
            f"fine/coarse phase ontology view "
            f"**{scope['phase_ontology_view_count']}개**"
        ),
        (
            f"- V12 phase-label ranking: fine "
            f"{scope['v12_fine_phase_ranking_count']}개, coarse "
            f"{scope['v12_coarse_phase_ranking_count']}개"
        ),
        (
            f"- Oracle phase-label ranking: fine "
            f"{scope['oracle_fine_phase_ranking_count']}개, coarse "
            f"{scope['oracle_coarse_phase_ranking_count']}개"
        ),
        "",
        (
            "Event와 phase는 배타적 의미 범주가 아니다. Event는 W5 raw "
            "event score의 instruction-balanced Top-5, phase는 같은 "
            "instruction 안의 phase-minus-mean-other W4/W5 양수·과반 "
            "Top-3 후보이다."
        ),
        "",
        "### 원본 통계 gate와 완화 ranking의 관계",
        "",
        (
            "- V12 원본 three-SAE matched + condition-local outer-Holm: "
            f"**{v12_gate['supported_count']}/"
            f"{v12_gate['family_cell_count']}**"
        ),
        (
            "- Oracle within-run 4-phase Holm: "
            f"**{oracle_within_gate['supported_count']}/"
            f"{oracle_within_gate['family_cell_count']}**, "
            "all-run phase-cell Holm: "
            f"**{oracle_all_gate['supported_count']}/"
            f"{oracle_all_gate['family_cell_count']}**"
        ),
        "",
        (
            "통계 gate는 overlay로 보존했지만 descriptive 후보의 exclusion "
            "gate로 쓰지 않았다. 즉 strict 결과가 0이라고 feature 발화를 "
            "없다고 결론내리지 않고, Hooked-SR 후보를 관찰할 기회를 유지했다."
        ),
        "",
        "### V12 event-ranked Top-5 재현 빈도",
        "",
        "| SAE | 15-cell recurrence |",
        "|---|---|",
    ]
    for sae_label in summary["method"]["sae_labels"]:
        lines.append(
            f"| {sae_label} | "
            f"{_format_recurrence(recurrence[sae_label]['event_ranked_top5'])} |"
        )

    lines.extend(
        [
            "",
            "### SAE와 coverage sensitivity",
            "",
            (
                "값은 feature score가 아니라 shortlist 교집합의 평균 크기다. "
                "Event 분모는 5, phase 분모는 3이다."
            ),
            "",
            (
                "| SAE | Event W4↔W5 | Event cov.3↔.4 / .3↔.5 | "
                "Fine cov.3↔.4 / .3↔.5 | "
                "Coarse cov.3↔.4 / .3↔.5 |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for sae_label in summary["method"]["sae_labels"]:
        row = stability[sae_label]
        cov34 = row["coverage_pairs"]["cov0p3__cov0p4"]
        cov35 = row["coverage_pairs"]["cov0p3__cov0p5"]
        lines.append(
            f"| {sae_label} | "
            f"{row['event_w4_w5_top5_mean_intersection']:.2f}/5 | "
            f"{cov34['event_top5_mean_intersection']:.2f} / "
            f"{cov35['event_top5_mean_intersection']:.2f} | "
            f"{cov34['fine_phase_top3_mean_intersection']:.2f} / "
            f"{cov35['fine_phase_top3_mean_intersection']:.2f} | "
            f"{cov34['coarse_phase_top3_mean_intersection']:.2f} / "
            f"{cov35['coarse_phase_top3_mean_intersection']:.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "Event W4/W5 안정성은 sae10k_bs8192가 가장 높고, phase "
                "coverage 안정성은 sae1p2k가 근소하게 가장 높다. sae10k는 "
                "반복 후보가 선명하지만 phase coverage overlap 자체의 1위는 "
                "아니다. 모든 SAE에서 phase .3↔.4가 .3↔.5보다 안정적이므로 "
                "`.3=discovery`, `.4=stability`, `.5=sparse sensitivity`로 "
                "보는 것이 가장 일관된다."
            ),
        ]
    )

    lines.extend(
        [
            "",
            "### V12 coarse phase-aligned Top-3 재현 빈도",
            "",
            "| SAE | reach | grasp | transport | terminal |",
            "|---|---|---|---|---|",
        ]
    )
    for sae_label in summary["method"]["sae_labels"]:
        phases = recurrence[sae_label]["phase_aligned_top3"]["coarse"]
        values = [
            _format_recurrence(phases.get(phase, []), limit=3)
            for phase in COARSE_PHASE_ORDER
        ]
        lines.append(f"| {sae_label} | " + " | ".join(values) + " |")

    primary_analysis = primary_cell["analysis"]
    primary_coarse = primary_analysis["phase_aligned"]["coarse"]
    primary_taxonomy = primary_coarse["taxonomy"]
    primary_event_text = ", ".join(
        f"F{feature_id}"
        for feature_id in primary_analysis["event_ranked"][
            "primary_top_feature_ids"
        ]
    )
    lines.extend(
        [
            "",
            "### Hooked-SR용 primary diagnostic arm",
            "",
            (
                "기존 선택인 `E3 / cov.3 / sae10k`를 바꾸지 않고 보면 "
                f"event Top-5는 {primary_event_text}이다."
            ),
            "",
            "| Coarse phase | Top-3 |",
            "|---|---|",
        ]
    )
    for phase in COARSE_PHASE_ORDER:
        result = primary_coarse["phases"].get(phase)
        feature_text = (
            ", ".join(
                f"F{feature_id}"
                for feature_id in result["primary_top_feature_ids"]
            )
            if result
            else "없음"
        )
        lines.append(f"| {phase} | {feature_text} |")
    lines.extend(
        [
            "",
            (
                f"이 12개 phase membership 중 event Top-5와 겹치는 "
                f"dual-ranked 후보는 "
                f"{', '.join(f'F{x}' for x in primary_taxonomy['dual_ranked']) or '없음'}"
                f"이고, phase-aligned only는 "
                f"{len(primary_taxonomy['phase_aligned_only'])}개다. "
                "dual-ranked를 버리지는 말고 broad-event control arm으로 "
                "분리해야 한다."
            ),
        ]
    )

    lines.extend(
        [
            "",
            "### Fine phase 재현 후보",
            "",
        ]
    )
    for sae_label in summary["method"]["sae_labels"]:
        lines.append(f"#### {sae_label}")
        lines.append("")
        fine = recurrence[sae_label]["phase_aligned_top3"]["fine"]
        for phase, rows in fine.items():
            lines.append(f"- {phase}: {_format_recurrence(rows, limit=4)}")
        lines.append("")

    lines.extend(
        [
            "### Oracle simulator-labeled reference",
            "",
            (
                "Oracle는 feature-ID coordinate가 같은지 checkpoint SHA로 "
                "검증했지만 V12 recurrence에 합산하지 않았다. 아래 분수는 "
                "Oracle 후보가 동일 SAE의 V12 15개 cell에 등장한 횟수다."
            ),
            "",
            "| SAE | Oracle event Top-5 → V12 recurrence |",
            "|---|---|",
        ]
    )
    for sae_label in summary["method"]["sae_labels"]:
        rows = summary["oracle_reference"][sae_label]["event_ranked_top5"]
        value = ", ".join(
            f"F{row['feature_id']} {row['v12_occurrence_count']}/15"
            for row in rows
        )
        lines.append(f"| {sae_label} | {value} |")
    lines.extend(
        [
            "",
            "| SAE | Event direct overlap /5 | Fine same-phase /3 | "
            "Coarse same-phase /3 |",
            "|---|---:|---:|---:|",
        ]
    )
    for sae_label in summary["method"]["sae_labels"]:
        reference = summary["oracle_reference"][sae_label]
        event_overlap = reference["event_overlap"][
            "mean_oracle_top5_features_present_per_v12_cell"
        ]
        fine_overlap = reference["phase_overlap"]["fine"][
            "mean_oracle_top3_features_present_per_eligible_v12_cell"
        ]
        coarse_overlap = reference["phase_overlap"]["coarse"][
            "mean_oracle_top3_features_present_per_eligible_v12_cell"
        ]
        lines.append(
            f"| {sae_label} | "
            f"{event_overlap:.2f} | {fine_overlap:.3f} | "
            f"{coarse_overlap:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "Oracle–V12의 same-phase overlap은 `summary.json`의 "
                "`oracle_reference.*.phase_aligned_top3`에 phase별로 "
                "전부 기록했다. 낮은 overlap은 task·ontology·anchor 차이와 "
                "분리되지 않으므로 feature 부재 증거가 아니다."
            ),
            "",
            "### Oracle 4-phase composition 고정 비교",
            "",
            (
                "비교 phase를 `reach-to-object`, `grasp`, `place`, "
                "`insert-settle` 네 개로 고정했다. 하나라도 없거나 "
                "task-local contrast가 불가능한 셀은 제외했고 missing phase를 "
                "보간하지 않았다."
            ),
            "",
            (
                f"네 phase를 한 exact instruction에서 모두 관측한 V12 "
                f"condition×coverage는 "
                f"**{strict_phase_match['v12_condition_coverage_count']}/"
                f"{strict_phase_match['v12_condition_coverage_total']}**다. "
                "따라서 strict Oracle-equivalent comparison은 불가능하고, "
                "아래는 suite-complete이면서 contrast는 instruction-local인 "
                "보수적 근사다."
            ),
            "",
            "| 10k 조건 | eligible coverage /3 | Oracle overlap /12 |",
            "|---|---:|---:|",
        ]
    )
    for condition_code, condition in fixed_10k["conditions"].items():
        mean_overlap = condition[
            "mean_overlap_memberships_per_eligible_cell"
        ]
        overlap_text = (
            f"{mean_overlap:.2f}" if mean_overlap is not None else "N/A"
        )
        lines.append(
            f"| {condition_code.upper()} | "
            f"{condition['eligible_cell_count']}/3 | {overlap_text} |"
        )
    lines.extend(
        [
            "",
            (
                "구성을 고정하면 10k에서 E3만 세 coverage가 모두 eligible이고 "
                "평균 overlap도 가장 높다. E0는 2개, E4는 1개 coverage만 "
                "eligible이며 E1/E2는 `insert-settle` 부재로 제외된다. 따라서 "
                "자유 phase 구성에서 보였던 E4 우세를 최종 결론으로 쓰지 않는다."
            ),
            "",
            "## 2. Confound audit",
            "",
            "| Gate | 판정 | 근거 |",
            "|---|---|---|",
        ]
    )
    for row in summary["confound_audit"]:
        lines.append(
            f"| {row['gate']} | **{row['status']}** | {row['evidence']} |"
        )
    lines.extend(
        [
            "",
            "## 3. Claim strength",
            "",
            "**diagnostic evidence**",
            "",
            (
                "- 현재 결과는 Hooked-SR 실험 후보를 정하는 관측 기반 "
                "ranking이다. phase 의미의 확인이나 인과 효과 검증이 아니다."
            ),
            (
                "- E0–E4와 coverage는 같은 rollout과 nested selection을 "
                "재사용하므로 recurrence는 기술 통계이며 독립 반복 수가 아니다."
            ),
            (
                "- Fine/coarse와 W4/W5도 같은 score pair의 sensitivity "
                "view이지 독립 검증이 아니다."
            ),
            "",
            "## 4. 판정 보류 주장",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in summary["held_claims"])
    lines.extend(
        [
            "",
            "## 산출물 사용법",
            "",
            (
                "이 문서는 반복 후보만 압축했다. 48개 셀 각각의 W4/W5/mean "
                "event Top-10, fine/coarse phase Top-10, MECE membership, "
                "coverage·condition sensitivity와 입력 SHA는 `summary.json`에 "
                "있다."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def analyze_event_phase_activation_grid(
    config: EventPhaseActivationGridConfig,
) -> dict[str, Any]:
    """Analyze every V12 and Oracle score pair and write two immutable files."""

    output_dir = config.output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite event/phase grid output: {output_dir}"
        )
    if not (
        config.event_artifact_top_n
        >= config.event_primary_top_n
        > 0
    ):
        raise ValueError("Invalid event shortlist sizes.")
    if not (
        config.phase_artifact_top_n
        >= config.phase_primary_top_n
        > 0
    ):
        raise ValueError("Invalid phase shortlist sizes.")

    v12_cells, v12_inputs, checkpoint_hashes = _v12_inventory(config)
    oracle_cells, oracle_input = _oracle_inventory(
        config,
        v12_checkpoint_hashes=checkpoint_hashes,
    )
    for cell in [*v12_cells, *oracle_cells]:
        score_pair = cell["score_pair"]
        cell["analysis"] = _analyze_score_pair(
            Path(score_pair["score_w4"]),
            Path(score_pair["score_w5"]),
            config=config,
        )

    recurrence = _recurrence_by_sae(v12_cells)
    oracle_reference = _oracle_reference_comparison(
        oracle_cells,
        recurrence,
    )
    oracle_fixed_comparison = _oracle_fixed_composition_comparison(
        v12_cells,
        oracle_cells,
        phase_primary_top_n=config.phase_primary_top_n,
    )
    v12_strict_supported = sum(
        item["source_statistical_gate"]["supported_count"]
        for item in v12_inputs
    )
    v12_strict_family_cells = sum(
        item["source_statistical_gate"]["family_cell_count"]
        for item in v12_inputs
    )
    oracle_statistical_cells = oracle_input["source_statistical_gate"]
    oracle_within_run_supported = [
        row
        for row in oracle_statistical_cells
        if row["holm_p_within_four_phases"] is not None
        and float(row["holm_p_within_four_phases"]) <= 0.05
    ]
    oracle_all_run_supported = [
        row
        for row in oracle_statistical_cells
        if row["holm_p_across_all_run_phase_cells"] is not None
        and float(row["holm_p_across_all_run_phase_cells"]) <= 0.05
    ]
    v12_fine_count = sum(
        len(cell["analysis"]["phase_aligned"]["fine"]["phases"])
        for cell in v12_cells
    )
    v12_coarse_count = sum(
        len(cell["analysis"]["phase_aligned"]["coarse"]["phases"])
        for cell in v12_cells
    )
    oracle_fine_count = sum(
        len(cell["analysis"]["phase_aligned"]["fine"]["phases"])
        for cell in oracle_cells
    )
    oracle_coarse_count = sum(
        len(cell["analysis"]["phase_aligned"]["coarse"]["phases"])
        for cell in oracle_cells
    )
    implementation_path = Path(__file__).resolve()
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_strength": "diagnostic_evidence",
        "scope": {
            "v12_score_pair_count": len(v12_cells),
            "oracle_score_pair_count": len(oracle_cells),
            "score_pair_count": len(v12_cells) + len(oracle_cells),
            "phase_ontology_view_count": 2
            * (len(v12_cells) + len(oracle_cells)),
            "v12_fine_phase_ranking_count": v12_fine_count,
            "v12_coarse_phase_ranking_count": v12_coarse_count,
            "oracle_fine_phase_ranking_count": oracle_fine_count,
            "oracle_coarse_phase_ranking_count": oracle_coarse_count,
            "checkpoint_coordinate_match": True,
        },
        "method": {
            "sae_labels": list(config.sae_labels),
            "coverages": list(config.coverages),
            "event_primary": (
                "mean W5 phase-row score within each exact instruction, then "
                "equal mean across exact instructions"
            ),
            "event_sensitivity_views": [
                "same task-balanced ranking for W4",
                "equal W4/W5 task-balanced mean",
            ],
            "phase_contrast": (
                "target phase minus mean of other phases observed in the same "
                "exact instruction"
            ),
            "phase_candidate_requirements": [
                "suite mean margin W4 > 0",
                "suite mean margin W5 > 0",
                "combined W4/W5 margin > 0 in a strict instruction majority",
            ],
            "event_artifact_top_n": config.event_artifact_top_n,
            "event_primary_top_n": config.event_primary_top_n,
            "phase_artifact_top_n": config.phase_artifact_top_n,
            "phase_primary_top_n": config.phase_primary_top_n,
            "discovery_non_gates": [
                "max-T p-value",
                "Holm correction",
                "three-SAE intersection",
                "task-mean rank",
                "window-mean rank or sign",
                "coverage intersection",
                "condition intersection",
            ],
            "recurrence_contract": (
                "descriptive occurrence over shared-rollout conditions and "
                "nested coverage views; not an independent replication count"
            ),
            "oracle_contract": (
                "separate simulator-labeled temporal-anchor reference; never "
                "pooled with V12 recurrence or compared by score magnitude"
            ),
            "oracle_fixed_phase_composition": {
                "target_phases": list(ORACLE_FINE_PHASES),
                "missing_phase_policy": (
                    "exclude comparison cell; never impute"
                ),
                "contrast_scope": (
                    "exact instruction local; suite completeness is not an "
                    "exact four-phase instruction match"
                ),
            },
        },
        "inputs": {
            "stage4_root": str(_canonical_path(config.stage4_root)),
            "v12_condition_coverage_summaries": v12_inputs,
            "oracle_summary": oracle_input,
            "checkpoint_sha256_by_sae": checkpoint_hashes,
            "implementation": {
                "path": str(implementation_path),
                "sha256": sha256_file(implementation_path),
            },
            "entrypoint": (
                {
                    "path": str(config.entrypoint.resolve()),
                    "sha256": sha256_file(config.entrypoint.resolve()),
                }
                if config.entrypoint is not None
                else None
            ),
        },
        "regression_fingerprints": {
            "v12_primary_rankings_sha256": _canonical_ranking_fingerprint(
                v12_cells
            ),
            "oracle_primary_rankings_sha256": _canonical_ranking_fingerprint(
                oracle_cells
            ),
        },
        "v12_cells": v12_cells,
        "oracle_cells": oracle_cells,
        "statistical_gate_overlay": {
            "contract": (
                "Source inferential results are reported as an overlay only; "
                "they do not exclude descriptive activation candidates."
            ),
            "v12_three_sae_matched_outer_holm": {
                "supported_count": v12_strict_supported,
                "family_cell_count": v12_strict_family_cells,
            },
            "oracle_within_run_four_phase_holm": {
                "supported_count": len(oracle_within_run_supported),
                "family_cell_count": len(oracle_statistical_cells),
                "supported_cells": oracle_within_run_supported,
            },
            "oracle_all_run_phase_cell_holm": {
                "supported_count": len(oracle_all_run_supported),
                "family_cell_count": len(oracle_statistical_cells),
                "supported_cells": oracle_all_run_supported,
            },
        },
        "v12_recurrence": recurrence,
        "nested_view_sensitivity": _nested_view_stability(v12_cells),
        "stability_headlines": _stability_headlines(
            v12_cells,
            event_primary_top_n=config.event_primary_top_n,
        ),
        "oracle_reference": oracle_reference,
        "oracle_fixed_phase_comparison": oracle_fixed_comparison,
        "confound_audit": _audit_rows(),
        "held_claims": [
            (
                "반복된 event-ranked feature가 task-general event feature라는 "
                "주장: confounded — 판정 보류"
            ),
            (
                "phase-aligned feature가 phase 의미 또는 causal control을 "
                "표현한다는 주장: confounded — 판정 보류"
            ),
            (
                "Oracle–V12 overlap 또는 non-overlap이 feature 의미를 확인·"
                "반증한다는 주장: confounded — 판정 보류"
            ),
            (
                "suite-complete 4-phase panel이 Oracle의 single-instruction "
                "phase composition과 동등하다는 주장: confounded — 판정 보류"
            ),
            (
                "후보 개수·score 크기로 SAE 품질 또는 Hooked-SR 개선을 "
                "판정하는 주장: confounded — 판정 보류"
            ),
        ],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "report": str(output_dir / "report.md"),
        },
    }
    report = _render_report(summary)
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "summary.json").open(
        "x",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": summary["outputs"]["summary"],
                "report": summary["outputs"]["report"],
                "score_pairs": summary["scope"]["score_pair_count"],
                "v12_fingerprint": summary["regression_fingerprints"][
                    "v12_primary_rankings_sha256"
                ],
                "oracle_fingerprint": summary["regression_fingerprints"][
                    "oracle_primary_rankings_sha256"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object.")
            rows.append(row)
    return rows


def _write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )


def _focused_file_spec(path: str | Path) -> dict[str, str]:
    canonical = _canonical_path(path)
    if not canonical.is_file():
        raise FileNotFoundError(canonical)
    return {"path": str(canonical), "sha256": sha256_file(canonical)}


def _require_recorded_hash(
    path: Path,
    expected_sha256: Any,
    *,
    context: str,
) -> str:
    actual = sha256_file(path)
    expected = str(expected_sha256 or "")
    if not expected or actual != expected:
        raise ValueError(
            f"{context}: SHA mismatch for {path}: {actual} != {expected!r}."
        )
    return actual


def _focused_score_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    score_definitions = dict(payload.get("score_definitions", {}))
    if score_definitions != FOCUSED_SCORE_DEFINITIONS:
        raise ValueError(f"{path}: focused score semantics changed.")
    selected_ids = [
        str(row["sample_id"]) for row in payload.get("selected_events", [])
    ]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(f"{path}: duplicate selected sample IDs.")
    metadata = {
        "source": dict(payload["source"]),
        "step_mapping": str(payload["step_mapping"]),
        "event_step_scale": int(payload["event_step_scale"]),
        "window_size": int(payload["window_size"]),
        "row_keys": list(payload["row_keys"]),
        "episode_group_keys": list(payload["episode_group_keys"]),
        "selected_sample_ids": selected_ids,
        "selection_counts": dict(payload["selection_counts"]),
        "score_definitions": score_definitions,
    }
    del payload
    return metadata


def _validate_score_pair_inventory(
    *,
    source_label: str,
    score_w4: Path,
    score_w5: Path,
    expected_rows: int,
    expected_episode_groups: int,
    expected_events: int,
    expected_event_step_scale: int,
    expected_w4_shifts: int,
    expected_w5_shifts: int,
    expected_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    w4 = _focused_score_metadata(score_w4)
    w5 = _focused_score_metadata(score_w5)
    if w4["source"] != w5["source"]:
        raise ValueError(f"{source_label}: W4/W5 score source dictionaries differ.")
    if w4["score_definitions"] != w5["score_definitions"]:
        raise ValueError(f"{source_label}: W4/W5 score definitions differ.")
    if w4["row_keys"] != w5["row_keys"]:
        raise ValueError(f"{source_label}: W4/W5 row inventories differ.")
    if w4["episode_group_keys"] != w5["episode_group_keys"]:
        raise ValueError(f"{source_label}: W4/W5 episode-group inventories differ.")
    ids_w4 = set(w4["selected_sample_ids"])
    ids_w5 = set(w5["selected_sample_ids"])
    if ids_w4 != ids_w5:
        raise ValueError(f"{source_label}: W4/W5 selected event identities differ.")
    if expected_sample_ids is not None and ids_w4 != expected_sample_ids:
        missing = sorted(expected_sample_ids - ids_w4)[:5]
        extra = sorted(ids_w4 - expected_sample_ids)[:5]
        raise ValueError(
            f"{source_label}: selected identities changed; "
            f"missing={missing}, extra={extra}."
        )
    for window, metadata, expected_shifts in (
        (4, w4, expected_w4_shifts),
        (5, w5, expected_w5_shifts),
    ):
        if metadata["window_size"] != window:
            raise ValueError(f"{source_label}: expected W{window} payload.")
        if metadata["event_step_scale"] != expected_event_step_scale:
            raise ValueError(
                f"{source_label}: unexpected W{window} event step scale."
            )
        counts = metadata["selection_counts"]
        checks = {
            "selected_events_after_activation_filter": expected_events,
            "skipped_missing_window_vectors": 0,
            "shifted_window_count": expected_shifts,
        }
        for key, expected in checks.items():
            if int(counts.get(key, -1)) != expected:
                raise ValueError(
                    f"{source_label}: W{window} {key}="
                    f"{counts.get(key)!r}, expected {expected}."
                )
    if len(w4["row_keys"]) != expected_rows:
        raise ValueError(
            f"{source_label}: row count {len(w4['row_keys'])} "
            f"!= {expected_rows}."
        )
    if len(w4["episode_group_keys"]) != expected_episode_groups:
        raise ValueError(
            f"{source_label}: episode-group count "
            f"{len(w4['episode_group_keys'])} != {expected_episode_groups}."
        )
    if len(ids_w4) != expected_events:
        raise ValueError(
            f"{source_label}: selected event count {len(ids_w4)} "
            f"!= {expected_events}."
        )
    return {
        "source": w4["source"],
        "step_mapping": w4["step_mapping"],
        "event_step_scale": w4["event_step_scale"],
        "selected_sample_ids": sorted(ids_w4),
        "num_rows": len(w4["row_keys"]),
        "num_episode_groups": len(w4["episode_group_keys"]),
        "num_selected_events": len(ids_w4),
        "w4_shifted_windows": int(
            w4["selection_counts"]["shifted_window_count"]
        ),
        "w5_shifted_windows": int(
            w5["selection_counts"]["shifted_window_count"]
        ),
        "w4_skipped_missing_window_vectors": int(
            w4["selection_counts"]["skipped_missing_window_vectors"]
        ),
        "w5_skipped_missing_window_vectors": int(
            w5["selection_counts"]["skipped_missing_window_vectors"]
        ),
        "score_definitions": w4["score_definitions"],
    }


def _focused_summary_paths(
    config: FocusedPhaseViewAnalysisConfig,
) -> dict[str, Path]:
    stage4_root = _canonical_path(config.stage4_root)
    return {
        source_label: (
            stage4_root
            / condition_id
            / "cov0p3"
            / "analysis"
            / "task_local_phase_feature_ranking"
            / "summary.json"
        )
        for source_label, condition_id in FOCUSED_V12_CONDITIONS.items()
    } | {"oracle_full": _canonical_path(config.oracle_summary)}


def _focused_source_inventory(
    config: FocusedPhaseViewAnalysisConfig,
) -> dict[str, dict[str, Any]]:
    checkpoint = _canonical_path(FOCUSED_CHECKPOINT_PATH)
    _require_recorded_hash(
        checkpoint,
        FOCUSED_CHECKPOINT_SHA256,
        context="fixed bs4096/10k checkpoint",
    )
    output: dict[str, dict[str, Any]] = {}
    for source_label, summary_path in _focused_summary_paths(config).items():
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        expected = FOCUSED_SOURCE_CONTRACT[source_label]
        summary = _load_json(summary_path)
        condition_id = str(summary.get("scope", {}).get("condition_id", ""))
        if condition_id != expected["condition_id"]:
            raise ValueError(
                f"{source_label}: condition {condition_id!r} "
                f"!= {expected['condition_id']!r}."
            )
        runs = summary.get("runs")
        if not isinstance(runs, dict) or "sae10k" not in runs:
            raise ValueError(f"{source_label}: missing sae10k run.")
        run = runs["sae10k"]
        run_checkpoint = _canonical_path(run["checkpoint"])
        if run_checkpoint != checkpoint:
            raise ValueError(
                f"{source_label}: checkpoint path differs from fixed 10k path."
            )
        if str(run.get("checkpoint_sha256", "")) != FOCUSED_CHECKPOINT_SHA256:
            raise ValueError(f"{source_label}: checkpoint SHA metadata differs.")

        score_w4 = _canonical_path(run["score_w4"])
        score_w5 = _canonical_path(run["score_w5"])
        _require_recorded_hash(
            score_w4,
            run.get("score_w4_sha256"),
            context=f"{source_label} W4",
        )
        _require_recorded_hash(
            score_w5,
            run.get("score_w5_sha256"),
            context=f"{source_label} W5",
        )
        inventory = _validate_score_pair_inventory(
            source_label=f"{source_label}/fine_original",
            score_w4=score_w4,
            score_w5=score_w5,
            expected_rows=int(expected["fine_rows"]),
            expected_episode_groups=int(expected["fine_episode_groups"]),
            expected_events=int(expected["selected_events"]),
            expected_event_step_scale=int(expected["event_step_scale"]),
            expected_w4_shifts=int(expected["w4_shifted_windows"]),
            expected_w5_shifts=int(expected["w5_shifted_windows"]),
        )
        score_source = inventory["source"]
        if inventory["step_mapping"] != "action_executed":
            raise ValueError(f"{source_label}: score step mapping is not fixed.")

        source_paths = {
            "event_features": _canonical_path(
                score_source["event_features_path"]
            ),
            "phase_assignments": _canonical_path(
                score_source["cluster_assignments_path"]
            ),
            "phase_groups": _canonical_path(
                score_source["cluster_annotations_path"]
            ),
            "prompt_records": _canonical_path(
                score_source["prompt_records_path"]
            ),
        }
        topk_dir = _canonical_path(score_source["topk_run_dir"])
        topk_manifest = topk_dir / "manifest.json"
        if not topk_manifest.is_file():
            raise FileNotFoundError(topk_manifest)
        topk_payload = _load_json(topk_manifest)
        if str(topk_payload.get("sae_sha256", "")) != FOCUSED_CHECKPOINT_SHA256:
            raise ValueError(f"{source_label}: Top-K checkpoint SHA differs.")
        if str(topk_payload.get("capture_target", "")) != "action_expert":
            raise ValueError(f"{source_label}: unexpected Top-K capture target.")
        if int(topk_payload.get("event_step_scale", -1)) != 5:
            raise ValueError(f"{source_label}: Top-K event step scale is not 5.")
        if str(run.get("topk_manifest_sha256", "")) != sha256_file(
            topk_manifest
        ):
            raise ValueError(f"{source_label}: Top-K manifest SHA differs.")

        recorded_source_hashes = {
            "event_features": score_source.get("event_features_sha256"),
            "phase_assignments": score_source.get(
                "cluster_assignments_sha256"
            ),
            "phase_groups": score_source.get(
                "cluster_annotations_sha256"
            ),
            "prompt_records": score_source.get("prompt_records_sha256"),
        }
        file_specs: dict[str, dict[str, str]] = {
            "summary": _focused_file_spec(summary_path),
            "score_w4": _focused_file_spec(score_w4),
            "score_w5": _focused_file_spec(score_w5),
            "checkpoint": _focused_file_spec(checkpoint),
            "topk_manifest": _focused_file_spec(topk_manifest),
        }
        for name, path in source_paths.items():
            actual_hash = _require_recorded_hash(
                path,
                recorded_source_hashes[name],
                context=f"{source_label} {name}",
            )
            file_specs[name] = {"path": str(path), "sha256": actual_hash}

        summary_inputs = summary.get("inputs", {})
        accepted = summary_inputs.get("accepted_annotations", {})
        accepted_path = _canonical_path(accepted["path"])
        _require_recorded_hash(
            accepted_path,
            accepted.get("sha256"),
            context=f"{source_label} accepted annotations",
        )
        file_specs["accepted_annotations"] = _focused_file_spec(accepted_path)
        assignments = _load_jsonl(source_paths["phase_assignments"])
        assignment_ids = [str(row["sample_id"]) for row in assignments]
        if len(assignment_ids) != len(set(assignment_ids)):
            raise ValueError(f"{source_label}: duplicate assignment sample IDs.")
        if set(assignment_ids) != set(inventory["selected_sample_ids"]):
            raise ValueError(
                f"{source_label}: source assignments and selected events differ."
            )
        output[source_label] = {
            "source_label": source_label,
            "condition_id": condition_id,
            "expected": dict(expected),
            "summary_path": summary_path,
            "score_w4": score_w4,
            "score_w5": score_w5,
            "checkpoint": checkpoint,
            "topk_dir": topk_dir,
            "event_features": source_paths["event_features"],
            "phase_assignments": source_paths["phase_assignments"],
            "phase_groups": source_paths["phase_groups"],
            "prompt_records": source_paths["prompt_records"],
            "score_step_mapping": inventory["step_mapping"],
            "score_event_step_scale": inventory["event_step_scale"],
            "topk_event_step_scale": int(topk_payload["event_step_scale"]),
            "capture_target": str(topk_payload["capture_target"]),
            "file_specs": file_specs,
            "fine_inventory": inventory,
        }
    return output


def regroup_phase_assignments_to_coarse(
    assignments: list[dict[str, Any]],
    phase_groups: list[dict[str, Any]],
    *,
    source_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Create an exact four-phase scorer input while preserving source rows.

    Every retained assignment is copied in full. Only the scorer-facing
    ``cluster_id``, ``phase_group_id``, and ``phase`` fields are replaced;
    their fine values remain in explicit ``source_*`` fields.
    """

    sample_ids = [str(row["sample_id"]) for row in assignments]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{source_label}: duplicate assignment sample IDs.")
    groups_by_id: dict[str, dict[str, Any]] = {}
    for group in phase_groups:
        group_id = str(group["phase_group_id"])
        if group_id in groups_by_id:
            raise ValueError(f"{source_label}: duplicate phase group {group_id}.")
        groups_by_id[group_id] = group

    if source_label == "oracle_full":
        invalid = {
            str(row["sample_id"]): errors
            for row in assignments
            if (
                errors := oracle_phase_entry_invariant_errors(
                    row,
                    required_window_size=5,
                )
            )
        }
        if invalid:
            sample = dict(list(invalid.items())[:5])
            raise ValueError(
                f"{source_label}: source Oracle clock invariant failed: {sample}."
            )

    excluded_counts: Counter[str] = Counter()
    unknown_counts: Counter[str] = Counter()
    prepared: list[tuple[dict[str, Any], str, str, str]] = []
    for row in assignments:
        fine_phase = str(row.get("phase") or "")
        if fine_phase in KNOWN_NON_PHASE_LABELS:
            excluded_counts[fine_phase] += 1
            continue
        coarse_phase = COARSE_PHASE_BY_ANNOTATION.get(fine_phase)
        if coarse_phase is None:
            unknown_counts[fine_phase] += 1
            continue
        source_group_id = str(row["phase_group_id"])
        source_group = groups_by_id.get(source_group_id)
        if source_group is None:
            raise ValueError(
                f"{source_label}: assignment references unknown group "
                f"{source_group_id!r}."
            )
        source_key = (
            str(source_group["task_description"]),
            str(source_group["phase"]),
        )
        row_key = (str(row["task_description"]), fine_phase)
        if source_key != row_key:
            raise ValueError(
                f"{source_label}: assignment/group key mismatch "
                f"{row_key!r} != {source_key!r}."
            )
        digest = hashlib.sha256(
            (
                source_label
                + "\0"
                + str(row["task_description"])
                + "\0"
                + coarse_phase
            ).encode("utf-8")
        ).hexdigest()[:16]
        derived_group_id = (
            f"focused_{source_label}_{coarse_phase}_{digest}"
        )
        prepared.append(
            (row, fine_phase, coarse_phase, derived_group_id)
        )
    if unknown_counts:
        raise ValueError(
            f"{source_label}: unknown fine phase labels "
            f"{dict(sorted(unknown_counts.items()))}."
        )

    derived_assignments: list[dict[str, Any]] = []
    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    source_rows_by_group: dict[str, list[dict[str, Any]]] = {}
    for row, fine_phase, coarse_phase, derived_group_id in prepared:
        derived = dict(row)
        derived.update(
            {
                "source_fine_phase": fine_phase,
                "source_phase_group_id": str(row["phase_group_id"]),
                "coarse_phase": coarse_phase,
                "derived_label_semantics": (
                    "deterministic ontology regroup; not a new direct "
                    "annotation or simulator state label"
                ),
                "cluster_id": derived_group_id,
                "phase_group_id": derived_group_id,
                "phase": coarse_phase,
            }
        )
        derived_assignments.append(derived)
        rows_by_group.setdefault(derived_group_id, []).append(derived)
        source_rows_by_group.setdefault(derived_group_id, []).append(row)

    derived_groups: list[dict[str, Any]] = []
    coarse_order = {phase: index for index, phase in enumerate(COARSE_PHASE_ORDER)}
    for group_id in sorted(
        rows_by_group,
        key=lambda value: (
            str(rows_by_group[value][0]["task_description"]),
            coarse_order[str(rows_by_group[value][0]["phase"])],
        ),
    ):
        rows = rows_by_group[group_id]
        source_rows = source_rows_by_group[group_id]
        task_description = str(rows[0]["task_description"])
        coarse_phase = str(rows[0]["phase"])
        member_ids = [str(row["sample_id"]) for row in rows]
        episode_nums = sorted({int(row["episode_num"]) for row in rows})
        source_group_ids = sorted(
            {str(row["phase_group_id"]) for row in source_rows}
        )
        source_group_records = [groups_by_id[value] for value in source_group_ids]
        total_episode_values = {
            int(group["total_task_episodes"]) for group in source_group_records
        }
        if len(total_episode_values) != 1:
            raise ValueError(
                f"{source_label}/{task_description}/{coarse_phase}: "
                "source total-task-episode counts differ."
            )
        total_task_episodes = next(iter(total_episode_values))
        source_fine_phases = sorted(
            {str(row["phase"]) for row in source_rows}
        )
        source_cluster_ids = sorted(
            {str(row["source_cluster_id"]) for row in source_rows}
        )
        representative_ids: list[str] = []
        member_set = set(member_ids)
        for group in source_group_records:
            for sample_id in group.get("representative_sample_ids", []):
                sample_id = str(sample_id)
                if sample_id in member_set and sample_id not in representative_ids:
                    representative_ids.append(sample_id)
        group_record: dict[str, Any] = {
            "format": "event_sae_exact_coarse_phase_group_v1",
            "cluster_id": group_id,
            "phase_group_id": group_id,
            "task_description": task_description,
            "phase": coarse_phase,
            "phrase": f"{coarse_phase} exact-regrouped phase group",
            "source_fine_phases": source_fine_phases,
            "source_phase_group_ids": source_group_ids,
            "source_cluster_ids": source_cluster_ids,
            "num_source_phase_groups": len(source_group_ids),
            "num_source_clusters": len(source_cluster_ids),
            "member_sample_ids": member_ids,
            "num_members": len(member_ids),
            "member_episode_nums": episode_nums,
            "total_task_episodes": total_task_episodes,
            "episode_coverage": (
                len(episode_nums) / total_task_episodes
                if total_task_episodes
                else 0.0
            ),
            "representative_sample_ids": representative_ids,
            "representative_clip_paths": [],
            "representative_frame_paths": [],
            "representative_progress_percents": [],
            "allowed_phase_labels": list(COARSE_PHASE_ORDER),
            "phase_scheme": "focused_exact_coarse4",
            "model": "deterministic_exact_coarse_regroup",
            "prompt_version": "none",
            "review_mode": "deterministic_ontology_view",
            "review_verdict": "derived_from_validated_fine_assignments",
            "actual_human_review_completed": False,
            "api_error": None,
            "parse_error": None,
            "derived_label_semantics": (
                "deterministic ontology regroup; exact scorer re-run; "
                "not a new direct annotation or simulator state label"
            ),
        }
        if source_label == "oracle_full":
            group_record.update(
                {
                    "source_oracle_phase_entry_invariants_validated": True,
                    "source_fine_label_semantics": (
                        "direct simulator-oracle phase-entry labels"
                    ),
                    "coarse_label_semantics": (
                        "derived deterministic ontology view; not a direct "
                        "simulator state label"
                    ),
                }
            )
        derived_groups.append(group_record)

    fine_episode_groups = {
        (
            str(row["task_description"]),
            str(row["phase_group_id"]),
            int(row["episode_num"]),
        )
        for row, _, _, _ in prepared
    }
    coarse_episode_groups = {
        (
            str(row["task_description"]),
            str(row["phase_group_id"]),
            int(row["episode_num"]),
        )
        for row in derived_assignments
    }
    phase_composition: dict[str, list[str]] = {}
    for group in derived_groups:
        phase_composition.setdefault(
            str(group["task_description"]), []
        ).append(str(group["phase"]))
    for task_description, phases in phase_composition.items():
        phase_composition[task_description] = sorted(
            phases,
            key=coarse_order.__getitem__,
        )
    diagnostics = {
        "source_assignment_count": len(assignments),
        "derived_assignment_count": len(derived_assignments),
        "source_phase_group_count": len(phase_groups),
        "derived_phase_group_count": len(derived_groups),
        "source_episode_phase_group_count": len(fine_episode_groups),
        "derived_episode_phase_group_count": len(coarse_episode_groups),
        "fine_to_coarse_episode_group_reduction": (
            len(fine_episode_groups) - len(coarse_episode_groups)
        ),
        "known_excluded_label_counts": dict(sorted(excluded_counts.items())),
        "unknown_label_counts": {},
        "source_fine_phase_counts": dict(
            sorted(Counter(str(row["phase"]) for row in assignments).items())
        ),
        "derived_coarse_phase_counts": dict(
            sorted(
                Counter(str(row["phase"]) for row in derived_assignments).items()
            )
        ),
        "task_phase_composition": dict(sorted(phase_composition.items())),
        "num_tasks": len(phase_composition),
        "num_contrastable_tasks": sum(
            len(phases) >= 2 for phases in phase_composition.values()
        ),
        "assignment_field_preservation": (
            "all source fields retained; scorer-facing cluster_id, "
            "phase_group_id, phase replaced with derived coarse values"
        ),
    }
    return derived_assignments, derived_groups, diagnostics


def _rank_focused_tasks(
    tasks: dict[str, TaskLocalPhaseScores],
    *,
    source_label: str,
    view: str,
    comparator_top_n: int,
    filtered_top_n: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rank all positive exact task×phase robust margins, then set-filter."""

    if comparator_top_n <= 0 or filtered_top_n <= 0:
        raise ValueError("Focused shortlist sizes must be positive.")
    cells: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    contrastable_tasks = 0
    dict_sizes: set[int] = set()
    for task_description, task in sorted(tasks.items()):
        dict_size = int(task.pair.phase_w5.shape[1])
        dict_sizes.add(dict_size)
        if len(task.pair.phases) >= 2:
            contrastable_tasks += 1
        task_mean_w4 = _validated_task_mean(task, window=4)
        task_mean_w5 = _validated_task_mean(task, window=5)
        task_rank_w4 = feature_ranks_descending(task_mean_w4)
        task_rank_w5 = feature_ranks_descending(task_mean_w5)
        task_control_ids = {
            int(feature_id)
            for feature_id in np.flatnonzero(
                (task_rank_w4 <= comparator_top_n)
                | (task_rank_w5 <= comparator_top_n)
            )
        }
        for phase_idx, phase in enumerate(task.pair.phases):
            window_rank_w4 = feature_ranks_descending(
                task.window_mean_w4[phase_idx]
            )
            window_rank_w5 = feature_ranks_descending(
                task.window_mean_w5[phase_idx]
            )
            window_control_ids = {
                int(feature_id)
                for feature_id in np.flatnonzero(
                    (window_rank_w4 <= comparator_top_n)
                    | (window_rank_w5 <= comparator_top_n)
                )
            }
            cell_id = (
                f"{source_label}/{view}/{task_description}/{phase}"
            )
            raw_rows: list[dict[str, Any]] = []
            if len(task.pair.phases) >= 2:
                robust_margin = task.pair.robust_margin[phase_idx]
                positive_ids = np.flatnonzero(robust_margin > 0)
                ordered_ids = sorted(
                    (int(value) for value in positive_ids),
                    key=lambda feature_id: (
                        -float(robust_margin[feature_id]),
                        feature_id,
                    ),
                )
                filtered_rank = 0
                for raw_rank, feature_id in enumerate(ordered_ids, start=1):
                    reasons = []
                    if window_rank_w4[feature_id] <= comparator_top_n:
                        reasons.append("window_mean_top20_w4")
                    if window_rank_w5[feature_id] <= comparator_top_n:
                        reasons.append("window_mean_top20_w5")
                    if task_rank_w4[feature_id] <= comparator_top_n:
                        reasons.append("task_mean_top20_w4")
                    if task_rank_w5[feature_id] <= comparator_top_n:
                        reasons.append("task_mean_top20_w5")
                    removed = bool(reasons)
                    if not removed:
                        filtered_rank += 1
                    row = {
                        "candidate_semantics": (
                            "unknown-combined temporal modulation"
                        ),
                        "source": source_label,
                        "view": view,
                        "cell_id": cell_id,
                        "task_description": task_description,
                        "phase": phase,
                        "feature_id": feature_id,
                        "raw_candidate_rank": raw_rank,
                        "event_score_w4": float(
                            task.pair.phase_w4[phase_idx, feature_id]
                        ),
                        "event_score_w5": float(
                            task.pair.phase_w5[phase_idx, feature_id]
                        ),
                        "event_delta_w4": float(
                            task.pair.margin_w4[phase_idx, feature_id]
                        ),
                        "event_delta_w5": float(
                            task.pair.margin_w5[phase_idx, feature_id]
                        ),
                        "robust_event_delta_min_w4_w5": float(
                            robust_margin[feature_id]
                        ),
                        "window_mean_rank_w4": int(
                            window_rank_w4[feature_id]
                        ),
                        "window_mean_rank_w5": int(
                            window_rank_w5[feature_id]
                        ),
                        "task_mean_rank_w4": int(task_rank_w4[feature_id]),
                        "task_mean_rank_w5": int(task_rank_w5[feature_id]),
                        "control_filter_reasons": reasons,
                        "control_filter_families": sorted(
                            {
                                reason.split("_top20_", 1)[0]
                                for reason in reasons
                            }
                        ),
                        "removed_by_control_filter": removed,
                        "passes_control_filter": not removed,
                        "filtered_rank": (
                            None if removed else filtered_rank
                        ),
                        "in_control_filtered_top10": (
                            not removed and filtered_rank <= filtered_top_n
                        ),
                        "numeric_control_subtraction_applied": False,
                    }
                    raw_rows.append(row)
                    candidate_rows.append(row)
            filtered_top = [
                {
                    "feature_id": int(row["feature_id"]),
                    "raw_candidate_rank": int(row["raw_candidate_rank"]),
                    "filtered_rank": int(row["filtered_rank"]),
                    "event_delta_w4": float(row["event_delta_w4"]),
                    "event_delta_w5": float(row["event_delta_w5"]),
                    "robust_event_delta_min_w4_w5": float(
                        row["robust_event_delta_min_w4_w5"]
                    ),
                }
                for row in raw_rows
                if row["in_control_filtered_top10"]
            ]
            cells.append(
                {
                    "cell_id": cell_id,
                    "task_description": task_description,
                    "phase": phase,
                    "contrastable": len(task.pair.phases) >= 2,
                    "noncontrastable_reason": (
                        None
                        if len(task.pair.phases) >= 2
                        else "exact task has fewer than two observed phases"
                    ),
                    "raw_positive_candidate_count": len(raw_rows),
                    "removed_by_control_count": sum(
                        bool(row["removed_by_control_filter"])
                        for row in raw_rows
                    ),
                    "control_filtered_candidate_count": sum(
                        bool(row["passes_control_filter"]) for row in raw_rows
                    ),
                    "window_mean_control_ids_w4_union_w5": sorted(
                        window_control_ids
                    ),
                    "task_mean_control_ids_w4_union_w5": sorted(
                        task_control_ids
                    ),
                    "control_union_ids": sorted(
                        window_control_ids | task_control_ids
                    ),
                    "removed_reason_counts": dict(
                        sorted(
                            Counter(
                                reason
                                for row in raw_rows
                                for reason in row["control_filter_reasons"]
                            ).items()
                        )
                    ),
                    "control_filtered_top10": filtered_top,
                }
            )
    if len(dict_sizes) != 1:
        raise ValueError(
            f"{source_label}/{view}: inconsistent dictionary sizes "
            f"{sorted(dict_sizes)}."
        )
    return (
        {
            "source": source_label,
            "view": view,
            "score_semantics": "exact scorer rows; no post-hoc score pooling",
            "candidate_semantics": "unknown-combined temporal modulation",
            "scope": {
                "num_tasks": len(tasks),
                "num_contrastable_tasks": contrastable_tasks,
                "num_task_phase_cells": len(cells),
                "num_contrastable_task_phase_cells": sum(
                    bool(cell["contrastable"]) for cell in cells
                ),
                "dict_size": next(iter(dict_sizes)),
                "raw_positive_candidate_count": sum(
                    int(cell["raw_positive_candidate_count"])
                    for cell in cells
                ),
                "removed_by_control_count": sum(
                    int(cell["removed_by_control_count"]) for cell in cells
                ),
                "control_filtered_candidate_count": sum(
                    int(cell["control_filtered_candidate_count"])
                    for cell in cells
                ),
                "control_filtered_top10_count": sum(
                    len(cell["control_filtered_top10"]) for cell in cells
                ),
            },
            "cells": cells,
        },
        candidate_rows,
    )


def _focused_confound_audit() -> list[dict[str, str]]:
    return [
        {
            "dimension": "length",
            "status": "FAIL",
            "evidence": (
                "W4/W5 is fixed, but episode length and phase dwell are not "
                "matched across task×phase cells."
            ),
        },
        {
            "dimension": "task_identity",
            "status": "PASS",
            "evidence": "Every phase contrast is within one exact instruction.",
        },
        {
            "dimension": "instruction_balance",
            "status": "PASS",
            "evidence": "No score or candidate ranking pools instructions.",
        },
        {
            "dimension": "in_sample_rescue",
            "status": "N/A",
            "evidence": "No detector or intervention is selected or evaluated.",
        },
        {
            "dimension": "rollout_pooling",
            "status": "PASS",
            "evidence": (
                "The scorer averages events within episode-phase groups, then "
                "weights episode groups equally."
            ),
        },
        {
            "dimension": "phase_dwell",
            "status": "FAIL",
            "evidence": (
                "Exact regrouping removes post-hoc score pooling but does not "
                "match dwell time or transition position."
            ),
        },
        {
            "dimension": "oracle_v12_anchor_clock_alignment",
            "status": "FAIL",
            "evidence": (
                "Oracle uses exact simulator phase-entry transitions at "
                "environment-step scale 1; V12 uses annotated waypoint "
                "anchors mapped at scale 5. Their clocks and anchor semantics "
                "are not matched."
            ),
        },
        {
            "dimension": "observation_not_causation",
            "status": "PASS",
            "evidence": (
                "Candidates are labeled unknown-combined temporal modulation, "
                "not causal or semantically identified features."
            ),
        },
        {
            "dimension": "scene_local_not_general",
            "status": "FAIL",
            "evidence": "No held-out scene or rollout replication is included.",
        },
    ]


def _render_focused_report(summary: dict[str, Any]) -> str:
    scope = summary["scope"]
    lines = [
        "# Oracle five-cell vs V12 E3/E4 focused phase views",
        "",
        "## 1. 수치와 범위",
        "",
        (
            f"- 고정 SAE: bs4096/10k, checkpoint SHA "
            f"`{summary['method']['checkpoint_sha256']}`"
        ),
        (
            f"- source 3개, view {scope['view_count']}개, exact task×phase "
            f"cell {scope['task_phase_cell_count']}개"
        ),
        (
            f"- raw positive 후보 {scope['raw_positive_candidate_count']}개, "
            f"control 제거 {scope['removed_by_control_count']}개, "
            f"cell별 filtered Top-10 합계 "
            f"{scope['control_filtered_top10_count']}개"
        ),
        "- 통계적 유의성 검정은 수행하지 않은 descriptive discovery이다.",
        "",
        "## 2. Confound audit",
        "",
        "| 항목 | 상태 | 근거 |",
        "| --- | --- | --- |",
    ]
    for row in summary["confound_audit"]:
        lines.append(
            f"| {row['dimension']} | {row['status']} | {row['evidence']} |"
        )
    lines.extend(
        [
            "",
            "## 3. Claim strength",
            "",
            f"- `{summary['claim_strength']}`",
            (
                "- 후보 의미: unknown-combined temporal modulation. "
                "semantic identity나 causal control을 뜻하지 않는다."
            ),
            "",
            "## 4. 판정 보류 주장",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in summary["held_claims"])
    lines.extend(
        [
            "",
            "## 방법 계약",
            "",
            (
                "- raw 후보는 exact task×phase에서 "
                "`min(Δevent W4, Δevent W5) > 0`인 모든 feature다."
            ),
            (
                "- control은 동일 row의 window_mean Top-20(W4∪W5)와 "
                "동일 task의 task_mean Top-20(W4∪W5)의 set exclusion이다."
            ),
            "- control score의 숫자 차감은 하지 않았다.",
            (
                "- coarse4는 fine score의 post-hoc approximation이 아니라 "
                "assignment를 exact regroup한 뒤 W4/W5 scorer를 재실행했다."
            ),
            "",
            "## View 요약",
            "",
            "| Source | View | Cells | Raw | Removed | Filtered Top-10 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for view in summary["views"]:
        view_scope = view["scope"]
        lines.append(
            f"| {view['source']} | {view['view']} | "
            f"{view_scope['num_task_phase_cells']} | "
            f"{view_scope['raw_positive_candidate_count']} | "
            f"{view_scope['removed_by_control_count']} | "
            f"{view_scope['control_filtered_top10_count']} |"
        )
    lines.extend(["", "### Cell별 filtered Top-10", ""])
    for view in summary["views"]:
        lines.append(f"#### {view['source']} / {view['view']}")
        lines.append("")
        for cell in view["cells"]:
            ids = [
                str(row["feature_id"])
                for row in cell["control_filtered_top10"]
            ]
            value = ", ".join(ids) if ids else "없음"
            suffix = (
                ""
                if cell["contrastable"]
                else f" ({cell['noncontrastable_reason']})"
            )
            lines.append(
                f"- {cell['task_description']} / {cell['phase']}: "
                f"{value}{suffix}"
            )
        lines.append("")
    lines.extend(
        [
            "## 산출물",
            "",
            "- 모든 raw 후보와 제거 이유·실제 rank: `candidates.jsonl`",
            "- 전체 계약·입력 SHA·cell 요약: `summary.json`",
            "- coarse 파생 입력과 exact W4/W5 score: `derived/`, `scores/`",
            "",
        ]
    )
    return "\n".join(lines)


def _public_focused_source(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_label": source["source_label"],
        "condition_id": source["condition_id"],
        "contract": source["expected"],
        "files": source["file_specs"],
        "score_contract": {
            "step_mapping": source["score_step_mapping"],
            "event_step_scale": source["score_event_step_scale"],
            "topk_event_step_scale": source["topk_event_step_scale"],
            "capture_target": source["capture_target"],
            "score_definitions": source["fine_inventory"][
                "score_definitions"
            ],
        },
        "fine_inventory": {
            key: source["fine_inventory"][key]
            for key in (
                "num_rows",
                "num_episode_groups",
                "num_selected_events",
                "w4_shifted_windows",
                "w5_shifted_windows",
                "w4_skipped_missing_window_vectors",
                "w5_skipped_missing_window_vectors",
            )
        },
    }


def analyze_focused_phase_views(
    config: FocusedPhaseViewAnalysisConfig,
) -> dict[str, Any]:
    """Run immutable fine-original and exact-coarse views for three sources."""

    output_dir = config.output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite focused phase-view output: {output_dir}"
        )
    if config.comparator_top_n != 20 or config.filtered_top_n != 10:
        raise ValueError(
            "Focused contract fixes comparator_top_n=20 and filtered_top_n=10."
        )

    scorer_implementation = _focused_file_spec(
        Path(score_cluster_features.__code__.co_filename)
    )
    sources = _focused_source_inventory(config)
    views: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    derived_inputs: dict[
        str,
        tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]],
    ] = {}
    for source_label, source in sources.items():
        fine_tasks = load_task_local_score_pair(
            source["score_w4"], source["score_w5"]
        )
        fine_view, fine_candidates = _rank_focused_tasks(
            fine_tasks,
            source_label=source_label,
            view="fine_original",
            comparator_top_n=config.comparator_top_n,
            filtered_top_n=config.filtered_top_n,
        )
        expected = source["expected"]
        if (
            fine_view["scope"]["num_task_phase_cells"]
            != int(expected["fine_rows"])
            or fine_view["scope"]["num_contrastable_tasks"]
            != int(expected["contrastable_tasks"])
        ):
            raise ValueError(f"{source_label}: fine view scope changed.")
        views.append(fine_view)
        candidate_rows.extend(fine_candidates)
        assignments = _load_jsonl(source["phase_assignments"])
        phase_groups = _load_jsonl(source["phase_groups"])
        derived = regroup_phase_assignments_to_coarse(
            assignments,
            phase_groups,
            source_label=source_label,
        )
        diagnostics = derived[2]
        checks = {
            "derived_assignment_count": int(expected["selected_events"]),
            "derived_phase_group_count": int(expected["coarse_rows"]),
            "derived_episode_phase_group_count": int(
                expected["coarse_episode_groups"]
            ),
            "num_contrastable_tasks": int(expected["contrastable_tasks"]),
        }
        for key, expected_value in checks.items():
            if int(diagnostics[key]) != expected_value:
                raise ValueError(
                    f"{source_label}: derived {key}={diagnostics[key]} "
                    f"!= {expected_value}."
                )
        derived_inputs[source_label] = derived

    output_dir.mkdir(parents=True, exist_ok=False)
    derived_records: dict[str, Any] = {}
    for source_label, source in sources.items():
        assignments, groups, diagnostics = derived_inputs[source_label]
        derived_root = output_dir / "derived" / source_label / "coarse4"
        assignment_path = derived_root / "phase_group_assignments.jsonl"
        group_path = derived_root / "phase_groups.jsonl"
        _write_jsonl_exclusive(assignment_path, assignments)
        _write_jsonl_exclusive(group_path, groups)
        score_root = output_dir / "scores" / source_label / "coarse4"
        score_w4 = score_root / "w4" / "event_feature_scores.pt"
        score_w5 = score_root / "w5" / "event_feature_scores.pt"
        for window, output_path in ((4, score_w4), (5, score_w5)):
            score_cluster_features(
                topk_run_dir=source["topk_dir"],
                event_features_path=source["event_features"],
                cluster_assignments_path=assignment_path,
                cluster_annotations_path=group_path,
                output_path=output_path,
                window_size=window,
                top_n=config.comparator_top_n,
                step_mapping=source["score_step_mapping"],
                event_step_scale=source["score_event_step_scale"],
                prompt_records_path=source["prompt_records"],
            )
        expected = source["expected"]
        coarse_inventory = _validate_score_pair_inventory(
            source_label=f"{source_label}/coarse4_exact",
            score_w4=score_w4,
            score_w5=score_w5,
            expected_rows=int(expected["coarse_rows"]),
            expected_episode_groups=int(expected["coarse_episode_groups"]),
            expected_events=int(expected["selected_events"]),
            expected_event_step_scale=int(expected["event_step_scale"]),
            expected_w4_shifts=int(expected["w4_shifted_windows"]),
            expected_w5_shifts=int(expected["w5_shifted_windows"]),
            expected_sample_ids=set(
                source["fine_inventory"]["selected_sample_ids"]
            ),
        )
        derived_assignment_ids = {
            str(row["sample_id"]) for row in assignments
        }
        if derived_assignment_ids != set(
            coarse_inventory["selected_sample_ids"]
        ):
            raise ValueError(
                f"{source_label}: derived assignments and coarse selected "
                "events differ."
            )
        coarse_tasks = load_task_local_score_pair(score_w4, score_w5)
        coarse_view, coarse_candidates = _rank_focused_tasks(
            coarse_tasks,
            source_label=source_label,
            view="coarse4_exact_rescore",
            comparator_top_n=config.comparator_top_n,
            filtered_top_n=config.filtered_top_n,
        )
        observed_phases = {
            str(cell["phase"]) for cell in coarse_view["cells"]
        }
        if not observed_phases.issubset(set(COARSE_PHASE_ORDER)):
            raise ValueError(
                f"{source_label}: coarse scorer emitted unknown phases."
            )
        if observed_phases != set(COARSE_PHASE_ORDER):
            raise ValueError(
                f"{source_label}: global coarse4 vocabulary is incomplete: "
                f"{sorted(observed_phases)}."
            )
        if (
            coarse_view["scope"]["num_task_phase_cells"]
            != int(expected["coarse_rows"])
            or coarse_view["scope"]["num_contrastable_tasks"]
            != int(expected["contrastable_tasks"])
        ):
            raise ValueError(f"{source_label}: coarse view scope changed.")
        views.append(coarse_view)
        candidate_rows.extend(coarse_candidates)
        derived_records[source_label] = {
            "regrouping": diagnostics,
            "phase_assignments": _focused_file_spec(assignment_path),
            "phase_groups": _focused_file_spec(group_path),
            "score_w4": _focused_file_spec(score_w4),
            "score_w5": _focused_file_spec(score_w5),
            "score_inventory": {
                key: coarse_inventory[key]
                for key in (
                    "num_rows",
                    "num_episode_groups",
                    "num_selected_events",
                    "w4_shifted_windows",
                    "w5_shifted_windows",
                    "w4_skipped_missing_window_vectors",
                    "w5_skipped_missing_window_vectors",
                )
            },
            "selected_event_identity_matches_fine_w4_w5": True,
            "score_definitions_match_fine_and_fixed_contract": (
                coarse_inventory["score_definitions"]
                == source["fine_inventory"]["score_definitions"]
                == FOCUSED_SCORE_DEFINITIONS
            ),
        }

    for source in sources.values():
        for file_spec in source["file_specs"].values():
            path = Path(file_spec["path"])
            if sha256_file(path) != file_spec["sha256"]:
                raise RuntimeError(f"Read-only input changed during analysis: {path}")
    scorer_path = Path(scorer_implementation["path"])
    if sha256_file(scorer_path) != scorer_implementation["sha256"]:
        raise RuntimeError(
            f"Scorer implementation changed during analysis: {scorer_path}"
        )

    view_order = {
        (source_label, view): index
        for index, (source_label, view) in enumerate(
            (
                ("v12_e3_cov0p3", "fine_original"),
                ("v12_e3_cov0p3", "coarse4_exact_rescore"),
                ("v12_e4_cov0p3", "fine_original"),
                ("v12_e4_cov0p3", "coarse4_exact_rescore"),
                ("oracle_full", "fine_original"),
                ("oracle_full", "coarse4_exact_rescore"),
            )
        )
    }
    views.sort(key=lambda row: view_order[(row["source"], row["view"])])
    candidate_rows.sort(
        key=lambda row: (
            view_order[(row["source"], row["view"])],
            row["task_description"],
            row["phase"],
            row["raw_candidate_rank"],
        )
    )
    candidates_path = output_dir / "candidates.jsonl"
    _write_jsonl_exclusive(candidates_path, candidate_rows)
    scope = {
        "source_count": len(sources),
        "view_count": len(views),
        "task_phase_cell_count": sum(
            int(view["scope"]["num_task_phase_cells"]) for view in views
        ),
        "raw_positive_candidate_count": len(candidate_rows),
        "removed_by_control_count": sum(
            bool(row["removed_by_control_filter"]) for row in candidate_rows
        ),
        "control_filtered_candidate_count": sum(
            bool(row["passes_control_filter"]) for row in candidate_rows
        ),
        "control_filtered_top10_count": sum(
            bool(row["in_control_filtered_top10"]) for row in candidate_rows
        ),
    }
    implementation_path = Path(__file__).resolve()
    summary: dict[str, Any] = {
        "schema_version": FOCUSED_SCHEMA_VERSION,
        "scope": scope,
        "method": {
            "checkpoint_coordinate": "bs4096/10k",
            "checkpoint_sha256": FOCUSED_CHECKPOINT_SHA256,
            "sources": [
                "V12 E3 cov0p3",
                "V12 E4 cov0p3",
                "full Oracle five-cell phase-entry panel",
            ],
            "views": ["fine_original", "coarse4_exact_rescore"],
            "coarse_phase_order": list(COARSE_PHASE_ORDER),
            "event_candidate": (
                "all exact task×phase features with "
                "min(delta_event_w4, delta_event_w5) > 0; each delta is "
                "target phase minus strongest other phase"
            ),
            "control_filter": (
                "set exclusion by target-row window_mean Top-20 W4 union W5 "
                "and exact-task task_mean Top-20 W4 union W5"
            ),
            "numeric_control_subtraction_applied": False,
            "filtered_shortlist_per_cell": 10,
            "candidate_semantics": "unknown-combined temporal modulation",
            "inference_contract": (
                "descriptive discovery only; statistical significance was "
                "neither required nor evaluated"
            ),
            "coarse_contract": (
                "deterministically regroup original assignments before "
                "sequential exact W4/W5 scorer re-runs; never post-hoc pool "
                "fine score rows"
            ),
        },
        "inputs": {
            "sources": {
                source_label: _public_focused_source(source)
                for source_label, source in sources.items()
            },
            "implementation": _focused_file_spec(implementation_path),
            "scorer_implementation": scorer_implementation,
            "entrypoint": (
                _focused_file_spec(config.entrypoint)
                if config.entrypoint is not None
                else None
            ),
        },
        "derived_exact_coarse_inputs_and_scores": derived_records,
        "views": views,
        "confound_audit": _focused_confound_audit(),
        "claim_strength": "diagnostic_evidence",
        "held_claims": [
            (
                "후보가 phase 의미를 직접 인코딩하거나 causal control을 "
                "제공한다는 주장: confounded — 판정 보류"
            ),
            (
                "Oracle–V12 일치 또는 불일치가 semantic identity를 "
                "확인·반증한다는 주장: confounded — 판정 보류"
            ),
            (
                "후보 수나 rank만으로 Hooked-SR 개선을 예측한다는 주장: "
                "confounded — 판정 보류"
            ),
            (
                "동일 rollout의 fine/coarse 및 W4/W5 view를 독립 반복으로 "
                "해석하는 주장: confounded — 판정 보류"
            ),
        ],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "outputs": {
            "root": str(output_dir),
            "summary": str(output_dir / "summary.json"),
            "report": str(output_dir / "report.md"),
            "candidates": str(candidates_path),
        },
    }
    report = _render_focused_report(summary)
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    with (output_dir / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(
        json.dumps(
            {
                "summary": summary["outputs"]["summary"],
                "report": summary["outputs"]["report"],
                "candidates": summary["outputs"]["candidates"],
                **scope,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


def _directional_recorded_file(
    record: Any,
    *,
    context: str,
) -> dict[str, str]:
    if not isinstance(record, dict):
        raise ValueError(f"{context}: expected a path/SHA record.")
    path_value = record.get("path")
    if not path_value:
        raise ValueError(f"{context}: missing path.")
    path = _canonical_path(path_value)
    actual_hash = _require_recorded_hash(
        path,
        record.get("sha256"),
        context=context,
    )
    return {"path": str(path), "sha256": actual_hash}


def _directional_focused_inventory(
    focused_summary_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, str]],
]:
    """Resolve only the immutable inputs needed by directional rescoring."""

    focused_summary_path = _canonical_path(focused_summary_path)
    focused_summary_spec = _focused_file_spec(focused_summary_path)
    focused_summary = _load_json(focused_summary_path)
    if focused_summary.get("schema_version") != FOCUSED_SCHEMA_VERSION:
        raise ValueError(
            "Directional analysis requires focused_phase_view_analysis_v1."
        )
    source_records = focused_summary.get("inputs", {}).get("sources")
    derived_records = focused_summary.get(
        "derived_exact_coarse_inputs_and_scores"
    )
    if not isinstance(source_records, dict) or set(source_records) != set(
        DIRECTIONAL_SOURCE_ORDER
    ):
        raise ValueError(
            "Focused summary source panel must contain Oracle, V12 E3, "
            "and V12 E4 exactly."
        )
    if not isinstance(derived_records, dict) or set(derived_records) != set(
        DIRECTIONAL_SOURCE_ORDER
    ):
        raise ValueError(
            "Focused summary must contain all three exact-coarse inputs."
        )

    readonly_specs: dict[str, dict[str, str]] = {
        "focused_summary": focused_summary_spec
    }
    inventory: dict[str, dict[str, Any]] = {}
    for source_label in DIRECTIONAL_SOURCE_ORDER:
        source = source_records[source_label]
        derived = derived_records[source_label]
        if not isinstance(source, dict) or not isinstance(derived, dict):
            raise ValueError(f"{source_label}: malformed focused source.")
        expected = FOCUSED_SOURCE_CONTRACT[source_label]
        contract = source.get("contract")
        if not isinstance(contract, dict):
            raise ValueError(f"{source_label}: missing focused contract.")
        for key, expected_value in expected.items():
            if contract.get(key) != expected_value:
                raise ValueError(
                    f"{source_label}: focused contract {key} changed: "
                    f"{contract.get(key)!r} != {expected_value!r}."
                )

        files = source.get("files")
        score_contract = source.get("score_contract")
        if not isinstance(files, dict) or not isinstance(score_contract, dict):
            raise ValueError(f"{source_label}: missing focused provenance.")
        required_source_files = (
            "checkpoint",
            "topk_manifest",
            "event_features",
            "phase_assignments",
            "phase_groups",
            "prompt_records",
            "score_w4",
            "score_w5",
        )
        source_specs = {
            name: _directional_recorded_file(
                files.get(name),
                context=f"{source_label} {name}",
            )
            for name in required_source_files
        }
        coarse_specs = {
            "phase_assignments": _directional_recorded_file(
                derived.get("phase_assignments"),
                context=f"{source_label} coarse phase assignments",
            ),
            "phase_groups": _directional_recorded_file(
                derived.get("phase_groups"),
                context=f"{source_label} coarse phase groups",
            ),
            "score_w4": _directional_recorded_file(
                derived.get("score_w4"),
                context=f"{source_label} coarse legacy W4 score",
            ),
            "score_w5": _directional_recorded_file(
                derived.get("score_w5"),
                context=f"{source_label} coarse legacy W5 score",
            ),
        }
        checkpoint = Path(source_specs["checkpoint"]["path"])
        if (
            checkpoint != _canonical_path(FOCUSED_CHECKPOINT_PATH)
            or source_specs["checkpoint"]["sha256"]
            != FOCUSED_CHECKPOINT_SHA256
        ):
            raise ValueError(
                f"{source_label}: fixed bs4096/10k checkpoint changed."
            )
        if score_contract.get("step_mapping") != "action_executed":
            raise ValueError(
                f"{source_label}: directional step mapping is not fixed."
            )
        if score_contract.get("capture_target") != "action_expert":
            raise ValueError(
                f"{source_label}: directional capture target changed."
            )
        if int(score_contract.get("event_step_scale", -1)) != int(
            expected["event_step_scale"]
        ):
            raise ValueError(
                f"{source_label}: score event-step scale changed."
            )

        fine_assignment_rows = _load_jsonl(
            Path(source_specs["phase_assignments"]["path"])
        )
        coarse_assignment_rows = _load_jsonl(
            Path(coarse_specs["phase_assignments"]["path"])
        )
        fine_ids = [str(row["sample_id"]) for row in fine_assignment_rows]
        coarse_ids = [
            str(row["sample_id"]) for row in coarse_assignment_rows
        ]
        if len(fine_ids) != len(set(fine_ids)):
            raise ValueError(f"{source_label}: duplicate fine sample IDs.")
        if len(coarse_ids) != len(set(coarse_ids)):
            raise ValueError(f"{source_label}: duplicate coarse sample IDs.")
        if set(fine_ids) != set(coarse_ids):
            raise ValueError(
                f"{source_label}: fine/coarse selected event IDs differ."
            )
        if len(fine_ids) != int(expected["selected_events"]):
            raise ValueError(
                f"{source_label}: selected event count changed."
            )
        fine_groups = _load_jsonl(Path(source_specs["phase_groups"]["path"]))
        coarse_groups = _load_jsonl(Path(coarse_specs["phase_groups"]["path"]))
        if len(fine_groups) != int(expected["fine_rows"]):
            raise ValueError(f"{source_label}: fine row count changed.")
        if len(coarse_groups) != int(expected["coarse_rows"]):
            raise ValueError(f"{source_label}: coarse row count changed.")

        for name, spec in source_specs.items():
            readonly_specs[f"{source_label}/source/{name}"] = spec
        for name, spec in coarse_specs.items():
            readonly_specs[f"{source_label}/coarse/{name}"] = spec
        inventory[source_label] = {
            "source_label": source_label,
            "source_kind": (
                "simulator_oracle"
                if source_label == "oracle_full"
                else "v12_annotation"
            ),
            "coverage_filter": (
                None if source_label == "oracle_full" else "cov0p3"
            ),
            "expected": dict(expected),
            "checkpoint": checkpoint,
            "topk_dir": Path(
                source_specs["topk_manifest"]["path"]
            ).parent,
            "event_features": Path(
                source_specs["event_features"]["path"]
            ),
            "prompt_records": Path(
                source_specs["prompt_records"]["path"]
            ),
            "step_mapping": str(score_contract["step_mapping"]),
            "event_step_scale": int(score_contract["event_step_scale"]),
            "selected_sample_ids": set(fine_ids),
            "files": source_specs,
            "views": {
                "fine_original": {
                    "phase_assignments": Path(
                        source_specs["phase_assignments"]["path"]
                    ),
                    "phase_groups": Path(
                        source_specs["phase_groups"]["path"]
                    ),
                    "expected_rows": int(expected["fine_rows"]),
                    "expected_episode_groups": int(
                        expected["fine_episode_groups"]
                    ),
                    "legacy_score_w4": Path(
                        source_specs["score_w4"]["path"]
                    ),
                    "legacy_score_w5": Path(
                        source_specs["score_w5"]["path"]
                    ),
                },
                "coarse4_exact_rescore": {
                    "phase_assignments": Path(
                        coarse_specs["phase_assignments"]["path"]
                    ),
                    "phase_groups": Path(
                        coarse_specs["phase_groups"]["path"]
                    ),
                    "expected_rows": int(expected["coarse_rows"]),
                    "expected_episode_groups": int(
                        expected["coarse_episode_groups"]
                    ),
                    "legacy_score_w4": Path(
                        coarse_specs["score_w4"]["path"]
                    ),
                    "legacy_score_w5": Path(
                        coarse_specs["score_w5"]["path"]
                    ),
                },
            },
        }
    return focused_summary_spec, inventory, readonly_specs


def _assert_directional_files_unchanged(
    specs: dict[str, dict[str, str]],
    *,
    kind: str,
) -> None:
    for label, spec in specs.items():
        path = Path(spec["path"])
        if sha256_file(path) != spec["sha256"]:
            raise RuntimeError(
                f"Read-only {kind} changed during directional analysis: "
                f"{label}: {path}"
            )


def _directional_boundary_evidence(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Describe local sparse evidence without claiming trace persistence."""

    support = candidate["episode_support"]["w5"]
    comparable = int(support.get("comparable") or 0)
    positive = int(support.get("positive") or 0)
    status = (
        "boundary-only"
        if comparable > 0 and positive > 0
        else "insufficient support"
    )
    return {
        "status": status,
        "evidence_scope": (
            "event-centered episode-group W5 template projections only"
        ),
        "positive_episode_groups": positive,
        "comparable_episode_groups": comparable,
        "sparse_full_trace_rescan_performed": False,
        "dense_recollection_required": False,
        "absence_is_censored_by_sparse_topk": None,
    }


def _directional_trace_not_measured(
    *,
    reason: str,
    boundary_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Represent candidates outside the deliberately narrow trace scope."""

    result: dict[str, Any] = {
        "status": "not-measured",
        "reason": reason,
        "ranking_support": False,
        "ranking_gate": False,
        "evidence_scope": "outside_lossless_sparse_trace_measurement_scope",
        "sparse_full_trace_rescan_performed": False,
        "dense_recollection_required": False,
        "absence_is_censored_by_sparse_topk": None,
    }
    if boundary_evidence is not None:
        result["boundary_evidence"] = boundary_evidence
    return result


def _directional_transition_boundary_evidence(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    components = {
        "on": _directional_boundary_evidence(candidate["on"]),
        "off": _directional_boundary_evidence(candidate["off"]),
    }
    pair_support = candidate.get("episode_pair_support", {})
    pair_positive = int(pair_support.get("positive") or 0)
    pair_comparable = int(pair_support.get("comparable") or 0)
    status = (
        "boundary-only"
        if pair_positive > 0 and pair_comparable > 0
        else "insufficient support"
    )
    return {
        "status": status,
        "components": components,
        "same_episode_pair_support": {
            "positive": pair_positive,
            "comparable": pair_comparable,
            "fraction": pair_support.get("fraction"),
            "display": pair_support.get("display", "N/A"),
        },
        "evidence_scope": (
            "same-episode event-centered ON/OFF episode-group "
            "projections only"
        ),
        "sparse_full_trace_rescan_performed": False,
        "dense_recollection_required": False,
        "absence_is_censored_by_sparse_topk": None,
    }


def _validate_directional_lossless_topk_manifest(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless omitted sparse entries are known exact zeros."""

    encoding_stats = manifest.get("encoding_stats")
    if not isinstance(encoding_stats, dict):
        raise ValueError("Top-K manifest is missing encoding_stats.")
    try:
        topk = int(manifest["topk"])
        maximum_positive = int(
            encoding_stats["max_positive_features_per_row"]
        )
        rows_over_topk = int(
            encoding_stats[
                "rows_with_more_positive_features_than_topk"
            ]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Top-K manifest has incomplete lossless encoding statistics."
        ) from error
    if (
        encoding_stats.get("lossless_topk") is not True
        or rows_over_topk != 0
        or topk <= 0
        or maximum_positive < 0
        or maximum_positive > topk
    ):
        raise ValueError(
            "Directional trace persistence requires a lossless Top-K "
            "artifact with zero rows exceeding Top-K."
        )
    return {
        "lossless_topk": True,
        "topk": topk,
        "max_positive_features_per_row": maximum_positive,
        "rows_with_more_positive_features_than_topk": rows_over_topk,
        "absence_is_exact_zero": True,
    }


def _directional_trace_candidate_specs(
    transition_rankings: dict[str, Any],
    *,
    display_top_n: int,
) -> list[tuple[str, str, int]]:
    specs: list[tuple[str, str, int]] = []
    for task_description, task_result in sorted(
        transition_rankings["tasks"].items()
    ):
        for transition in transition_rankings[
            "ordered_transition_pairs"
        ]:
            candidates = task_result["transitions"][transition][
                "candidates"
            ]
            if candidates is None:
                continue
            specs.extend(
                (
                    task_description,
                    transition,
                    int(candidate["feature_id"]),
                )
                for candidate in candidates[:display_top_n]
            )
    if len(specs) != len(set(specs)):
        raise ValueError("Displayed state-pair trace keys are duplicated.")
    return specs


def _directional_trace_anchors(
    selected_events: list[dict[str, Any]],
    *,
    task_description: str,
    transition: str,
) -> tuple[list[dict[str, int]], dict[str, int]]:
    """Use earliest ON entry and first later OFF entry in each episode."""

    earlier_phase, later_phase = transition.split("->", maxsplit=1)
    by_episode: dict[int, dict[str, list[int]]] = {}
    for event in selected_events:
        if str(event["task_description"]) != task_description:
            continue
        phase = str(event["phase"])
        if phase not in (earlier_phase, later_phase):
            continue
        episode_num = int(event["episode_num"])
        phase_steps = by_episode.setdefault(episode_num, {})
        phase_steps.setdefault(phase, []).append(
            int(event["event_center_step"])
        )

    anchors: list[dict[str, int]] = []
    counters: Counter[str] = Counter()
    for episode_num, phase_steps in sorted(by_episode.items()):
        earlier_steps = sorted(set(phase_steps.get(earlier_phase, [])))
        later_steps = sorted(set(phase_steps.get(later_phase, [])))
        if not earlier_steps or not later_steps:
            counters["missing_phase_anchor"] += 1
            continue
        on_step = earlier_steps[0]
        later_after_on = [step for step in later_steps if step > on_step]
        if not later_after_on:
            counters["no_later_anchor_strictly_after_on"] += 1
            continue
        anchors.append(
            {
                "episode_num": episode_num,
                "on_step": on_step,
                "off_step": later_after_on[0],
            }
        )
    counters["anchor_episode_count"] = len(anchors)
    return anchors, dict(counters)


def _directional_complete_trace_values(
    timestep_vectors: dict[tuple[int, int], torch.Tensor],
    *,
    episode_num: int,
    steps: range,
    feature_id: int,
) -> list[float] | None:
    values: list[float] = []
    for step in steps:
        vector = timestep_vectors.get((episode_num, step))
        if vector is None or feature_id < 0 or feature_id >= vector.numel():
            return None
        values.append(float(vector[feature_id]))
    return values


def _directional_evaluate_trace_persistence(
    *,
    selected_events: list[dict[str, Any]],
    timestep_vectors: dict[tuple[int, int], torch.Tensor],
    transition_rankings: dict[str, Any],
    display_top_n: int,
    config: DirectionalTracePersistenceConfig,
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    dict[str, Any],
]:
    """Evaluate ON, dwell, and OFF patterns without changing rank membership."""

    candidate_specs = _directional_trace_candidate_specs(
        transition_rankings,
        display_top_n=display_top_n,
    )
    anchors_by_cell: dict[
        tuple[str, str],
        tuple[list[dict[str, int]], dict[str, int]],
    ] = {}
    for task_description, transition, _feature_id in candidate_specs:
        cell_key = (task_description, transition)
        if cell_key not in anchors_by_cell:
            anchors_by_cell[cell_key] = _directional_trace_anchors(
                selected_events,
                task_description=task_description,
                transition=transition,
            )

    results: dict[tuple[str, str, int], dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    window = int(config.local_window_size)
    for task_description, transition, feature_id in candidate_specs:
        anchors, anchor_counters = anchors_by_cell[
            (task_description, transition)
        ]
        episode_patterns: list[dict[str, Any]] = []
        exclusions: Counter[str] = Counter()
        for anchor in anchors:
            episode_num = anchor["episode_num"]
            on_step = anchor["on_step"]
            off_step = anchor["off_step"]
            if on_step < window:
                exclusions["incomplete_on_pre_window"] += 1
                continue
            if off_step - on_step < config.minimum_interval_steps:
                exclusions["interval_too_short"] += 1
                continue
            segment_ranges = {
                "on_pre": range(on_step - window, on_step),
                "on_post": range(on_step, on_step + window),
                "on_to_off": range(on_step, off_step),
                "off_post": range(off_step, off_step + window),
            }
            segments = {
                name: _directional_complete_trace_values(
                    timestep_vectors,
                    episode_num=episode_num,
                    steps=steps,
                    feature_id=feature_id,
                )
                for name, steps in segment_ranges.items()
            }
            missing_segments = [
                name for name, values in segments.items() if values is None
            ]
            if missing_segments:
                exclusions["missing_timestep_vectors"] += 1
                continue
            on_pre = float(np.mean(segments["on_pre"]))
            on_post = float(np.mean(segments["on_post"]))
            interval = float(np.mean(segments["on_to_off"]))
            off_post = float(np.mean(segments["off_post"]))
            interval_prevalence = float(
                np.mean(
                    np.asarray(segments["on_to_off"])
                    > config.activation_epsilon
                )
            )
            on_detected = on_post >= max(
                on_pre + config.on_minimum_absolute_increase,
                on_pre * config.on_minimum_ratio,
            )
            interval_sustained = (
                interval
                >= (
                    config.interval_minimum_on_post_fraction
                    * on_post
                )
                and interval_prevalence
                >= config.interval_minimum_activation_prevalence
            )
            off_detected = (
                off_post
                <= config.off_maximum_interval_fraction * interval
                and interval - off_post
                >= config.off_minimum_absolute_decrease
            )
            component_count = sum(
                (on_detected, interval_sustained, off_detected)
            )
            episode_patterns.append(
                {
                    "episode_num": episode_num,
                    "on_anchor_step": on_step,
                    "off_anchor_step": off_step,
                    "on_pre_w5_mean": on_pre,
                    "on_post_w5_mean": on_post,
                    "on_to_off_mean": interval,
                    "on_to_off_activation_prevalence": (
                        interval_prevalence
                    ),
                    "off_post_w5_mean": off_post,
                    "on_detected": on_detected,
                    "interval_sustained": interval_sustained,
                    "off_detected": off_detected,
                    "full_pattern": component_count == 3,
                    "two_of_three_pattern": component_count >= 2,
                }
            )

        comparable = len(episode_patterns)
        full_count = sum(row["full_pattern"] for row in episode_patterns)
        component_count = sum(
            row["two_of_three_pattern"] for row in episode_patterns
        )
        full_ratio = full_count / comparable if comparable else None
        component_ratio = (
            component_count / comparable if comparable else None
        )
        if comparable < config.minimum_comparable_episodes:
            status = "insufficient support"
        elif (
            full_ratio is not None
            and full_ratio
            >= config.confirmed_minimum_full_repeat_ratio
        ):
            status = "confirmed"
        elif (
            full_ratio is not None
            and component_ratio is not None
            and (
                full_ratio
                >= config.partial_minimum_full_repeat_ratio
                or component_ratio
                >= config.partial_minimum_component_repeat_ratio
            )
        ):
            status = "partial"
        else:
            status = "boundary-only"
        status_counts[status] += 1
        results[(task_description, transition, feature_id)] = {
            "status": status,
            "ranking_support": status in {"confirmed", "partial"},
            "ranking_gate": False,
            "evidence_scope": (
                "lossless sparse full trajectory for displayed coarse "
                "state-pair candidates only"
            ),
            "sparse_full_trace_rescan_performed": True,
            "dense_recollection_required": False,
            "absence_is_censored_by_sparse_topk": False,
            "anchor_episode_count": len(anchors),
            "anchor_counters": anchor_counters,
            "comparable_episode_count": comparable,
            "excluded_episode_counts": dict(exclusions),
            "full_pattern_episode_count": full_count,
            "full_pattern_repeat_ratio": full_ratio,
            "two_of_three_episode_count": component_count,
            "two_of_three_repeat_ratio": component_ratio,
            "episode_patterns": episode_patterns,
        }
    return results, {
        "measured_candidate_count": len(candidate_specs),
        "status_counts": dict(status_counts),
        "ranking_gate": False,
    }


def _directional_source_trace_persistence(
    *,
    source_label: str,
    source: dict[str, Any],
    score_w5: Path,
    transition_rankings: dict[str, Any],
    display_top_n: int,
    config: DirectionalTracePersistenceConfig,
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    dict[str, Any],
]:
    """Scan one source's lossless sparse trace exactly once."""

    artifact = open_sparse_topk_artifact(source["topk_dir"])
    lossless_contract = _validate_directional_lossless_topk_manifest(
        artifact.manifest
    )
    score_payload = torch.load(
        score_w5,
        map_location="cpu",
        weights_only=False,
    )
    selected_events = list(score_payload.get("selected_events", []))
    if not selected_events:
        raise ValueError(f"{source_label}: W5 score has no selected events.")
    dict_size = int(artifact.manifest["dict_size"])
    episode_to_task_id: dict[int, int] = {}
    _merge_episode_task_ids(
        episode_to_task_id,
        selected_events,
        source_name=f"{source_label} selected_events",
    )
    _merge_episode_task_ids(
        episode_to_task_id,
        _load_jsonl(source["prompt_records"]),
        source_name=f"{source_label} prompt_records",
    )
    task_id_set = {
        int(event["task_id"]) for event in selected_events
    }
    (
        timestep_vectors,
        task_means,
        task_counts,
        aggregate_manifest,
        load_counters,
    ) = aggregate_sparse_activations_by_timestep(
        source["topk_dir"],
        step_mapping=source["step_mapping"],
        episode_to_task_id=episode_to_task_id,
        task_id_set=task_id_set,
        dict_size=dict_size,
    )
    try:
        if aggregate_manifest != artifact.manifest:
            raise RuntimeError(
                f"{source_label}: Top-K manifest changed during trace scan."
            )
        results, evaluation_audit = (
            _directional_evaluate_trace_persistence(
                selected_events=selected_events,
                timestep_vectors=timestep_vectors,
                transition_rankings=transition_rankings,
                display_top_n=display_top_n,
                config=config,
            )
        )
    finally:
        del timestep_vectors
        del task_means
        del task_counts
        del score_payload
    return results, {
        "source": source_label,
        "scan_count": 1,
        "scope": "displayed_coarse_state_pair_top10_only",
        "lossless_topk": lossless_contract,
        "load_counters": load_counters,
        **evaluation_audit,
    }


def _directional_score_compatibility(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    matrix_contract = payload.get("template_matrix_contract")
    if not isinstance(matrix_contract, dict):
        raise ValueError(f"{path}: missing directional matrix contract.")
    required_keys = {
        *(f"matrix_{template}" for template in DIRECTIONAL_TEMPLATE_NAMES),
        *(
            f"episode_group_matrix_{template}"
            for template in DIRECTIONAL_TEMPLATE_NAMES
        ),
        "matrix_template_max",
        "matrix_raw",
    }
    missing = sorted(required_keys - set(payload))
    if missing:
        raise ValueError(f"{path}: missing directional tensors {missing}.")
    result = {
        "window_size": int(payload["window_size"]),
        "legacy_raw_relation": dict(
            matrix_contract.get("raw_vs_template_max", {})
        ),
        "episode_group_raw_equals_template_max": bool(
            matrix_contract.get(
                "episode_group_raw_equals_template_max",
                False,
            )
        ),
        "row_raw_equals_episode_group_raw_mean": bool(
            matrix_contract.get(
                "row_raw_equals_episode_group_raw_mean",
                False,
            )
        ),
        "combined_role": (
            "compatibility_only; never used as a directional label or gate"
        ),
    }
    del payload
    return result


def _directional_legacy_reproduction(
    *,
    legacy_path: Path,
    directional_path: Path,
) -> dict[str, Any]:
    """Require the new scorer to reproduce the immutable legacy raw matrix."""

    legacy = torch.load(
        legacy_path,
        map_location="cpu",
        weights_only=False,
    )
    directional = torch.load(
        directional_path,
        map_location="cpu",
        weights_only=False,
    )
    row_keys_equal = legacy.get("row_keys") == directional.get("row_keys")
    episode_group_keys_equal = legacy.get(
        "episode_group_keys"
    ) == directional.get("episode_group_keys")
    legacy_raw = legacy.get("matrix_raw")
    directional_raw = directional.get("matrix_raw")
    if not isinstance(legacy_raw, torch.Tensor) or not isinstance(
        directional_raw,
        torch.Tensor,
    ):
        raise ValueError("Legacy reproduction requires matrix_raw tensors.")
    shape_equal = legacy_raw.shape == directional_raw.shape
    exact_equal = shape_equal and torch.equal(
        legacy_raw,
        directional_raw,
    )
    rtol = 1e-6
    atol = 1e-7
    tolerance_equal = shape_equal and torch.allclose(
        legacy_raw,
        directional_raw,
        rtol=rtol,
        atol=atol,
    )
    max_abs = (
        float((legacy_raw - directional_raw).abs().max())
        if shape_equal and legacy_raw.numel()
        else (0.0 if shape_equal else None)
    )
    reproduced = bool(
        row_keys_equal
        and episode_group_keys_equal
        and tolerance_equal
    )
    result = {
        "legacy_score": _focused_file_spec(legacy_path),
        "directional_score": _focused_file_spec(directional_path),
        "row_keys_equal": row_keys_equal,
        "episode_group_keys_equal": episode_group_keys_equal,
        "matrix_raw_shape_equal": shape_equal,
        "matrix_raw_exact_torch_equal": exact_equal,
        "matrix_raw_tolerance_equal": tolerance_equal,
        "matrix_raw_comparison": {
            "rtol": rtol,
            "atol": atol,
            "max_abs": max_abs,
        },
        "reproduced": reproduced,
        "contract": (
            "row/group identity must be exact; matrix_raw must be exact or "
            "within the stated numerical tolerance"
        ),
    }
    del legacy
    del directional
    if not reproduced:
        raise ValueError(
            "Directional scorer failed immutable legacy reproduction: "
            f"{legacy_path} -> {directional_path}: {result}"
        )
    return result


def _directional_phase_tables_and_records(
    *,
    source_label: str,
    source_kind: str,
    view: str,
    phase_rankings: dict[str, Any],
    phase_recurrence: dict[str, Any],
    display_top_n: int,
    low_coverage_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    task_rows = []
    for task_description, task_result in sorted(
        phase_rankings["tasks"].items()
    ):
        phase_cells = []
        for phase in phase_rankings["phase_order"]:
            phase_result = task_result["phases"][phase]
            coverage = phase_result.get("phase_coverage")
            low_coverage = (
                coverage is not None
                and float(coverage) < low_coverage_threshold
            )
            template_cells = {}
            for template in DIRECTIONAL_TEMPLATE_NAMES:
                template_result = phase_result["templates"][template]
                candidates = template_result["candidates"]
                if candidates is None:
                    top_candidates = None
                else:
                    top_candidates = []
                    for display_rank, candidate in enumerate(
                        candidates,
                        start=1,
                    ):
                        in_display = display_rank <= display_top_n
                        persistence = _directional_trace_not_measured(
                            reason=(
                                "phase_and_pulse_candidates_are_outside_"
                                "coarse_state_pair_trace_scope"
                            ),
                            boundary_evidence=(
                                _directional_boundary_evidence(candidate)
                            ),
                        )
                        records.append(
                            {
                                "record_type": "phase_feature",
                                "source": source_label,
                                "source_kind": source_kind,
                                "view": view,
                                "task_description": task_description,
                                "phase": phase,
                                "template": template,
                                "feature_id": int(
                                    candidate["feature_id"]
                                ),
                                "display_rank": display_rank,
                                "in_display_top10": in_display,
                                "low_coverage": low_coverage,
                                "coverage_marker": (
                                    "†" if low_coverage else None
                                ),
                                "persistence": persistence,
                                "candidate": candidate,
                            }
                        )
                        if in_display:
                            top_candidates.append(
                                {
                                    "feature_id": int(
                                        candidate["feature_id"]
                                    ),
                                    "rank": display_rank,
                                    "margin_w5": float(
                                        candidate["w5"]["margin"]
                                    ),
                                    "margin_percentile_w5": float(
                                        candidate["w5"][
                                            "margin_percentile"
                                        ]
                                    ),
                                    "control_overlap_w5": bool(
                                        candidate["controls"]["w5"][
                                            "overlap"
                                        ]
                                    ),
                                    "w4_positive": candidate[
                                        "w4_sensitivity"
                                    ]["positive_margin"],
                                    "persistence_status": persistence[
                                        "status"
                                    ],
                                }
                            )
                template_cells[template] = {
                    "status": template_result["status"],
                    "display": template_result["display"],
                    "num_candidates": template_result[
                        "num_candidates"
                    ],
                    "top10": top_candidates,
                }
            phase_cells.append(
                {
                    "phase": phase,
                    "status": phase_result["status"],
                    "display": phase_result["display"],
                    "reason": phase_result.get("reason"),
                    "episode_coverage": coverage,
                    "low_coverage": low_coverage,
                    "coverage_marker": "†" if low_coverage else None,
                    "templates": template_cells,
                }
            )
        task_rows.append(
            {
                "task_description": task_description,
                "observed_phases": task_result["observed_phases"],
                "phase_cells": phase_cells,
            }
        )

    recurrence_rows = []
    for phase, phase_result in phase_recurrence["phases"].items():
        for template, template_result in phase_result["templates"].items():
            features = template_result["features"]
            compact_features = []
            for display_rank, feature in enumerate(features, start=1):
                in_display = display_rank <= display_top_n
                records.append(
                    {
                        "record_type": "phase_recurrence",
                        "source": source_label,
                        "source_kind": source_kind,
                        "view": view,
                        "phase": phase,
                        "template": template,
                        "feature_id": int(feature["feature_id"]),
                        "display_rank": display_rank,
                        "in_display_top10": in_display,
                        "recurrence": feature,
                    }
                )
                if in_display:
                    compact_features.append(
                        {
                            "feature_id": int(feature["feature_id"]),
                            "rank": display_rank,
                            "support_over_eligible": feature[
                                "support_over_eligible"
                            ],
                            "eligible_over_expected": feature[
                                "eligible_over_expected"
                            ],
                            "support_over_expected": feature[
                                "support_over_expected"
                            ],
                            "global_strict": bool(
                                feature["global_strict"]
                            ),
                            "global_relaxed": bool(
                                feature["global_relaxed"]
                            ),
                            "families": feature["families"],
                        }
                    )
            recurrence_rows.append(
                {
                    "phase": phase,
                    "template": template,
                    "status": template_result["status"],
                    "display": template_result["display"],
                    "available_task_count": template_result[
                        "available_task_count"
                    ],
                    "expected_task_count": template_result[
                        "expected_task_count"
                    ],
                    "num_features": template_result["num_features"],
                    "top10": compact_features,
                }
            )
    return (
        {
            "phase_vocabulary": list(phase_rankings["phase_order"]),
            "phase_vocabulary_role": (
                "deterministic observed-label display vocabulary; "
                "fine labels do not imply one cross-task temporal order"
                if view == "fine_original"
                else "fixed coarse temporal order"
            ),
            "tasks": task_rows,
            "recurrence": {
                "task_family_contract": phase_recurrence[
                    "task_family_contract"
                ],
                "global_contract": phase_recurrence["global_contract"],
                "rows": recurrence_rows,
            },
        },
        records,
    )


def _directional_transition_tables_and_records(
    *,
    source_label: str,
    source_kind: str,
    view: str,
    transition_rankings: dict[str, Any],
    transition_recurrence: dict[str, Any],
    trace_persistence_by_candidate: dict[
        tuple[str, str, int],
        dict[str, Any],
    ],
    display_top_n: int,
    low_coverage_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    task_rows = []
    for task_description, task_result in sorted(
        transition_rankings["tasks"].items()
    ):
        transition_cells = []
        for transition in transition_rankings[
            "ordered_transition_pairs"
        ]:
            transition_result = task_result["transitions"][transition]
            candidates = transition_result["candidates"]
            if candidates is None:
                top_candidates = None
            else:
                top_candidates = []
                for display_rank, candidate in enumerate(
                    candidates,
                    start=1,
                ):
                    low_coverage = (
                        float(candidate["coverage"]["minimum"])
                        < low_coverage_threshold
                    )
                    in_display = display_rank <= display_top_n
                    candidate_key = (
                        task_description,
                        transition,
                        int(candidate["feature_id"]),
                    )
                    if in_display:
                        if candidate_key not in (
                            trace_persistence_by_candidate
                        ):
                            raise RuntimeError(
                                "Missing trace persistence for displayed "
                                f"state-pair candidate {candidate_key!r}."
                            )
                        persistence = trace_persistence_by_candidate[
                            candidate_key
                        ]
                        persistence = {
                            **persistence,
                            "boundary_evidence": (
                                _directional_transition_boundary_evidence(
                                    candidate
                                )
                            ),
                        }
                    else:
                        persistence = _directional_trace_not_measured(
                            reason="outside_displayed_state_pair_top10",
                            boundary_evidence=(
                                _directional_transition_boundary_evidence(
                                    candidate
                                )
                            ),
                        )
                    records.append(
                        {
                            "record_type": "state_pair",
                            "source": source_label,
                            "source_kind": source_kind,
                            "view": view,
                            "task_description": task_description,
                            "transition": transition,
                            "feature_id": int(candidate["feature_id"]),
                            "display_rank": display_rank,
                            "in_display_top10": in_display,
                            "low_coverage": low_coverage,
                            "coverage_marker": (
                                "†" if low_coverage else None
                            ),
                            "persistence": persistence,
                            "candidate": candidate,
                        }
                    )
                    if in_display:
                        top_candidates.append(
                            {
                                "feature_id": int(
                                    candidate["feature_id"]
                                ),
                                "rank": display_rank,
                                "pair_score_w5": float(
                                    candidate["pair_score_w5"]
                                ),
                                "margin_percentile_w5": float(
                                    candidate[
                                        "conservative_margin_percentile_w5"
                                    ]
                                ),
                                "minimum_coverage": float(
                                    candidate["coverage"]["minimum"]
                                ),
                                "low_coverage": low_coverage,
                                "coverage_marker": (
                                    "†" if low_coverage else None
                                ),
                                "control_overlap_w5": bool(
                                    candidate["control_overlap_w5"]
                                ),
                                "persistence_status": persistence[
                                    "status"
                                ],
                            }
                        )
            transition_cells.append(
                {
                    "transition": transition,
                    "status": transition_result["status"],
                    "display": transition_result["display"],
                    "reason": transition_result.get("reason"),
                    "num_candidates": transition_result[
                        "num_candidates"
                    ],
                    "top10": top_candidates,
                }
            )
        task_rows.append(
            {
                "task_description": task_description,
                "transition_cells": transition_cells,
            }
        )

    recurrence_rows = []
    for transition, transition_result in transition_recurrence[
        "transitions"
    ].items():
        compact_features = []
        for display_rank, feature in enumerate(
            transition_result["features"],
            start=1,
        ):
            in_display = display_rank <= display_top_n
            records.append(
                {
                    "record_type": "state_pair_recurrence",
                    "source": source_label,
                    "source_kind": source_kind,
                    "view": view,
                    "transition": transition,
                    "feature_id": int(feature["feature_id"]),
                    "display_rank": display_rank,
                    "in_display_top10": in_display,
                    "recurrence": feature,
                }
            )
            if in_display:
                compact_features.append(
                    {
                        "feature_id": int(feature["feature_id"]),
                        "rank": display_rank,
                        "support_over_eligible": feature[
                            "support_over_eligible"
                        ],
                        "eligible_over_expected": feature[
                            "eligible_over_expected"
                        ],
                        "support_over_expected": feature[
                            "support_over_expected"
                        ],
                        "global_strict": bool(feature["global_strict"]),
                        "global_relaxed": bool(feature["global_relaxed"]),
                        "families": feature["families"],
                    }
                )
        recurrence_rows.append(
            {
                "transition": transition,
                "status": transition_result["status"],
                "display": transition_result["display"],
                "available_task_count": transition_result[
                    "available_task_count"
                ],
                "expected_task_count": transition_result[
                    "expected_task_count"
                ],
                "num_features": transition_result["num_features"],
                "top10": compact_features,
            }
        )
    return (
        {
            "status": "available",
            "display": None,
            "scope": "coarse4_only",
            "phase_order": list(COARSE_PHASE_ORDER),
            "pair_count": len(
                transition_rankings["ordered_transition_pairs"]
            ),
            "pair_definition": transition_rankings["pair_definition"],
            "tasks": task_rows,
            "recurrence": {
                "task_family_contract": transition_recurrence[
                    "task_family_contract"
                ],
                "global_contract": transition_recurrence[
                    "global_contract"
                ],
                "rows": recurrence_rows,
            },
        },
        records,
    )


def _directional_task_family(task_description: str) -> str:
    return (
        "drawer"
        if "drawer" in task_description.casefold()
        else "object"
    )


def _directional_probe_ranking(
    candidate_records: list[dict[str, Any]],
    *,
    tables: dict[str, list[dict[str, Any]]],
    display_top_n: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Link same feature IDs without combining raw scores across sources."""

    availability: dict[tuple[Any, ...], str] = {}
    for panel_views in tables.values():
        for view_record in panel_views:
            source = view_record["source"]
            view = view_record["view"]
            for task in view_record["phase_table"]["tasks"]:
                task_description = task["task_description"]
                for phase_cell in task["phase_cells"]:
                    for template, template_cell in phase_cell[
                        "templates"
                    ].items():
                        availability[
                            (
                                "phase",
                                source,
                                view,
                                task_description,
                                phase_cell["phase"],
                                template,
                            )
                        ] = template_cell["status"]
            state_pairs = view_record["state_pairs"]
            if state_pairs.get("status") == "not_applicable":
                continue
            for task in state_pairs["tasks"]:
                task_description = task["task_description"]
                for transition_cell in task["transition_cells"]:
                    availability[
                        (
                            "state_pair",
                            source,
                            view,
                            task_description,
                            transition_cell["transition"],
                        )
                    ] = transition_cell["status"]

    recurrence_lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in candidate_records:
        if record["record_type"] == "phase_recurrence":
            key = (
                "phase",
                record["source"],
                record["view"],
                record["phase"],
                record["template"],
                int(record["feature_id"]),
            )
        elif record["record_type"] == "state_pair_recurrence":
            key = (
                "state_pair",
                record["source"],
                record["view"],
                record["transition"],
                int(record["feature_id"]),
            )
        else:
            continue
        recurrence_lookup[key] = record["recurrence"]

    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    oracle_feature_contexts: dict[
        tuple[str, str, str, int],
        list[dict[str, Any]],
    ] = {}
    for record in candidate_records:
        if record["record_type"] == "phase_feature":
            key = (
                "phase",
                record["view"],
                record["task_description"],
                record["phase"],
                record["template"],
                int(record["feature_id"]),
            )
        elif record["record_type"] == "state_pair":
            key = (
                "state_pair",
                record["view"],
                record["task_description"],
                record["transition"],
                int(record["feature_id"]),
            )
        else:
            continue
        grouped.setdefault(key, {})[record["source"]] = record
        if record["source"] == "oracle_full":
            oracle_feature_contexts.setdefault(
                (
                    record["record_type"],
                    record["view"],
                    record["task_description"],
                    int(record["feature_id"]),
                ),
                [],
            ).append(record)

    probe_records: list[dict[str, Any]] = []
    groups_by_cell: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    e_sources = ("v12_e3_cov0p3", "v12_e4_cov0p3")
    for key, by_source in grouped.items():
        if not any(source in by_source for source in e_sources):
            continue
        record_type = key[0]
        if record_type == "phase":
            (
                _kind,
                view,
                task_description,
                phase,
                template,
                feature_id,
            ) = key
            cell_key = (
                "phase",
                view,
                task_description,
                phase,
                template,
            )
            recurrence_keys = {
                source: (
                    "phase",
                    source,
                    view,
                    phase,
                    template,
                    feature_id,
                )
                for source in by_source
            }
        else:
            (
                _kind,
                view,
                task_description,
                transition,
                feature_id,
            ) = key
            cell_key = (
                "state_pair",
                view,
                task_description,
                transition,
            )
            recurrence_keys = {
                source: (
                    "state_pair",
                    source,
                    view,
                    transition,
                    feature_id,
                )
                for source in by_source
            }

        e3_e4_match = all(source in by_source for source in e_sources)
        if record_type == "phase":
            availability_keys = {
                source: (
                    "phase",
                    source,
                    view,
                    task_description,
                    phase,
                    template,
                )
                for source in (*e_sources, "oracle_full")
            }
        else:
            availability_keys = {
                source: (
                    "state_pair",
                    source,
                    view,
                    task_description,
                    transition,
                )
                for source in (*e_sources, "oracle_full")
            }
        e_availability = {
            source: availability.get(availability_keys[source])
            for source in e_sources
        }
        if e3_e4_match:
            e3_e4_match_status = "✓"
        elif any(
            status in (None, "missing")
            for status in e_availability.values()
        ):
            e3_e4_match_status = "N/A"
        else:
            e3_e4_match_status = "—"
        oracle = by_source.get("oracle_full")
        oracle_match = oracle is not None
        oracle_cell_availability = availability.get(
            availability_keys["oracle_full"]
        )
        alternate_oracle_contexts = oracle_feature_contexts.get(
            (
                (
                    "phase_feature"
                    if record_type == "phase"
                    else "state_pair"
                ),
                view,
                task_description,
                int(feature_id),
            ),
            [],
        )
        oracle_feature_present = bool(alternate_oracle_contexts)
        oracle_partial_match = oracle_feature_present and not oracle_match
        if oracle is None:
            oracle_coverage = None
            oracle_low_coverage = False
        elif record_type == "phase":
            oracle_coverage = float(
                oracle["candidate"]["phase_coverage"]
            )
            oracle_low_coverage = bool(oracle["low_coverage"])
        else:
            oracle_coverage = float(
                oracle["candidate"]["coverage"]["minimum"]
            )
            oracle_low_coverage = bool(oracle["low_coverage"])
        if oracle_match:
            match_status = "✓†" if oracle_low_coverage else "✓"
        elif oracle_cell_availability in (None, "missing"):
            match_status = "N/A"
        elif oracle_partial_match:
            match_status = "partial"
        else:
            match_status = "—"

        task_family = _directional_task_family(task_description)
        family_strict_by_source = {}
        for source, recurrence_key in recurrence_keys.items():
            recurrence = recurrence_lookup.get(recurrence_key)
            family_strict_by_source[source] = bool(
                recurrence
                and recurrence.get("families", {})
                .get(task_family, {})
                .get("strict", False)
            )
        persistence_by_source = {
            source: record["persistence"]["status"]
            for source, record in by_source.items()
        }
        local_rank_by_source = {
            source: int(record["display_rank"])
            for source, record in by_source.items()
        }
        e_local_ranks = [
            local_rank_by_source[source]
            for source in e_sources
            if source in local_rank_by_source
        ]
        conservative_e_local_rank = max(e_local_ranks)
        family_strict_count = sum(family_strict_by_source.values())
        persistence_support_sources = sorted(
            source
            for source, status in persistence_by_source.items()
            if status in {"confirmed", "partial"}
        )
        persistence_support_count = len(persistence_support_sources)
        persistence_measured = any(
            status
            in {
                "confirmed",
                "partial",
                "boundary-only",
                "insufficient support",
            }
            for status in persistence_by_source.values()
        )
        probe = {
            "record_type": (
                "phase_probe"
                if record_type == "phase"
                else "state_pair_probe"
            ),
            "view": view,
            "task_description": task_description,
            "feature_id": int(feature_id),
            "e3_e4_same_feature_direction": e3_e4_match,
            "e3_e4_match_status": e3_e4_match_status,
            "e3_e4_cell_availability": e_availability,
            "oracle_same_feature_direction": oracle_match,
            "oracle_cell_availability": oracle_cell_availability,
            "oracle_feature_present": oracle_feature_present,
            "oracle_partial_different_direction": oracle_partial_match,
            "oracle_alternate_contexts": [
                (
                    {
                        "phase": record["phase"],
                        "template": record["template"],
                    }
                    if record["record_type"] == "phase_feature"
                    else {"transition": record["transition"]}
                )
                for record in alternate_oracle_contexts
                if record is not oracle
            ],
            "oracle_coverage": oracle_coverage,
            "oracle_low_coverage": oracle_low_coverage,
            "match_status": match_status,
            "support_sources": sorted(by_source),
            "family": task_family,
            "family_strict_by_source": family_strict_by_source,
            "family_strict_source_count": family_strict_count,
            "persistence_by_source": persistence_by_source,
            "persistence_support_source_count": (
                persistence_support_count
            ),
            "persistence_ranking_status": (
                (
                    "diagnostic_support_from_confirmed_or_partial_trace"
                    if persistence_support_sources
                    else (
                        "measured_neutral_without_confirmed_or_partial_trace"
                        if persistence_measured
                        else "not_measured_neutral"
                    )
                )
            ),
            "persistence_support_sources": persistence_support_sources,
            "instruction_local_rank_by_source": local_rank_by_source,
            "conservative_e3_e4_local_rank": (
                conservative_e_local_rank
            ),
            "raw_scores_aggregated_across_sources": False,
            "ranking_components": {
                "e3_e4_pair_match": e3_e4_match,
                "e3_e4_match_status": e3_e4_match_status,
                "oracle_same_direction": oracle_match,
                "oracle_partial_different_direction": (
                    oracle_partial_match
                ),
                "oracle_coverage": oracle_coverage,
                "family_strict_source_count": family_strict_count,
                "persistence_support_source_count": (
                    persistence_support_count
                ),
                "conservative_instruction_local_rank": (
                    conservative_e_local_rank
                ),
            },
        }
        if record_type == "phase":
            probe.update({"phase": phase, "template": template})
        else:
            probe.update({"transition": transition})
        groups_by_cell.setdefault(cell_key, []).append(probe)

    table_cells = []
    for cell_key, probes in sorted(groups_by_cell.items()):
        probes.sort(
            key=lambda row: (
                -int(row["e3_e4_same_feature_direction"]),
                -(
                    1
                    if row["e3_e4_match_status"] == "—"
                    else 0
                ),
                -int(row["oracle_same_feature_direction"]),
                -int(row["oracle_partial_different_direction"]),
                int(row["oracle_low_coverage"]),
                -(
                    float(row["oracle_coverage"])
                    if row["oracle_coverage"] is not None
                    else -1.0
                ),
                -int(row["family_strict_source_count"]),
                -int(row["persistence_support_source_count"]),
                int(row["conservative_e3_e4_local_rank"]),
                int(row["feature_id"]),
            )
        )
        for probe_rank, probe in enumerate(probes, start=1):
            probe["probe_rank"] = probe_rank
            probe["in_display_top10"] = probe_rank <= display_top_n
            probe_records.append(probe)
        if cell_key[0] == "phase":
            (
                _kind,
                view,
                task_description,
                phase,
                template,
            ) = cell_key
            identity = {
                "kind": "phase",
                "view": view,
                "task_description": task_description,
                "phase": phase,
                "template": template,
            }
        else:
            (
                _kind,
                view,
                task_description,
                transition,
            ) = cell_key
            identity = {
                "kind": "state_pair",
                "view": view,
                "task_description": task_description,
                "transition": transition,
            }
        table_cells.append(
            {
                **identity,
                "candidate_count": len(probes),
                "top10": [
                    {
                        "feature_id": row["feature_id"],
                        "probe_rank": row["probe_rank"],
                        "match_status": row["match_status"],
                        "e3_e4_same_feature_direction": row[
                            "e3_e4_same_feature_direction"
                        ],
                        "e3_e4_match_status": row[
                            "e3_e4_match_status"
                        ],
                        "oracle_same_feature_direction": row[
                            "oracle_same_feature_direction"
                        ],
                        "oracle_coverage": row["oracle_coverage"],
                        "family_strict_source_count": row[
                            "family_strict_source_count"
                        ],
                        "persistence_support_source_count": row[
                            "persistence_support_source_count"
                        ],
                        "instruction_local_rank_by_source": row[
                            "instruction_local_rank_by_source"
                        ],
                    }
                    for row in probes[:display_top_n]
                ],
            }
        )
    return (
        {
            "schema_version": "directional_cross_source_probe_ranking_v1",
            "source_scope": [
                "v12_e3_cov0p3",
                "v12_e4_cov0p3",
                "oracle_full",
            ],
            "same_feature_coordinate": (
                "fixed bs4096/10k SAE feature ID"
            ),
            "ranking_priority": [
                "E3/E4 same-feature same-direction match",
                "Oracle same direction, then Oracle phase coverage",
                "task-family strict recurrence",
                (
                    "confirmed/partial lossless-trace persistence support; "
                    "boundary-only, insufficient, and unmeasured are neutral"
                ),
                "conservative instruction-local rank",
                "feature ID deterministic tie-break",
            ],
            "match_status": {
                "✓": (
                    "Oracle has the same task/view/feature/phase-or-pair "
                    "direction"
                ),
                "✓†": (
                    "Oracle has the same task/view/feature/phase-or-pair "
                    "direction with coverage below 0.30"
                ),
                "partial": (
                    "Oracle has the same task/view/feature but a different "
                    "phase/template direction or transition pair"
                ),
                "—": "Oracle has no same task/view feature candidate",
                "N/A": (
                    "Oracle phase/transition cell is missing or "
                    "noncontrastable"
                ),
            },
            "raw_score_aggregation_across_sources": False,
            "cells": table_cells,
        },
        probe_records,
    )


def _directional_feature_list(
    template_cell: dict[str, Any],
) -> str:
    if template_cell["status"] == "missing":
        return "N/A"
    if template_cell["status"] == "no_candidate":
        return "—"
    return ", ".join(
        f"f{row['feature_id']}" for row in template_cell["top10"]
    )


def _directional_recurrence_list(row: dict[str, Any]) -> str:
    if row["status"] == "missing":
        return "N/A"
    if row["status"] == "no_candidate":
        return "—"
    values = []
    for feature in row["top10"]:
        family_values = "; ".join(
            (
                f"{family}="
                f"{summary['support_over_eligible']['display']}/"
                f"{summary['eligible_over_expected']['display']}/"
                f"{summary['support_over_expected']['display']}"
            )
            for family, summary in feature["families"].items()
        )
        values.append(
            (
                f"f{feature['feature_id']} "
                f"(global="
                f"{feature['support_over_eligible']['display']}/"
                f"{feature['eligible_over_expected']['display']}/"
                f"{feature['support_over_expected']['display']}; "
                f"{family_values})"
            )
        )
    return ", ".join(values)


def _render_directional_report(summary: dict[str, Any]) -> str:
    scope = summary["scope"]
    lines = [
        "# Directional W5 phase-feature discovery",
        "",
        "W5 양의 phase-vs-strongest-rest margin만 후보와 순위를 결정한다. "
        "W4 및 window/task-mean overlap은 sensitivity flag일 뿐 gate가 아니다.",
        "",
        "표의 `N/A`는 phase 누락 또는 비교 불가능, `—`는 비교는 가능하지만 "
        "양의 W5 후보가 없음을 뜻한다.",
        "",
        "## Scope",
        "",
        (
            f"- Sources {scope['source_count']}, views {scope['view_count']}, "
            f"new directional scores {scope['score_artifact_count']}"
        ),
        (
            "- Immutable legacy matrix_raw reproduction: "
            f"{scope['legacy_score_reproduction_count']}/"
            f"{scope['legacy_score_reproduction_expected']}"
        ),
        (
            f"- All phase candidates {scope['phase_candidate_count']}, "
            f"all state-pair candidates "
            f"{scope['state_pair_candidate_count']}; report는 cell별 "
            "Top-10만 표시"
        ),
        (
            f"- Lossless sparse trace scans {scope['trace_scan_count']} "
            f"(source당 1회), measured state-pair candidates "
            f"{scope['trace_measured_candidate_count']}: "
            f"{scope['trace_status_counts']}"
        ),
        "",
        "## Confound audit",
        "",
        "| Audit | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for audit in summary["confound_audit"]:
        lines.append(
            f"| {audit['dimension']} | {audit['status']} | "
            f"{audit['evidence']} |"
        )
    lines.extend(
        [
            "",
            "## Claim strength",
            "",
            f"Claim strength: `{summary['claim_strength']}`",
            "",
            "## Held claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in summary["held_claims"])
    lines.append("")
    panel_titles = (
        ("oracle", "Oracle five-cell panel"),
        ("v12_e3_e4", "V12 E3/E4 comparison panel"),
    )
    for panel_key, panel_title in panel_titles:
        lines.extend([f"## {panel_title}", ""])
        for view in summary["tables"][panel_key]:
            lines.extend(
                [
                    f"### {view['source']} / {view['view']}",
                    "",
                    "| Task | Phase | Pulse Top-10 | Step-up Top-10 | "
                    "Step-down Top-10 | Coverage |",
                    "| --- | --- | --- | --- | --- | ---: |",
                ]
            )
            for task in view["phase_table"]["tasks"]:
                for cell in task["phase_cells"]:
                    coverage = (
                        "N/A"
                        if cell["episode_coverage"] is None
                        else (
                            f"{float(cell['episode_coverage']):.2f}"
                            f"{cell['coverage_marker'] or ''}"
                        )
                    )
                    lines.append(
                        f"| {task['task_description']} | {cell['phase']} | "
                        f"{_directional_feature_list(cell['templates']['pulse'])} | "
                        f"{_directional_feature_list(cell['templates']['step_up'])} | "
                        f"{_directional_feature_list(cell['templates']['step_down'])} | "
                        f"{coverage} |"
                    )
            lines.extend(
                [
                    "",
                    "#### Phase recurrence Top-10",
                    "",
                    "| Phase | Template | Features "
                    "(support/eligible; eligible/expected; "
                    "support/expected) |",
                    "| --- | --- | --- |",
                ]
            )
            for row in view["phase_table"]["recurrence"]["rows"]:
                lines.append(
                    f"| {row['phase']} | {row['template']} | "
                    f"{_directional_recurrence_list(row)} |"
                )
            state_pairs = view["state_pairs"]
            if state_pairs["status"] == "not_applicable":
                lines.extend(
                    [
                        "",
                        "- Fine view state pair: N/A — 원본 fine label에 "
                        "단일 cross-task 시간 순서를 강제하지 않았다.",
                        "",
                    ]
                )
                continue
            lines.extend(
                [
                    "",
                    "#### Coarse4 ON→OFF state pairs",
                    "",
                    "| Task | Pair | Same-feature Top-10 |",
                    "| --- | --- | --- |",
                ]
            )
            for task in state_pairs["tasks"]:
                for cell in task["transition_cells"]:
                    if cell["status"] == "missing":
                        value = "N/A"
                    elif cell["status"] == "no_candidate":
                        value = "—"
                    else:
                        value = ", ".join(
                            (
                                f"f{candidate['feature_id']}"
                                f"{candidate['coverage_marker'] or ''}"
                                f"[{candidate['persistence_status']}]"
                            )
                            for candidate in cell["top10"]
                        )
                    lines.append(
                        f"| {task['task_description']} | "
                        f"{cell['transition']} | {value} |"
                    )
            lines.extend(
                [
                    "",
                    "#### State-pair recurrence Top-10",
                    "",
                    "| Pair | Features "
                    "(support/eligible; eligible/expected; "
                    "support/expected) |",
                    "| --- | --- |",
                ]
            )
            for row in state_pairs["recurrence"]["rows"]:
                lines.append(
                    f"| {row['transition']} | "
                    f"{_directional_recurrence_list(row)} |"
                )
            lines.append("")

    lines.extend(
        [
            "## E3/E4–Oracle probe ranking",
            "",
            "Raw score를 source 사이에서 더하거나 평균하지 않았다. "
            "동일 10k SAE feature ID를 아래 우선순위로 사전식 정렬했다.",
            "",
        ]
    )
    lines.extend(
        f"{index}. {priority}"
        for index, priority in enumerate(
            summary["probe_ranking"]["ranking_priority"],
            start=1,
        )
    )
    lines.extend(
        [
            "",
            "| View | Task | Direction | Probe Top-10 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for cell in summary["probe_ranking"]["cells"]:
        if cell["kind"] == "phase":
            direction = f"{cell['phase']} / {cell['template']}"
        else:
            direction = cell["transition"]
        probes = ", ".join(
            (
                f"f{row['feature_id']} "
                f"E3∩E4={row['e3_e4_match_status']} "
                f"Oracle={row['match_status']}"
            )
            for row in cell["top10"]
        )
        lines.append(
            f"| {cell['view']} | {cell['task_description']} | "
            f"{direction} | {probes or '—'} |"
        )
    lines.extend(
        [
            "",
            "## Legacy combined-score compatibility",
            "",
            "| Source | View | Window | Differing row-feature values | "
            "Max absolute difference | Legacy matrix_raw reproduced |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for panel_key, _title in panel_titles:
        for view in summary["tables"][panel_key]:
            for window_label in ("w4", "w5"):
                compatibility = view["compatibility"][window_label]
                relation = compatibility["legacy_raw_relation"]
                reproduction = view["legacy_reproduction"][
                    window_label
                ]
                reproduction_value = (
                    "exact"
                    if reproduction["matrix_raw_exact_torch_equal"]
                    else (
                        "tolerance"
                        if reproduction["matrix_raw_tolerance_equal"]
                        else "FAIL"
                    )
                )
                lines.append(
                    f"| {view['source']} | {view['view']} | "
                    f"{compatibility['window_size']} | "
                    f"{relation.get('num_diff', 'N/A')} | "
                    f"{relation.get('max_abs', 'N/A')} | "
                    f"{reproduction_value} |"
                )
    lines.extend(
        [
            "",
            "## Evidence limits",
            "",
            "- `†` episode coverage < 0.30. Oracle에는 cov0p3 filter를 "
            "적용하지 않았으며 low-support diagnostic으로만 해석한다.",
            "- persistence는 lossless sparse trace에서 coarse state-pair "
            "cell별 표시 Top-10만 측정했다. phase/pulse와 나머지 후보는 "
            "not-measured neutral이다.",
            "- confirmed/partial만 probe 정렬의 diagnostic support이며, "
            "어떤 persistence 상태도 후보 membership gate가 아니다.",
            "- Oracle과 V12는 anchor clock이 달라 score 크기를 직접 "
            "비교하지 않는다.",
            "- 동일 feature ID 연결은 동일한 10k SAE checkpoint "
            "coordinate 안에서만 유효하다.",
            "",
            "## Artifacts",
            "",
            f"- Summary: `{summary['outputs']['summary']}`",
            f"- All candidates: `{summary['outputs']['candidates']['path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_directional_phase_views(
    config: DirectionalPhaseViewAnalysisConfig,
) -> dict[str, Any]:
    """Rescore and report W5-primary directionality over six focused views."""

    output_dir = config.output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite directional phase-view output: "
            f"{output_dir}"
        )
    if (
        config.display_top_n != 10
        or config.control_top_n != 20
        or config.low_coverage_threshold != 0.3
    ):
        raise ValueError(
            "Directional contract fixes display_top_n=10, "
            "control_top_n=20, and low_coverage_threshold=0.3."
        )
    (
        focused_summary_spec,
        sources,
        readonly_specs,
    ) = _directional_focused_inventory(config.focused_summary)
    implementation_specs = {
        "analysis": _focused_file_spec(Path(__file__).resolve()),
        "scorer": _focused_file_spec(
            Path(score_cluster_features.__code__.co_filename)
        ),
        "directional_ranking": _focused_file_spec(
            Path(discover_directional_phase_features.__code__.co_filename)
        ),
    }
    if config.entrypoint is not None:
        implementation_specs["entrypoint"] = _focused_file_spec(
            config.entrypoint
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    tables: dict[str, list[dict[str, Any]]] = {
        "oracle": [],
        "v12_e3_e4": [],
    }
    candidate_records: list[dict[str, Any]] = []
    score_artifact_count = 0
    legacy_reproduction_count = 0
    trace_scan_count = 0
    trace_audits: dict[str, dict[str, Any]] = {}
    for source_label in DIRECTIONAL_SOURCE_ORDER:
        source = sources[source_label]
        for view in DIRECTIONAL_VIEW_ORDER:
            view_input = source["views"][view]
            score_root = output_dir / "scores" / source_label / view
            score_w4 = score_root / "w4" / "event_feature_scores.pt"
            score_w5 = score_root / "w5" / "event_feature_scores.pt"
            for window, output_path in ((4, score_w4), (5, score_w5)):
                score_cluster_features(
                    topk_run_dir=source["topk_dir"],
                    event_features_path=source["event_features"],
                    cluster_assignments_path=view_input[
                        "phase_assignments"
                    ],
                    cluster_annotations_path=view_input["phase_groups"],
                    output_path=output_path,
                    window_size=window,
                    top_n=config.control_top_n,
                    step_mapping=source["step_mapping"],
                    event_step_scale=source["event_step_scale"],
                    prompt_records_path=source["prompt_records"],
                )
                score_artifact_count += 1
            expected = source["expected"]
            score_inventory = _validate_score_pair_inventory(
                source_label=f"{source_label}/{view}",
                score_w4=score_w4,
                score_w5=score_w5,
                expected_rows=int(view_input["expected_rows"]),
                expected_episode_groups=int(
                    view_input["expected_episode_groups"]
                ),
                expected_events=int(expected["selected_events"]),
                expected_event_step_scale=int(
                    expected["event_step_scale"]
                ),
                expected_w4_shifts=int(expected["w4_shifted_windows"]),
                expected_w5_shifts=int(expected["w5_shifted_windows"]),
                expected_sample_ids=source["selected_sample_ids"],
            )

            directional_config: DirectionalDiscoveryConfig
            if view == "fine_original":
                tasks = load_directional_template_scores(
                    score_w5=score_w5,
                    score_w4=score_w4,
                )
                if len(tasks) != 5:
                    raise ValueError(
                        f"{source_label}/{view}: expected five task cells."
                    )
                fine_vocabulary = tuple(
                    sorted(
                        {
                            phase
                            for task in tasks.values()
                            for phase in task.phases
                        }
                    )
                )
                directional_config = DirectionalDiscoveryConfig(
                    control_top_n=config.control_top_n,
                    phase_order=fine_vocabulary,
                )
                phase_rankings = rank_directional_phase_candidates(
                    tasks,
                    config=directional_config,
                )
                phase_recurrence = (
                    summarize_directional_phase_recurrence(
                        phase_rankings,
                        config=directional_config,
                    )
                )
                discovery_contract = {
                    "primary_discovery_window": 5,
                    "w4_policy": "sensitivity_only_never_a_gate",
                    "control_policy": "diagnostic_flags_only",
                    "candidate_membership": (
                        "all_positive_w5_phase_vs_strongest_rest_margins"
                    ),
                    "fine_phase_vocabulary": (
                        "lexicographically sorted observed score-row labels; "
                        "not a temporal order"
                    ),
                    "state_pair_policy": (
                        "not_applicable_for_fine_original"
                    ),
                }
                transition_rankings = None
                transition_recurrence = None
            else:
                directional_config = DirectionalDiscoveryConfig(
                    control_top_n=config.control_top_n,
                    phase_order=COARSE_PHASE_ORDER,
                )
                discovery = discover_directional_phase_features(
                    score_w5=score_w5,
                    score_w4=score_w4,
                    config=directional_config,
                )
                phase_rankings = discovery["phase_rankings"]
                phase_recurrence = discovery["phase_recurrence"]
                transition_rankings = discovery[
                    "transition_rankings"
                ]
                transition_recurrence = discovery[
                    "transition_recurrence"
                ]
                discovery_contract = discovery["contract"]
                expected_pairs = [
                    f"{earlier}->{later}"
                    for earlier_index, earlier in enumerate(
                        COARSE_PHASE_ORDER
                    )
                    for later in COARSE_PHASE_ORDER[
                        earlier_index + 1 :
                    ]
                ]
                if transition_rankings[
                    "ordered_transition_pairs"
                ] != expected_pairs:
                    raise ValueError(
                        f"{source_label}/{view}: expected six coarse pairs."
                    )
                if source_label in trace_audits:
                    raise RuntimeError(
                        f"{source_label}: duplicate sparse trace scan."
                    )
                (
                    trace_persistence_by_candidate,
                    trace_audit,
                ) = _directional_source_trace_persistence(
                    source_label=source_label,
                    source=source,
                    score_w5=score_w5,
                    transition_rankings=transition_rankings,
                    display_top_n=config.display_top_n,
                    config=config.trace_persistence,
                )
                trace_scan_count += 1
                trace_audits[source_label] = trace_audit

            phase_table, phase_records = (
                _directional_phase_tables_and_records(
                    source_label=source_label,
                    source_kind=source["source_kind"],
                    view=view,
                    phase_rankings=phase_rankings,
                    phase_recurrence=phase_recurrence,
                    display_top_n=config.display_top_n,
                    low_coverage_threshold=(
                        config.low_coverage_threshold
                    ),
                )
            )
            candidate_records.extend(phase_records)
            if transition_rankings is None:
                state_pairs: dict[str, Any] = {
                    "status": "not_applicable",
                    "display": "N/A",
                    "reason": (
                        "fine labels have no forced cross-task temporal order"
                    ),
                }
            else:
                state_pairs, transition_records = (
                    _directional_transition_tables_and_records(
                        source_label=source_label,
                        source_kind=source["source_kind"],
                        view=view,
                        transition_rankings=transition_rankings,
                        transition_recurrence=transition_recurrence,
                        trace_persistence_by_candidate=(
                            trace_persistence_by_candidate
                        ),
                        display_top_n=config.display_top_n,
                        low_coverage_threshold=(
                            config.low_coverage_threshold
                        ),
                    )
                )
                candidate_records.extend(transition_records)

            score_specs = {
                "w4": _focused_file_spec(score_w4),
                "w5": _focused_file_spec(score_w5),
            }
            legacy_reproduction = {
                "w4": _directional_legacy_reproduction(
                    legacy_path=view_input["legacy_score_w4"],
                    directional_path=score_w4,
                ),
                "w5": _directional_legacy_reproduction(
                    legacy_path=view_input["legacy_score_w5"],
                    directional_path=score_w5,
                ),
            }
            legacy_reproduction_count += sum(
                bool(record["reproduced"])
                for record in legacy_reproduction.values()
            )
            table = {
                "source": source_label,
                "source_kind": source["source_kind"],
                "coverage_filter": source["coverage_filter"],
                "view": view,
                "score_artifacts": score_specs,
                "score_inventory": {
                    key: score_inventory[key]
                    for key in (
                        "num_rows",
                        "num_episode_groups",
                        "num_selected_events",
                        "w4_shifted_windows",
                        "w5_shifted_windows",
                    )
                },
                "discovery_contract": discovery_contract,
                "phase_table": phase_table,
                "state_pairs": state_pairs,
                "compatibility": {
                    "w4": _directional_score_compatibility(score_w4),
                    "w5": _directional_score_compatibility(score_w5),
                },
                "legacy_reproduction": legacy_reproduction,
            }
            panel = (
                "oracle"
                if source_label == "oracle_full"
                else "v12_e3_e4"
            )
            tables[panel].append(table)

    if score_artifact_count != 12:
        raise RuntimeError(
            f"Directional scorer call count {score_artifact_count} != 12."
        )
    if legacy_reproduction_count != 12:
        raise RuntimeError(
            "Directional legacy reproduction did not pass for all "
            f"12 score artifacts: {legacy_reproduction_count}/12."
        )
    if trace_scan_count != len(DIRECTIONAL_SOURCE_ORDER):
        raise RuntimeError(
            "Expected exactly one sparse full-trace scan per source: "
            f"{trace_scan_count}/{len(DIRECTIONAL_SOURCE_ORDER)}."
        )
    probe_ranking, probe_records = _directional_probe_ranking(
        candidate_records,
        tables=tables,
        display_top_n=config.display_top_n,
    )
    candidate_records.extend(probe_records)
    candidates_path = output_dir / "candidates.jsonl"
    _write_jsonl_exclusive(candidates_path, candidate_records)
    _assert_directional_files_unchanged(
        readonly_specs,
        kind="input",
    )
    _assert_directional_files_unchanged(
        implementation_specs,
        kind="implementation",
    )

    scope = {
        "source_count": 3,
        "view_count": 6,
        "score_artifact_count": score_artifact_count,
        "legacy_score_reproduction_count": (
            legacy_reproduction_count
        ),
        "legacy_score_reproduction_expected": 12,
        "phase_candidate_count": sum(
            row["record_type"] == "phase_feature"
            for row in candidate_records
        ),
        "displayed_phase_candidate_count": sum(
            row["record_type"] == "phase_feature"
            and row["in_display_top10"]
            for row in candidate_records
        ),
        "state_pair_candidate_count": sum(
            row["record_type"] == "state_pair"
            for row in candidate_records
        ),
        "displayed_state_pair_candidate_count": sum(
            row["record_type"] == "state_pair"
            and row["in_display_top10"]
            for row in candidate_records
        ),
        "trace_scan_count": trace_scan_count,
        "trace_measured_candidate_count": sum(
            int(audit["measured_candidate_count"])
            for audit in trace_audits.values()
        ),
        "trace_status_counts": dict(
            Counter(
                record["persistence"]["status"]
                for record in candidate_records
                if record["record_type"] == "state_pair"
                and record["in_display_top10"]
            )
        ),
        "candidate_record_count": len(candidate_records),
        "probe_candidate_count": len(probe_records),
        "displayed_probe_candidate_count": sum(
            bool(row["in_display_top10"]) for row in probe_records
        ),
    }
    candidates_spec = _focused_file_spec(candidates_path)
    summary: dict[str, Any] = {
        "schema_version": DIRECTIONAL_PHASE_VIEW_SCHEMA_VERSION,
        "scope": scope,
        "method": {
            "checkpoint_coordinate": "bs4096/10k",
            "checkpoint_sha256": FOCUSED_CHECKPOINT_SHA256,
            "primary_window": 5,
            "sensitivity_window": 4,
            "candidate_rule": (
                "for each named template independently, keep every feature "
                "whose target-phase W5 score minus strongest other observed "
                "phase W5 score is positive"
            ),
            "display_top_n": config.display_top_n,
            "controls": (
                "window_mean/task_mean Top-20 overlap flags only; never a "
                "membership, ordering, eligibility, or recurrence gate"
            ),
            "state_pair": (
                "coarse4 only; same feature, earlier step_up plus later "
                "step_down, both positive; score=min(component margins)"
            ),
            "state_pair_scope": "all_six_ordered_coarse_phase_pairs",
            "fine_transition_policy": (
                "not computed; observed fine vocabulary has no forced "
                "cross-task temporal order"
            ),
            "missing_display": "N/A",
            "no_candidate_display": "—",
            "low_coverage_threshold": config.low_coverage_threshold,
            "low_coverage_marker": "†",
            "persistence": {
                "status_scope": (
                    "confirmed, partial, boundary-only, or insufficient "
                    "support for displayed coarse state-pair Top-10 only"
                ),
                "source_trace_scan_count": trace_scan_count,
                "exactly_one_sparse_scan_per_source": True,
                "dense_recollection_required": False,
                "lossless_topk_absence_is_exact_zero": True,
                "ranking_gate": False,
                "phase_and_pulse_policy": "not_measured_neutral",
                "unselected_state_pair_policy": "not_measured_neutral",
                "thresholds": {
                    "local_window_size": (
                        config.trace_persistence.local_window_size
                    ),
                    "minimum_interval_steps": (
                        config.trace_persistence.minimum_interval_steps
                    ),
                    "activation_epsilon": (
                        config.trace_persistence.activation_epsilon
                    ),
                    "on_minimum_absolute_increase": (
                        config.trace_persistence
                        .on_minimum_absolute_increase
                    ),
                    "on_minimum_ratio": (
                        config.trace_persistence.on_minimum_ratio
                    ),
                    "interval_minimum_on_post_fraction": (
                        config.trace_persistence
                        .interval_minimum_on_post_fraction
                    ),
                    "interval_minimum_activation_prevalence": (
                        config.trace_persistence
                        .interval_minimum_activation_prevalence
                    ),
                    "off_maximum_interval_fraction": (
                        config.trace_persistence
                        .off_maximum_interval_fraction
                    ),
                    "off_minimum_absolute_decrease": (
                        config.trace_persistence
                        .off_minimum_absolute_decrease
                    ),
                    "minimum_comparable_episodes": (
                        config.trace_persistence
                        .minimum_comparable_episodes
                    ),
                    "confirmed_minimum_full_repeat_ratio": (
                        config.trace_persistence
                        .confirmed_minimum_full_repeat_ratio
                    ),
                    "partial_minimum_full_repeat_ratio": (
                        config.trace_persistence
                        .partial_minimum_full_repeat_ratio
                    ),
                    "partial_minimum_component_repeat_ratio": (
                        config.trace_persistence
                        .partial_minimum_component_repeat_ratio
                    ),
                },
            },
            "combined_score_role": "compatibility_table_only",
            "oracle_v12_magnitude_comparison": (
                "prohibited because anchor clocks differ"
            ),
        },
        "feature_identity_contract": {
            "same_checkpoint_sha256": FOCUSED_CHECKPOINT_SHA256,
            "same_integer_feature_ids_link_oracle_v12": True,
            "coordinate": "fixed bs4096/10k SAE dictionary",
            "semantic_identity_established": False,
        },
        "inputs": {
            "focused_summary": focused_summary_spec,
            "readonly_files": readonly_specs,
            "implementations": implementation_specs,
            "trace_audits": trace_audits,
            "before_after_hash_verification": {
                "input_files_unchanged": True,
                "implementation_files_unchanged": True,
            },
        },
        "tables": tables,
        "probe_ranking": probe_ranking,
        "confound_audit": _focused_confound_audit(),
        "claim_strength": "diagnostic_evidence",
        "held_claims": [
            (
                "Directional margin이 phase semantics 또는 causal control을 "
                "입증한다는 주장: confounded — 판정 보류"
            ),
            (
                "Lossless sparse ON–dwell–OFF pattern이 phase semantics나 "
                "causal control을 입증한다는 주장: task/scene/dwell/clock "
                "교란으로 confounded — 판정 보류"
            ),
            (
                "Oracle–V12 score magnitude 또는 rank 차이가 annotation "
                "우열을 입증한다는 주장: clock mismatch로 "
                "confounded — 판정 보류"
            ),
        ],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "outputs": {
            "root": str(output_dir),
            "summary": str(output_dir / "summary.json"),
            "report": str(output_dir / "report.md"),
            "candidates": candidates_spec,
        },
    }
    report = _render_directional_report(summary)
    report_path = output_dir / "report.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    summary["outputs"]["report_spec"] = _focused_file_spec(report_path)
    summary_path = output_dir / "summary.json"
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "report": str(report_path),
                "candidates": str(candidates_path),
                **scope,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


__all__ = [
    "DirectionalPhaseViewAnalysisConfig",
    "DirectionalTracePersistenceConfig",
    "EventPhaseActivationGridConfig",
    "FocusedPhaseViewAnalysisConfig",
    "ORACLE_FINE_PHASES",
    "analyze_directional_phase_views",
    "analyze_event_phase_activation_grid",
    "analyze_focused_phase_views",
    "build_fixed_phase_composition_cells",
    "build_fine_phase_cells",
    "classify_event_phase_candidates",
    "regroup_phase_assignments_to_coarse",
    "rank_phase_cells",
    "rank_task_balanced_event_features",
]
