"""Rank phase-associated SAE features within exact instructions.

This module owns task-local score loading, provenance validation, annotation
confidence diagnostics, decoder-matched ranking, and immutable JSON/Markdown
report generation. Shared inference and checkpoint-stability primitives live
in ``event_sae.scoring.phase_selectivity``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from event_sae import resolve_groot_artifact_path, sha256_file
import event_sae.scoring.rankings as ranking_tools
from event_sae.groot.oracle_phase_keyframes import (
    oracle_phase_entry_invariant_errors,
)
from event_sae.scoring.phase_selectivity import (
    PhaseFeatureRun,
    WindowRobustPhaseScores,
    episode_stratified_phase_permutations,
    feature_ranks_descending,
    holm_adjusted_p_values,
    match_decoder_features_across_checkpoints,
    max_t_p_value,
    phase_vs_rest_margin,
    phasewise_max_feature_null,
    summarize_episode_paired_support,
)


ANNOTATION_CONFIDENCE_TIERS = (
    "strong",
    "majority",
    "plurality",
    "user-directed",
)
ORACLE_ANNOTATION_CONFIDENCE_TIER = "simulator-oracle"

COARSE_PHASE_ORDER = (
    "reach",
    "grasp",
    "transport",
    "terminal",
)

DIRECTIONAL_TEMPLATE_NAMES = (
    "pulse",
    "step_up",
    "step_down",
)
DIRECTIONAL_MISSING_VALUE = "N/A"
DIRECTIONAL_NO_CANDIDATE = "—"
DEFAULT_TASK_FAMILY_SIZES = (
    ("drawer", 2),
    ("object", 3),
)

COARSE_PHASE_BY_ANNOTATION = {
    "reach-to-handle": "reach",
    "reach-to-object": "reach",
    "grasp": "grasp",
    "grasp-handle": "grasp",
    "contact": "grasp",
    "pull": "transport",
    "push-back": "transport",
    "transport": "transport",
    "disengage": "terminal",
    "insert-settle": "terminal",
    "open-done": "terminal",
    "place": "terminal",
    "terminal": "terminal",
}


CONDITION_TASK_PHASE_INFERENCE_SCOPE = "condition_task_phase_best"
ROBUST_WINDOW_HALF_WIDTH = 5


@dataclass(frozen=True)
class TaskLocalPhaseRankingConfig:
    """Configuration for task-local phase rankings in one condition."""

    runs: tuple[PhaseFeatureRun, ...]
    reference_label: str
    condition_id: str
    accepted_annotations: Path
    phase_groups: Path
    phase_assignments: Path
    output_dir: Path
    entrypoint: Path | None = None
    num_permutations: int = 5_000
    seed: int = 20_260_725
    chunk_size: int = 100
    top_n: int = 10
    alpha: float = 0.05
    expected_step_mapping: str = "action_executed"
    score_event_step_scale: int = 5
    topk_event_step_scale: int = 5
    expected_capture_target: str = "action_expert"


@dataclass
class TaskLocalPhaseScores:
    """W4/W5 scores and auxiliary matrices for one exact instruction."""

    task_description: str
    row_keys: list[dict[str, Any]]
    pair: WindowRobustPhaseScores
    group_event_counts: list[int]
    window_mean_w4: np.ndarray
    window_mean_w5: np.ndarray
    task_mean_w4: np.ndarray
    task_mean_w5: np.ndarray


@dataclass(frozen=True)
class DirectionalDiscoveryConfig:
    """Configuration for W5-primary directional feature discovery.

    ``task_family_by_description`` is optional for the five canonical cells.
    When omitted, drawer instructions are recognized by the word ``drawer``
    and pick/place instructions are assigned to the object family.  Supplying
    the mapping makes the family contract fully explicit for other suites.
    """

    control_top_n: int = 20
    phase_order: tuple[str, ...] = COARSE_PHASE_ORDER
    expected_family_sizes: tuple[
        tuple[str, int], ...
    ] = DEFAULT_TASK_FAMILY_SIZES
    task_family_by_description: tuple[tuple[str, str], ...] = ()
    global_strict_task_count: int = 5
    global_relaxed_task_count: int = 3


@dataclass
class DirectionalTemplateTaskScores:
    """Directional template matrices for one exact instruction.

    W5 is the primary discovery view.  Every W4 field is optional sensitivity
    metadata and is deliberately kept separate so that it cannot become an
    implicit candidate or eligibility gate.
    """

    task_description: str
    row_keys: list[dict[str, Any]]
    phases: list[str]
    group_keys: list[tuple[int, str]]
    group_event_counts: list[int]
    phase_coverage: dict[str, float]
    phase_template_scores_w5: dict[str, np.ndarray]
    group_template_scores_w5: dict[str, np.ndarray]
    window_mean_w5: np.ndarray
    task_mean_w5: np.ndarray
    phase_template_scores_w4: dict[str, np.ndarray] | None
    group_template_scores_w4: dict[str, np.ndarray] | None
    window_mean_w4: np.ndarray | None
    task_mean_w4: np.ndarray | None


@dataclass(frozen=True)
class CoarsePhaseCandidateRankingConfig:
    """Configuration for descriptive suite-level coarse-phase candidates."""

    stage4_root: Path
    output_dir: Path
    primary_condition: str
    sensitivity_condition: str
    discovery_coverage: str = "cov0p3"
    stability_coverage: str = "cov0p4"
    artifact_top_n: int = 10
    candidate_pool_n: int = 5
    shortlist_n: int = 3
    expected_conditions: int = 5
    entrypoint: Path | None = None


def _annotation_confidence_tier(
    annotation: dict[str, Any],
    *,
    cluster_id: str,
) -> str:
    """Return a validated annotation confidence/provenance tier."""

    tier = str(annotation.get("confidence_tier", "")).strip()
    if tier not in (
        *ANNOTATION_CONFIDENCE_TIERS,
        ORACLE_ANNOTATION_CONFIDENCE_TIER,
    ):
        raise ValueError(
            f"{cluster_id}: unsupported annotation confidence tier {tier!r}."
        )
    if tier == "user-directed":
        override = annotation.get("phase_override")
        if (
            annotation.get("status") != "user-directed-phase-override"
            or not isinstance(override, dict)
            or override.get("applied") is not True
        ):
            raise ValueError(
                f"{cluster_id}: user-directed confidence lacks explicit "
                "phase-override provenance."
            )
    if tier == ORACLE_ANNOTATION_CONFIDENCE_TIER:
        expected_fields = {
            "phase_source": "simulator_oracle_env_step_phases",
            "label_source": "env_step_phases",
            "model": "simulator_oracle_labeler",
            "review_mode": "programmatic_oracle",
            "review_verdict": "oracle_generated",
            "actual_human_review_completed": False,
            "oracle_upper_bound": True,
        }
        mismatches = {
            key: {
                "expected": expected,
                "actual": annotation.get(key),
            }
            for key, expected in expected_fields.items()
            if annotation.get(key) != expected
        }
        oracle_provenance = annotation.get("oracle_provenance")
        expected_oracle_provenance = {
            "source": "trusted_rollout_env_step_phases",
            "label_resolution": "environment_state",
            "generation": "programmatic",
            "upper_bound": True,
        }
        if mismatches or oracle_provenance != expected_oracle_provenance:
            raise ValueError(
                f"{cluster_id}: simulator-oracle confidence lacks explicit "
                "direct-label provenance."
            )
    return tier


def _annotation_label_provenance(
    annotation: dict[str, Any],
    *,
    cluster_id: str,
) -> dict[str, Any]:
    """Keep the compact source-label provenance needed for claim auditing."""

    tier = _annotation_confidence_tier(
        annotation,
        cluster_id=cluster_id,
    )
    override = annotation.get("phase_override")
    override_payload = override if isinstance(override, dict) else {}
    provenance = {
        "confidence_tier": tier,
        "status": str(annotation.get("status") or ""),
        "phase_source": str(annotation.get("phase_source") or ""),
        "phase_override": {
            "applied": bool(override_payload.get("applied", False)),
            "authorized_by": override_payload.get("authorized_by"),
            "reason": override_payload.get("reason"),
            "formal_blind_review_completed": bool(
                override_payload.get("formal_blind_review_completed", False)
            ),
            "provider_response_generated": override_payload.get(
                "provider_response_generated"
            ),
        },
    }
    if tier == ORACLE_ANNOTATION_CONFIDENCE_TIER:
        provenance["simulator_oracle"] = {
            "label_source": str(annotation["label_source"]),
            "model": str(annotation["model"]),
            "review_mode": str(annotation["review_mode"]),
            "review_verdict": str(annotation["review_verdict"]),
            "actual_human_review_completed": bool(
                annotation["actual_human_review_completed"]
            ),
            "oracle_upper_bound": bool(annotation["oracle_upper_bound"]),
            "oracle_provenance": dict(annotation["oracle_provenance"]),
        }
    return provenance


def _inferential_family_description(
    *,
    condition_id: str,
    hypothesis_count: int,
) -> str:
    """Describe the condition-local multiplicity family without aliases."""

    return (
        f"{int(hypothesis_count)} decoder-matched task-phase "
        "best-candidate hypotheses "
        f"within condition {condition_id!r}"
    )


def _phase_means(
    group_matrix: np.ndarray,
    group_keys: list[tuple[int, str]],
    phases: list[str],
) -> np.ndarray:
    return np.stack(
        [
            group_matrix[
                np.asarray([phase == target for _, phase in group_keys], dtype=bool)
            ].mean(axis=0)
            for target in phases
        ],
        axis=0,
    )


def _as_numpy_matrix(payload: dict[str, Any], key: str) -> np.ndarray:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"Score artifact missing two-dimensional tensor {key!r}.")
    matrix = value.detach().cpu().numpy().astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Score artifact contains non-finite values in {key!r}.")
    return matrix


def _score_row_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["task_description"]),
        str(row["cluster_id"]),
        str(row["phase"]),
    )


def _episode_group_identity(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["cluster_id"]),
        int(row["episode_num"]),
        int(row.get("num_events", 0)),
    )


def load_task_local_score_pair(
    score_w4: Path,
    score_w5: Path,
) -> dict[str, TaskLocalPhaseScores]:
    """Split a W4/W5 score pair into exact-instruction phase comparisons.

    Phase names may repeat across instructions.  Each instruction must contain
    at most one score row per phase, which is the phase-group builder contract.
    """

    payload_w4 = torch.load(score_w4, map_location="cpu", weights_only=False)
    payload_w5 = torch.load(score_w5, map_location="cpu", weights_only=False)
    if int(payload_w4.get("window_size", -1)) != 4:
        raise ValueError(f"Expected a W4 score artifact: {score_w4}")
    if int(payload_w5.get("window_size", -1)) != 5:
        raise ValueError(f"Expected a W5 score artifact: {score_w5}")

    rows_w4 = list(payload_w4.get("row_keys", []))
    rows_w5 = list(payload_w5.get("row_keys", []))
    identities_w4 = [_score_row_identity(row) for row in rows_w4]
    identities_w5 = [_score_row_identity(row) for row in rows_w5]
    if identities_w4 != identities_w5:
        raise ValueError("W4/W5 score-row contracts differ.")
    group_rows_w4 = list(payload_w4.get("episode_group_keys", []))
    group_rows_w5 = list(payload_w5.get("episode_group_keys", []))
    if [
        _episode_group_identity(row) for row in group_rows_w4
    ] != [
        _episode_group_identity(row) for row in group_rows_w5
    ]:
        raise ValueError("W4/W5 episode-group contracts differ.")

    matrices_w4 = {
        key: _as_numpy_matrix(payload_w4, key)
        for key in (
            "episode_group_matrix_raw",
            "matrix_raw",
            "matrix_window_mean",
            "matrix_task_mean",
        )
    }
    matrices_w5 = {
        key: _as_numpy_matrix(payload_w5, key)
        for key in (
            "episode_group_matrix_raw",
            "matrix_raw",
            "matrix_window_mean",
            "matrix_task_mean",
        )
    }
    if matrices_w4["matrix_raw"].shape != matrices_w5["matrix_raw"].shape:
        raise ValueError("W4/W5 dictionary dimensions differ.")

    cluster_to_row = {
        str(row["cluster_id"]): row for row in rows_w4
    }
    if len(cluster_to_row) != len(rows_w4):
        raise ValueError("Score rows contain duplicate cluster IDs.")
    tasks: dict[str, list[int]] = {}
    for row_idx, row in enumerate(rows_w4):
        tasks.setdefault(str(row["task_description"]), []).append(row_idx)

    output: dict[str, TaskLocalPhaseScores] = {}
    for task_description, row_indices in tasks.items():
        task_rows = [rows_w4[index] for index in row_indices]
        phases = [str(row["phase"]) for row in task_rows]
        if len(phases) != len(set(phases)):
            raise ValueError(
                f"{task_description!r} contains more than one row for a phase."
            )
        cluster_ids = {str(row["cluster_id"]) for row in task_rows}
        group_indices = [
            index
            for index, group in enumerate(group_rows_w4)
            if str(group["cluster_id"]) in cluster_ids
        ]
        group_keys = [
            (
                int(group_rows_w4[index]["episode_num"]),
                str(cluster_to_row[str(group_rows_w4[index]["cluster_id"])]["phase"]),
            )
            for index in group_indices
        ]
        group_event_counts = [
            int(group_rows_w4[index].get("num_events", 1))
            for index in group_indices
        ]
        if any(count <= 0 for count in group_event_counts):
            raise ValueError(
                f"{task_description!r}: episode-group event counts must be positive."
            )
        group_w4 = matrices_w4["episode_group_matrix_raw"][group_indices]
        group_w5 = matrices_w5["episode_group_matrix_raw"][group_indices]
        phase_w4 = _phase_means(group_w4, group_keys, phases)
        phase_w5 = _phase_means(group_w5, group_keys, phases)
        artifact_w4 = matrices_w4["matrix_raw"][row_indices]
        artifact_w5 = matrices_w5["matrix_raw"][row_indices]
        if not np.allclose(phase_w4, artifact_w4, rtol=1e-5, atol=1e-5):
            raise ValueError(
                f"{task_description!r}: reconstructed W4 phase means disagree."
            )
        if not np.allclose(phase_w5, artifact_w5, rtol=1e-5, atol=1e-5):
            raise ValueError(
                f"{task_description!r}: reconstructed W5 phase means disagree."
            )
        if len(phases) >= 2:
            margin_w4 = phase_vs_rest_margin(phase_w4)
            margin_w5 = phase_vs_rest_margin(phase_w5)
            robust_margin = np.minimum(margin_w4, margin_w5)
        else:
            margin_w4 = np.full_like(phase_w4, np.nan)
            margin_w5 = np.full_like(phase_w5, np.nan)
            robust_margin = np.full_like(phase_w5, np.nan)
        output[task_description] = TaskLocalPhaseScores(
            task_description=task_description,
            row_keys=task_rows,
            pair=WindowRobustPhaseScores(
                phases=phases,
                group_keys=group_keys,
                group_w4=group_w4,
                group_w5=group_w5,
                phase_w4=phase_w4,
                phase_w5=phase_w5,
                margin_w4=margin_w4,
                margin_w5=margin_w5,
                robust_margin=robust_margin,
            ),
            group_event_counts=group_event_counts,
            window_mean_w4=matrices_w4["matrix_window_mean"][row_indices],
            window_mean_w5=matrices_w5["matrix_window_mean"][row_indices],
            task_mean_w4=matrices_w4["matrix_task_mean"][row_indices],
            task_mean_w5=matrices_w5["matrix_task_mean"][row_indices],
        )
    return output


@dataclass
class _DirectionalScoreArtifact:
    """Validated directional matrices from one score artifact."""

    path: Path
    window_size: int
    row_keys: list[dict[str, Any]]
    group_rows: list[dict[str, Any]]
    phase_templates: dict[str, np.ndarray]
    group_templates: dict[str, np.ndarray]
    matrix_raw: np.ndarray
    group_matrix_raw: np.ndarray
    matrix_template_max: np.ndarray
    window_mean: np.ndarray
    task_mean: np.ndarray
    source_identity: dict[str, Any]
    step_mapping: str


def _validate_directional_contract_metadata(
    payload: dict[str, Any],
    *,
    score_path: Path,
) -> None:
    """Require the scorer's explicit directional decomposition contract."""

    required_tensor_keys = {
        *(f"episode_group_matrix_{name}" for name in DIRECTIONAL_TEMPLATE_NAMES),
        *(f"matrix_{name}" for name in DIRECTIONAL_TEMPLATE_NAMES),
        "matrix_template_max",
    }
    missing_tensor_keys = sorted(
        key for key in required_tensor_keys if key not in payload
    )
    definitions = payload.get("directional_score_definitions")
    template_contract = payload.get("template_matrix_contract")
    metadata_errors: set[str] = set()
    if not isinstance(definitions, dict):
        metadata_errors.add("directional_score_definitions")
    else:
        required_definition_keys = {
            "episode_group_matrix_template",
            "episode_group_matrix_raw",
            "matrix_template",
            "matrix_template_max",
            "matrix_raw",
        }
        missing_definitions = sorted(
            required_definition_keys - set(definitions)
        )
        if missing_definitions:
            metadata_errors.add(
                "directional_score_definitions."
                + ",".join(missing_definitions)
            )
    if not isinstance(template_contract, dict):
        metadata_errors.add("template_matrix_contract")
    else:
        if tuple(template_contract.get("template_names", ())) != (
            DIRECTIONAL_TEMPLATE_NAMES
        ):
            metadata_errors.add(
                "template_matrix_contract.template_names"
            )
        if (
            template_contract.get(
                "episode_group_raw_equals_template_max"
            )
            is not True
        ):
            metadata_errors.add(
                "template_matrix_contract."
                "episode_group_raw_equals_template_max"
            )
        if "raw_vs_template_max" not in template_contract:
            metadata_errors.add(
                "template_matrix_contract.raw_vs_template_max"
            )
    if missing_tensor_keys or metadata_errors:
        details = []
        if missing_tensor_keys:
            details.append(
                "missing tensors=" + ", ".join(missing_tensor_keys)
            )
        if metadata_errors:
            details.append(
                "missing/invalid metadata="
                + ", ".join(sorted(metadata_errors))
            )
        raise ValueError(
            f"{score_path}: directional template score contract is absent or "
            f"incomplete ({'; '.join(details)}). Recompute this score "
            "artifact with the directional score-matrix implementation."
        )


def _validate_matrix_shape(
    matrix: np.ndarray,
    *,
    expected_rows: int,
    expected_features: int,
    key: str,
    score_path: Path,
) -> None:
    if matrix.shape != (expected_rows, expected_features):
        raise ValueError(
            f"{score_path}: {key!r} has shape {matrix.shape}, expected "
            f"{(expected_rows, expected_features)}."
        )


def _load_directional_score_artifact(
    score_path: Path,
    *,
    expected_window_size: int,
) -> _DirectionalScoreArtifact:
    """Load and independently verify a directional score artifact."""

    score_path = Path(score_path)
    payload = torch.load(
        score_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{score_path}: score artifact must contain a mapping.")
    actual_window_size = int(payload.get("window_size", -1))
    if actual_window_size != expected_window_size:
        raise ValueError(
            f"{score_path}: expected W{expected_window_size}, found "
            f"W{actual_window_size}."
        )
    _validate_directional_contract_metadata(
        payload,
        score_path=score_path,
    )
    source = payload.get("source")
    source_identity_fields = (
        "contract_version",
        "topk_manifest_sha256",
        "event_features_sha256",
        "cluster_assignments_sha256",
        "cluster_annotations_sha256",
        "prompt_records_sha256",
        "dict_size",
        "topk",
        "layer",
        "sae_sha256",
        "capture_target",
        "event_step_scale",
        "activation_source_manifest_sha256",
        "trajectory_manifest_sha256",
    )
    if not isinstance(source, dict):
        raise ValueError(
            f"{score_path}: directional score artifact is missing source "
            "provenance."
        )
    missing_source_fields = [
        field for field in source_identity_fields if field not in source
    ]
    if missing_source_fields:
        raise ValueError(
            f"{score_path}: directional source provenance is missing "
            f"{missing_source_fields}."
        )
    source_identity = {
        field: source[field] for field in source_identity_fields
    }
    step_mapping = str(payload.get("step_mapping", "")).strip()
    if not step_mapping:
        raise ValueError(
            f"{score_path}: directional score artifact is missing "
            "step_mapping."
        )

    row_keys = list(payload.get("row_keys", []))
    group_rows = list(payload.get("episode_group_keys", []))
    if not row_keys:
        raise ValueError(f"{score_path}: row_keys must not be empty.")
    if not group_rows:
        raise ValueError(
            f"{score_path}: episode_group_keys must not be empty."
        )
    if not all(isinstance(row, dict) for row in row_keys):
        raise ValueError(f"{score_path}: every score row key must be a mapping.")
    if not all(isinstance(row, dict) for row in group_rows):
        raise ValueError(
            f"{score_path}: every episode-group key must be a mapping."
        )

    phase_templates = {
        template: _as_numpy_matrix(payload, f"matrix_{template}")
        for template in DIRECTIONAL_TEMPLATE_NAMES
    }
    group_templates = {
        template: _as_numpy_matrix(
            payload,
            f"episode_group_matrix_{template}",
        )
        for template in DIRECTIONAL_TEMPLATE_NAMES
    }
    matrix_raw = _as_numpy_matrix(payload, "matrix_raw")
    group_matrix_raw = _as_numpy_matrix(
        payload,
        "episode_group_matrix_raw",
    )
    matrix_template_max = _as_numpy_matrix(
        payload,
        "matrix_template_max",
    )
    window_mean = _as_numpy_matrix(payload, "matrix_window_mean")
    task_mean = _as_numpy_matrix(payload, "matrix_task_mean")

    num_rows, num_features = matrix_raw.shape
    if num_rows != len(row_keys) or num_features <= 0:
        raise ValueError(
            f"{score_path}: matrix_raw shape does not match row_keys."
        )
    num_groups = len(group_rows)
    for template, matrix in phase_templates.items():
        _validate_matrix_shape(
            matrix,
            expected_rows=num_rows,
            expected_features=num_features,
            key=f"matrix_{template}",
            score_path=score_path,
        )
    for template, matrix in group_templates.items():
        _validate_matrix_shape(
            matrix,
            expected_rows=num_groups,
            expected_features=num_features,
            key=f"episode_group_matrix_{template}",
            score_path=score_path,
        )
    for key, matrix in (
        ("episode_group_matrix_raw", group_matrix_raw),
        ("matrix_template_max", matrix_template_max),
        ("matrix_window_mean", window_mean),
        ("matrix_task_mean", task_mean),
    ):
        expected_rows = (
            num_groups if key == "episode_group_matrix_raw" else num_rows
        )
        _validate_matrix_shape(
            matrix,
            expected_rows=expected_rows,
            expected_features=num_features,
            key=key,
            score_path=score_path,
        )

    group_template_max = np.maximum.reduce(
        [group_templates[name] for name in DIRECTIONAL_TEMPLATE_NAMES]
    )
    if not np.array_equal(group_matrix_raw, group_template_max):
        raise ValueError(
            f"{score_path}: episode_group_matrix_raw is not the exact "
            "feature-wise maximum of the three template matrices."
        )
    row_template_max = np.maximum.reduce(
        [phase_templates[name] for name in DIRECTIONAL_TEMPLATE_NAMES]
    )
    if not np.array_equal(matrix_template_max, row_template_max):
        raise ValueError(
            f"{score_path}: matrix_template_max is not the exact "
            "feature-wise maximum of the three row template matrices."
        )

    cluster_to_row_index: dict[str, int] = {}
    for row_index, row in enumerate(row_keys):
        identity = _score_row_identity(row)
        cluster_id = identity[1]
        if cluster_id in cluster_to_row_index:
            raise ValueError(
                f"{score_path}: duplicate score-row cluster_id {cluster_id!r}."
            )
        cluster_to_row_index[cluster_id] = row_index
    cluster_to_group_indices: dict[str, list[int]] = {
        cluster_id: [] for cluster_id in cluster_to_row_index
    }
    group_identities: set[tuple[str, int]] = set()
    for group_index, group in enumerate(group_rows):
        cluster_id = str(group["cluster_id"])
        if cluster_id not in cluster_to_group_indices:
            raise ValueError(
                f"{score_path}: episode group references unknown cluster "
                f"{cluster_id!r}."
            )
        episode_identity = (cluster_id, int(group["episode_num"]))
        if episode_identity in group_identities:
            raise ValueError(
                f"{score_path}: duplicate episode group {episode_identity!r}."
            )
        group_identities.add(episode_identity)
        event_count = int(group.get("num_events", 0))
        if event_count <= 0:
            raise ValueError(
                f"{score_path}: episode-group event counts must be positive."
            )
        cluster_to_group_indices[cluster_id].append(group_index)

    for cluster_id, row_index in cluster_to_row_index.items():
        group_indices = cluster_to_group_indices[cluster_id]
        if not group_indices:
            raise ValueError(
                f"{score_path}: score row {cluster_id!r} has no episode groups."
            )
        for template in DIRECTIONAL_TEMPLATE_NAMES:
            reconstructed = group_templates[template][group_indices].mean(
                axis=0
            )
            if not np.allclose(
                phase_templates[template][row_index],
                reconstructed,
                rtol=1e-5,
                atol=1e-5,
            ):
                raise ValueError(
                    f"{score_path}: matrix_{template} row {cluster_id!r} "
                    "does not equal the episode-balanced template-group mean."
                )
        reconstructed_raw = group_matrix_raw[group_indices].mean(axis=0)
        if not np.allclose(
            matrix_raw[row_index],
            reconstructed_raw,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError(
                f"{score_path}: matrix_raw row {cluster_id!r} does not equal "
                "the episode-balanced group-raw mean."
            )

    return _DirectionalScoreArtifact(
        path=score_path,
        window_size=actual_window_size,
        row_keys=row_keys,
        group_rows=group_rows,
        phase_templates=phase_templates,
        group_templates=group_templates,
        matrix_raw=matrix_raw,
        group_matrix_raw=group_matrix_raw,
        matrix_template_max=matrix_template_max,
        window_mean=window_mean,
        task_mean=task_mean,
        source_identity=source_identity,
        step_mapping=step_mapping,
    )


def _validate_directional_pair_compatibility(
    primary: _DirectionalScoreArtifact,
    sensitivity: _DirectionalScoreArtifact,
) -> None:
    if [
        _score_row_identity(row) for row in primary.row_keys
    ] != [
        _score_row_identity(row) for row in sensitivity.row_keys
    ]:
        raise ValueError("W5/W4 directional score-row contracts differ.")
    if [
        _episode_group_identity(row) for row in primary.group_rows
    ] != [
        _episode_group_identity(row) for row in sensitivity.group_rows
    ]:
        raise ValueError("W5/W4 directional episode-group contracts differ.")
    if primary.matrix_raw.shape != sensitivity.matrix_raw.shape:
        raise ValueError("W5/W4 directional dictionary dimensions differ.")
    if primary.source_identity != sensitivity.source_identity:
        differing_fields = sorted(
            field
            for field in primary.source_identity
            if primary.source_identity[field]
            != sensitivity.source_identity[field]
        )
        raise ValueError(
            "W5/W4 directional source/SAE provenance differs in "
            f"{differing_fields}."
        )
    if primary.step_mapping != sensitivity.step_mapping:
        raise ValueError("W5/W4 directional step_mapping values differ.")


def load_directional_template_scores(
    score_w5: Path,
    score_w4: Path | None = None,
) -> dict[str, DirectionalTemplateTaskScores]:
    """Load W5-primary directional scores, with optional W4 sensitivity.

    The function intentionally rejects legacy score artifacts that do not
    carry the three named template tensors and their explicit scorer contract.
    W4 absence is represented by ``None`` fields.  A supplied but invalid W4
    artifact raises instead of being silently converted to missing data.
    """

    primary = _load_directional_score_artifact(
        Path(score_w5),
        expected_window_size=5,
    )
    sensitivity = None
    if score_w4 is not None:
        sensitivity = _load_directional_score_artifact(
            Path(score_w4),
            expected_window_size=4,
        )
        _validate_directional_pair_compatibility(primary, sensitivity)

    cluster_to_row = {
        str(row["cluster_id"]): row for row in primary.row_keys
    }
    task_row_indices: dict[str, list[int]] = {}
    for row_index, row in enumerate(primary.row_keys):
        task_row_indices.setdefault(
            str(row["task_description"]),
            [],
        ).append(row_index)

    tasks: dict[str, DirectionalTemplateTaskScores] = {}
    for task_description, row_indices in task_row_indices.items():
        task_rows = [primary.row_keys[index] for index in row_indices]
        phases = [str(row["phase"]) for row in task_rows]
        if len(phases) != len(set(phases)):
            raise ValueError(
                f"{task_description!r} contains more than one score row for "
                "the same phase."
            )
        cluster_ids = {str(row["cluster_id"]) for row in task_rows}
        group_indices = [
            index
            for index, group in enumerate(primary.group_rows)
            if str(group["cluster_id"]) in cluster_ids
        ]
        group_keys = [
            (
                int(primary.group_rows[index]["episode_num"]),
                str(
                    cluster_to_row[
                        str(primary.group_rows[index]["cluster_id"])
                    ]["phase"]
                ),
            )
            for index in group_indices
        ]
        group_event_counts = [
            int(primary.group_rows[index]["num_events"])
            for index in group_indices
        ]
        phase_coverage = {}
        for row in task_rows:
            phase = str(row["phase"])
            coverage = float(row.get("episode_coverage", 0.0))
            if not np.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
                raise ValueError(
                    f"{task_description!r}/{phase!r}: episode coverage must "
                    "be finite and lie in [0, 1]."
                )
            phase_coverage[phase] = coverage

        task_mean_w5 = primary.task_mean[row_indices]
        if not np.allclose(
            task_mean_w5,
            task_mean_w5[0][None, :],
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError(
                f"{task_description!r}: W5 task-mean rows disagree."
            )
        task_mean_w4 = None
        if sensitivity is not None:
            task_mean_w4 = sensitivity.task_mean[row_indices]
            if not np.allclose(
                task_mean_w4,
                task_mean_w4[0][None, :],
                rtol=1e-5,
                atol=1e-5,
            ):
                raise ValueError(
                    f"{task_description!r}: W4 task-mean rows disagree."
                )

        tasks[task_description] = DirectionalTemplateTaskScores(
            task_description=task_description,
            row_keys=task_rows,
            phases=phases,
            group_keys=group_keys,
            group_event_counts=group_event_counts,
            phase_coverage=phase_coverage,
            phase_template_scores_w5={
                template: primary.phase_templates[template][row_indices]
                for template in DIRECTIONAL_TEMPLATE_NAMES
            },
            group_template_scores_w5={
                template: primary.group_templates[template][group_indices]
                for template in DIRECTIONAL_TEMPLATE_NAMES
            },
            window_mean_w5=primary.window_mean[row_indices],
            task_mean_w5=task_mean_w5,
            phase_template_scores_w4=(
                {
                    template: sensitivity.phase_templates[template][
                        row_indices
                    ]
                    for template in DIRECTIONAL_TEMPLATE_NAMES
                }
                if sensitivity is not None
                else None
            ),
            group_template_scores_w4=(
                {
                    template: sensitivity.group_templates[template][
                        group_indices
                    ]
                    for template in DIRECTIONAL_TEMPLATE_NAMES
                }
                if sensitivity is not None
                else None
            ),
            window_mean_w4=(
                sensitivity.window_mean[row_indices]
                if sensitivity is not None
                else None
            ),
            task_mean_w4=task_mean_w4,
        )
    return tasks


def _empirical_percentiles(values: np.ndarray) -> np.ndarray:
    """Return right-continuous empirical percentiles, with larger being better."""

    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Percentiles require a non-empty one-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError("Percentiles require finite values.")
    ordered = np.sort(values)
    return (
        100.0
        * np.searchsorted(ordered, values, side="right")
        / float(len(values))
    )


def _episode_support_summary(
    *,
    group_matrix: np.ndarray,
    group_keys: list[tuple[int, str]],
    target_phase: str,
    feature_id: int,
) -> dict[str, Any]:
    """Count exact-episode target-vs-strongest-rest positive margins."""

    positive_episode_ids = []
    comparable_episode_ids = []
    for episode_num in sorted({episode for episode, _ in group_keys}):
        target_indices = [
            index
            for index, (episode, phase) in enumerate(group_keys)
            if episode == episode_num and phase == target_phase
        ]
        other_indices = [
            index
            for index, (episode, phase) in enumerate(group_keys)
            if episode == episode_num and phase != target_phase
        ]
        if not target_indices or not other_indices:
            continue
        if len(target_indices) != 1:
            raise ValueError(
                "Expected one directional episode group per task, episode, "
                "and phase."
            )
        comparable_episode_ids.append(int(episode_num))
        target_score = group_matrix[target_indices[0], feature_id]
        strongest_rest = group_matrix[other_indices, feature_id].max()
        if target_score - strongest_rest > 0:
            positive_episode_ids.append(int(episode_num))
    positive = len(positive_episode_ids)
    comparable = len(comparable_episode_ids)
    return {
        "positive": positive,
        "comparable": comparable,
        "fraction": (
            float(positive / comparable) if comparable > 0 else None
        ),
        "display": (
            f"{positive}/{comparable}"
            if comparable > 0
            else DIRECTIONAL_MISSING_VALUE
        ),
        "positive_episode_ids": positive_episode_ids,
        "comparable_episode_ids": comparable_episode_ids,
    }


def _paired_episode_support_summary(
    on_support: dict[str, Any],
    off_support: dict[str, Any],
) -> dict[str, Any]:
    """Intersect ON/OFF evidence within the same exact-task episodes."""

    on_positive = set(on_support["positive_episode_ids"])
    off_positive = set(off_support["positive_episode_ids"])
    on_comparable = set(on_support["comparable_episode_ids"])
    off_comparable = set(off_support["comparable_episode_ids"])
    positive_episode_ids = sorted(on_positive & off_positive)
    comparable_episode_ids = sorted(on_comparable & off_comparable)
    if not set(positive_episode_ids).issubset(comparable_episode_ids):
        raise ValueError(
            "Paired positive episodes must be a subset of paired comparable "
            "episodes."
        )
    positive = len(positive_episode_ids)
    comparable = len(comparable_episode_ids)
    return {
        "positive": positive,
        "comparable": comparable,
        "fraction": (
            float(positive / comparable) if comparable > 0 else None
        ),
        "display": (
            f"{positive}/{comparable}"
            if comparable > 0
            else DIRECTIONAL_MISSING_VALUE
        ),
        "positive_episode_ids": positive_episode_ids,
        "comparable_episode_ids": comparable_episode_ids,
    }


def _missing_sensitivity_payload() -> dict[str, Any]:
    return {
        "status": "not_supplied",
        "display": DIRECTIONAL_MISSING_VALUE,
        "raw_score": None,
        "raw_score_rank": None,
        "margin": None,
        "margin_rank": None,
        "margin_percentile": None,
        "positive_margin": None,
    }


def _missing_control_sensitivity_payload() -> dict[str, Any]:
    return {
        "status": "not_supplied",
        "display": DIRECTIONAL_MISSING_VALUE,
        "window_mean_rank": None,
        "task_mean_rank": None,
        "overlaps_window_top_n": None,
        "overlaps_task_top_n": None,
        "overlap": None,
    }


def _missing_episode_support_payload() -> dict[str, Any]:
    return {
        "status": "not_supplied",
        "display": DIRECTIONAL_MISSING_VALUE,
        "positive": None,
        "comparable": None,
        "fraction": None,
        "positive_episode_ids": None,
        "comparable_episode_ids": None,
    }


def _validate_directional_config(
    config: DirectionalDiscoveryConfig,
) -> None:
    if config.control_top_n <= 0:
        raise ValueError("control_top_n must be positive.")
    if len(config.phase_order) < 2:
        raise ValueError("phase_order must contain at least two phases.")
    if len(config.phase_order) != len(set(config.phase_order)):
        raise ValueError("phase_order contains duplicate phases.")
    expected_sizes = dict(config.expected_family_sizes)
    if (
        len(expected_sizes) != len(config.expected_family_sizes)
        or any(count <= 0 for count in expected_sizes.values())
    ):
        raise ValueError(
            "expected_family_sizes must contain unique positive family sizes."
        )
    if sum(expected_sizes.values()) != config.global_strict_task_count:
        raise ValueError(
            "Expected family sizes must sum to global_strict_task_count."
        )
    if not (
        1
        <= config.global_relaxed_task_count
        <= config.global_strict_task_count
    ):
        raise ValueError(
            "global_relaxed_task_count must lie within the expected suite."
        )


def rank_directional_phase_candidates(
    tasks: dict[str, DirectionalTemplateTaskScores],
    *,
    config: DirectionalDiscoveryConfig = DirectionalDiscoveryConfig(),
) -> dict[str, Any]:
    """Keep every positive W5 directional candidate for every task phase.

    Candidate membership is exactly ``W5 margin > 0``, where margin is the
    named-template phase score minus the strongest other observed phase score
    in the same exact task.  Top-N controls are flags only.  W4 measurements
    are sensitivity annotations only and never affect membership, ordering,
    or eligibility.
    """

    _validate_directional_config(config)
    if not tasks:
        raise ValueError("Directional discovery requires at least one task.")
    task_results: dict[str, Any] = {}
    for task_description, task in sorted(tasks.items()):
        if task_description != task.task_description:
            raise ValueError("Directional task mapping key and payload differ.")
        num_phases = len(task.phases)
        if num_phases == 0:
            raise ValueError(f"{task_description!r}: phases must not be empty.")
        phase_to_index = {
            phase: index for index, phase in enumerate(task.phases)
        }
        if len(phase_to_index) != num_phases:
            raise ValueError(
                f"{task_description!r}: duplicate phase labels."
            )
        num_features = task.window_mean_w5.shape[1]
        if task.window_mean_w5.shape != (num_phases, num_features):
            raise ValueError(
                f"{task_description!r}: W5 window-mean shape disagrees."
            )
        if task.task_mean_w5.shape != (num_phases, num_features):
            raise ValueError(
                f"{task_description!r}: W5 task-mean shape disagrees."
            )
        has_w4 = task.phase_template_scores_w4 is not None
        if has_w4 != (task.group_template_scores_w4 is not None):
            raise ValueError(
                f"{task_description!r}: partial W4 template payload."
            )
        if has_w4 != (task.window_mean_w4 is not None):
            raise ValueError(
                f"{task_description!r}: partial W4 control payload."
            )
        if has_w4 != (task.task_mean_w4 is not None):
            raise ValueError(
                f"{task_description!r}: partial W4 task-mean payload."
            )

        w5_margins = {}
        w5_margin_percentiles = {}
        w5_margin_ranks = {}
        w5_raw_ranks = {}
        w4_margins: dict[str, np.ndarray] = {}
        w4_margin_percentiles: dict[str, np.ndarray] = {}
        w4_margin_ranks: dict[str, np.ndarray] = {}
        w4_raw_ranks: dict[str, np.ndarray] = {}
        for template in DIRECTIONAL_TEMPLATE_NAMES:
            matrix_w5 = task.phase_template_scores_w5[template]
            if matrix_w5.shape != (num_phases, num_features):
                raise ValueError(
                    f"{task_description!r}: W5 {template} shape disagrees."
                )
            if num_phases >= 2:
                margin_matrix_w5 = phase_vs_rest_margin(matrix_w5)
                w5_margins[template] = margin_matrix_w5
                w5_margin_percentiles[template] = np.stack(
                    [
                        _empirical_percentiles(row)
                        for row in margin_matrix_w5
                    ]
                )
                w5_margin_ranks[template] = np.stack(
                    [
                        feature_ranks_descending(row)
                        for row in margin_matrix_w5
                    ]
                )
            w5_raw_ranks[template] = np.stack(
                [feature_ranks_descending(row) for row in matrix_w5]
            )
            if has_w4:
                assert task.phase_template_scores_w4 is not None
                matrix_w4 = task.phase_template_scores_w4[template]
                if matrix_w4.shape != (num_phases, num_features):
                    raise ValueError(
                        f"{task_description!r}: W4 {template} shape "
                        "disagrees."
                    )
                if num_phases >= 2:
                    margin_matrix_w4 = phase_vs_rest_margin(matrix_w4)
                    w4_margins[template] = margin_matrix_w4
                    w4_margin_percentiles[template] = np.stack(
                        [
                            _empirical_percentiles(row)
                            for row in margin_matrix_w4
                        ]
                    )
                    w4_margin_ranks[template] = np.stack(
                        [
                            feature_ranks_descending(row)
                            for row in margin_matrix_w4
                        ]
                    )
                w4_raw_ranks[template] = np.stack(
                    [feature_ranks_descending(row) for row in matrix_w4]
                )

        window_ranks_w5 = np.stack(
            [
                feature_ranks_descending(row)
                for row in task.window_mean_w5
            ]
        )
        task_ranks_w5 = np.stack(
            [
                feature_ranks_descending(row)
                for row in task.task_mean_w5
            ]
        )
        window_ranks_w4 = None
        task_ranks_w4 = None
        if has_w4:
            assert task.window_mean_w4 is not None
            assert task.task_mean_w4 is not None
            window_ranks_w4 = np.stack(
                [
                    feature_ranks_descending(row)
                    for row in task.window_mean_w4
                ]
            )
            task_ranks_w4 = np.stack(
                [
                    feature_ranks_descending(row)
                    for row in task.task_mean_w4
                ]
            )

        phase_results: dict[str, Any] = {}
        for phase in config.phase_order:
            phase_index = phase_to_index.get(phase)
            if phase_index is None or num_phases < 2:
                reason = (
                    "phase_not_observed"
                    if phase_index is None
                    else "fewer_than_two_observed_phases"
                )
                phase_results[phase] = {
                    "status": "missing",
                    "display": DIRECTIONAL_MISSING_VALUE,
                    "reason": reason,
                    "phase_coverage": (
                        task.phase_coverage.get(phase)
                        if phase_index is not None
                        else None
                    ),
                    "templates": {
                        template: {
                            "status": "missing",
                            "display": DIRECTIONAL_MISSING_VALUE,
                            "candidates": None,
                            "num_candidates": None,
                        }
                        for template in DIRECTIONAL_TEMPLATE_NAMES
                    },
                }
                continue

            template_results = {}
            total_candidates = 0
            for template in DIRECTIONAL_TEMPLATE_NAMES:
                margin_w5 = w5_margins[template][phase_index]
                positive_feature_ids = np.flatnonzero(margin_w5 > 0)
                positive_feature_ids = np.asarray(
                    sorted(
                        positive_feature_ids.tolist(),
                        key=lambda feature_id: (
                            -float(margin_w5[feature_id]),
                            int(feature_id),
                        ),
                    ),
                    dtype=np.int64,
                )
                candidates = []
                for feature_value in positive_feature_ids:
                    feature_id = int(feature_value)
                    window_rank_w5 = int(
                        window_ranks_w5[phase_index, feature_id]
                    )
                    task_rank_w5 = int(
                        task_ranks_w5[phase_index, feature_id]
                    )
                    overlaps_window_w5 = (
                        window_rank_w5 <= config.control_top_n
                    )
                    overlaps_task_w5 = (
                        task_rank_w5 <= config.control_top_n
                    )
                    control_overlap_w5 = (
                        overlaps_window_w5 or overlaps_task_w5
                    )

                    sensitivity_payload = _missing_sensitivity_payload()
                    control_sensitivity = (
                        _missing_control_sensitivity_payload()
                    )
                    support_w4 = _missing_episode_support_payload()
                    if has_w4:
                        assert task.phase_template_scores_w4 is not None
                        assert task.group_template_scores_w4 is not None
                        assert window_ranks_w4 is not None
                        assert task_ranks_w4 is not None
                        margin_value_w4 = float(
                            w4_margins[template][
                                phase_index,
                                feature_id,
                            ]
                        )
                        sensitivity_payload = {
                            "status": "available",
                            "display": None,
                            "raw_score": float(
                                task.phase_template_scores_w4[template][
                                    phase_index,
                                    feature_id,
                                ]
                            ),
                            "raw_score_rank": int(
                                w4_raw_ranks[template][
                                    phase_index,
                                    feature_id,
                                ]
                            ),
                            "margin": margin_value_w4,
                            "margin_rank": int(
                                w4_margin_ranks[template][
                                    phase_index,
                                    feature_id,
                                ]
                            ),
                            "margin_percentile": float(
                                w4_margin_percentiles[template][
                                    phase_index,
                                    feature_id,
                                ]
                            ),
                            "positive_margin": margin_value_w4 > 0,
                        }
                        window_rank_w4 = int(
                            window_ranks_w4[phase_index, feature_id]
                        )
                        task_rank_w4 = int(
                            task_ranks_w4[phase_index, feature_id]
                        )
                        overlaps_window_w4 = (
                            window_rank_w4 <= config.control_top_n
                        )
                        overlaps_task_w4 = (
                            task_rank_w4 <= config.control_top_n
                        )
                        control_sensitivity = {
                            "status": "available",
                            "display": None,
                            "window_mean_rank": window_rank_w4,
                            "task_mean_rank": task_rank_w4,
                            "overlaps_window_top_n": overlaps_window_w4,
                            "overlaps_task_top_n": overlaps_task_w4,
                            "overlap": (
                                overlaps_window_w4 or overlaps_task_w4
                            ),
                        }
                        support_w4 = {
                            "status": "available",
                            **_episode_support_summary(
                                group_matrix=(
                                    task.group_template_scores_w4[template]
                                ),
                                group_keys=task.group_keys,
                                target_phase=phase,
                                feature_id=feature_id,
                            ),
                        }

                    candidates.append(
                        {
                            "feature_id": feature_id,
                            "template": template,
                            "task_description": task_description,
                            "phase": phase,
                            "phase_coverage": float(
                                task.phase_coverage[phase]
                            ),
                            "w5": {
                                "raw_score": float(
                                    task.phase_template_scores_w5[template][
                                        phase_index,
                                        feature_id,
                                    ]
                                ),
                                "raw_score_rank": int(
                                    w5_raw_ranks[template][
                                        phase_index,
                                        feature_id,
                                    ]
                                ),
                                "margin": float(
                                    margin_w5[feature_id]
                                ),
                                "margin_rank": int(
                                    w5_margin_ranks[template][
                                        phase_index,
                                        feature_id,
                                    ]
                                ),
                                "margin_percentile": float(
                                    w5_margin_percentiles[template][
                                        phase_index,
                                        feature_id,
                                    ]
                                ),
                            },
                            "w4_sensitivity": sensitivity_payload,
                            "episode_support": {
                                "w5": _episode_support_summary(
                                    group_matrix=(
                                        task.group_template_scores_w5[
                                            template
                                        ]
                                    ),
                                    group_keys=task.group_keys,
                                    target_phase=phase,
                                    feature_id=feature_id,
                                ),
                                "w4_sensitivity": support_w4,
                            },
                            "controls": {
                                "w5": {
                                    "window_mean_rank": window_rank_w5,
                                    "task_mean_rank": task_rank_w5,
                                    "overlaps_window_top_n": (
                                        overlaps_window_w5
                                    ),
                                    "overlaps_task_top_n": (
                                        overlaps_task_w5
                                    ),
                                    "overlap": control_overlap_w5,
                                },
                                "w4_sensitivity": control_sensitivity,
                            },
                            "eligibility": {
                                "eligible": True,
                                "basis": (
                                    "positive_w5_phase_vs_strongest_rest_"
                                    "margin"
                                ),
                                "w4_is_gate": False,
                                "controls_are_gate": False,
                                "reasons": [],
                            },
                        }
                    )
                total_candidates += len(candidates)
                template_results[template] = {
                    "status": (
                        "available" if candidates else "no_candidate"
                    ),
                    "display": (
                        None if candidates else DIRECTIONAL_NO_CANDIDATE
                    ),
                    "num_candidates": len(candidates),
                    "num_eligible": sum(
                        candidate["eligibility"]["eligible"]
                        for candidate in candidates
                    ),
                    "candidates": candidates,
                }
            phase_results[phase] = {
                "status": (
                    "available" if total_candidates else "no_candidate"
                ),
                "display": (
                    None
                    if total_candidates
                    else DIRECTIONAL_NO_CANDIDATE
                ),
                "phase_coverage": float(task.phase_coverage[phase]),
                "num_candidates": total_candidates,
                "templates": template_results,
            }
        task_results[task_description] = {
            "observed_phases": list(task.phases),
            "unranked_observed_phases": [
                phase
                for phase in task.phases
                if phase not in config.phase_order
            ],
            "w4_sensitivity_status": (
                "available" if has_w4 else "not_supplied"
            ),
            "phases": phase_results,
        }

    return {
        "schema_version": "directional_phase_candidates_v1",
        "primary_window": 5,
        "sensitivity_window": (
            4
            if all(
                task.phase_template_scores_w4 is not None
                for task in tasks.values()
            )
            else None
        ),
        "phase_order": list(config.phase_order),
        "template_names": list(DIRECTIONAL_TEMPLATE_NAMES),
        "control_top_n": config.control_top_n,
        "candidate_membership": "positive_w5_phase_vs_strongest_rest_margin",
        "candidate_truncation": "none",
        "w4_policy": "sensitivity_only_never_a_gate",
        "control_policy": (
            "w5_and_w4_top_n_overlap_are_diagnostic_flags_only; "
            "never_membership_order_or_eligibility_gates"
        ),
        "percentile_definition": (
            "right-continuous empirical percentile among all SAE features; "
            "larger is better"
        ),
        "sentinels": {
            "missing": DIRECTIONAL_MISSING_VALUE,
            "no_candidate": DIRECTIONAL_NO_CANDIDATE,
        },
        "tasks": task_results,
    }


def _candidate_by_feature(
    phase_result: dict[str, Any],
    *,
    template: str,
) -> dict[int, dict[str, Any]] | None:
    template_result = phase_result["templates"][template]
    candidates = template_result["candidates"]
    if candidates is None:
        return None
    return {
        int(candidate["feature_id"]): candidate
        for candidate in candidates
    }


def _transition_component(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": candidate["phase"],
        "template": candidate["template"],
        "phase_coverage": candidate["phase_coverage"],
        "w5": dict(candidate["w5"]),
        "w4_sensitivity": dict(candidate["w4_sensitivity"]),
        "episode_support": dict(candidate["episode_support"]),
        "controls": dict(candidate["controls"]),
        "eligibility": dict(candidate["eligibility"]),
    }


def rank_directional_transition_candidates(
    phase_rankings: dict[str, Any],
    *,
    config: DirectionalDiscoveryConfig = DirectionalDiscoveryConfig(),
) -> dict[str, Any]:
    """Pair earlier ON and later OFF candidates for all ordered phase pairs."""

    _validate_directional_config(config)
    if phase_rankings.get("schema_version") != (
        "directional_phase_candidates_v1"
    ):
        raise ValueError("Unsupported directional phase-ranking schema.")
    if tuple(phase_rankings.get("phase_order", ())) != config.phase_order:
        raise ValueError("Phase-ranking order and transition config differ.")
    task_phase_results = phase_rankings.get("tasks")
    if not isinstance(task_phase_results, dict) or not task_phase_results:
        raise ValueError("Directional phase rankings contain no tasks.")

    ordered_pairs = [
        (earlier, later)
        for earlier_index, earlier in enumerate(config.phase_order)
        for later in config.phase_order[earlier_index + 1 :]
    ]
    task_results: dict[str, Any] = {}
    for task_description, task_result in sorted(task_phase_results.items()):
        transition_results = {}
        for earlier_phase, later_phase in ordered_pairs:
            transition_name = f"{earlier_phase}->{later_phase}"
            earlier_result = task_result["phases"][earlier_phase]
            later_result = task_result["phases"][later_phase]
            if (
                earlier_result["status"] == "missing"
                or later_result["status"] == "missing"
            ):
                transition_results[transition_name] = {
                    "status": "missing",
                    "display": DIRECTIONAL_MISSING_VALUE,
                    "earlier_phase": earlier_phase,
                    "later_phase": later_phase,
                    "reason": "one_or_both_phases_missing",
                    "num_candidates": None,
                    "num_eligible": None,
                    "candidates": None,
                }
                continue
            on_by_feature = _candidate_by_feature(
                earlier_result,
                template="step_up",
            )
            off_by_feature = _candidate_by_feature(
                later_result,
                template="step_down",
            )
            if on_by_feature is None or off_by_feature is None:
                raise ValueError(
                    "Observed phase unexpectedly lacks candidate collection."
                )
            feature_ids = sorted(
                set(on_by_feature) & set(off_by_feature)
            )
            candidates = []
            for feature_id in feature_ids:
                on_candidate = on_by_feature[feature_id]
                off_candidate = off_by_feature[feature_id]
                pair_score_w5 = min(
                    float(on_candidate["w5"]["margin"]),
                    float(off_candidate["w5"]["margin"]),
                )
                conservative_percentile_w5 = min(
                    float(on_candidate["w5"]["margin_percentile"]),
                    float(off_candidate["w5"]["margin_percentile"]),
                )
                w4_on = on_candidate["w4_sensitivity"]
                w4_off = off_candidate["w4_sensitivity"]
                if (
                    w4_on["status"] == "available"
                    and w4_off["status"] == "available"
                ):
                    pair_score_w4 = min(
                        float(w4_on["margin"]),
                        float(w4_off["margin"]),
                    )
                    w4_sensitivity = {
                        "status": "available",
                        "display": None,
                        "pair_score": pair_score_w4,
                        "conservative_margin_percentile": min(
                            float(w4_on["margin_percentile"]),
                            float(w4_off["margin_percentile"]),
                        ),
                        "both_component_margins_positive": (
                            float(w4_on["margin"]) > 0
                            and float(w4_off["margin"]) > 0
                        ),
                    }
                else:
                    w4_sensitivity = {
                        "status": "not_supplied",
                        "display": DIRECTIONAL_MISSING_VALUE,
                        "pair_score": None,
                        "conservative_margin_percentile": None,
                        "both_component_margins_positive": None,
                    }

                control_overlap_w5 = bool(
                    on_candidate["controls"]["w5"]["overlap"]
                    or off_candidate["controls"]["w5"]["overlap"]
                )
                on_w4_controls = on_candidate["controls"][
                    "w4_sensitivity"
                ]
                off_w4_controls = off_candidate["controls"][
                    "w4_sensitivity"
                ]
                if (
                    on_w4_controls["status"] == "available"
                    and off_w4_controls["status"] == "available"
                ):
                    control_overlap_w4 = bool(
                        on_w4_controls["overlap"]
                        or off_w4_controls["overlap"]
                    )
                else:
                    control_overlap_w4 = None
                on_episode_support = on_candidate["episode_support"]
                off_episode_support = off_candidate["episode_support"]
                episode_pair_support = {
                    "status": "available",
                    **_paired_episode_support_summary(
                        on_episode_support["w5"],
                        off_episode_support["w5"],
                    ),
                    "basis": (
                        "intersection of same-episode earlier step_up and "
                        "later step_down comparable/positive evidence"
                    ),
                    "ranking_gate": False,
                }
                on_w4_support = on_episode_support["w4_sensitivity"]
                off_w4_support = off_episode_support["w4_sensitivity"]
                if (
                    on_w4_support["status"] == "available"
                    and off_w4_support["status"] == "available"
                ):
                    episode_pair_support["w4_sensitivity"] = {
                        "status": "available",
                        **_paired_episode_support_summary(
                            on_w4_support,
                            off_w4_support,
                        ),
                        "ranking_gate": False,
                    }
                else:
                    episode_pair_support["w4_sensitivity"] = (
                        _missing_episode_support_payload()
                    )
                coverage_values = [
                    float(on_candidate["phase_coverage"]),
                    float(off_candidate["phase_coverage"]),
                ]
                candidates.append(
                    {
                        "feature_id": feature_id,
                        "task_description": task_description,
                        "earlier_phase": earlier_phase,
                        "later_phase": later_phase,
                        "on_template": "step_up",
                        "off_template": "step_down",
                        "pair_score_w5": pair_score_w5,
                        "conservative_margin_percentile_w5": (
                            conservative_percentile_w5
                        ),
                        "w4_sensitivity": w4_sensitivity,
                        "episode_pair_support": episode_pair_support,
                        "coverage": {
                            "earlier": coverage_values[0],
                            "later": coverage_values[1],
                            "minimum": min(coverage_values),
                            "maximum": max(coverage_values),
                        },
                        "on": _transition_component(on_candidate),
                        "off": _transition_component(off_candidate),
                        "control_overlap_w5": control_overlap_w5,
                        "control_overlap_w4_sensitivity": (
                            control_overlap_w4
                        ),
                        "eligibility": {
                            "eligible": True,
                            "basis": (
                                "both_required_w5_component_margins_are_"
                                "positive"
                            ),
                            "w4_is_gate": False,
                            "controls_are_gate": False,
                            "reasons": [],
                        },
                    }
                )
            candidates.sort(
                key=lambda candidate: (
                    -float(candidate["pair_score_w5"]),
                    int(candidate["feature_id"]),
                )
            )
            transition_results[transition_name] = {
                "status": (
                    "available" if candidates else "no_candidate"
                ),
                "display": (
                    None
                    if candidates
                    else DIRECTIONAL_NO_CANDIDATE
                ),
                "earlier_phase": earlier_phase,
                "later_phase": later_phase,
                "num_candidates": len(candidates),
                "num_eligible": sum(
                    candidate["eligibility"]["eligible"]
                    for candidate in candidates
                ),
                "candidates": candidates,
            }
        task_results[task_description] = {
            "w4_sensitivity_status": task_result[
                "w4_sensitivity_status"
            ],
            "transitions": transition_results,
        }

    return {
        "schema_version": "directional_transition_candidates_v1",
        "phase_order": list(config.phase_order),
        "ordered_transition_pairs": [
            f"{earlier}->{later}" for earlier, later in ordered_pairs
        ],
        "pair_scope": "all_ordered_earlier_later_pairs_not_only_adjacent",
        "pair_definition": (
            "same feature; earlier positive W5 step_up margin and later "
            "positive W5 step_down margin; score=min(component margins)"
        ),
        "pair_percentile_definition": (
            "minimum of the two component empirical margin percentiles"
        ),
        "pulse_policy": "reported_per_phase_but_not_used_in_transitions",
        "episode_pair_support_policy": (
            "same-task episode intersection of earlier step_up and later "
            "step_down comparable/positive evidence; diagnostic only"
        ),
        "w4_policy": "sensitivity_only_never_a_gate",
        "control_policy": (
            "window/task-mean overlap is diagnostic only and never changes "
            "transition membership, ordering, or eligibility"
        ),
        "sentinels": {
            "missing": DIRECTIONAL_MISSING_VALUE,
            "no_candidate": DIRECTIONAL_NO_CANDIDATE,
        },
        "tasks": task_results,
    }


def _resolve_directional_task_families(
    task_descriptions: list[str],
    *,
    config: DirectionalDiscoveryConfig,
) -> dict[str, str]:
    explicit = dict(config.task_family_by_description)
    if len(explicit) != len(config.task_family_by_description):
        raise ValueError(
            "task_family_by_description contains duplicate task names."
        )
    expected_sizes = dict(config.expected_family_sizes)
    unknown_explicit_families = sorted(
        set(explicit.values()) - set(expected_sizes)
    )
    if unknown_explicit_families:
        raise ValueError(
            "Explicit task-family mapping uses unknown families: "
            f"{unknown_explicit_families}"
        )

    resolved = {}
    for task_description in task_descriptions:
        if task_description in explicit:
            resolved[task_description] = explicit[task_description]
            continue
        lowered = task_description.casefold()
        if "drawer" in lowered:
            resolved[task_description] = "drawer"
        elif any(
            token in lowered
            for token in ("pick", "place", "object")
        ):
            resolved[task_description] = "object"
        else:
            raise ValueError(
                f"Cannot infer task family for {task_description!r}; provide "
                "task_family_by_description explicitly."
            )
    unexpected_explicit_tasks = sorted(set(explicit) - set(task_descriptions))
    if unexpected_explicit_tasks:
        raise ValueError(
            "Explicit task-family mapping contains tasks absent from scores: "
            f"{unexpected_explicit_tasks}"
        )

    actual_counts = Counter(resolved.values())
    if dict(actual_counts) != expected_sizes:
        raise ValueError(
            "Directional family contract differs from expected cell counts: "
            f"expected={expected_sizes}, actual={dict(actual_counts)}."
        )
    return resolved


def _metric_distribution(
    values: list[float],
    *,
    empty_display: str,
) -> dict[str, Any]:
    if not values:
        return {
            "values": [],
            "median": None,
            "minimum": None,
            "display": empty_display,
        }
    return {
        "values": [float(value) for value in values],
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "display": None,
    }


def _coverage_distribution(
    candidates: list[dict[str, Any]],
    *,
    empty_display: str,
) -> dict[str, Any]:
    if not candidates:
        return {
            "minimum": None,
            "maximum": None,
            "display": empty_display,
        }
    coverage_values = [
        float(value)
        for candidate in candidates
        for value in (
            candidate["coverage"]["minimum"],
            candidate["coverage"]["maximum"],
        )
    ]
    return {
        "minimum": min(coverage_values),
        "maximum": max(coverage_values),
        "display": None,
    }


def _episode_pair_support_distribution(
    candidates: list[dict[str, Any]],
    *,
    empty_display: str,
) -> dict[str, Any]:
    """Summarize paired-episode evidence without turning it into a gate."""

    supports = [
        candidate["episode_pair_support"]
        for candidate in candidates
    ]
    if not supports:
        return {
            "measured_task_count": 0,
            "comparable_task_count": 0,
            "positive_task_count": 0,
            "fully_positive_task_count": 0,
            "positive_episode_count_range": {
                "minimum": None,
                "maximum": None,
                "median": None,
            },
            "comparable_episode_count_range": {
                "minimum": None,
                "maximum": None,
                "median": None,
            },
            "fraction": {
                "values": [],
                "median": None,
                "minimum": None,
                "display": empty_display,
            },
            "task_support": [],
            "display": empty_display,
            "ranking_gate": False,
        }
    positive_counts = [int(support["positive"]) for support in supports]
    comparable_counts = [
        int(support["comparable"]) for support in supports
    ]
    fractions = [
        float(support["fraction"])
        for support in supports
        if support["fraction"] is not None
    ]
    task_support = [
        {
            "task_description": candidate["task_description"],
            "positive": int(candidate["episode_pair_support"]["positive"]),
            "comparable": int(
                candidate["episode_pair_support"]["comparable"]
            ),
            "fraction": candidate["episode_pair_support"]["fraction"],
            "display": candidate["episode_pair_support"]["display"],
            "positive_episode_ids": list(
                candidate["episode_pair_support"][
                    "positive_episode_ids"
                ]
            ),
            "comparable_episode_ids": list(
                candidate["episode_pair_support"][
                    "comparable_episode_ids"
                ]
            ),
        }
        for candidate in candidates
    ]
    return {
        "measured_task_count": len(supports),
        "comparable_task_count": sum(
            count > 0 for count in comparable_counts
        ),
        "positive_task_count": sum(count > 0 for count in positive_counts),
        "fully_positive_task_count": sum(
            comparable > 0 and positive == comparable
            for positive, comparable in zip(
                positive_counts,
                comparable_counts,
                strict=True,
            )
        ),
        "positive_episode_count_range": {
            "minimum": min(positive_counts),
            "maximum": max(positive_counts),
            "median": float(np.median(positive_counts)),
        },
        "comparable_episode_count_range": {
            "minimum": min(comparable_counts),
            "maximum": max(comparable_counts),
            "median": float(np.median(comparable_counts)),
        },
        "fraction": _metric_distribution(
            fractions,
            empty_display=DIRECTIONAL_MISSING_VALUE,
        ),
        "task_support": task_support,
        "display": (
            None if fractions else DIRECTIONAL_MISSING_VALUE
        ),
        "ranking_gate": False,
    }


def _transition_candidate_lookup(
    transition_result: dict[str, Any],
) -> dict[int, dict[str, Any]] | None:
    candidates = transition_result["candidates"]
    if candidates is None:
        return None
    return {
        int(candidate["feature_id"]): candidate
        for candidate in candidates
    }


def _family_transition_feature_summary(
    *,
    feature_id: int,
    task_names: list[str],
    expected_count: int,
    transition_by_task: dict[str, dict[str, Any]],
    include_episode_pair_support: bool,
) -> dict[str, Any]:
    task_cells = {}
    available_tasks = []
    support_candidates = []
    for task_name in task_names:
        task_transition = transition_by_task[task_name]
        candidate_lookup = _transition_candidate_lookup(task_transition)
        if candidate_lookup is None:
            task_cells[task_name] = {
                "status": "missing",
                "display": DIRECTIONAL_MISSING_VALUE,
                "candidate": None,
            }
            continue
        available_tasks.append(task_name)
        candidate = candidate_lookup.get(feature_id)
        if candidate is None:
            task_cells[task_name] = {
                "status": "no_candidate",
                "display": DIRECTIONAL_NO_CANDIDATE,
                "candidate": None,
            }
            continue
        support_candidates.append(candidate)
        if not candidate["eligibility"]["eligible"]:
            raise ValueError(
                "Positive W5 transition candidate was marked ineligible."
            )
        candidate_summary = {
            "pair_score_w5": candidate["pair_score_w5"],
            "conservative_margin_percentile_w5": candidate[
                "conservative_margin_percentile_w5"
            ],
            "coverage": dict(candidate["coverage"]),
            "control_overlap_w5": candidate[
                "control_overlap_w5"
            ],
            "control_overlap_w4_sensitivity": candidate[
                "control_overlap_w4_sensitivity"
            ],
            "eligible": True,
        }
        if include_episode_pair_support:
            candidate_summary["episode_pair_support"] = dict(
                candidate["episode_pair_support"]
            )
        task_cells[task_name] = {
            "status": "supported",
            "display": None,
            "candidate": candidate_summary,
        }

    if not available_tasks:
        status = "missing"
        display = DIRECTIONAL_MISSING_VALUE
    elif not support_candidates:
        status = "no_candidate"
        display = DIRECTIONAL_NO_CANDIDATE
    else:
        status = "available"
        display = None
    support_percentiles = [
        float(candidate["conservative_margin_percentile_w5"])
        for candidate in support_candidates
    ]
    empty_metric_display = (
        DIRECTIONAL_MISSING_VALUE
        if not available_tasks
        else DIRECTIONAL_NO_CANDIDATE
    )
    available_count = len(available_tasks)
    support_count = len(support_candidates)
    relaxed_threshold = expected_count // 2 + 1
    summary = {
        "status": status,
        "display": display,
        "expected_task_count": expected_count,
        "available_task_count": available_count,
        "available_tasks": available_tasks,
        "support_task_count": support_count,
        "eligible_task_count": available_count,
        "support_over_eligible": {
            "support": support_count,
            "eligible": available_count,
            "fraction": (
                float(support_count / available_count)
                if available_count > 0
                else None
            ),
            "display": (
                f"{support_count}/{available_count}"
                if available_count > 0
                else DIRECTIONAL_MISSING_VALUE
            ),
        },
        "eligible_over_expected": {
            "eligible": available_count,
            "expected": expected_count,
            "fraction": float(available_count / expected_count),
            "display": f"{available_count}/{expected_count}",
        },
        "support_over_expected": {
            "support": support_count,
            "expected": expected_count,
            "fraction": float(support_count / expected_count),
            "display": f"{support_count}/{expected_count}",
        },
        "strict": (
            available_count == expected_count
            and support_count == expected_count
        ),
        "relaxed": support_count >= relaxed_threshold,
        "relaxed_threshold": relaxed_threshold,
        "support_percentile": _metric_distribution(
            support_percentiles,
            empty_display=empty_metric_display,
        ),
        "support_coverage_range": _coverage_distribution(
            support_candidates,
            empty_display=empty_metric_display,
        ),
        "task_cells": task_cells,
    }
    if include_episode_pair_support:
        summary["episode_pair_support"] = (
            _episode_pair_support_distribution(
                support_candidates,
                empty_display=empty_metric_display,
            )
        )
    return summary


def summarize_directional_transition_recurrence(
    transition_rankings: dict[str, Any],
    *,
    config: DirectionalDiscoveryConfig = DirectionalDiscoveryConfig(),
) -> dict[str, Any]:
    """Aggregate transition recurrence across drawer-2 and object-3 cells."""

    _validate_directional_config(config)
    if transition_rankings.get("schema_version") != (
        "directional_transition_candidates_v1"
    ):
        raise ValueError("Unsupported directional transition schema.")
    task_results = transition_rankings.get("tasks")
    if not isinstance(task_results, dict) or not task_results:
        raise ValueError("Directional transition rankings contain no tasks.")
    include_episode_pair_support = bool(
        transition_rankings.get("episode_pair_support_policy")
    )
    task_names = sorted(task_results)
    families = _resolve_directional_task_families(
        task_names,
        config=config,
    )
    expected_sizes = dict(config.expected_family_sizes)
    tasks_by_family = {
        family: [
            task_name
            for task_name in task_names
            if families[task_name] == family
        ]
        for family in expected_sizes
    }

    transition_names = list(
        transition_rankings["ordered_transition_pairs"]
    )
    aggregated_transitions = {}
    for transition_name in transition_names:
        transition_by_task = {
            task_name: task_results[task_name]["transitions"][
                transition_name
            ]
            for task_name in task_names
        }
        available_task_count = sum(
            result["status"] != "missing"
            for result in transition_by_task.values()
        )
        feature_ids = sorted(
            {
                int(candidate["feature_id"])
                for result in transition_by_task.values()
                if result["candidates"] is not None
                for candidate in result["candidates"]
            }
        )
        feature_summaries = []
        for feature_id in feature_ids:
            family_summaries = {
                family: _family_transition_feature_summary(
                    feature_id=feature_id,
                    task_names=tasks_by_family[family],
                    expected_count=expected_count,
                    transition_by_task=transition_by_task,
                    include_episode_pair_support=(
                        include_episode_pair_support
                    ),
                )
                for family, expected_count in expected_sizes.items()
            }
            support_candidates = []
            task_cells = {}
            for task_name in task_names:
                transition_result = transition_by_task[task_name]
                candidate_lookup = _transition_candidate_lookup(
                    transition_result
                )
                if candidate_lookup is None:
                    task_cells[task_name] = {
                        "family": families[task_name],
                        "status": "missing",
                        "display": DIRECTIONAL_MISSING_VALUE,
                    }
                    continue
                candidate = candidate_lookup.get(feature_id)
                if candidate is None:
                    task_cells[task_name] = {
                        "family": families[task_name],
                        "status": "no_candidate",
                        "display": DIRECTIONAL_NO_CANDIDATE,
                    }
                    continue
                support_candidates.append(candidate)
                if not candidate["eligibility"]["eligible"]:
                    raise ValueError(
                        "Positive W5 transition candidate was marked "
                        "ineligible."
                    )
                task_cells[task_name] = {
                    "family": families[task_name],
                    "status": "supported",
                    "display": None,
                    "pair_score_w5": candidate["pair_score_w5"],
                    "conservative_margin_percentile_w5": candidate[
                        "conservative_margin_percentile_w5"
                    ],
                    "coverage": dict(candidate["coverage"]),
                }
                if include_episode_pair_support:
                    task_cells[task_name]["episode_pair_support"] = dict(
                        candidate["episode_pair_support"]
                    )

            support_percentiles = [
                float(candidate["conservative_margin_percentile_w5"])
                for candidate in support_candidates
            ]
            support_family_counts = {
                family: family_summary["support_task_count"]
                for family, family_summary in family_summaries.items()
            }
            family_support_medians = [
                family_summary["support_percentile"]["median"]
                for family_summary in family_summaries.values()
            ]
            both_families_have_support = all(
                count > 0 for count in support_family_counts.values()
            )
            all_families_have_available_cells = all(
                family_summary["available_task_count"] > 0
                for family_summary in family_summaries.values()
            )
            if (
                both_families_have_support
                and all(
                    value is not None
                    for value in family_support_medians
                )
            ):
                family_balanced_percentile = {
                    "value": float(
                        np.mean(family_support_medians)
                    ),
                    "family_medians": {
                        family: family_summaries[family][
                            "support_percentile"
                        ]["median"]
                        for family in expected_sizes
                    },
                    "status": "available",
                    "display": None,
                }
            else:
                family_balanced_percentile = {
                    "value": None,
                    "family_medians": {
                        family: family_summaries[family][
                            "support_percentile"
                        ]["median"]
                        for family in expected_sizes
                    },
                    "status": (
                        "no_cross_family_support"
                        if all_families_have_available_cells
                        else "missing_family"
                    ),
                    "display": (
                        DIRECTIONAL_NO_CANDIDATE
                        if all_families_have_available_cells
                        else DIRECTIONAL_MISSING_VALUE
                    ),
                }

            support_count = len(support_candidates)
            empty_metric_display = (
                DIRECTIONAL_NO_CANDIDATE
                if available_task_count > 0
                else DIRECTIONAL_MISSING_VALUE
            )
            feature_summaries.append(
                {
                    "feature_id": feature_id,
                    "support_task_count": support_count,
                    "eligible_task_count": available_task_count,
                    "support_over_eligible": {
                        "support": support_count,
                        "eligible": available_task_count,
                        "fraction": (
                            float(support_count / available_task_count)
                            if available_task_count > 0
                            else None
                        ),
                        "display": (
                            f"{support_count}/{available_task_count}"
                            if available_task_count > 0
                            else DIRECTIONAL_MISSING_VALUE
                        ),
                    },
                    "eligible_over_expected": {
                        "eligible": available_task_count,
                        "expected": config.global_strict_task_count,
                        "fraction": float(
                            available_task_count
                            / config.global_strict_task_count
                        ),
                        "display": (
                            f"{available_task_count}/"
                            f"{config.global_strict_task_count}"
                        ),
                    },
                    "support_over_expected": {
                        "support": support_count,
                        "expected": config.global_strict_task_count,
                        "fraction": float(
                            support_count
                            / config.global_strict_task_count
                        ),
                        "display": (
                            f"{support_count}/"
                            f"{config.global_strict_task_count}"
                        ),
                    },
                    "global_strict": (
                        available_task_count
                        == config.global_strict_task_count
                        and support_count
                        == config.global_strict_task_count
                    ),
                    "global_relaxed": (
                        support_count
                        >= config.global_relaxed_task_count
                        and both_families_have_support
                    ),
                    "global_relaxed_requirements": {
                        "minimum_support_tasks": (
                            config.global_relaxed_task_count
                        ),
                        "requires_each_family": True,
                    },
                    "support_percentile": _metric_distribution(
                        support_percentiles,
                        empty_display=empty_metric_display,
                    ),
                    "support_coverage_range": _coverage_distribution(
                        support_candidates,
                        empty_display=empty_metric_display,
                    ),
                    "family_balanced_percentile": (
                        family_balanced_percentile
                    ),
                    **(
                        {
                            "episode_pair_support": (
                                _episode_pair_support_distribution(
                                    support_candidates,
                                    empty_display=empty_metric_display,
                                )
                            )
                        }
                        if include_episode_pair_support
                        else {}
                    ),
                    "families": family_summaries,
                    "task_cells": task_cells,
                }
            )
        feature_summaries.sort(
            key=lambda row: (
                -int(row["global_strict"]),
                -int(row["global_relaxed"]),
                -int(row["support_task_count"]),
                -(
                    float(row["family_balanced_percentile"]["value"])
                    if row["family_balanced_percentile"]["value"]
                    is not None
                    else -1.0
                ),
                int(row["feature_id"]),
            )
        )
        if available_task_count == 0:
            transition_status = "missing"
            transition_display = DIRECTIONAL_MISSING_VALUE
        elif not feature_ids:
            transition_status = "no_candidate"
            transition_display = DIRECTIONAL_NO_CANDIDATE
        else:
            transition_status = "available"
            transition_display = None
        aggregated_transitions[transition_name] = {
            "status": transition_status,
            "display": transition_display,
            "available_task_count": available_task_count,
            "expected_task_count": config.global_strict_task_count,
            "num_features": len(feature_summaries),
            "features": feature_summaries,
            "transition_repeat": {
                "raw_feature_count": len(feature_summaries),
                "supported_feature_count": sum(
                    row["support_task_count"] > 0
                    for row in feature_summaries
                ),
                "supported_in_at_least_two_tasks": [
                    row["feature_id"]
                    for row in feature_summaries
                    if row["support_task_count"] >= 2
                ],
                "global_strict_feature_ids": [
                    row["feature_id"]
                    for row in feature_summaries
                    if row["global_strict"]
                ],
                "global_relaxed_feature_ids": [
                    row["feature_id"]
                    for row in feature_summaries
                    if row["global_relaxed"]
                ],
                "family_strict_feature_ids": {
                    family: [
                        row["feature_id"]
                        for row in feature_summaries
                        if row["families"][family]["strict"]
                    ]
                    for family in expected_sizes
                },
                "family_relaxed_feature_ids": {
                    family: [
                        row["feature_id"]
                        for row in feature_summaries
                        if row["families"][family]["relaxed"]
                    ]
                    for family in expected_sizes
                },
                **(
                    {
                        "features_with_positive_episode_pair_support": [
                            row["feature_id"]
                            for row in feature_summaries
                            if row["episode_pair_support"][
                                "positive_task_count"
                            ]
                            > 0
                        ],
                        "features_with_fully_positive_episode_pair_support": [
                            row["feature_id"]
                            for row in feature_summaries
                            if row["episode_pair_support"][
                                "fully_positive_task_count"
                            ]
                            > 0
                        ],
                    }
                    if include_episode_pair_support
                    else {}
                ),
            },
        }

    return {
        "schema_version": "directional_transition_recurrence_v1",
        "task_family_contract": {
            "expected_family_sizes": expected_sizes,
            "task_family_by_description": families,
            "denominators": {
                "support_over_eligible": (
                    "candidate support divided by phase/transition-available "
                    "cells"
                ),
                "eligible_over_expected": (
                    "phase/transition-available cells divided by expected "
                    "family cells"
                ),
            },
            "candidate_metric_scope": (
                "percentile and coverage distributions contain supporting "
                "positive-margin candidates only; an available cell without "
                "support has no candidate metric"
            ),
            "family_strict": (
                "complete expected-cell availability and candidate support "
                "in every expected family cell"
            ),
            "family_relaxed": (
                "candidate support in a strict majority of expected family "
                "cells: drawer 2/2 and object 2/3"
            ),
        },
        "global_contract": {
            "strict": (
                "complete expected-cell availability and candidate support "
                f"in all {config.global_strict_task_count} cells"
            ),
            "relaxed": (
                "candidate support in at least "
                f"{config.global_relaxed_task_count}_of_"
                f"{config.global_strict_task_count}_cells_and_both_families"
            ),
            "family_balanced_percentile": (
                "unweighted mean of each family's median supported "
                "conservative transition percentile"
            ),
        },
        **(
            {
                "episode_pair_support_contract": (
                    "family/global ranges summarize same-task episodes where "
                    "both ON and OFF components were comparable/positive; "
                    "evidence only, never a ranking gate"
                )
            }
            if include_episode_pair_support
            else {}
        ),
        "sentinels": {
            "missing": DIRECTIONAL_MISSING_VALUE,
            "no_candidate": DIRECTIONAL_NO_CANDIDATE,
        },
        "transitions": aggregated_transitions,
    }


def summarize_directional_phase_recurrence(
    phase_rankings: dict[str, Any],
    *,
    config: DirectionalDiscoveryConfig = DirectionalDiscoveryConfig(),
) -> dict[str, Any]:
    """Aggregate each phase/template candidate from task to family to suite."""

    _validate_directional_config(config)
    if phase_rankings.get("schema_version") != (
        "directional_phase_candidates_v1"
    ):
        raise ValueError("Unsupported directional phase-ranking schema.")
    if tuple(phase_rankings.get("phase_order", ())) != config.phase_order:
        raise ValueError("Phase-ranking order and recurrence config differ.")
    task_phase_results = phase_rankings.get("tasks")
    if not isinstance(task_phase_results, dict) or not task_phase_results:
        raise ValueError("Directional phase rankings contain no tasks.")

    recurrence_keys = [
        f"{phase}:{template}"
        for phase in config.phase_order
        for template in DIRECTIONAL_TEMPLATE_NAMES
    ]
    recurrence_tasks = {}
    for task_name, task_result in task_phase_results.items():
        task_recurrence = {}
        for phase in config.phase_order:
            phase_result = task_result["phases"][phase]
            for template in DIRECTIONAL_TEMPLATE_NAMES:
                recurrence_key = f"{phase}:{template}"
                template_result = phase_result["templates"][template]
                candidates = template_result["candidates"]
                if candidates is None:
                    converted_candidates = None
                else:
                    converted_candidates = [
                        {
                            "feature_id": int(candidate["feature_id"]),
                            "pair_score_w5": float(
                                candidate["w5"]["margin"]
                            ),
                            "conservative_margin_percentile_w5": float(
                                candidate["w5"]["margin_percentile"]
                            ),
                            "coverage": {
                                "earlier": float(
                                    candidate["phase_coverage"]
                                ),
                                "later": float(
                                    candidate["phase_coverage"]
                                ),
                                "minimum": float(
                                    candidate["phase_coverage"]
                                ),
                                "maximum": float(
                                    candidate["phase_coverage"]
                                ),
                            },
                            "control_overlap_w5": bool(
                                candidate["controls"]["w5"]["overlap"]
                            ),
                            "control_overlap_w4_sensitivity": (
                                candidate["controls"]["w4_sensitivity"][
                                    "overlap"
                                ]
                            ),
                            "eligibility": dict(
                                candidate["eligibility"]
                            ),
                        }
                        for candidate in candidates
                    ]
                task_recurrence[recurrence_key] = {
                    "status": template_result["status"],
                    "display": template_result["display"],
                    "num_candidates": template_result[
                        "num_candidates"
                    ],
                    "num_eligible": template_result.get("num_eligible"),
                    "candidates": converted_candidates,
                }
        recurrence_tasks[task_name] = {
            "transitions": task_recurrence,
        }

    generic_recurrence = summarize_directional_transition_recurrence(
        {
            "schema_version": "directional_transition_candidates_v1",
            "ordered_transition_pairs": recurrence_keys,
            "tasks": recurrence_tasks,
        },
        config=config,
    )
    phase_summaries = {}
    for phase in config.phase_order:
        template_summaries = {}
        for template in DIRECTIONAL_TEMPLATE_NAMES:
            recurrence_key = f"{phase}:{template}"
            template_summary = dict(
                generic_recurrence["transitions"][recurrence_key]
            )
            template_summary["phase"] = phase
            template_summary["template"] = template
            template_summary["phase_repeat"] = template_summary.pop(
                "transition_repeat"
            )
            template_summaries[template] = template_summary
        if all(
            summary["status"] == "missing"
            for summary in template_summaries.values()
        ):
            phase_status = "missing"
            phase_display = DIRECTIONAL_MISSING_VALUE
        elif all(
            summary["status"] != "available"
            for summary in template_summaries.values()
        ):
            phase_status = "no_candidate"
            phase_display = DIRECTIONAL_NO_CANDIDATE
        else:
            phase_status = "available"
            phase_display = None
        phase_summaries[phase] = {
            "status": phase_status,
            "display": phase_display,
            "templates": template_summaries,
        }

    global_contract = dict(generic_recurrence["global_contract"])
    global_contract["family_balanced_percentile"] = (
        "unweighted mean of each family's median supported W5 phase-margin "
        "percentile"
    )
    return {
        "schema_version": "directional_phase_recurrence_v1",
        "recurrence_scope": (
            "each named template independently within each ordered phase"
        ),
        "task_family_contract": generic_recurrence[
            "task_family_contract"
        ],
        "global_contract": global_contract,
        "sentinels": generic_recurrence["sentinels"],
        "phases": phase_summaries,
    }


def discover_directional_phase_features(
    score_w5: Path,
    *,
    score_w4: Path | None = None,
    config: DirectionalDiscoveryConfig = DirectionalDiscoveryConfig(),
) -> dict[str, Any]:
    """Run reusable directional discovery without writing any artifacts."""

    tasks = load_directional_template_scores(
        score_w5=Path(score_w5),
        score_w4=(Path(score_w4) if score_w4 is not None else None),
    )
    phase_rankings = rank_directional_phase_candidates(
        tasks,
        config=config,
    )
    transition_rankings = rank_directional_transition_candidates(
        phase_rankings,
        config=config,
    )
    phase_recurrence = summarize_directional_phase_recurrence(
        phase_rankings,
        config=config,
    )
    recurrence = summarize_directional_transition_recurrence(
        transition_rankings,
        config=config,
    )
    return {
        "schema_version": "directional_phase_discovery_v1",
        "inputs": {
            "score_w5": str(Path(score_w5)),
            "score_w4": (
                str(Path(score_w4)) if score_w4 is not None else None
            ),
            "w4_sensitivity_status": (
                "available" if score_w4 is not None else "not_supplied"
            ),
        },
        "contract": {
            "primary_discovery_window": 5,
            "w4_policy": "sensitivity_only_never_a_gate",
            "phase_margin": (
                "exact-task named-template phase score minus strongest "
                "other observed phase"
            ),
            "candidate_membership": "all_positive_w5_margins_preserved",
            "control_policy": (
                "window/task-mean Top-N overlap is diagnostic metadata only "
                "and never changes membership, ordering, eligibility, or "
                "recurrence support"
            ),
            "transition_scope": (
                "all ordered earlier-to-later phase pairs"
            ),
            "transition_pair": (
                "same-feature earlier step_up ON plus later step_down OFF; "
                "pair score is the minimum component W5 margin"
            ),
            "episode_pair_support": (
                "intersection of ON/OFF comparable and positive episode IDs "
                "within the exact task; evidence only, never a gate"
            ),
            "pulse_policy": (
                "ranked separately per phase and excluded from ON/OFF pairs"
            ),
            "required_score_artifact_keys": [
                *(
                    f"episode_group_matrix_{template}"
                    for template in DIRECTIONAL_TEMPLATE_NAMES
                ),
                *(
                    f"matrix_{template}"
                    for template in DIRECTIONAL_TEMPLATE_NAMES
                ),
                "matrix_template_max",
                "directional_score_definitions",
                "template_matrix_contract",
            ],
            "legacy_raw_relation": (
                "matrix_raw is not required to equal matrix_template_max; "
                "mean(max(template)) and max(mean(template)) do not commute"
            ),
            "missing_value": DIRECTIONAL_MISSING_VALUE,
            "no_candidate_value": DIRECTIONAL_NO_CANDIDATE,
        },
        "phase_rankings": phase_rankings,
        "phase_recurrence": phase_recurrence,
        "transition_rankings": transition_rankings,
        "transition_recurrence": recurrence,
    }


@dataclass
class _CoarseTaskPhaseScores:
    """Post-hoc coarse phases for one exact instruction."""

    task_description: str
    phases: list[str]
    event_w4: np.ndarray
    event_w5: np.ndarray
    window_mean_w5: np.ndarray
    task_mean_w5: np.ndarray
    num_source_episode_groups: int
    num_coarse_episode_groups: int
    num_approximated_episode_groups: int


def _mean_other_phase_margin(matrix: np.ndarray) -> np.ndarray:
    """Subtract the mean of the other observed phases from every phase."""

    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("Mean-other phase margin requires at least two phases.")
    return matrix - (
        (matrix.sum(axis=0, keepdims=True) - matrix)
        / float(matrix.shape[0] - 1)
    )


def _collapse_task_to_coarse_phases(
    task: TaskLocalPhaseScores,
) -> _CoarseTaskPhaseScores:
    """Collapse fine labels without pooling episodes or instructions.

    Episode-group vectors are first combined within
    ``(episode, coarse_phase)`` using selected-event counts, then averaged
    equally across episodes.  When two fine phase groups share that key, their
    already-maximized template scores are combined post hoc; the caller reports
    this small approximation count explicitly.
    """

    if len(task.group_event_counts) != len(task.pair.group_keys):
        raise ValueError(
            f"{task.task_description!r}: group counts and keys differ."
        )
    if task.pair.group_w4.shape != task.pair.group_w5.shape:
        raise ValueError(
            f"{task.task_description!r}: W4/W5 group matrices differ."
        )

    episode_groups: dict[
        tuple[int, str],
        list[tuple[int, np.ndarray, np.ndarray]],
    ] = {}
    num_source_episode_groups = 0
    for group_idx, ((episode_num, fine_phase), event_count) in enumerate(
        zip(task.pair.group_keys, task.group_event_counts, strict=True)
    ):
        coarse_phase = COARSE_PHASE_BY_ANNOTATION.get(fine_phase)
        if coarse_phase is None:
            continue
        episode_groups.setdefault((episode_num, coarse_phase), []).append(
            (
                event_count,
                task.pair.group_w4[group_idx],
                task.pair.group_w5[group_idx],
            )
        )
        num_source_episode_groups += 1

    phase_episode_w4: dict[str, list[np.ndarray]] = {}
    phase_episode_w5: dict[str, list[np.ndarray]] = {}
    approximated_groups = 0
    for (_, coarse_phase), pieces in episode_groups.items():
        total_events = sum(piece[0] for piece in pieces)
        if total_events <= 0:
            raise ValueError("Coarse episode groups require positive event counts.")
        if len(pieces) > 1:
            approximated_groups += 1
        combined_w4 = sum(
            event_count * values_w4
            for event_count, values_w4, _ in pieces
        ) / float(total_events)
        combined_w5 = sum(
            event_count * values_w5
            for event_count, _, values_w5 in pieces
        ) / float(total_events)
        phase_episode_w4.setdefault(coarse_phase, []).append(combined_w4)
        phase_episode_w5.setdefault(coarse_phase, []).append(combined_w5)

    phases = [
        phase for phase in COARSE_PHASE_ORDER if phase in phase_episode_w5
    ]
    if not phases:
        raise ValueError(
            f"{task.task_description!r}: no supported coarse phases."
        )
    event_w4 = np.stack(
        [np.mean(phase_episode_w4[phase], axis=0) for phase in phases]
    )
    event_w5 = np.stack(
        [np.mean(phase_episode_w5[phase], axis=0) for phase in phases]
    )

    phase_row_indices: dict[str, list[int]] = {}
    for row_idx, fine_phase in enumerate(task.pair.phases):
        coarse_phase = COARSE_PHASE_BY_ANNOTATION.get(fine_phase)
        if coarse_phase is not None:
            phase_row_indices.setdefault(coarse_phase, []).append(row_idx)
    window_mean_w5 = np.stack(
        [
            task.window_mean_w5[phase_row_indices[phase]].mean(axis=0)
            for phase in phases
        ]
    )
    included_row_indices = [
        row_idx
        for indices in phase_row_indices.values()
        for row_idx in indices
    ]
    task_means = task.task_mean_w5[included_row_indices]
    if not np.allclose(
        task_means,
        task_means[0][None, :],
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError(
            f"{task.task_description!r}: task-mean rows disagree."
        )

    return _CoarseTaskPhaseScores(
        task_description=task.task_description,
        phases=phases,
        event_w4=event_w4,
        event_w5=event_w5,
        window_mean_w5=window_mean_w5,
        task_mean_w5=task_means[0],
        num_source_episode_groups=num_source_episode_groups,
        num_coarse_episode_groups=len(episode_groups),
        num_approximated_episode_groups=approximated_groups,
    )


def _coarse_phase_cells(
    tasks: dict[str, TaskLocalPhaseScores],
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, int]]:
    """Build instruction-local coarse-phase contrasts for one SAE run."""

    cells: dict[str, dict[str, dict[str, np.ndarray]]] = {
        phase: {} for phase in COARSE_PHASE_ORDER
    }
    diagnostics = {
        "num_tasks": len(tasks),
        "num_contrastable_tasks": 0,
        "num_source_episode_groups": 0,
        "num_coarse_episode_groups": 0,
        "num_approximated_episode_groups": 0,
    }
    for task_description, task in tasks.items():
        coarse = _collapse_task_to_coarse_phases(task)
        diagnostics["num_source_episode_groups"] += (
            coarse.num_source_episode_groups
        )
        diagnostics["num_coarse_episode_groups"] += (
            coarse.num_coarse_episode_groups
        )
        diagnostics["num_approximated_episode_groups"] += (
            coarse.num_approximated_episode_groups
        )
        if len(coarse.phases) < 2:
            continue
        diagnostics["num_contrastable_tasks"] += 1
        margin_w4 = _mean_other_phase_margin(coarse.event_w4)
        margin_w5 = _mean_other_phase_margin(coarse.event_w5)
        window_margin_w5 = _mean_other_phase_margin(
            coarse.window_mean_w5
        )
        for phase_idx, phase in enumerate(coarse.phases):
            cells[phase][task_description] = {
                "margin_w4": margin_w4[phase_idx],
                "margin_w5": margin_w5[phase_idx],
                "window_margin_w5": window_margin_w5[phase_idx],
                "task_mean_w5": coarse.task_mean_w5,
            }
    return (
        {phase: rows for phase, rows in cells.items() if rows},
        diagnostics,
    )


def _rank_coarse_phase_cells(
    cells: dict[str, dict[str, np.ndarray]],
    *,
    artifact_top_n: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Macro-rank one coarse phase across exact instructions."""

    if not cells:
        raise ValueError("At least one instruction cell is required.")
    instructions = sorted(cells)
    margin_w4_by_instruction = np.stack(
        [cells[instruction]["margin_w4"] for instruction in instructions]
    )
    margin_w5_by_instruction = np.stack(
        [cells[instruction]["margin_w5"] for instruction in instructions]
    )
    combined_by_instruction = (
        margin_w4_by_instruction + margin_w5_by_instruction
    ) / 2.0
    margin_w4 = margin_w4_by_instruction.mean(axis=0)
    margin_w5 = margin_w5_by_instruction.mean(axis=0)
    combined_margin = combined_by_instruction.mean(axis=0)
    window_margin_w5 = np.stack(
        [cells[instruction]["window_margin_w5"] for instruction in instructions]
    ).mean(axis=0)
    task_mean_w5 = np.stack(
        [cells[instruction]["task_mean_w5"] for instruction in instructions]
    ).mean(axis=0)
    positive_instruction_count = np.count_nonzero(
        combined_by_instruction > 0,
        axis=0,
    )
    required_positive_instructions = len(instructions) // 2 + 1
    eligible = (
        (margin_w4 > 0)
        & (margin_w5 > 0)
        & (positive_instruction_count >= required_positive_instructions)
    )
    eligible_ids = np.flatnonzero(eligible)
    ordered = eligible_ids[
        np.argsort(-combined_margin[eligible_ids], kind="stable")
    ]
    rank_combined = feature_ranks_descending(combined_margin)
    rank_w4 = feature_ranks_descending(margin_w4)
    rank_w5 = feature_ranks_descending(margin_w5)
    task_mean_rank_w5 = feature_ranks_descending(task_mean_w5)

    candidates = []
    for rank, feature_id in enumerate(
        ordered[:artifact_top_n],
        start=1,
    ):
        feature_idx = int(feature_id)
        candidates.append(
            {
                "rank": rank,
                "feature_id": feature_idx,
                "event_margin_mean_w4_w5": float(
                    combined_margin[feature_idx]
                ),
                "margin_w4": float(margin_w4[feature_idx]),
                "margin_w5": float(margin_w5[feature_idx]),
                "rank_w4": int(rank_w4[feature_idx]),
                "rank_w5": int(rank_w5[feature_idx]),
                "positive_instruction_count": int(
                    positive_instruction_count[feature_idx]
                ),
                "instruction_count": len(instructions),
                "instruction_support_fraction": float(
                    positive_instruction_count[feature_idx]
                    / len(instructions)
                ),
                "window_mean_margin_w5": float(
                    window_margin_w5[feature_idx]
                ),
                "window_mean_same_sign": bool(
                    window_margin_w5[feature_idx] > 0
                ),
                "task_mean_rank_w5": int(
                    task_mean_rank_w5[feature_idx]
                ),
                "task_mean_top20": bool(
                    task_mean_rank_w5[feature_idx] <= 20
                ),
            }
        )
    return (
        {
            "instructions": instructions,
            "instruction_count": len(instructions),
            "required_positive_instruction_count": (
                required_positive_instructions
            ),
            "eligible_feature_count": int(np.count_nonzero(eligible)),
            "top_candidates": candidates,
        },
        {
            "eligible": eligible,
            "order": ordered,
            "rank_combined": rank_combined,
            "combined_margin": combined_margin,
        },
    )


def _canonical_artifact_path(path: str | Path) -> Path:
    return resolve_groot_artifact_path(path).resolve()


def _selected_event_identity(
    row: dict[str, Any],
) -> tuple[str, str, str, str, int, int]:
    return (
        str(row["sample_id"]),
        str(row["task_description"]),
        str(row["cluster_id"]),
        str(row["phase"]),
        int(row["episode_num"]),
        int(row["waypoint_step"]),
    )


def _validate_task_local_score_provenance(
    spec: PhaseFeatureRun,
    config: TaskLocalPhaseRankingConfig,
    topk_manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, str, str, str, int, int]]]:
    payload_w4 = torch.load(spec.score_w4, map_location="cpu", weights_only=False)
    payload_w5 = torch.load(spec.score_w5, map_location="cpu", weights_only=False)
    source_w4 = payload_w4.get("source")
    source_w5 = payload_w5.get("source")
    if not isinstance(source_w4, dict) or source_w4 != source_w5:
        raise ValueError(f"{spec.label}: W4/W5 score source contracts differ.")
    if source_w4.get("contract_version") != "event_feature_score_source_v2":
        raise ValueError(
            f"{spec.label}: score source lacks content-hash lineage."
        )
    expected_paths = {
        "topk_run_dir": spec.topk_dir,
        "cluster_annotations_path": config.phase_groups,
        "cluster_assignments_path": config.phase_assignments,
    }
    for key, expected_path in expected_paths.items():
        actual_value = source_w4.get(key)
        if actual_value is None or _canonical_artifact_path(
            actual_value
        ) != _canonical_artifact_path(expected_path):
            raise ValueError(
                f"{spec.label}: score source {key!r} does not match input."
            )
    event_features_path = source_w4.get("event_features_path")
    if event_features_path is None or not _canonical_artifact_path(
        event_features_path
    ).is_file():
        raise ValueError(f"{spec.label}: score event-feature source is missing.")
    current_content_hashes = {
        "topk_manifest_sha256": sha256_file(spec.topk_dir / "manifest.json"),
        "event_features_sha256": sha256_file(
            _canonical_artifact_path(event_features_path)
        ),
        "cluster_assignments_sha256": sha256_file(
            _canonical_artifact_path(config.phase_assignments)
        ),
        "cluster_annotations_sha256": sha256_file(
            _canonical_artifact_path(config.phase_groups)
        ),
    }
    for key, current_hash in current_content_hashes.items():
        if source_w4.get(key) != current_hash:
            raise ValueError(
                f"{spec.label}: score source content hash {key!r} differs."
            )
    prompt_records_path = source_w4.get("prompt_records_path")
    prompt_records_sha256 = source_w4.get("prompt_records_sha256")
    if prompt_records_path is not None:
        if not _canonical_artifact_path(prompt_records_path).is_file():
            raise ValueError(f"{spec.label}: prompt-record source is missing.")
        if prompt_records_sha256 != sha256_file(
            _canonical_artifact_path(prompt_records_path)
        ):
            raise ValueError(
                f"{spec.label}: prompt-record source content hash differs."
            )
    if _canonical_artifact_path(
        source_w4.get("sae_path", "")
    ) != _canonical_artifact_path(spec.checkpoint):
        raise ValueError(f"{spec.label}: score SAE source does not match checkpoint.")
    if _canonical_artifact_path(
        topk_manifest.get("sae_path", "")
    ) != _canonical_artifact_path(spec.checkpoint):
        raise ValueError(
            f"{spec.label}: Top-K SAE source does not match checkpoint."
        )
    checkpoint_sha256 = sha256_file(spec.checkpoint)
    if source_w4.get("sae_sha256") != checkpoint_sha256:
        raise ValueError(f"{spec.label}: score SAE content hash differs.")
    for key in (
        "activation_source_manifest_sha256",
        "trajectory_manifest_sha256",
    ):
        if source_w4.get(key) != topk_manifest.get(key):
            raise ValueError(
                f"{spec.label}: score and Top-K {key!r} differ."
            )
    for payload_name, payload in (("W4", payload_w4), ("W5", payload_w5)):
        if payload.get("step_mapping") != config.expected_step_mapping:
            raise ValueError(
                f"{spec.label}: {payload_name} step mapping is not "
                f"{config.expected_step_mapping!r}."
            )
        if int(payload.get("event_step_scale", -1)) != int(
            config.score_event_step_scale
        ):
            raise ValueError(
                f"{spec.label}: {payload_name} score event-step scale differs."
            )
    if int(source_w4.get("event_step_scale", -1)) != int(
        config.score_event_step_scale
    ):
        raise ValueError(f"{spec.label}: score-source event-step scale differs.")
    if source_w4.get("capture_target") != config.expected_capture_target:
        raise ValueError(f"{spec.label}: score capture target differs.")
    if topk_manifest.get("capture_target") != config.expected_capture_target:
        raise ValueError(f"{spec.label}: Top-K capture target differs.")
    if int(topk_manifest.get("event_step_scale", -1)) != int(
        config.topk_event_step_scale
    ):
        raise ValueError(f"{spec.label}: Top-K event-step scale differs.")
    for key in ("dict_size", "topk", "layer"):
        if int(source_w4.get(key, -1)) != int(topk_manifest.get(key, -2)):
            raise ValueError(
                f"{spec.label}: score and Top-K {key!r} contracts differ."
            )
    if topk_manifest.get("format") != "token_topk_sparse_v1":
        raise ValueError(f"{spec.label}: unsupported Top-K format.")
    if bool(topk_manifest.get("partial", False)):
        raise ValueError(f"{spec.label}: partial Top-K input is not allowed.")
    encoding_stats = topk_manifest.get("encoding_stats", {})
    if not bool(encoding_stats.get("lossless_topk", False)):
        raise ValueError(f"{spec.label}: Top-K input is not lossless.")

    selected_w4 = [
        _selected_event_identity(row)
        for row in payload_w4.get("selected_events", [])
    ]
    selected_w5 = [
        _selected_event_identity(row)
        for row in payload_w5.get("selected_events", [])
    ]
    if selected_w4 != selected_w5:
        raise ValueError(f"{spec.label}: W4/W5 selected-event contracts differ.")
    if len(selected_w4) != len(set(selected_w4)):
        raise ValueError(f"{spec.label}: selected-event identities are duplicated.")
    selected_sha256 = hashlib.sha256(
        json.dumps(
            selected_w4,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    source_contract_sha256 = hashlib.sha256(
        json.dumps(
            source_w4,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return (
        {
            "score_source_contract_version": str(
                source_w4["contract_version"]
            ),
            "score_source_contract_sha256": source_contract_sha256,
            "topk_manifest_sha256": current_content_hashes[
                "topk_manifest_sha256"
            ],
            "event_features_path": str(
                _canonical_artifact_path(event_features_path)
            ),
            "event_features_sha256": current_content_hashes[
                "event_features_sha256"
            ],
            "phase_groups_sha256": current_content_hashes[
                "cluster_annotations_sha256"
            ],
            "phase_assignments_sha256": current_content_hashes[
                "cluster_assignments_sha256"
            ],
            "prompt_records_sha256": prompt_records_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "selected_event_contract_sha256": selected_sha256,
            "num_selected_events": len(selected_w4),
            "activation_source_manifest_sha256": str(
                topk_manifest.get("activation_source_manifest_sha256", "")
            ),
            "trajectory_manifest_sha256": str(
                topk_manifest.get("trajectory_manifest_sha256", "")
            ),
            "dict_size": int(topk_manifest["dict_size"]),
            "activation_dim": int(topk_manifest["activation_dim"]),
            "topk": int(topk_manifest["topk"]),
            "layer": int(topk_manifest["layer"]),
            "capture_target": str(topk_manifest["capture_target"]),
            "score_step_mapping": str(payload_w4["step_mapping"]),
            "score_event_step_scale": int(payload_w4["event_step_scale"]),
            "topk_event_step_scale": int(
                topk_manifest["event_step_scale"]
            ),
            "lossless_topk": True,
        },
        selected_w4,
    )


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _confidence_and_observation_diagnostics(
    *,
    accepted_annotations: Path,
    phase_groups: Path,
    phase_assignments: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    accepted = _load_jsonl_rows(accepted_annotations)
    groups = _load_jsonl_rows(phase_groups)
    assignments = _load_jsonl_rows(phase_assignments)
    annotation_by_cluster = {
        str(row["cluster_id"]): row for row in accepted
    }
    if len(annotation_by_cluster) != len(accepted):
        raise ValueError("Accepted annotations contain duplicate cluster IDs.")
    sample_ids = [str(row["sample_id"]) for row in assignments]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Phase assignments contain duplicate sample IDs.")

    assignments_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in assignments:
        key = (str(row["task_description"]), str(row["phase"]))
        assignments_by_key.setdefault(key, []).append(row)

    output: dict[tuple[str, str], dict[str, Any]] = {}
    grouped_source_cluster_ids: set[str] = set()
    for group in groups:
        key = (str(group["task_description"]), str(group["phase"]))
        if key in output:
            raise ValueError(f"Duplicate task/phase group: {key}")
        source_cluster_ids = [
            str(cluster_id) for cluster_id in group["source_cluster_ids"]
        ]
        if len(source_cluster_ids) != len(set(source_cluster_ids)):
            raise ValueError(f"{key}: duplicate source cluster IDs.")
        overlapping = grouped_source_cluster_ids.intersection(source_cluster_ids)
        if overlapping:
            raise ValueError(
                f"Source clusters appear in multiple phase groups: {overlapping}"
            )
        grouped_source_cluster_ids.update(source_cluster_ids)
        missing = [
            cluster_id
            for cluster_id in source_cluster_ids
            if cluster_id not in annotation_by_cluster
        ]
        if missing:
            raise ValueError(f"Phase group references unknown annotations: {missing}")
        for cluster_id in source_cluster_ids:
            annotation = annotation_by_cluster[cluster_id]
            annotation_key = (
                str(annotation["task_description"]),
                str(annotation["phase"]),
            )
            if annotation_key != key:
                raise ValueError(
                    f"{cluster_id}: annotation {annotation_key} disagrees "
                    f"with phase group {key}."
                )
        source_label_provenance = {
            cluster_id: _annotation_label_provenance(
                annotation_by_cluster[cluster_id],
                cluster_id=cluster_id,
            )
            for cluster_id in source_cluster_ids
        }
        confidence_tiers = ANNOTATION_CONFIDENCE_TIERS
        if any(
            row["confidence_tier"] == ORACLE_ANNOTATION_CONFIDENCE_TIER
            for row in source_label_provenance.values()
        ):
            confidence_tiers = (
                *confidence_tiers,
                ORACLE_ANNOTATION_CONFIDENCE_TIER,
            )
        source_tiers = Counter(
            row["confidence_tier"]
            for row in source_label_provenance.values()
        )
        rows = assignments_by_key.get(key, [])
        if len(rows) != int(group["num_members"]):
            raise ValueError(f"{key}: phase-group assignment count disagrees.")
        group_member_ids = [str(value) for value in group["member_sample_ids"]]
        assignment_member_ids = [str(row["sample_id"]) for row in rows]
        if sorted(group_member_ids) != sorted(assignment_member_ids):
            raise ValueError(f"{key}: phase-group member inventory disagrees.")
        event_tiers = Counter()
        episode_success: dict[int, bool] = {}
        for row in rows:
            source_cluster_id = str(row["source_cluster_id"])
            if source_cluster_id not in source_cluster_ids:
                raise ValueError(
                    f"{key}: assignment references an unrelated source cluster."
                )
            event_tiers[
                source_label_provenance[source_cluster_id][
                    "confidence_tier"
                ]
            ] += 1
            episode_num = int(row["episode_num"])
            row_success = bool(row["success"])
            previous_success = episode_success.setdefault(
                episode_num, row_success
            )
            if previous_success != row_success:
                raise ValueError(
                    f"{key}: success is inconsistent within episode "
                    f"{episode_num}."
                )
        if {str(row["source_cluster_id"]) for row in rows} != set(
            source_cluster_ids
        ):
            raise ValueError(f"{key}: source-cluster assignment set disagrees.")
        progress = np.asarray(
            [float(row["progress_percent"]) for row in rows],
            dtype=np.float64,
        )
        episode_success_values = np.asarray(
            list(episode_success.values()),
            dtype=np.float64,
        )
        if source_tiers[ORACLE_ANNOTATION_CONFIDENCE_TIER] == len(
            source_cluster_ids
        ):
            claim_limit = "simulator_oracle_phase_labels"
        elif source_tiers[ORACLE_ANNOTATION_CONFIDENCE_TIER] > 0:
            claim_limit = "includes_simulator_oracle_phase_labels"
        elif source_tiers["user-directed"] == len(source_cluster_ids):
            claim_limit = "user_directed_phase_override_only"
        elif source_tiers["user-directed"] > 0:
            claim_limit = "includes_user_directed_phase_override"
        elif source_tiers["strong"] == len(source_cluster_ids):
            claim_limit = "strong_labels_only"
        elif source_tiers["plurality"] == len(source_cluster_ids):
            claim_limit = "two_of_five_plurality_only"
        elif source_tiers["strong"] == 0 and source_tiers["plurality"] > 0:
            claim_limit = "no_strong_support_includes_two_of_five_plurality"
        elif source_tiers["strong"] == 0:
            claim_limit = "majority_only_no_strong_support"
        elif source_tiers["plurality"] > 0:
            claim_limit = "includes_two_of_five_plurality_support"
        else:
            claim_limit = "majority_or_strong_with_some_strong_support"
        output[key] = {
            "phase_group_id": str(group["phase_group_id"]),
            "source_cluster_ids": source_cluster_ids,
            "source_cluster_confidence": {
                tier: int(source_tiers[tier])
                for tier in confidence_tiers
            },
            "event_membership_confidence": {
                tier: int(event_tiers[tier])
                for tier in confidence_tiers
            },
            "source_cluster_label_provenance": source_label_provenance,
            "claim_limit": claim_limit,
            "num_events": len(rows),
            "num_episodes": len(episode_success),
            "median_event_progress": float(np.median(progress)),
            "event_progress_min": float(progress.min()),
            "event_progress_max": float(progress.max()),
            "episode_success_rate": float(episode_success_values.mean()),
        }
    if set(assignments_by_key) != set(output):
        extra = set(assignments_by_key).difference(output)
        missing = set(output).difference(assignments_by_key)
        raise ValueError(
            "Phase-group and assignment task/phase sets differ: "
            f"extra={sorted(extra)}, missing={sorted(missing)}"
        )
    if grouped_source_cluster_ids != set(annotation_by_cluster):
        unused = set(annotation_by_cluster).difference(grouped_source_cluster_ids)
        unknown = grouped_source_cluster_ids.difference(annotation_by_cluster)
        raise ValueError(
            "Accepted annotation and phase-group source sets differ: "
            f"unused={sorted(unused)}, unknown={sorted(unknown)}"
        )
    return output


def _derived_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _permutation_diagnostics(
    group_keys: list[tuple[int, str]],
    permutations: np.ndarray,
) -> dict[str, int]:
    episode_phases: dict[int, set[str]] = {}
    for episode_num, phase in group_keys:
        episode_phases.setdefault(episode_num, set()).add(phase)
    return {
        "num_episode_groups": len(group_keys),
        "num_episodes": len(episode_phases),
        "exchangeable_episodes": sum(
            len(phases) >= 2 for phases in episode_phases.values()
        ),
        "unique_permutation_assignments": len(
            {row.tobytes() for row in permutations}
        ),
    }


def _task_local_rank_cache(
    bundle: TaskLocalPhaseScores,
) -> dict[str, list[np.ndarray] | np.ndarray]:
    pair = bundle.pair
    window_margin_w4 = phase_vs_rest_margin(bundle.window_mean_w4)
    window_margin_w5 = phase_vs_rest_margin(bundle.window_mean_w5)
    return {
        "robust_rank": [
            feature_ranks_descending(row) for row in pair.robust_margin
        ],
        "raw_rank_w4": [
            feature_ranks_descending(row) for row in pair.phase_w4
        ],
        "raw_rank_w5": [
            feature_ranks_descending(row) for row in pair.phase_w5
        ],
        "task_mean_rank_w5": [
            feature_ranks_descending(row) for row in bundle.task_mean_w5
        ],
        "window_margin_w4": window_margin_w4,
        "window_margin_w5": window_margin_w5,
    }


def _describe_task_phase_candidate(
    bundle: TaskLocalPhaseScores,
    null_max: np.ndarray,
    rank_cache: dict[str, list[np.ndarray] | np.ndarray],
    *,
    phase_idx: int,
    feature_idx: int,
) -> dict[str, Any]:
    pair = bundle.pair
    window_margin_w4 = rank_cache["window_margin_w4"]
    window_margin_w5 = rank_cache["window_margin_w5"]
    assert isinstance(window_margin_w4, np.ndarray)
    assert isinstance(window_margin_w5, np.ndarray)
    robust_window_margin = min(
        float(window_margin_w4[phase_idx, feature_idx]),
        float(window_margin_w5[phase_idx, feature_idx]),
    )
    return {
        "feature_id": int(feature_idx),
        "phase_score_w4": float(pair.phase_w4[phase_idx, feature_idx]),
        "phase_score_w5": float(pair.phase_w5[phase_idx, feature_idx]),
        "margin_w4": float(pair.margin_w4[phase_idx, feature_idx]),
        "margin_w5": float(pair.margin_w5[phase_idx, feature_idx]),
        "robust_margin": float(pair.robust_margin[phase_idx, feature_idx]),
        "robust_rank": int(rank_cache["robust_rank"][phase_idx][feature_idx]),
        "raw_rank_w4": int(rank_cache["raw_rank_w4"][phase_idx][feature_idx]),
        "raw_rank_w5": int(rank_cache["raw_rank_w5"][phase_idx][feature_idx]),
        "window_mean_margin_w4": float(
            window_margin_w4[phase_idx, feature_idx]
        ),
        "window_mean_margin_w5": float(
            window_margin_w5[phase_idx, feature_idx]
        ),
        "robust_window_mean_margin": robust_window_margin,
        "persistent_window_same_sign": robust_window_margin > 0,
        "task_mean_rank_w5": int(
            rank_cache["task_mean_rank_w5"][phase_idx][feature_idx]
        ),
        "max_t_p": max_t_p_value(
            null_max[:, phase_idx],
            float(pair.robust_margin[phase_idx, feature_idx]),
        ),
        "episode_paired_support": summarize_episode_paired_support(
            pair, phase_idx, feature_idx
        ),
    }


def _render_task_local_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Task-local phase-feature ranking",
        "",
        (
            f"- Condition: `{summary['scope']['condition_id']}`"
        ),
        (
            f"- Scope: {summary['scope']['num_tasks']} tasks, "
            f"{summary['scope']['num_phase_rows']} phase rows, "
            f"{summary['scope']['num_contrastable_phase_rows']} contrastable rows"
        ),
        (
            f"- Checkpoints: {summary['scope']['num_runs']}; "
            f"dictionary size: {summary['scope']['dict_size']}"
        ),
        (
            f"- Permutations: {summary['permutation']['num_permutations']} "
            "episode-stratified assignments per task"
        ),
        "- Claim strength: **diagnostic evidence**",
        (
            "- Statistical support (sole inferential family): "
            f"{summary['scope']['matched_statistically_supported']['count']}/"
            f"{summary['scope']['matched_statistically_supported']['total']} "
            "decoder-matched task-phase hypotheses in "
            f"`{summary['scope']['condition_id']}`"
        ),
        (
            "- Cross-instruction sensitivity: "
            f"{summary['scope']['cross_instruction_descriptive']['num_phase_rows']} "
            "phase rows; descriptive only"
        ),
        "- Overall verdict: **confounded — 판정 보류**",
        "",
        "The feature ID in the first column belongs to the reference SAE. "
        "The other two IDs are decoder mutual-nearest matches; they are not "
        "assumed to share integer IDs.",
        "",
        "## Best positive-margin diagnostic candidate per task and phase",
        "",
        "| Task | Phase | Reference feature | Matched feature IDs | "
        "Min cosine | p(all 3) | Holm | Inferential support | Label limit |",
        "|---|---|---:|---|---:|---:|---:|---|---|",
    ]
    reference_label = summary["decoder_matching"]["reference_label"]
    for cell in summary["matched_task_phase_results"]:
        candidates = cell["top_candidates"]
        if candidates:
            best = candidates[0]
            reference_feature = best["feature_ids"][reference_label]
            matched_ids = ", ".join(
                f"{label}:{feature_id}"
                for label, feature_id in best["feature_ids"].items()
            )
            min_cosine = f"{best['min_decoder_cosine']:.3f}"
            p_value = f"{best['p_all3_conjunction']:.4g}"
        else:
            reference_feature = "N/A"
            matched_ids = "no all-positive decoder triplet"
            min_cosine = "N/A"
            p_value = "1"
        lines.append(
            f"| {cell['task_description']} | {cell['phase']} | "
            f"{reference_feature} | {matched_ids} | {min_cosine} | "
            f"{p_value} | "
            f"{cell['best_holm_p']:.4g} | "
            f"{str(cell['statistically_supported']).lower()} | "
            f"{cell['label_diagnostics']['claim_limit']} |"
        )
    lines.extend(
        [
            "",
            "## Cross-instruction candidates",
            "",
            "This table is a descriptive sensitivity analysis and is not a "
            "second inferential family.",
            "",
            "| Phase | Instructions | Reference feature | Min cosine | "
            "p(all cells) | Descriptive Holm sensitivity | Scope | Label limits |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in summary["cross_instruction_results"]:
        candidates = row["top_candidates"]
        if candidates:
            best = candidates[0]
            feature_id = best["feature_ids"][reference_label]
            min_cosine = f"{best['min_decoder_cosine']:.3f}"
            p_value = f"{best['p_all_cells_conjunction']:.4g}"
        else:
            feature_id = "N/A"
            min_cosine = "N/A"
            p_value = "1"
        label_limits = ", ".join(
            sorted(set(row["label_claim_limits"].values()))
        )
        lines.append(
            f"| {row['phase']} | {row['num_instructions']} | {feature_id} | "
            f"{min_cosine} | {p_value} | {row['best_holm_p']:.4g} | "
            f"{row['inference_scope']} | "
            f"{label_limits} |"
        )
    lines.extend(
        [
            "",
            "## Confound audit",
            "",
            "| Gate | Status | Evidence |",
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
            "These rankings are observational diagnostics. A positive margin "
            "means the event-aligned template score is larger than every other "
            "observed phase in the same exact instruction for both W4 and W5. "
            "The Oracle centers have exact environment execution-time labels, "
            "but this does not establish causal control, fresh observation of "
            "every intermediate state inside an open-loop action chunk, or "
            "detector performance.",
            "",
        ]
    )
    return "\n".join(lines)


def rank_task_local_phase_features(
    config: TaskLocalPhaseRankingConfig,
) -> dict[str, Any]:
    """Rank phase-associated features within exact instructions.

    The analysis is immutable, uses W4/W5 robust margins, corrects each
    phase-wise feature search with a max-T permutation null, and joins three
    SAE dictionaries through strict all-pair decoder mutual-nearest matches.
    """

    if config.num_permutations <= 0:
        raise ValueError("num_permutations must be positive.")
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if config.top_n <= 0:
        raise ValueError("top_n must be positive.")
    if not 0.0 < config.alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one.")
    if config.score_event_step_scale <= 0:
        raise ValueError("score_event_step_scale must be positive.")
    if config.topk_event_step_scale <= 0:
        raise ValueError("topk_event_step_scale must be positive.")
    output_dir = Path(config.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite task-local phase output: {output_dir}"
        )
    specs = list(config.runs)
    if len(specs) != 3:
        raise ValueError("Task-local checkpoint analysis requires three runs.")
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError("Run labels must be unique.")
    if config.reference_label not in labels:
        raise ValueError(f"Unknown reference label: {config.reference_label}")
    for path in (
        config.accepted_annotations,
        config.phase_groups,
        config.phase_assignments,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_hashes: dict[str, str] = {}
    topk_manifest_hashes: dict[str, str] = {}
    topk_manifests: dict[str, dict[str, Any]] = {}
    for spec in specs:
        for path in (spec.checkpoint, spec.score_w4, spec.score_w5):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest_path = spec.topk_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        checkpoint_hash = sha256_file(spec.checkpoint)
        with manifest_path.open("r", encoding="utf-8") as handle:
            topk_manifest = json.load(handle)
        if str(topk_manifest.get("sae_sha256", "")) != checkpoint_hash:
            raise ValueError(
                f"{spec.label}: checkpoint hash does not match Top-K manifest."
            )
        checkpoint_hashes[spec.label] = checkpoint_hash
        topk_manifest_hashes[spec.label] = sha256_file(manifest_path)
        topk_manifests[spec.label] = topk_manifest

    score_provenance: dict[str, dict[str, Any]] = {}
    selected_event_contracts: dict[
        str, list[tuple[str, str, str, str, int, int]]
    ] = {}
    for spec in specs:
        provenance, selected_events = _validate_task_local_score_provenance(
            spec,
            config,
            topk_manifests[spec.label],
        )
        score_provenance[spec.label] = provenance
        selected_event_contracts[spec.label] = selected_events
    reference_selected_events = selected_event_contracts[config.reference_label]
    for label in labels:
        if selected_event_contracts[label] != reference_selected_events:
            raise ValueError(
                f"{label}: selected-event order differs across checkpoints."
            )
    for key in (
        "event_features_sha256",
        "selected_event_contract_sha256",
        "activation_source_manifest_sha256",
        "trajectory_manifest_sha256",
        "dict_size",
        "activation_dim",
        "topk",
        "layer",
        "capture_target",
        "score_step_mapping",
        "score_event_step_scale",
        "topk_event_step_scale",
        "lossless_topk",
    ):
        if len({score_provenance[label][key] for label in labels}) != 1:
            raise ValueError(
                f"Score/Top-K provenance differs across checkpoints for {key!r}."
            )
    phase_assignment_rows = _load_jsonl_rows(config.phase_assignments)
    assignment_events = [
        _selected_event_identity(row) for row in phase_assignment_rows
    ]
    exact_phase_entry_errors = {
        str(row.get("sample_id") or f"row_{row_index}"): errors
        for row_index, row in enumerate(phase_assignment_rows)
        if (
            errors := oracle_phase_entry_invariant_errors(
                row,
                required_window_size=ROBUST_WINDOW_HALF_WIDTH,
            )
        )
    }
    all_assignments_are_oracle_phase_entries = bool(
        phase_assignment_rows
    ) and not exact_phase_entry_errors
    action_token_offset_counts: Counter[int] = Counter()
    activation_record_phase_comparison_count = 0
    activation_record_phase_mismatch_count = 0
    observation_boundary_count = 0
    for row in phase_assignment_rows:
        if row.get("action_token_offset") is not None:
            try:
                action_token_offset_counts[int(row["action_token_offset"])] += 1
            except (TypeError, ValueError):
                pass
        activation_record_phase = row.get("activation_record_phase")
        if activation_record_phase is not None:
            activation_record_phase_comparison_count += 1
            activation_record_phase_mismatch_count += int(
                str(activation_record_phase) != str(row.get("phase") or "")
            )
        observation_boundary_count += int(
            row.get("observation_record_index") is not None
        )
    if len(assignment_events) != len(set(assignment_events)):
        raise ValueError("Phase assignments contain duplicate event identities.")
    if set(assignment_events) != set(reference_selected_events):
        raise ValueError(
            "Score selected-event inventory differs from phase assignments."
        )

    bundles = {
        spec.label: load_task_local_score_pair(spec.score_w4, spec.score_w5)
        for spec in specs
    }
    reference_bundles = bundles[config.reference_label]
    task_order = list(reference_bundles)
    for label in labels:
        if list(bundles[label]) != task_order:
            raise ValueError(f"{label}: task order differs across checkpoints.")
        for task_description in task_order:
            reference = reference_bundles[task_description].pair
            candidate = bundles[label][task_description].pair
            if (
                candidate.phases != reference.phases
                or candidate.group_keys != reference.group_keys
            ):
                raise ValueError(
                    f"{label}/{task_description}: phase-group contract differs."
                )
            if candidate.group_w4.shape[1] != reference.group_w4.shape[1]:
                raise ValueError(
                    f"{label}/{task_description}: dictionary size differs."
                )
    dict_size = next(iter(reference_bundles.values())).pair.group_w4.shape[1]
    diagnostics = _confidence_and_observation_diagnostics(
        accepted_annotations=config.accepted_annotations,
        phase_groups=config.phase_groups,
        phase_assignments=config.phase_assignments,
    )
    for task_description, bundle in reference_bundles.items():
        for row in bundle.row_keys:
            key = (task_description, str(row["phase"]))
            if key not in diagnostics:
                raise ValueError(f"Missing label diagnostics for {key}.")
            if str(row["cluster_id"]) != diagnostics[key]["phase_group_id"]:
                raise ValueError(f"{key}: score row phase-group ID differs.")
            if int(row["num_events"]) != diagnostics[key]["num_events"]:
                raise ValueError(f"{key}: score/event diagnostic count differs.")

    contrastable_tasks = [
        task
        for task in task_order
        if len(reference_bundles[task].pair.phases) >= 2
    ]
    if not contrastable_tasks:
        raise ValueError("No exact instruction contains at least two phases.")
    rank_caches = {
        label: {
            task: _task_local_rank_cache(bundles[label][task])
            for task in contrastable_tasks
        }
        for label in labels
    }
    permutation_metadata: dict[str, dict[str, Any]] = {}
    nulls: dict[str, dict[str, np.ndarray]] = {label: {} for label in labels}
    for task_description in contrastable_tasks:
        pair = reference_bundles[task_description].pair
        task_permutations, assignment_sha256 = (
            episode_stratified_phase_permutations(
                pair.group_keys,
                pair.phases,
                num_permutations=config.num_permutations,
                seed=_derived_seed(config.seed, task_description),
            )
        )
        permutation_metadata[task_description] = {
            "assignment_sha256": assignment_sha256,
            "phases": pair.phases,
            **_permutation_diagnostics(pair.group_keys, task_permutations),
        }
        for label in labels:
            nulls[label][task_description] = phasewise_max_feature_null(
                bundles[label][task_description].pair,
                task_permutations,
                chunk_size=config.chunk_size,
            )

    local_results: dict[str, dict[str, dict[str, Any]]] = {
        label: {} for label in labels
    }
    local_best_p: dict[str, float] = {}
    for label in labels:
        for task_description in contrastable_tasks:
            bundle = bundles[label][task_description]
            task_results: dict[str, Any] = {}
            for phase_idx, phase in enumerate(bundle.pair.phases):
                positive_features = np.flatnonzero(
                    bundle.pair.robust_margin[phase_idx] > 0
                )
                ordered = positive_features[
                    np.argsort(
                        -bundle.pair.robust_margin[
                            phase_idx, positive_features
                        ],
                        kind="stable",
                    )
                ]
                candidates = [
                    _describe_task_phase_candidate(
                        bundle,
                        nulls[label][task_description],
                        rank_caches[label][task_description],
                        phase_idx=phase_idx,
                        feature_idx=int(feature_idx),
                    )
                    for feature_idx in ordered[: config.top_n]
                ]
                key = f"{label}::{task_description}::{phase}"
                local_best_p[key] = (
                    float(candidates[0]["max_t_p"]) if candidates else 1.0
                )
                task_results[phase] = {
                    "positive_feature_count": int(len(positive_features)),
                    "top_candidates": candidates,
                }
            local_results[label][task_description] = task_results
    local_holm = holm_adjusted_p_values(local_best_p)
    for label in labels:
        for task_description in contrastable_tasks:
            for phase, row in local_results[label][task_description].items():
                row["best_holm_p_across_all_run_task_phase_cells"] = local_holm[
                    f"{label}::{task_description}::{phase}"
                ]
                row["statistically_supported"] = False
                row["inference_scope"] = "descriptive_only"

    decoder = match_decoder_features_across_checkpoints(
        specs,
        config.reference_label,
        expected_dict_size=dict_size,
        expected_activation_dim=int(
            topk_manifests[config.reference_label]["activation_dim"]
        ),
    )
    strict_triplets = decoder["triplets"]
    matched_results: list[dict[str, Any]] = []
    matched_best_p: dict[str, float] = {}
    for task_description in contrastable_tasks:
        reference_pair = reference_bundles[task_description].pair
        for phase_idx, phase in enumerate(reference_pair.phases):
            eligible: list[dict[str, Any]] = []
            for triplet in strict_triplets:
                per_run_brief: dict[str, dict[str, float | int]] = {}
                all_positive = True
                for label in labels:
                    feature_idx = int(triplet["feature_ids"][label])
                    pair = bundles[label][task_description].pair
                    margin_w4 = float(pair.margin_w4[phase_idx, feature_idx])
                    margin_w5 = float(pair.margin_w5[phase_idx, feature_idx])
                    robust_margin = min(margin_w4, margin_w5)
                    all_positive &= margin_w4 > 0 and margin_w5 > 0
                    per_run_brief[label] = {
                        "feature_id": feature_idx,
                        "robust_margin": robust_margin,
                        "robust_rank": int(
                            rank_caches[label][task_description]["robust_rank"][
                                phase_idx
                            ][feature_idx]
                        ),
                        "max_t_p": max_t_p_value(
                            nulls[label][task_description][:, phase_idx],
                            robust_margin,
                        ),
                    }
                if not all_positive:
                    continue
                eligible.append(
                    {
                        "triplet": triplet,
                        "p_all3_conjunction": max(
                            float(row["max_t_p"])
                            for row in per_run_brief.values()
                        ),
                        "worst_robust_rank": max(
                            int(row["robust_rank"])
                            for row in per_run_brief.values()
                        ),
                    }
                )
            eligible.sort(
                key=lambda row: (
                    row["p_all3_conjunction"],
                    row["worst_robust_rank"],
                    -row["triplet"]["min_decoder_cosine"],
                )
            )
            top_candidates: list[dict[str, Any]] = []
            for eligible_row in eligible[: config.top_n]:
                triplet = eligible_row["triplet"]
                per_run = {
                    label: _describe_task_phase_candidate(
                        bundles[label][task_description],
                        nulls[label][task_description],
                        rank_caches[label][task_description],
                        phase_idx=phase_idx,
                        feature_idx=int(triplet["feature_ids"][label]),
                    )
                    for label in labels
                }
                top_candidates.append(
                    {
                        "feature_ids": triplet["feature_ids"],
                        "decoder_cosines": triplet["decoder_cosines"],
                        "min_decoder_cosine": triplet["min_decoder_cosine"],
                        "p_all3_conjunction": eligible_row[
                            "p_all3_conjunction"
                        ],
                        "worst_robust_rank": eligible_row[
                            "worst_robust_rank"
                        ],
                        "per_run": per_run,
                    }
                )
            cell_key = f"{task_description}::{phase}"
            matched_best_p[cell_key] = (
                float(top_candidates[0]["p_all3_conjunction"])
                if top_candidates
                else 1.0
            )
            matched_results.append(
                {
                    "task_description": task_description,
                    "phase": phase,
                    "label_diagnostics": diagnostics[
                        (task_description, phase)
                    ],
                    "eligible_triplet_count": len(eligible),
                    "top_candidates": top_candidates,
                }
            )
    matched_holm = holm_adjusted_p_values(matched_best_p)
    for row in matched_results:
        row["best_holm_p"] = matched_holm[
            f"{row['task_description']}::{row['phase']}"
        ]
        row["statistically_supported"] = (
            row["best_holm_p"] <= config.alpha
        )

    phase_to_tasks: dict[str, list[str]] = {}
    for task_description in contrastable_tasks:
        for phase in reference_bundles[task_description].pair.phases:
            phase_to_tasks.setdefault(phase, []).append(task_description)
    cross_instruction_results: list[dict[str, Any]] = []
    cross_best_p: dict[str, float] = {}
    for phase, tasks in sorted(phase_to_tasks.items()):
        if len(tasks) < 2:
            continue
        eligible = []
        for triplet in strict_triplets:
            all_positive = True
            cell_p_values: list[float] = []
            worst_rank = 0
            for task_description in tasks:
                phase_idx = bundles[config.reference_label][
                    task_description
                ].pair.phases.index(phase)
                for label in labels:
                    feature_idx = int(triplet["feature_ids"][label])
                    pair = bundles[label][task_description].pair
                    margin_w4 = float(pair.margin_w4[phase_idx, feature_idx])
                    margin_w5 = float(pair.margin_w5[phase_idx, feature_idx])
                    robust_margin = min(margin_w4, margin_w5)
                    all_positive &= margin_w4 > 0 and margin_w5 > 0
                    cell_p_values.append(
                        max_t_p_value(
                            nulls[label][task_description][:, phase_idx],
                            robust_margin,
                        )
                    )
                    worst_rank = max(
                        worst_rank,
                        int(
                            rank_caches[label][task_description][
                                "robust_rank"
                            ][phase_idx][feature_idx]
                        ),
                    )
            if all_positive:
                eligible.append(
                    {
                        "triplet": triplet,
                        "p_all_cells_conjunction": max(cell_p_values),
                        "worst_robust_rank": worst_rank,
                    }
                )
        eligible.sort(
            key=lambda row: (
                row["p_all_cells_conjunction"],
                row["worst_robust_rank"],
                -row["triplet"]["min_decoder_cosine"],
            )
        )
        top_candidates = []
        for eligible_row in eligible[: config.top_n]:
            triplet = eligible_row["triplet"]
            per_instruction = {}
            for task_description in tasks:
                phase_idx = bundles[config.reference_label][
                    task_description
                ].pair.phases.index(phase)
                per_instruction[task_description] = {
                    label: _describe_task_phase_candidate(
                        bundles[label][task_description],
                        nulls[label][task_description],
                        rank_caches[label][task_description],
                        phase_idx=phase_idx,
                        feature_idx=int(triplet["feature_ids"][label]),
                    )
                    for label in labels
                }
            top_candidates.append(
                {
                    "feature_ids": triplet["feature_ids"],
                    "decoder_cosines": triplet["decoder_cosines"],
                    "min_decoder_cosine": triplet["min_decoder_cosine"],
                    "p_all_cells_conjunction": eligible_row[
                        "p_all_cells_conjunction"
                    ],
                    "worst_robust_rank": eligible_row["worst_robust_rank"],
                    "per_instruction": per_instruction,
                }
            )
        cross_best_p[phase] = (
            float(top_candidates[0]["p_all_cells_conjunction"])
            if top_candidates
            else 1.0
        )
        cross_instruction_results.append(
            {
                "phase": phase,
                "instructions": tasks,
                "num_instructions": len(tasks),
                "label_claim_limits": {
                    task: diagnostics[(task, phase)]["claim_limit"]
                    for task in tasks
                },
                "eligible_triplet_count": len(eligible),
                "top_candidates": top_candidates,
            }
        )
    cross_holm = holm_adjusted_p_values(cross_best_p)
    for row in cross_instruction_results:
        row["best_holm_p"] = cross_holm[row["phase"]]
        row["statistically_supported"] = False
        row["inference_scope"] = "descriptive_only"

    noncontrastable = [
        {
            "task_description": task,
            "phases": reference_bundles[task].pair.phases,
            "reason": "only one accepted phase; phase-vs-rest is not identifiable",
        }
        for task in task_order
        if task not in contrastable_tasks
    ]
    progress_gaps = []
    success_gaps = []
    for task in contrastable_tasks:
        phase_diagnostics = [
            diagnostics[(task, phase)]
            for phase in reference_bundles[task].pair.phases
        ]
        progress_values = [
            row["median_event_progress"] for row in phase_diagnostics
        ]
        success_values = [
            row["episode_success_rate"] for row in phase_diagnostics
        ]
        progress_gaps.append(max(progress_values) - min(progress_values))
        success_gaps.append(max(success_values) - min(success_values))

    decoder_summary = {
        key: value for key, value in decoder.items() if key != "triplets"
    }
    decoder_summary["strict_triplet_contract_sha256"] = hashlib.sha256(
        json.dumps(
            strict_triplets,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    entrypoint = (
        Path(config.entrypoint).resolve()
        if config.entrypoint is not None
        else None
    )
    analysis_config = {
        "condition_id": config.condition_id,
        "reference_label": config.reference_label,
        "run_labels": labels,
        "num_permutations": config.num_permutations,
        "seed": config.seed,
        "chunk_size": config.chunk_size,
        "top_n": config.top_n,
        "alpha": config.alpha,
        "expected_step_mapping": config.expected_step_mapping,
        "score_event_step_scale": config.score_event_step_scale,
        "topk_event_step_scale": config.topk_event_step_scale,
        "expected_capture_target": config.expected_capture_target,
    }
    analysis_config_sha256 = hashlib.sha256(
        json.dumps(
            analysis_config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    matched_supported = sum(
        bool(row["statistically_supported"]) for row in matched_results
    )
    inferential_family_description = _inferential_family_description(
        condition_id=config.condition_id,
        hypothesis_count=len(matched_results),
    )
    confidence_tiers = [
        *ANNOTATION_CONFIDENCE_TIERS,
        *(
            [ORACLE_ANNOTATION_CONFIDENCE_TIER]
            if any(
                ORACLE_ANNOTATION_CONFIDENCE_TIER
                in row["source_cluster_confidence"]
                for row in diagnostics.values()
            )
            else []
        ),
    ]
    annotation_confidence_counts = {
        tier: sum(
            row["source_cluster_confidence"].get(tier, 0)
            for row in diagnostics.values()
        )
        for tier in confidence_tiers
    }
    label_confidence_parts = [
        f"{annotation_confidence_counts[tier]} {tier}"
        for tier in confidence_tiers
        if annotation_confidence_counts[tier]
    ]
    label_limitations = []
    if annotation_confidence_counts["user-directed"]:
        label_limitations.append(
            f"{annotation_confidence_counts['user-directed']} user-directed "
            "phase override label(s)"
        )
    if annotation_confidence_counts["plurality"]:
        label_limitations.append(
            f"{annotation_confidence_counts['plurality']} plurality label(s)"
        )
    label_confidence_evidence = (
        "Source annotation confidence counts are "
        + ", ".join(label_confidence_parts)
        + ". "
        + (
            "The analysis includes "
            + " and ".join(label_limitations)
            + "; "
            if label_limitations
            else ""
        )
        + "source provenance remains explicit in each task-phase diagnostic; "
        "no strong-label-only recomputation is claimed."
    )
    oracle_label_count = annotation_confidence_counts.get(
        ORACLE_ANNOTATION_CONFIDENCE_TIER,
        0,
    )
    all_labels_are_simulator_oracle = (
        oracle_label_count > 0
        and oracle_label_count == sum(annotation_confidence_counts.values())
    )
    exact_oracle_phase_entries = (
        all_labels_are_simulator_oracle
        and all_assignments_are_oracle_phase_entries
    )
    if all_labels_are_simulator_oracle:
        label_confidence_evidence = (
            f"All {oracle_label_count} source annotations are direct "
            "simulator-oracle environment-state labels; the trusted rollout "
            "source, programmatic generation mode, and upper-bound scope are "
            "preserved in each task-phase diagnostic."
        )
    summary: dict[str, Any] = {
        "schema_version": "task_local_phase_feature_ranking_v2",
        "scope": {
            "condition_id": config.condition_id,
            "num_runs": len(specs),
            "num_tasks": len(task_order),
            "num_contrastable_tasks": len(contrastable_tasks),
            "num_phase_rows": sum(
                len(bundle.pair.phases)
                for bundle in reference_bundles.values()
            ),
            "num_contrastable_phase_rows": sum(
                len(reference_bundles[task].pair.phases)
                for task in contrastable_tasks
            ),
            "dict_size": int(dict_size),
            "task_order": task_order,
            "alpha": config.alpha,
            "matched_statistically_supported": {
                "count": matched_supported,
                "total": len(matched_results),
            },
            "inferential_family": {
                "condition_id": config.condition_id,
                "cell_count": len(matched_results),
                "task_phase_hypothesis_count": len(matched_results),
                "description": inferential_family_description,
            },
            "annotation_confidence_counts": annotation_confidence_counts,
            "cross_instruction_descriptive": {
                "num_phase_rows": len(cross_instruction_results),
                "inferential_family": False,
            },
            "exact_phase_entry_validation": {
                "required_window_half_width": ROBUST_WINDOW_HALF_WIDTH,
                "num_assignments": len(phase_assignment_rows),
                "num_invalid_assignments": len(exact_phase_entry_errors),
                "sample_errors": dict(
                    list(exact_phase_entry_errors.items())[:10]
                ),
                "action_token_offset_counts": {
                    str(offset): int(count)
                    for offset, count in sorted(
                        action_token_offset_counts.items()
                    )
                },
                "observation_boundary_count": observation_boundary_count,
                "activation_record_phase_comparison_count": (
                    activation_record_phase_comparison_count
                ),
                "activation_record_phase_mismatch_count": (
                    activation_record_phase_mismatch_count
                ),
            },
            "selected_event_scope": {
                "num_events": len(phase_assignment_rows),
                "num_episodes": len(
                    {
                        (
                            str(row["task_description"]),
                            int(row["episode_num"]),
                        )
                        for row in phase_assignment_rows
                    }
                ),
                "num_episode_phase_groups": sum(
                    int(row["num_episodes"])
                    for row in diagnostics.values()
                ),
            },
        },
        "analysis_config": {
            **analysis_config,
            "sha256": analysis_config_sha256,
        },
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "torch_deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "platform": platform.platform(),
            "byte_order": sys.byteorder,
        },
        "inputs": {
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "implementation_dependencies": {
                "decoder_matching": {
                    "path": str(Path(ranking_tools.__file__).resolve()),
                    "sha256": sha256_file(
                        Path(ranking_tools.__file__).resolve()
                    ),
                },
            },
            "entrypoint": (
                {
                    "path": str(entrypoint),
                    "sha256": sha256_file(entrypoint),
                }
                if entrypoint is not None
                else None
            ),
            "accepted_annotations": {
                "path": str(config.accepted_annotations.resolve()),
                "sha256": sha256_file(config.accepted_annotations),
            },
            "phase_groups": {
                "path": str(config.phase_groups.resolve()),
                "sha256": sha256_file(config.phase_groups),
            },
            "phase_assignments": {
                "path": str(config.phase_assignments.resolve()),
                "sha256": sha256_file(config.phase_assignments),
            },
        },
        "runs": {
            spec.label: {
                "checkpoint": str(spec.checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_hashes[spec.label],
                "score_w4": str(spec.score_w4.resolve()),
                "score_w4_sha256": sha256_file(spec.score_w4),
                "score_w5": str(spec.score_w5.resolve()),
                "score_w5_sha256": sha256_file(spec.score_w5),
                "topk_dir": str(spec.topk_dir.resolve()),
                "topk_manifest_sha256": topk_manifest_hashes[spec.label],
                "score_provenance": score_provenance[spec.label],
                "task_phase_results": local_results[spec.label],
            }
            for spec in specs
        },
        "permutation": {
            "method": (
                "within-episode phase-label shuffle, independently per exact "
                "instruction; shared across W4/W5 and checkpoints; phase-wise "
                "max-T over all dictionary features"
            ),
            "seed": config.seed,
            "num_permutations": config.num_permutations,
            "tasks": permutation_metadata,
        },
        "decoder_matching": decoder_summary,
        "matched_task_phase_results": matched_results,
        "matched_outer_holm": matched_holm,
        "cross_instruction_results": cross_instruction_results,
        "cross_instruction_descriptive_holm_sensitivity": cross_holm,
        "noncontrastable_tasks": noncontrastable,
        "confound_audit": [
            {
                "gate": "Task identity",
                "status": "PASS",
                "evidence": "Every contrast is within one exact instruction.",
            },
            {
                "gate": "Length",
                "status": "FAIL",
                "evidence": (
                    "W4/W5 agreement is required, but episode length and phase "
                    "opportunity are not matched."
                ),
            },
            {
                "gate": "Instruction balance",
                "status": "N/A",
                "evidence": (
                    "Local contrasts do not pool instructions; cross-instruction "
                    "rows are same-sign diagnostics, not a balanced test."
                ),
            },
            {
                "gate": "In-sample rescue",
                "status": "N/A",
                "evidence": "No detector or intervention performance is claimed.",
            },
            {
                "gate": "Rollout pooling",
                "status": "PASS",
                "evidence": "Scores are averaged by phase and episode first.",
            },
            {
                "gate": "Phase / dwell",
                "status": "FAIL",
                "evidence": (
                    f"Maximum within-task median event-progress gap is "
                    f"{max(progress_gaps):.3f}; maximum episode success-rate gap is "
                    f"{max(success_gaps):.3f}."
                ),
            },
            {
                "gate": "Feature multiplicity",
                "status": "PASS",
                "evidence": (
                    "Per-phase max-T covers all dictionary features; Holm "
                    "covers the sole inferential family of "
                    f"{inferential_family_description}. Per-checkpoint and "
                    "cross-instruction rows are descriptive only."
                ),
            },
            {
                "gate": "Label confidence",
                "status": (
                    "PASS" if all_labels_are_simulator_oracle else "FAIL"
                ),
                "evidence": label_confidence_evidence,
            },
            {
                "gate": "Observation != causation",
                "status": "PASS",
                "evidence": "Outputs are labeled observational diagnostics.",
            },
            {
                "gate": "Scene-local != general",
                "status": "FAIL",
                "evidence": "No held-out scene or instruction is evaluated.",
            },
            {
                "gate": "Checkpoint independence",
                "status": "FAIL",
                "evidence": (
                    "The three related SAE checkpoints measure dictionary "
                    "stability, not independent replication."
                ),
            },
            {
                "gate": "Exact phase entry",
                "status": "PASS" if exact_oracle_phase_entries else "FAIL",
                "evidence": (
                    "Every selected center is a simulator-oracle "
                    "env_step_phases transition entry, aligned from state s_k "
                    "to the execution-time token for outgoing action a_k; "
                    "initial and terminal states without a complete causal "
                    "window were excluded. This does not mean the policy "
                    "re-observed every intermediate s_k inside a five-action "
                    "open-loop chunk."
                    if exact_oracle_phase_entries
                    else (
                        f"{len(exact_phase_entry_errors)}/"
                        f"{len(phase_assignment_rows)} assignments fail the "
                        "explicit simulator-transition, state/action clock, "
                        "or complete centered-W5 invariant."
                    )
                ),
            },
        ],
        "claim_strength": "diagnostic_evidence",
        "claim_contract": {
            "robust_margin": (
                "minimum of W4 and W5 target phase score minus the strongest "
                "other observed phase score in the same exact instruction"
            ),
            "score_direction": (
                "event-aligned score is the maximum of pulse, step-up, and "
                "step-down templates; activation direction is not identified"
            ),
            "checkpoint_scope": (
                "decoder-matched checkpoint robustness, not independent "
                "replication"
            ),
            "statistical_support": (
                "the sole inferential family is the "
                f"{inferential_family_description}; support requires phase-wise "
                "feature max-T and outer Holm at the configured alpha"
            ),
            "descriptive_families": (
                "per-checkpoint and cross-instruction rankings are sensitivity "
                "diagnostics only and cannot set statistically_supported=true"
            ),
            "condition_scope": (
                f"multiplicity control is local to condition "
                f"{config.condition_id!r}; selecting across conditions or "
                "coverage thresholds requires an additional outer correction"
            ),
            "clock_scope": (
                "Oracle labels are exact at environment execution time. "
                "Action-token latents can have been computed at an earlier "
                "policy-inference boundary inside the five-action open-loop "
                "chunk, so fresh-state observation is not claimed."
            ),
            "prohibited_claims": [
                "causal phase-control feature",
                "exact phase-entry detector",
                "task-independent phase feature without held-out validation",
            ],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    null_path = output_dir / "permutation_null_max.npz"
    null_arrays: dict[str, np.ndarray] = {}
    null_keys: dict[str, dict[str, str]] = {label: {} for label in labels}
    for run_idx, label in enumerate(labels):
        for task_idx, task_description in enumerate(contrastable_tasks):
            key = f"run_{run_idx}_task_{task_idx}"
            null_arrays[key] = nulls[label][task_description]
            null_keys[label][task_description] = key
    np.savez_compressed(null_path, **null_arrays)
    summary["permutation"]["null_archive"] = str(null_path)
    summary["permutation"]["null_keys"] = null_keys
    summary["permutation"]["null_archive_sha256"] = sha256_file(null_path)

    ranking_rows: list[dict[str, Any]] = []
    for label, task_results in local_results.items():
        for task_description, phase_results in task_results.items():
            for phase, phase_result in phase_results.items():
                for rank, candidate in enumerate(
                    phase_result["top_candidates"], start=1
                ):
                    ranking_rows.append(
                        {
                            "ranking": "task_local_checkpoint",
                            "condition_id": config.condition_id,
                            "checkpoint": label,
                            "task_description": task_description,
                            "phase": phase,
                            "rank": rank,
                            "claim_strength": "diagnostic_evidence",
                            "label_diagnostics": diagnostics[
                                (task_description, phase)
                            ],
                            "descriptive_holm_sensitivity_p": phase_result[
                                "best_holm_p_across_all_run_task_phase_cells"
                            ],
                            "statistically_supported": (
                                rank == 1
                                and phase_result["statistically_supported"]
                            ),
                            "inference_scope": "descriptive_only",
                            **candidate,
                        }
                    )
    for cell in matched_results:
        for rank, candidate in enumerate(cell["top_candidates"], start=1):
            ranking_rows.append(
                {
                    "ranking": "task_local_decoder_matched",
                    "condition_id": config.condition_id,
                    "task_description": cell["task_description"],
                    "phase": cell["phase"],
                    "rank": rank,
                    "claim_strength": "diagnostic_evidence",
                    "label_diagnostics": cell["label_diagnostics"],
                    "cell_best_outer_holm_p": cell["best_holm_p"],
                    "statistically_supported": (
                        rank == 1 and cell["statistically_supported"]
                    ),
                    "inference_scope": (
                        CONDITION_TASK_PHASE_INFERENCE_SCOPE
                        if rank == 1
                        else "descriptive_only"
                    ),
                    **candidate,
                }
            )
    for phase_result in cross_instruction_results:
        for rank, candidate in enumerate(
            phase_result["top_candidates"], start=1
        ):
            ranking_rows.append(
                {
                    "ranking": "cross_instruction_decoder_matched",
                    "condition_id": config.condition_id,
                    "phase": phase_result["phase"],
                    "instructions": phase_result["instructions"],
                    "rank": rank,
                    "claim_strength": "diagnostic_evidence",
                    "label_claim_limits": phase_result[
                        "label_claim_limits"
                    ],
                    "descriptive_holm_sensitivity_p": phase_result[
                        "best_holm_p"
                    ],
                    "statistically_supported": (
                        rank == 1
                        and phase_result["statistically_supported"]
                    ),
                    "inference_scope": "descriptive_only",
                    **candidate,
                }
            )
    rankings_path = output_dir / "rankings.jsonl"
    with rankings_path.open("w", encoding="utf-8") as handle:
        for row in ranking_rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    summary["outputs"] = {
        "rankings": str(rankings_path),
        "num_ranking_rows": len(ranking_rows),
        "report": str(output_dir / "report.md"),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    report_path = output_dir / "report.md"
    report_path.write_text(
        _render_task_local_markdown(summary),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "report": str(report_path),
                "rankings": str(rankings_path),
                "num_ranking_rows": len(ranking_rows),
                "matched_best_holm": matched_holm,
                "cross_instruction_descriptive_holm_sensitivity": cross_holm,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


def _load_coarse_condition_coverage(
    summary_path: Path,
    *,
    artifact_top_n: int,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]:
    """Load one existing condition/coverage score suite without rescoring."""

    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    condition_id = str(source_summary["scope"]["condition_id"])
    reference_label = str(source_summary["analysis_config"]["reference_label"])
    run_labels = [
        str(label) for label in source_summary["analysis_config"]["run_labels"]
    ]
    source_runs = source_summary["runs"]
    if set(run_labels) != set(source_runs):
        raise ValueError(
            f"{summary_path}: analysis run labels and result runs differ."
        )

    result_runs: dict[str, Any] = {}
    internal_runs: dict[
        str,
        dict[str, dict[str, dict[str, np.ndarray]]],
    ] = {}
    shared_diagnostics: dict[str, int] | None = None
    reference_phase_instructions: dict[str, list[str]] | None = None
    for label in run_labels:
        run = source_runs[label]
        score_w4 = _canonical_artifact_path(run["score_w4"])
        score_w5 = _canonical_artifact_path(run["score_w5"])
        expected_hashes = {
            score_w4: str(run["score_w4_sha256"]),
            score_w5: str(run["score_w5_sha256"]),
        }
        for score_path, expected_hash in expected_hashes.items():
            if not score_path.is_file():
                raise FileNotFoundError(score_path)
            if sha256_file(score_path) != expected_hash:
                raise ValueError(
                    f"{label}: score artifact hash changed: {score_path}"
                )

        tasks = load_task_local_score_pair(score_w4, score_w5)
        cells, diagnostics = _coarse_phase_cells(tasks)
        phase_instructions = {
            phase: sorted(rows) for phase, rows in cells.items()
        }
        if shared_diagnostics is None:
            shared_diagnostics = diagnostics
            reference_phase_instructions = phase_instructions
        elif (
            diagnostics != shared_diagnostics
            or phase_instructions != reference_phase_instructions
        ):
            raise ValueError(
                f"{condition_id}: coarse observation inventory differs by SAE."
            )

        phase_results = {}
        for phase in COARSE_PHASE_ORDER:
            if phase not in cells:
                continue
            phase_result, _ = _rank_coarse_phase_cells(
                cells[phase],
                artifact_top_n=artifact_top_n,
            )
            phase_results[phase] = phase_result
        result_runs[label] = {
            "checkpoint": str(_canonical_artifact_path(run["checkpoint"])),
            "checkpoint_sha256": str(run["checkpoint_sha256"]),
            "score_w4": str(score_w4),
            "score_w4_sha256": str(run["score_w4_sha256"]),
            "score_w5": str(score_w5),
            "score_w5_sha256": str(run["score_w5_sha256"]),
            "phases": phase_results,
        }
        internal_runs[label] = cells

    if shared_diagnostics is None or reference_phase_instructions is None:
        raise ValueError(f"{summary_path}: no SAE runs were found.")
    return (
        {
            "condition_id": condition_id,
            "reference_label": reference_label,
            "source_summary": str(summary_path.resolve()),
            "source_summary_sha256": sha256_file(summary_path),
            "diagnostics": shared_diagnostics,
            "phase_instruction_counts": {
                phase: len(instructions)
                for phase, instructions in reference_phase_instructions.items()
            },
            "num_contrastable_phase_cells": sum(
                len(instructions)
                for instructions in reference_phase_instructions.values()
            ),
            "runs": result_runs,
        },
        internal_runs,
    )


def _top_feature_ids(
    phase_result: dict[str, Any],
    top_n: int,
) -> list[int]:
    return [
        int(candidate["feature_id"])
        for candidate in phase_result["top_candidates"][:top_n]
    ]


def _coverage_stability_audit(
    discovery_result: dict[str, Any],
    discovery_internal: dict[
        str,
        dict[str, dict[str, dict[str, np.ndarray]]],
    ],
    stability_result: dict[str, Any],
    stability_internal: dict[
        str,
        dict[str, dict[str, dict[str, np.ndarray]]],
    ],
    *,
    shortlist_n: int,
    candidate_pool_n: int,
    stability_top_n: int,
) -> dict[str, Any]:
    """Compare nested coverages on only their shared instruction cells."""

    output: dict[str, Any] = {}
    for label, discovery_phases in discovery_internal.items():
        if label not in stability_internal:
            raise ValueError(f"Stability coverage lacks SAE run {label!r}.")
        run_output: dict[str, Any] = {}
        for phase in COARSE_PHASE_ORDER:
            discovery_cells = discovery_phases.get(phase)
            stability_cells = stability_internal[label].get(phase)
            if discovery_cells is None or stability_cells is None:
                run_output[phase] = {
                    "status": "not_observed",
                    "shared_instruction_count": 0,
                    "shortlist_overlap": None,
                    "candidate_pool_overlap": None,
                }
                continue
            shared_instructions = sorted(
                set(discovery_cells).intersection(stability_cells)
            )
            if not shared_instructions:
                run_output[phase] = {
                    "status": "no_shared_instructions",
                    "shared_instruction_count": 0,
                    "shortlist_overlap": None,
                    "candidate_pool_overlap": None,
                }
                continue
            discovery_shared = {
                instruction: discovery_cells[instruction]
                for instruction in shared_instructions
            }
            stability_shared = {
                instruction: stability_cells[instruction]
                for instruction in shared_instructions
            }
            discovery_ranked, discovery_arrays = _rank_coarse_phase_cells(
                discovery_shared,
                artifact_top_n=stability_top_n,
            )
            stability_ranked, stability_arrays = _rank_coarse_phase_cells(
                stability_shared,
                artifact_top_n=stability_top_n,
            )
            discovery_shortlist = set(
                _top_feature_ids(discovery_ranked, shortlist_n)
            )
            stability_shortlist = set(
                _top_feature_ids(stability_ranked, shortlist_n)
            )
            discovery_pool = set(
                _top_feature_ids(discovery_ranked, candidate_pool_n)
            )
            stability_pool = set(
                _top_feature_ids(stability_ranked, candidate_pool_n)
            )
            phase_audit = {
                "status": "shared_instruction_sensitivity",
                "shared_instructions": shared_instructions,
                "shared_instruction_count": len(shared_instructions),
                "shortlist_overlap": len(
                    discovery_shortlist.intersection(stability_shortlist)
                ),
                "shortlist_size_discovery": len(discovery_shortlist),
                "shortlist_size_stability": len(stability_shortlist),
                "candidate_pool_overlap": len(
                    discovery_pool.intersection(stability_pool)
                ),
                "candidate_pool_size_discovery": len(discovery_pool),
                "candidate_pool_size_stability": len(stability_pool),
            }
            full_phase_result = discovery_result["runs"][label]["phases"].get(
                phase
            )
            if full_phase_result is not None:
                for candidate in full_phase_result["top_candidates"]:
                    feature_id = int(candidate["feature_id"])
                    eligible_at_stability = bool(
                        stability_arrays["eligible"][feature_id]
                    )
                    stability_rank = int(
                        stability_arrays["rank_combined"][feature_id]
                    )
                    candidate["coverage_stability"] = {
                        "shared_instruction_count": len(shared_instructions),
                        "eligible_at_stability": eligible_at_stability,
                        "stability_rank": stability_rank,
                        "stability_top_n": stability_top_n,
                        "within_stability_top_n": bool(
                            eligible_at_stability
                            and stability_rank <= stability_top_n
                        ),
                    }
            phase_audit["discovery_top_features"] = _top_feature_ids(
                discovery_ranked,
                stability_top_n,
            )
            phase_audit["stability_top_features"] = _top_feature_ids(
                stability_ranked,
                stability_top_n,
            )
            phase_audit["discovery_eligible_feature_count"] = int(
                np.count_nonzero(discovery_arrays["eligible"])
            )
            phase_audit["stability_eligible_feature_count"] = int(
                np.count_nonzero(stability_arrays["eligible"])
            )
            run_output[phase] = phase_audit
        output[label] = run_output
    return output


def _geometry_sensitivity_audit(
    primary: dict[str, Any],
    sensitivity: dict[str, Any],
    *,
    shortlist_n: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare coordinate conditions without treating them as replication."""

    output: dict[str, Any] = {}
    rescue_candidates: list[dict[str, Any]] = []
    for label, primary_run in primary["runs"].items():
        sensitivity_run = sensitivity["runs"].get(label)
        if sensitivity_run is None:
            raise ValueError(f"Sensitivity condition lacks SAE run {label!r}.")
        run_output: dict[str, Any] = {}
        for phase in COARSE_PHASE_ORDER:
            primary_phase = primary_run["phases"].get(phase)
            sensitivity_phase = sensitivity_run["phases"].get(phase)
            if primary_phase is None or sensitivity_phase is None:
                run_output[phase] = {
                    "status": "not_observed",
                    "shortlist_overlap": None,
                }
                continue
            primary_ids = _top_feature_ids(primary_phase, shortlist_n)
            sensitivity_ids = _top_feature_ids(
                sensitivity_phase,
                shortlist_n,
            )
            overlap = sorted(set(primary_ids).intersection(sensitivity_ids))
            phase_output = {
                "status": "coordinate_condition_sensitivity",
                "primary_top_features": primary_ids,
                "sensitivity_top_features": sensitivity_ids,
                "shortlist_overlap": len(overlap),
                "overlapping_features": overlap,
            }
            if not overlap and sensitivity_phase["top_candidates"]:
                rescue = {
                    "run_label": label,
                    "phase": phase,
                    "reason": "zero_top3_overlap_with_primary_condition",
                    **sensitivity_phase["top_candidates"][0],
                }
                rescue_candidates.append(rescue)
                phase_output["rescue_feature_id"] = int(rescue["feature_id"])
            run_output[phase] = phase_output
        output[label] = run_output
    return output, rescue_candidates


def _render_coarse_phase_markdown(summary: dict[str, Any]) -> str:
    """Render the concise Korean handoff report for relaxed candidates."""

    contract = summary["candidate_contract"]
    primary_condition = contract["primary_condition"]
    sensitivity_condition = contract["sensitivity_condition"]
    discovery_coverage = contract["discovery_coverage"]
    stability_coverage = contract["stability_coverage"]
    primary = summary["conditions"][primary_condition]["coverages"][
        discovery_coverage
    ]
    sensitivity = summary["conditions"][sensitivity_condition]["coverages"][
        discovery_coverage
    ]
    reference_label = primary["reference_label"]

    lines = [
        "# Relaxed coarse-phase feature ranking",
        "",
        "## 1. Result numbers and scope",
        "",
        (
            f"- Primary: `{primary_condition}` / `{discovery_coverage}` / "
            f"`{reference_label}`"
        ),
        (
            "- Shortlist unit: `(SAE checkpoint, coarse phase)`; "
            f"Top-{contract['shortlist_n']}"
        ),
        (
            f"- Primary memberships: {contract['primary_membership_count']}; "
            f"geometry rescues: {contract['rescue_membership_count']}; "
            f"deduplicated `(SAE, feature)` keys: "
            f"{contract['unique_sae_feature_count']}"
        ),
        (
            f"- `{discovery_coverage}` is discovery; `{stability_coverage}` "
            "is shared-instruction sensitivity only."
        ),
        "",
        "### Condition inventory",
        "",
        "| Condition | Reach n | Grasp n | Transport n | Terminal n | Cells |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition_id, condition in summary["conditions"].items():
        cell = condition["coverages"][discovery_coverage]
        counts = cell["phase_instruction_counts"]
        lines.append(
            f"| `{condition_id.split('_', 1)[0].upper()}` "
            f"| {counts.get('reach', 0)} "
            f"| {counts.get('grasp', 0)} "
            f"| {counts.get('transport', 0)} "
            f"| {counts.get('terminal', 0)} "
            f"| {cell['num_contrastable_phase_cells']} |"
        )

    lines.extend(
        [
            "",
            f"### Primary `{reference_label}` Top-{contract['shortlist_n']}",
            "",
            (
                "| Phase | Instruction n | E3 candidates | cov.4 Top-10 "
                "| E4 sensitivity | E3↔E4 |"
            ),
            "| --- | ---: | --- | ---: | --- | ---: |",
        ]
    )
    geometry = summary["geometry_sensitivity"][reference_label]
    primary_phases = primary["runs"][reference_label]["phases"]
    sensitivity_phases = sensitivity["runs"][reference_label]["phases"]
    for phase in COARSE_PHASE_ORDER:
        primary_phase = primary_phases.get(phase)
        sensitivity_phase = sensitivity_phases.get(phase)
        if primary_phase is None:
            continue
        primary_ids = _top_feature_ids(
            primary_phase,
            contract["shortlist_n"],
        )
        sensitivity_ids = (
            _top_feature_ids(sensitivity_phase, contract["shortlist_n"])
            if sensitivity_phase is not None
            else []
        )
        stable_count = sum(
            bool(
                candidate.get("coverage_stability", {}).get(
                    "within_stability_top_n",
                    False,
                )
            )
            for candidate in primary_phase["top_candidates"][
                : contract["shortlist_n"]
            ]
        )
        overlap = geometry[phase]["shortlist_overlap"]
        lines.append(
            f"| {phase} | {primary_phase['instruction_count']} "
            f"| {', '.join(f'F{value}' for value in primary_ids)} "
            f"| {stable_count}/{len(primary_ids)} "
            f"| {', '.join(f'F{value}' for value in sensitivity_ids) or 'N/A'} "
            f"| {overlap if overlap is not None else 'N/A'} |"
        )

    reference_candidates = [
        candidate
        for phase_result in primary_phases.values()
        for candidate in phase_result["top_candidates"][
            : contract["shortlist_n"]
        ]
    ]
    task_mean_flags = [
        f"F{candidate['feature_id']} ({phase})"
        for phase, phase_result in primary_phases.items()
        for candidate in phase_result["top_candidates"][
            : contract["shortlist_n"]
        ]
        if candidate["task_mean_top20"]
    ]
    stable_reference_count = sum(
        candidate.get("coverage_stability", {}).get(
            "within_stability_top_n",
            False,
        )
        for candidate in reference_candidates
    )
    lines.extend(
        [
            "",
            (
                f"- cov.4 shared-instruction Top-10 유지: "
                f"{stable_reference_count}/{len(reference_candidates)}"
            ),
            (
                "- task-mean Top-20 confound flag: "
                + (", ".join(task_mean_flags) if task_mean_flags else "없음")
            ),
            (
                "- window-mean same-sign은 event score가 이미 mean-centered라 "
                "탈락 조건으로 쓰지 않고 진단값으로만 보존했다."
            ),
        ]
    )

    lines.extend(
        [
            "",
            "### All SAE primary shortlists",
            "",
            "| SAE | Reach | Grasp | Transport | Terminal |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for label, run in primary["runs"].items():
        values = []
        for phase in COARSE_PHASE_ORDER:
            phase_result = run["phases"].get(phase)
            feature_ids = (
                _top_feature_ids(phase_result, contract["shortlist_n"])
                if phase_result is not None
                else []
            )
            values.append(", ".join(f"F{value}" for value in feature_ids) or "N/A")
        lines.append(f"| `{label}` | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## 2. Confound audit",
            "",
            "| Gate | Result | Evidence |",
            "| --- | --- | --- |",
        ]
    )
    for row in summary["confound_audit"]:
        lines.append(
            f"| {row['gate']} | **{row['result']}** | {row['evidence']} |"
        )

    lines.extend(
        [
            "",
            "## 3. Claim strength",
            "",
            f"**{summary['claim_strength']}**",
            "",
            (
                "이 결과는 Hooked-SR용 `phase-aligned intervention candidate` "
                "shortlist다. p-value, Holm, max-T, 3-SAE 교집합은 discovery "
                "gate에서 제거했다. task/window mean은 탈락 gate가 아니라 "
                "persistent confound 표식이다."
            ),
            "",
            "## 4. Held claims",
            "",
            (
                "- 일반적인 coarse-phase feature: "
                "**confounded — 판정 보류**"
            ),
            (
                "- 특정 phase의 인과적 SR 변화: "
                "**confounded — 판정 보류**"
            ),
            (
                "- 새로운 scene/task 일반성: "
                "**confounded — 판정 보류**"
            ),
            "",
            "## Method boundary",
            "",
            (
                "각 instruction 안에서 coarse phase score에서 다른 관측 "
                "phase들의 평균을 뺀 뒤, W4/W5 margin을 평균하고 instruction을 "
                "동일 가중했다. 후보는 W4·W5 suite margin이 모두 양수이고 "
                "instruction 과반에서 combined margin이 양수인 feature다."
            ),
            (
                "Fine phase 둘이 동일 episode/coarse phase로 합쳐지는 경우에는 "
                "이미 template-max된 episode-group score를 event-count 가중 "
                "평균하므로 정확한 재채점이 아닌 명시적 post-hoc 근사다."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def rank_coarse_phase_candidates(
    config: CoarsePhaseCandidateRankingConfig,
) -> dict[str, Any]:
    """Create a relaxed, immutable coarse-phase discovery artifact."""

    stage4_root = _canonical_artifact_path(config.stage4_root)
    output_dir = config.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite coarse-phase output: {output_dir}"
        )
    if not stage4_root.is_dir():
        raise FileNotFoundError(stage4_root)
    if not (
        config.artifact_top_n
        >= config.candidate_pool_n
        >= config.shortlist_n
        > 0
    ):
        raise ValueError(
            "Require artifact_top_n >= candidate_pool_n >= shortlist_n > 0."
        )

    coverage_paths: dict[str, list[Path]] = {}
    for coverage in (
        config.discovery_coverage,
        config.stability_coverage,
    ):
        paths = sorted(
            stage4_root.glob(
                f"e*/{coverage}/analysis/"
                "task_local_phase_feature_ranking/summary.json"
            )
        )
        if len(paths) != config.expected_conditions:
            raise ValueError(
                f"{coverage}: expected {config.expected_conditions} condition "
                f"summaries, found {len(paths)}."
            )
        coverage_paths[coverage] = paths

    conditions: dict[str, dict[str, Any]] = {}
    internal: dict[
        str,
        dict[
            str,
            dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
        ],
    ] = {}
    input_summaries = []
    for coverage, paths in coverage_paths.items():
        for path in paths:
            result, internal_runs = _load_coarse_condition_coverage(
                path,
                artifact_top_n=config.artifact_top_n,
            )
            condition_id = result["condition_id"]
            condition = conditions.setdefault(
                condition_id,
                {
                    "reference_label": result["reference_label"],
                    "coverages": {},
                },
            )
            if condition["reference_label"] != result["reference_label"]:
                raise ValueError(
                    f"{condition_id}: reference SAE label changed by coverage."
                )
            condition["coverages"][coverage] = result
            internal.setdefault(condition_id, {})[coverage] = internal_runs
            input_summaries.append(
                {
                    "condition_id": condition_id,
                    "coverage": coverage,
                    "path": result["source_summary"],
                    "sha256": result["source_summary_sha256"],
                }
            )

    condition_ids = set(conditions)
    required_conditions = {
        config.primary_condition,
        config.sensitivity_condition,
    }
    if not required_conditions.issubset(condition_ids):
        missing = sorted(required_conditions - condition_ids)
        raise ValueError(f"Required ranking conditions are missing: {missing}")

    coverage_stability = {}
    for condition_id, condition in conditions.items():
        coverage_stability[condition_id] = _coverage_stability_audit(
            condition["coverages"][config.discovery_coverage],
            internal[condition_id][config.discovery_coverage],
            condition["coverages"][config.stability_coverage],
            internal[condition_id][config.stability_coverage],
            shortlist_n=config.shortlist_n,
            candidate_pool_n=config.candidate_pool_n,
            stability_top_n=config.artifact_top_n,
        )

    primary = conditions[config.primary_condition]["coverages"][
        config.discovery_coverage
    ]
    sensitivity = conditions[config.sensitivity_condition]["coverages"][
        config.discovery_coverage
    ]
    geometry_sensitivity, rescue_candidates = _geometry_sensitivity_audit(
        primary,
        sensitivity,
        shortlist_n=config.shortlist_n,
    )

    primary_memberships = []
    unique_sae_features: set[tuple[str, int]] = set()
    for label, run in primary["runs"].items():
        for phase in COARSE_PHASE_ORDER:
            phase_result = run["phases"].get(phase)
            if phase_result is None:
                continue
            for candidate in phase_result["top_candidates"][
                : config.shortlist_n
            ]:
                row = {
                    "run_label": label,
                    "phase": phase,
                    **candidate,
                }
                primary_memberships.append(row)
                unique_sae_features.add(
                    (label, int(candidate["feature_id"]))
                )
    for candidate in rescue_candidates:
        unique_sae_features.add(
            (
                str(candidate["run_label"]),
                int(candidate["feature_id"]),
            )
        )

    confound_audit = [
        {
            "gate": "Length",
            "result": "FAIL",
            "evidence": (
                "점수는 episode-balanced지만 upstream recurring-cluster "
                "선정에서는 긴 episode가 더 많은 선택 기회를 가질 수 있다."
            ),
        },
        {
            "gate": "Task identity",
            "result": "PASS",
            "evidence": (
                "정확히 같은 instruction 안에서 phase contrast를 계산한 뒤 "
                "instruction-balanced macro 평균을 냈다."
            ),
        },
        {
            "gate": "Instruction balance",
            "result": "PASS",
            "evidence": (
                "각 eligible instruction은 동일 가중 cell 하나로 기여하고 "
                "phase별 instruction 수를 함께 기록했다."
            ),
        },
        {
            "gate": "In-sample rescue",
            "result": "FAIL",
            "evidence": (
                "현재 150개 source episode에서 후보를 골랐고 held-out "
                "Hooked-SR rollout 평가는 아직 없다."
            ),
        },
        {
            "gate": "Rollout pooling",
            "result": "PASS",
            "evidence": (
                "event score를 episode, exact instruction 순으로 균형화하여 "
                "event row를 독립 표본처럼 취급하지 않았다."
            ),
        },
        {
            "gate": "Phase/dwell",
            "result": "FAIL",
            "evidence": (
                "primary condition의 reach는 supporting instruction이 1개이고 "
                "관측 phase prevalence도 불균형하다."
            ),
        },
        {
            "gate": "Observation != causation",
            "result": "PASS",
            "evidence": (
                "diagnostic intervention candidate만 명명하며 인과 또는 SR "
                "효과를 주장하지 않는다."
            ),
        },
        {
            "gate": "Scene-local != general",
            "result": "FAIL",
            "evidence": (
                "모든 source observation은 현재 5개 cell과 2개 task "
                "family에서만 왔다."
            ),
        },
    ]

    summary = {
        "schema_version": "groot_relaxed_coarse_phase_ranking_v1",
        "claim_strength": "diagnostic_evidence",
        "method": {
            "coarse_phase_order": list(COARSE_PHASE_ORDER),
            "coarse_phase_by_annotation": dict(
                sorted(COARSE_PHASE_BY_ANNOTATION.items())
            ),
            "unsupported_annotation_policy": (
                "mixed and wrong-grasp remain outcome/failure labels and are "
                "excluded from the canonical phase ranking"
            ),
            "episode_aggregation": (
                "event-count weighted within (episode, coarse phase), then "
                "equal mean across episodes"
            ),
            "instruction_contrast": (
                "phase score minus mean of the other observed phases"
            ),
            "event_window_size_aggregation": (
                "equal mean of W4 and W5 event margins"
            ),
            "instruction_aggregation": "equal macro mean",
            "descriptive_candidate_requirements": [
                "suite margin W4 > 0",
                "suite margin W5 > 0",
                "combined margin > 0 in a strict majority of instructions",
            ],
            "discovery_non_gates": [
                "max-T p-value",
                "Holm correction",
                "three-SAE intersection",
                "task-mean rank",
                "window-mean sign",
                "coverage-threshold intersection",
            ],
            "post_hoc_approximation": (
                "Combining multiple fine groups in one episode/coarse phase "
                "averages already template-maximized scores."
            ),
        },
        "candidate_contract": {
            "primary_condition": config.primary_condition,
            "sensitivity_condition": config.sensitivity_condition,
            "discovery_coverage": config.discovery_coverage,
            "stability_coverage": config.stability_coverage,
            "artifact_top_n": config.artifact_top_n,
            "candidate_pool_n": config.candidate_pool_n,
            "shortlist_n": config.shortlist_n,
            "primary_membership_count": len(primary_memberships),
            "rescue_policy": (
                "Add the sensitivity-condition rank-1 only when the primary "
                "and sensitivity Top-3 have zero overlap for the same "
                "(SAE, coarse phase)."
            ),
            "rescue_membership_count": len(rescue_candidates),
            "informed_membership_count": (
                len(primary_memberships) + len(rescue_candidates)
            ),
            "unique_sae_feature_count": len(unique_sae_features),
            "same_integer_id_across_saes": "distinct_candidates",
        },
        "inputs": {
            "stage4_root": str(stage4_root),
            "source_summaries": input_summaries,
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
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
        "conditions": conditions,
        "coverage_stability": coverage_stability,
        "geometry_sensitivity": geometry_sensitivity,
        "primary_shortlist": primary_memberships,
        "geometry_rescue_candidates": rescue_candidates,
        "confound_audit": confound_audit,
        "claim_limits": [
            "Coarse labels were defined after annotation and are descriptive.",
            "Nested coverage thresholds are sensitivity views, not replication.",
            "E3 and E4 share source episodes and differ in geometry definition.",
            "Feature score scale is not a checkpoint-quality metric.",
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

    output_dir.mkdir(parents=True)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    report_path = output_dir / "report.md"
    report_path.write_text(
        _render_coarse_phase_markdown(summary),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "report": str(report_path),
                "primary_memberships": len(primary_memberships),
                "geometry_rescues": len(rescue_candidates),
                "unique_sae_features": len(unique_sae_features),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


__all__ = [
    "COARSE_PHASE_BY_ANNOTATION",
    "COARSE_PHASE_ORDER",
    "CoarsePhaseCandidateRankingConfig",
    "DIRECTIONAL_MISSING_VALUE",
    "DIRECTIONAL_NO_CANDIDATE",
    "DIRECTIONAL_TEMPLATE_NAMES",
    "DirectionalDiscoveryConfig",
    "DirectionalTemplateTaskScores",
    "TaskLocalPhaseRankingConfig",
    "TaskLocalPhaseScores",
    "discover_directional_phase_features",
    "load_directional_template_scores",
    "load_task_local_score_pair",
    "rank_directional_phase_candidates",
    "rank_directional_transition_candidates",
    "rank_coarse_phase_candidates",
    "rank_task_local_phase_features",
    "summarize_directional_phase_recurrence",
    "summarize_directional_transition_recurrence",
]
