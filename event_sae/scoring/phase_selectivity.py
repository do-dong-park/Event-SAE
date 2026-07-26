"""Analyze phase-selective feature stability across SAE checkpoints.

The common statistical layer owns episode-stratified permutations, phase-wise
max-T inference, Holm adjustment, decoder matching, and checkpoint stability.
Task-local loading, diagnostics, ranking, and report generation live in
``event_sae.scoring.task_phase_ranking``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from event_sae import sha256_file
import event_sae.scoring.rankings as ranking_tools


PHASE_CHECKPOINT_STABILITY_FORMAT = "oracle_phase_multi_sae_v1"


@dataclass(frozen=True)
class PhaseFeatureRun:
    """Inputs for one SAE checkpoint in a phase-stability analysis."""

    label: str
    checkpoint: Path
    score_w4: Path
    score_w5: Path
    topk_dir: Path


@dataclass(frozen=True)
class CheckpointPhaseStabilityConfig:
    """Configuration for one three-checkpoint phase-stability analysis."""

    runs: tuple[PhaseFeatureRun, ...]
    reference_label: str
    output_dir: Path
    num_permutations: int = 5_000
    seed: int = 20_260_724
    chunk_size: int = 100


@dataclass
class WindowRobustPhaseScores:
    """Episode-group and phase-level scores for the W4/W5 sensitivity pair."""

    phases: list[str]
    group_keys: list[tuple[int, str]]
    group_w4: np.ndarray
    group_w5: np.ndarray
    phase_w4: np.ndarray
    phase_w5: np.ndarray
    margin_w4: np.ndarray
    margin_w5: np.ndarray
    robust_margin: np.ndarray


def phase_vs_rest_margin(matrix: np.ndarray) -> np.ndarray:
    """Return each phase score minus the strongest other-phase score."""

    if matrix.shape[-2] < 2:
        raise ValueError("phase-vs-rest requires at least two phases")
    margins = np.empty_like(matrix)
    for phase_idx in range(matrix.shape[-2]):
        other_indices = [
            idx for idx in range(matrix.shape[-2]) if idx != phase_idx
        ]
        margins[..., phase_idx, :] = (
            matrix[..., phase_idx, :]
            - matrix[..., other_indices, :].max(axis=-2)
        )
    return margins


def _load_single_instruction_score_pair(
    spec: PhaseFeatureRun,
) -> WindowRobustPhaseScores:
    from event_sae.scoring.task_phase_ranking import load_task_local_score_pair

    task_scores = load_task_local_score_pair(spec.score_w4, spec.score_w5)
    if len(task_scores) != 1:
        raise ValueError(
            f"{spec.label}: checkpoint stability requires exactly one "
            f"instruction, found {len(task_scores)}."
        )
    return next(iter(task_scores.values())).pair


def episode_stratified_phase_permutations(
    group_keys: list[tuple[int, str]],
    phases: list[str],
    *,
    num_permutations: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Shuffle phase labels within episodes and return assignments plus SHA."""

    if num_permutations <= 0:
        raise ValueError("num_permutations must be positive")
    phase_to_idx = {phase: idx for idx, phase in enumerate(phases)}
    labels = np.asarray(
        [phase_to_idx[phase] for _, phase in group_keys],
        dtype=np.int16,
    )
    episode_groups: dict[int, list[int]] = {}
    for group_idx, (episode, _) in enumerate(group_keys):
        episode_groups.setdefault(episode, []).append(group_idx)
    groups = [indices for _, indices in sorted(episode_groups.items())]
    rng = np.random.default_rng(seed)
    permutations = np.tile(labels, (num_permutations, 1))
    for permutation_idx in range(num_permutations):
        for indices in groups:
            permutations[permutation_idx, indices] = rng.permutation(labels[indices])
    digest = hashlib.sha256(permutations.tobytes()).hexdigest()
    return permutations, digest


def phasewise_max_feature_null(
    pair: WindowRobustPhaseScores,
    permutations: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    num_permutations = permutations.shape[0]
    num_phases = len(pair.phases)
    null_max = np.empty((num_permutations, num_phases), dtype=np.float64)
    phase_counts = np.bincount(
        permutations[0].astype(np.int64),
        minlength=num_phases,
    ).astype(np.float64)
    for start in range(0, num_permutations, chunk_size):
        stop = min(start + chunk_size, num_permutations)
        labels = permutations[start:stop]
        means_w4 = np.empty(
            (stop - start, num_phases, pair.group_w4.shape[1]),
            dtype=np.float64,
        )
        means_w5 = np.empty_like(means_w4)
        for phase_idx in range(num_phases):
            weights = (labels == phase_idx).astype(np.float64)
            means_w4[:, phase_idx, :] = (
                weights @ pair.group_w4
            ) / phase_counts[phase_idx]
            means_w5[:, phase_idx, :] = (
                weights @ pair.group_w5
            ) / phase_counts[phase_idx]
        robust = np.minimum(
            phase_vs_rest_margin(means_w4),
            phase_vs_rest_margin(means_w5),
        )
        null_max[start:stop] = robust.max(axis=-1)
    return null_max


def feature_ranks_descending(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def max_t_p_value(null_max: np.ndarray, observed: float) -> float:
    """Monte Carlo max-T p-value with the finite-sample plus-one correction."""

    if len(null_max) == 0:
        raise ValueError("null_max must contain at least one permutation")
    return float(
        (1 + np.count_nonzero(null_max >= observed))
        / (len(null_max) + 1)
    )


def summarize_episode_paired_support(
    pair: WindowRobustPhaseScores,
    phase_idx: int,
    feature_idx: int,
) -> dict[str, int]:
    target_phase = pair.phases[phase_idx]
    episodes = sorted({episode for episode, _ in pair.group_keys})
    supported = 0
    comparable = 0
    for episode in episodes:
        target_rows = [
            idx
            for idx, key in enumerate(pair.group_keys)
            if key == (episode, target_phase)
        ]
        other_rows = [
            idx
            for idx, (row_episode, phase) in enumerate(pair.group_keys)
            if row_episode == episode and phase != target_phase
        ]
        if not target_rows or not other_rows:
            continue
        if len(target_rows) != 1:
            raise ValueError("Expected one aggregated row per episode and phase.")
        comparable += 1
        target_idx = target_rows[0]
        margin_w4 = (
            pair.group_w4[target_idx, feature_idx]
            - pair.group_w4[other_rows, feature_idx].mean()
        )
        margin_w5 = (
            pair.group_w5[target_idx, feature_idx]
            - pair.group_w5[other_rows, feature_idx].mean()
        )
        supported += int(margin_w4 > 0 and margin_w5 > 0)
    return {"positive": supported, "comparable": comparable}


def _executed_alive_features(
    topk_dir: Path,
    dict_size: int,
) -> dict[str, Any]:
    """Summarize executed-token support in transfer Top-K artifacts.

    Transfer shards may store action-local token offsets or global model-token
    indices.  This explicit contract is intentionally separate from the
    generic ``rankings.alive_feature_ids`` step-mapping reader.
    """

    manifest_path = topk_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    global_action_start = int(manifest["source_action_token_slice"]["start"])
    executed_action_steps = int(manifest["executed_action_steps"])
    action_horizon = int(manifest["action_horizon"])
    alive = np.zeros(dict_size, dtype=bool)
    executed_rows = 0
    for shard in manifest["shards"]:
        payload = torch.load(
            topk_dir / str(shard["path"]),
            map_location="cpu",
            weights_only=False,
        )
        token_idx = payload["token_idx"]
        # Transfer Top-K shards store action-local offsets [0, horizon), while
        # older collectors may preserve global model-token indices.
        if int(token_idx.max().item()) < action_horizon:
            action_start = 0
        else:
            action_start = global_action_start
        action_stop = action_start + executed_action_steps
        row_mask = (token_idx >= action_start) & (token_idx < action_stop)
        ids = payload["top_feature_ids"][row_mask]
        vals = payload["top_feature_vals"][row_mask]
        positive_ids = ids[vals > 0].to(dtype=torch.int64).unique().cpu().numpy()
        positive_ids = positive_ids[(positive_ids >= 0) & (positive_ids < dict_size)]
        alive[positive_ids] = True
        executed_rows += int(row_mask.sum().item())
    return {
        "alive_mask": alive,
        "alive_features": int(alive.sum()),
        "executed_token_rows": executed_rows,
        "lossless_topk": bool(manifest["encoding_stats"]["lossless_topk"]),
        "mean_positive_features_per_all_rows": float(
            manifest["encoding_stats"]["mean_positive_features_per_row"]
        ),
        "max_positive_features_per_all_rows": int(
            manifest["encoding_stats"]["max_positive_features_per_row"]
        ),
        "sae_sha256": str(manifest["sae_sha256"]),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _summarize_checkpoint_phase_candidates(
    pair: WindowRobustPhaseScores,
    null_max: np.ndarray,
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for phase_idx, phase in enumerate(pair.phases):
        robust_order = np.argsort(-pair.robust_margin[phase_idx], kind="stable")
        raw_order_w4 = np.argsort(-pair.phase_w4[phase_idx], kind="stable")
        raw_order_w5 = np.argsort(-pair.phase_w5[phase_idx], kind="stable")
        stable_top10 = [
            int(feature_idx)
            for feature_idx in raw_order_w5[:10]
            if feature_idx in set(raw_order_w4[:10].tolist())
        ]
        candidates = []
        for robust_rank, feature_idx in enumerate(robust_order[:top_n], start=1):
            observed = float(pair.robust_margin[phase_idx, feature_idx])
            candidates.append(
                {
                    "feature_id": int(feature_idx),
                    "robust_rank": robust_rank,
                    "raw_rank_w4": int(
                        feature_ranks_descending(pair.phase_w4[phase_idx])[feature_idx]
                    ),
                    "raw_rank_w5": int(
                        feature_ranks_descending(pair.phase_w5[phase_idx])[feature_idx]
                    ),
                    "margin_w4": float(pair.margin_w4[phase_idx, feature_idx]),
                    "margin_w5": float(pair.margin_w5[phase_idx, feature_idx]),
                    "robust_margin": observed,
                    "max_t_p": max_t_p_value(
                        null_max[:, phase_idx],
                        observed,
                    ),
                    "episode_paired_support": summarize_episode_paired_support(
                        pair,
                        phase_idx,
                        int(feature_idx),
                    ),
                }
            )
        best = candidates[0]
        output[phase] = {
            "stable_event_aligned_top10_intersection": stable_top10,
            "top_robust_candidates": candidates,
            "best_feature": best["feature_id"],
            "best_robust_margin": best["robust_margin"],
            "null_max_95": float(np.quantile(null_max[:, phase_idx], 0.95)),
            "best_max_t_p": best["max_t_p"],
        }
    return output


def _quantiles(values: np.ndarray) -> dict[str, float]:
    levels = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    return {f"{level:.2f}": float(np.quantile(values, level)) for level in levels}


def holm_adjusted_p_values(
    raw_p: dict[str, float],
) -> dict[str, float]:
    """Apply Holm's step-down family-wise error correction."""

    ordered = sorted(raw_p, key=raw_p.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (total - rank) * raw_p[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def _safe_quantiles(values: np.ndarray) -> dict[str, float] | None:
    return _quantiles(values) if values.size else None


def match_decoder_features_across_checkpoints(
    specs: list[PhaseFeatureRun],
    reference_label: str,
    *,
    expected_dict_size: int,
    expected_activation_dim: int | None = None,
) -> dict[str, Any]:
    if len(specs) != 3:
        raise ValueError("Strict decoder matching requires exactly three runs.")
    labels = [spec.label for spec in specs]
    if reference_label not in labels:
        raise ValueError(f"Unknown reference label: {reference_label}")
    targets = [label for label in labels if label != reference_label]
    decoders = {
        spec.label: ranking_tools.load_normalized_decoder_columns(
            spec.checkpoint
        )
        for spec in specs
    }
    for label, decoder in decoders.items():
        if decoder.shape[1] != expected_dict_size:
            raise ValueError(
                f"{label}: decoder dictionary size differs from score matrix."
            )
        if (
            expected_activation_dim is not None
            and decoder.shape[0] != expected_activation_dim
        ):
            raise ValueError(
                f"{label}: decoder activation dimension differs from Top-K."
            )
    ref_to_a = ranking_tools.match_mutual_nearest_decoder_features(
        decoders[reference_label], decoders[targets[0]]
    )
    ref_to_b = ranking_tools.match_mutual_nearest_decoder_features(
        decoders[reference_label], decoders[targets[1]]
    )
    a_to_b = ranking_tools.match_mutual_nearest_decoder_features(
        decoders[targets[0]], decoders[targets[1]]
    )
    triplets: list[dict[str, Any]] = []
    anchor_mutual_count = 0
    for ref_idx in range(decoders[reference_label].shape[1]):
        a_idx = int(ref_to_a["left_to_right"][ref_idx])
        b_idx = int(ref_to_b["left_to_right"][ref_idx])
        if not (
            bool(ref_to_a["mutual"][ref_idx])
            and bool(ref_to_b["mutual"][ref_idx])
        ):
            continue
        anchor_mutual_count += 1
        if not (
            bool(a_to_b["mutual"][a_idx])
            and int(a_to_b["left_to_right"][a_idx]) == b_idx
        ):
            continue
        cosines = {
            f"{reference_label}__{targets[0]}": float(
                ref_to_a["cosine"][ref_idx]
            ),
            f"{reference_label}__{targets[1]}": float(
                ref_to_b["cosine"][ref_idx]
            ),
            f"{targets[0]}__{targets[1]}": float(a_to_b["cosine"][a_idx]),
        }
        triplets.append(
            {
                "feature_ids": {
                    reference_label: ref_idx,
                    targets[0]: a_idx,
                    targets[1]: b_idx,
                },
                "decoder_cosines": cosines,
                "min_decoder_cosine": min(cosines.values()),
            }
        )
    min_cosines = np.asarray(
        [row["min_decoder_cosine"] for row in triplets],
        dtype=np.float64,
    )
    return {
        "reference_label": reference_label,
        "target_labels": targets,
        "method": (
            "L2-normalized decoder columns; all three pairwise edges must be "
            "mutual nearest neighbors; no cosine cutoff"
        ),
        "pairwise_mnn": {
            f"{reference_label}__{targets[0]}": {
                "count": ref_to_a["mutual_count"],
                "cosine_quantiles": _safe_quantiles(
                    ref_to_a["mutual_cosines"]
                ),
            },
            f"{reference_label}__{targets[1]}": {
                "count": ref_to_b["mutual_count"],
                "cosine_quantiles": _safe_quantiles(
                    ref_to_b["mutual_cosines"]
                ),
            },
            f"{targets[0]}__{targets[1]}": {
                "count": a_to_b["mutual_count"],
                "cosine_quantiles": _safe_quantiles(
                    a_to_b["mutual_cosines"]
                ),
            },
        },
        "anchor_mutual_triplets": anchor_mutual_count,
        "strict_all_pair_mnn_triplets": len(triplets),
        "strict_same_integer_triplets": sum(
            len(set(row["feature_ids"].values())) == 1 for row in triplets
        ),
        "strict_min_cosine_quantiles": _safe_quantiles(min_cosines),
        "triplets": triplets,
    }


def _summarize_decoder_matched_conjunction(
    specs: list[PhaseFeatureRun],
    pairs: dict[str, WindowRobustPhaseScores],
    nulls: dict[str, np.ndarray],
    alive: dict[str, np.ndarray],
    reference_label: str,
) -> dict[str, Any]:
    labels = [spec.label for spec in specs]
    matched = match_decoder_features_across_checkpoints(
        specs,
        reference_label,
        expected_dict_size=pairs[reference_label].group_w4.shape[1],
    )
    strict_triplets = [dict(row) for row in matched["triplets"]]
    strict_min_cosines = np.asarray(
        [row["min_decoder_cosine"] for row in strict_triplets],
        dtype=np.float64,
    )
    for row in strict_triplets:
        row["all_pilot_executed_alive"] = all(
            bool(alive[label][feature_idx])
            for label, feature_idx in row["feature_ids"].items()
        )
        row["min_cosine_empirical_cdf"] = float(
            np.mean(strict_min_cosines <= row["min_decoder_cosine"])
        )

    phase_candidates: dict[str, list[dict[str, Any]]] = {}
    best_raw_p: dict[str, float] = {}
    phases = pairs[reference_label].phases
    for phase_idx, phase in enumerate(phases):
        candidates = []
        rank_cache = {
            label: {
                "robust": feature_ranks_descending(pairs[label].robust_margin[phase_idx]),
                "raw_w4": feature_ranks_descending(pairs[label].phase_w4[phase_idx]),
                "raw_w5": feature_ranks_descending(pairs[label].phase_w5[phase_idx]),
            }
            for label in labels
        }
        for triplet in strict_triplets:
            if not triplet["all_pilot_executed_alive"]:
                continue
            per_run: dict[str, Any] = {}
            all_positive = True
            for label in labels:
                feature_idx = int(triplet["feature_ids"][label])
                margin_w4 = float(pairs[label].margin_w4[phase_idx, feature_idx])
                margin_w5 = float(pairs[label].margin_w5[phase_idx, feature_idx])
                robust = min(margin_w4, margin_w5)
                all_positive &= margin_w4 > 0 and margin_w5 > 0
                per_run[label] = {
                    "feature_id": feature_idx,
                    "raw_rank_w4": int(rank_cache[label]["raw_w4"][feature_idx]),
                    "raw_rank_w5": int(rank_cache[label]["raw_w5"][feature_idx]),
                    "robust_rank": int(rank_cache[label]["robust"][feature_idx]),
                    "margin_w4": margin_w4,
                    "margin_w5": margin_w5,
                    "robust_margin": robust,
                    "max_t_p": max_t_p_value(
                        nulls[label][:, phase_idx],
                        robust,
                    ),
                    "episode_paired_support": summarize_episode_paired_support(
                        pairs[label],
                        phase_idx,
                        feature_idx,
                    ),
                }
            if not all_positive:
                continue
            p_all3 = max(row["max_t_p"] for row in per_run.values())
            candidates.append(
                {
                    "feature_ids": triplet["feature_ids"],
                    "decoder_cosines": triplet["decoder_cosines"],
                    "min_decoder_cosine": triplet["min_decoder_cosine"],
                    "min_cosine_empirical_cdf": triplet[
                        "min_cosine_empirical_cdf"
                    ],
                    "per_run": per_run,
                    "p_all3_conjunction": p_all3,
                }
            )
        candidates.sort(
            key=lambda row: (
                row["p_all3_conjunction"],
                max(item["robust_rank"] for item in row["per_run"].values()),
            )
        )
        phase_candidates[phase] = candidates
        best_raw_p[phase] = (
            float(candidates[0]["p_all3_conjunction"]) if candidates else 1.0
        )
    holm = holm_adjusted_p_values(best_raw_p)
    for phase, candidates in phase_candidates.items():
        if candidates:
            candidates[0]["holm_p_across_four_phase_best"] = holm[phase]

    return {
        **{key: value for key, value in matched.items() if key != "triplets"},
        "phase_candidate_counts": {
            phase: len(rows) for phase, rows in phase_candidates.items()
        },
        "phase_candidates": phase_candidates,
        "phase_best_p_all3": best_raw_p,
        "phase_best_holm_p": holm,
    }


def analyze_checkpoint_phase_stability(
    config: CheckpointPhaseStabilityConfig,
) -> dict[str, Any]:
    """Run the W4/W5 phase-selectivity and decoder-matching analysis.

    The output directory must not already exist so a rerun cannot silently
    replace a canonical report.
    """

    if config.num_permutations <= 0:
        raise ValueError("num_permutations must be positive.")
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    output_dir = Path(config.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite phase-stability output: {output_dir}"
        )
    specs = list(config.runs)
    if len(specs) != 3:
        raise ValueError(
            "Phase checkpoint stability requires exactly three runs."
        )
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError("Run labels must be unique.")
    if config.reference_label not in labels:
        raise ValueError(
            f"Unknown reference label: {config.reference_label}"
        )
    for spec in specs:
        for label, path in (
            ("checkpoint", spec.checkpoint),
            ("W4 score", spec.score_w4),
            ("W5 score", spec.score_w5),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{spec.label} {label}: {path}")
        if not spec.topk_dir.is_dir():
            raise FileNotFoundError(
                f"{spec.label} Top-K directory: {spec.topk_dir}"
            )

    pairs = {
        spec.label: _load_single_instruction_score_pair(spec)
        for spec in specs
    }
    first_pair = pairs[specs[0].label]
    for spec in specs[1:]:
        pair = pairs[spec.label]
        if (
            pair.phases != first_pair.phases
            or pair.group_keys != first_pair.group_keys
        ):
            raise ValueError(
                f"{spec.label}: phase-group contract differs across SAEs."
            )
        if pair.group_w4.shape[1] != first_pair.group_w4.shape[1]:
            raise ValueError(f"{spec.label}: dictionary size differs across SAEs.")

    checkpoint_hashes = {
        spec.label: sha256_file(spec.checkpoint) for spec in specs
    }
    raw_alive_payload = {
        spec.label: _executed_alive_features(
            spec.topk_dir,
            first_pair.group_w4.shape[1],
        )
        for spec in specs
    }
    for spec in specs:
        manifest_hash = str(raw_alive_payload[spec.label]["sae_sha256"])
        checkpoint_hash = checkpoint_hashes[spec.label]
        if manifest_hash != checkpoint_hash:
            raise ValueError(
                f"{spec.label}: Top-K SAE hash {manifest_hash} does not "
                f"match checkpoint hash {checkpoint_hash}."
            )
    alive_masks = {
        label: row["alive_mask"] for label, row in raw_alive_payload.items()
    }
    alive_payload = {
        label: {
            key: value
            for key, value in row.items()
            if key != "alive_mask"
        }
        for label, row in raw_alive_payload.items()
    }

    permutations, permutation_sha256 = (
        episode_stratified_phase_permutations(
            first_pair.group_keys,
            first_pair.phases,
            num_permutations=config.num_permutations,
            seed=config.seed,
        )
    )
    nulls = {
        spec.label: phasewise_max_feature_null(
            pairs[spec.label],
            permutations,
            chunk_size=config.chunk_size,
        )
        for spec in specs
    }
    local = {
        spec.label: _summarize_checkpoint_phase_candidates(
            pairs[spec.label],
            nulls[spec.label],
        )
        for spec in specs
    }
    local_raw_p = {
        label: {
            phase: float(row["best_max_t_p"])
            for phase, row in phase_rows.items()
        }
        for label, phase_rows in local.items()
    }
    local_holm_by_run = {
        label: holm_adjusted_p_values(phase_p)
        for label, phase_p in local_raw_p.items()
    }
    global_cell_p = {
        f"{label}::{phase}": value
        for label, phase_p in local_raw_p.items()
        for phase, value in phase_p.items()
    }
    local_holm_all_cells = holm_adjusted_p_values(global_cell_p)
    for label, phase_rows in local.items():
        for phase, row in phase_rows.items():
            row["best_holm_p_across_four_phases"] = local_holm_by_run[label][
                phase
            ]
            row["best_holm_p_across_all_run_phase_cells"] = (
                local_holm_all_cells[f"{label}::{phase}"]
            )
    matched = _summarize_decoder_matched_conjunction(
        specs,
        pairs,
        nulls,
        alive_masks,
        config.reference_label,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    null_path = output_dir / "permutation_null_max.npz"
    np.savez_compressed(
        null_path,
        **{label: values for label, values in nulls.items()},
    )
    summary = {
        "schema_version": PHASE_CHECKPOINT_STABILITY_FORMAT,
        "scope": {
            "num_runs": len(specs),
            "num_phases": len(first_pair.phases),
            "phases": first_pair.phases,
            "num_phase_episode_groups": len(first_pair.group_keys),
            "dict_size": int(first_pair.group_w4.shape[1]),
        },
        "permutation": {
            "method": (
                "shuffle only the phase rows present within each episode; "
                "reuse assignments across W4/W5 and all SAE checkpoints; "
                "phase-wise max-T over all dictionary features"
            ),
            "seed": int(config.seed),
            "num_permutations": int(config.num_permutations),
            "assignment_matrix_sha256": permutation_sha256,
            "null_archive": str(null_path),
            "null_array_sha256": {
                label: hashlib.sha256(values.tobytes()).hexdigest()
                for label, values in nulls.items()
            },
        },
        "runs": {
            spec.label: {
                "checkpoint": str(spec.checkpoint),
                "checkpoint_sha256": checkpoint_hashes[spec.label],
                "score_w4": str(spec.score_w4),
                "score_w4_sha256": sha256_file(spec.score_w4),
                "score_w5": str(spec.score_w5),
                "score_w5_sha256": sha256_file(spec.score_w5),
                "topk_dir": str(spec.topk_dir),
                **alive_payload[spec.label],
                "phase_results": local[spec.label],
            }
            for spec in specs
        },
        "local_outer_adjustment": {
            "holm_within_each_run_across_four_phases": local_holm_by_run,
            "holm_across_all_run_phase_cells": local_holm_all_cells,
        },
        "decoder_matching": matched,
        "claim_contract": {
            "episode_paired_support": (
                "among episodes containing the target and at least one other "
                "phase, target score exceeds the mean of other observed phase "
                "scores in both W4 and W5; auxiliary diagnostic only"
            ),
            "p_all3": (
                "maximum of the three per-checkpoint phase-wise max-T p-values; "
                "an intersection-union test without checkpoint-independence"
            ),
            "outer_adjustment": (
                "Holm correction across the four phase-best conjunction "
                "p-values"
            ),
            "replication_scope": (
                "checkpoint robustness only; all checkpoints share a training "
                "source and all scores share the same eight evaluation episodes"
            ),
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "null_archive": str(null_path),
                "permutation_sha256": permutation_sha256,
                "phase_best_p_all3": matched["phase_best_p_all3"],
                "phase_best_holm_p": matched["phase_best_holm_p"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


__all__ = [
    "PHASE_CHECKPOINT_STABILITY_FORMAT",
    "CheckpointPhaseStabilityConfig",
    "PhaseFeatureRun",
    "WindowRobustPhaseScores",
    "analyze_checkpoint_phase_stability",
    "episode_stratified_phase_permutations",
    "feature_ranks_descending",
    "holm_adjusted_p_values",
    "match_decoder_features_across_checkpoints",
    "max_t_p_value",
    "phase_vs_rest_margin",
    "phasewise_max_feature_null",
    "summarize_episode_paired_support",
]
