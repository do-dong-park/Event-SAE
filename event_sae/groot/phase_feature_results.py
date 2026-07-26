"""Phase-feature score loading, heatmaps, and evidence summaries.

This module owns checkpoint-local score interpretation. It does not serve HTTP,
load Oracle media, or mutate any historical experiment artifact.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from event_sae import load_pipeline_profile, resolve_groot_artifact_path
from event_sae import sha256_file as _sha256
from event_sae.events.io import load_jsonl


RESULT_RANKINGS = (
    "event_aligned",
    "window_mean",
    "task_mean",
    "random_alive",
)


PHASE_FEATURE_HEATMAP_FORMAT = "event_sae_phase_feature_heatmap_v1"


PHASE_FEATURE_HEATMAP_RANKINGS = {
    "event_aligned": "matrix_raw",
}


PHASE_FEATURE_OVERVIEW_FORMAT = "event_sae_stage4_phase_feature_overview_v1"


STAGE4_GRID_SUMMARY_FORMAT = "event_sae_stage4_grid_summary_v1"


PHASE_FEATURE_CANDIDATE_TOP_K = 20


PHASE_FEATURE_TEMPORAL_SEMANTICS = "unknown-combined"


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower() or "task"


def format_checkpoint_label(sae_id: str) -> str:
    if "sae1p2k" in sae_id:
        return "SAE 1.2k"
    if "sae10k_bs8192" in sae_id:
        return "SAE 10k · bs8192"
    if "sae10k" in sae_id:
        return "SAE 10k"
    return sae_id


def compact_ranking_row(row: dict) -> dict:
    """Return the result fields that are useful in the browser."""

    browser_row = {
        key: row[key]
        for key in (
            "ranking",
            "task_description",
            "cluster_id",
            "phrase",
            "phase",
            "feature_id",
            "rank",
            "score",
        )
        if key in row
    }
    top_features = row.get("top_features")
    if top_features is not None:
        if not isinstance(top_features, list):
            raise ValueError("Ranking top_features must be a list")
        browser_row["top_features"] = [
            {
                "feature_id": int(feature["feature_id"]),
                **(
                    {"score": float(feature["score"])}
                    if feature.get("score") is not None
                    else {}
                ),
            }
            for feature in top_features
        ]
    if browser_row.get("score") is not None:
        browser_row["score"] = float(browser_row["score"])
        if not math.isfinite(browser_row["score"]):
            raise ValueError("Ranking contains a non-finite score")
    if browser_row.get("feature_id") is not None:
        browser_row["feature_id"] = int(browser_row["feature_id"])
    if browser_row.get("rank") is not None:
        browser_row["rank"] = int(browser_row["rank"])
    return browser_row


def _rank_feature_rows(values: np.ndarray) -> np.ndarray:
    """Return deterministic one-based feature ranks for every finite row."""

    if values.ndim != 2:
        raise ValueError("Feature score matrix must be two-dimensional")
    feature_ids = np.arange(values.shape[1], dtype=np.int64)
    ranks = np.full(values.shape, -1, dtype=np.int32)
    for row_index, row in enumerate(values):
        finite = np.isfinite(row)
        if not finite.any():
            continue
        finite_ids = feature_ids[finite]
        order = finite_ids[
            np.lexsort((finite_ids, -row[finite]))
        ]
        ranks[row_index, order] = np.arange(
            1,
            len(order) + 1,
            dtype=np.int32,
        )
    return ranks


def _task_local_phase_selectivity(
    matrix: np.ndarray,
    row_keys: list[dict],
) -> np.ndarray:
    """Compute phase-vs-rest margins inside one exact instruction/cell."""

    margins = np.full(matrix.shape, np.nan, dtype=np.float64)
    rows_by_task: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row_index, row in enumerate(row_keys):
        rows_by_task[_row_instruction_key(row)].append(row_index)
    for task_rows in rows_by_task.values():
        for row_index in task_rows:
            phase = str(row_keys[row_index]["phase"])
            rest = [
                other_index
                for other_index in task_rows
                if str(row_keys[other_index]["phase"]) != phase
            ]
            if not rest:
                continue
            margins[row_index] = (
                matrix[row_index] - np.max(matrix[rest], axis=0)
            )
    return margins


def _row_instruction_key(row: dict) -> tuple[str, ...]:
    """Return the exact instruction/cell identity used for phase contrast."""

    description = str(row["task_description"])
    cell_id = str(row.get("cell_id") or "").strip()
    if cell_id:
        return ("cell", cell_id, description)
    task_id = row.get("raw_task_id", row.get("task_id"))
    if task_id is not None:
        return ("task_id", str(task_id), description)
    return ("instruction", description)


def _row_cell_key(row: dict) -> tuple[str, str] | None:
    """Return a stable distinct-cell key, or None when it is unavailable."""

    cell_id = str(row.get("cell_id") or "").strip()
    if not cell_id:
        return None
    return (cell_id, str(row["task_description"]))


def _row_family_key(row: dict) -> str | None:
    """Return only an explicit canonical family; never infer it from task_id."""

    family = str(row.get("task_family_id") or "").strip()
    return family or None


def _empty_task_identity_registry() -> dict:
    return {
        "by_task": {},
        "by_description": {},
        "meta": {
            "available": False,
            "source_path": None,
            "source_sha256": None,
            "num_source_episodes": None,
            "num_instruction_cells": None,
            "num_task_families": None,
            "task_families": [],
        },
    }


def load_controlled_task_identity_registry(experiment_root: Path) -> dict:
    """Load canonical family/cell identities from the frozen trajectory source.

    Bundles without the frozen profile keep family evidence unavailable instead
    of guessing from numeric task IDs or instruction wording.
    """

    root = resolve_groot_artifact_path(experiment_root).resolve()
    manifest_path = root / "experiment_manifest.json"
    if not manifest_path.is_file():
        return _empty_task_identity_registry()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_profile_path = str((manifest.get("profile") or {}).get("path") or "")
    if not raw_profile_path:
        return _empty_task_identity_registry()
    profile_path = Path(raw_profile_path).expanduser()
    repo_root = root.parents[3] if len(root.parents) > 3 else root.parent
    if not profile_path.is_absolute():
        profile_path = repo_root / profile_path
    if not profile_path.is_file():
        relocated = repo_root / "configs" / "groot" / profile_path.name
        if relocated.is_file():
            profile_path = relocated
        else:
            return _empty_task_identity_registry()

    profile = load_pipeline_profile(profile_path.resolve())
    trajectory_path = profile.path_value("source", "trajectory_records_path")
    if not trajectory_path.is_file():
        return _empty_task_identity_registry()
    trajectory_rows = load_jsonl(trajectory_path)
    if not trajectory_rows:
        raise ValueError("Controlled trajectory identity source is empty")

    by_task: dict[tuple[int, str], dict] = {}
    by_description: dict[str, dict] = {}
    episode_ids: set[int] = set()
    cells: set[str] = set()
    family_cells: dict[str, set[str]] = defaultdict(set)
    for source_row in trajectory_rows:
        task_id = int(source_row["task_id"])
        description = str(source_row["task_description"]).strip()
        cell_id = str(source_row.get("cell_id") or "").strip()
        family = str(source_row.get("task_family") or "").strip()
        if not description or not cell_id or not family:
            raise ValueError(
                "Controlled trajectory row is missing task_description, "
                "cell_id, or task_family"
            )
        identity = {
            "raw_task_id": task_id,
            "instruction_id": f"task{task_id}:{_slugify(description)}",
            "cell_id": cell_id,
            "task_family_id": family,
            "task_family_label": family,
            "task_identity_source": "source_trajectory",
        }
        task_key = (task_id, description)
        existing = by_task.get(task_key)
        if existing is not None and existing != identity:
            raise ValueError(f"Inconsistent task identity for {task_key}")
        by_task[task_key] = identity
        description_identity = by_description.get(description)
        if description_identity is not None and description_identity != identity:
            raise ValueError(
                f"Task description maps to multiple controlled cells: {description}"
            )
        by_description[description] = identity
        episode_ids.add(int(source_row["episode_num"]))
        cells.add(cell_id)
        family_cells[family].add(cell_id)

    return {
        "by_task": by_task,
        "by_description": by_description,
        "meta": {
            "available": True,
            "source_path": str(trajectory_path.resolve()),
            "source_sha256": _sha256(trajectory_path),
            "num_source_episodes": len(episode_ids),
            "num_instruction_cells": len(cells),
            "num_task_families": len(family_cells),
            "task_families": [
                {
                    "task_family_id": family,
                    "task_family_label": family,
                    "cell_ids": sorted(family_cell_ids),
                    "cell_count": len(family_cell_ids),
                }
                for family, family_cell_ids in sorted(family_cells.items())
            ],
        },
    }


def decorate_score_task_identities(
    score_data: dict,
    registry: dict | None,
) -> dict:
    """Attach explicit instruction/cell/family identity to score rows."""

    registry = registry or _empty_task_identity_registry()
    decorated_rows: list[dict] = []
    for raw_row in score_data["row_keys"]:
        row = dict(raw_row)
        task_id = row.get("raw_task_id", row.get("task_id"))
        description = str(row["task_description"])
        mapped = None
        if task_id is not None:
            mapped = registry["by_task"].get((int(task_id), description))
        if mapped is None:
            mapped = registry["by_description"].get(description)
        if mapped is not None:
            for field in (
                "raw_task_id",
                "instruction_id",
                "cell_id",
                "task_family_id",
                "task_family_label",
            ):
                existing = row.get(field)
                if existing not in (None, "") and str(existing) != str(mapped[field]):
                    raise ValueError(
                        f"Score row {description!r} conflicts with canonical {field}"
                    )
                row[field] = mapped[field]
            row["task_identity_source"] = mapped["task_identity_source"]
        else:
            row["raw_task_id"] = task_id
            row.setdefault(
                "instruction_id",
                (
                    f"task{task_id}:{_slugify(description)}"
                    if task_id is not None
                    else f"instruction:{_slugify(description)}"
                ),
            )
            row.setdefault("cell_id", None)
            row.setdefault("task_family_id", None)
            row.setdefault("task_family_label", None)
            row.setdefault("task_identity_source", "unavailable")
        decorated_rows.append(row)
    decorated = dict(score_data)
    decorated["row_keys"] = decorated_rows
    decorated["task_identity_meta"] = registry["meta"]
    return decorated


def _event_aligned_evidence_ladder(
    *,
    row_index: int,
    feature_id: int,
    row_keys: list[dict],
    selectivity: np.ndarray,
    selectivity_ranks: np.ndarray,
) -> dict:
    """Summarize instruction, family, and cross-family evidence.

    Family support counts distinct canonical cells. Missing family/cell identity
    is reported as unavailable and never inferred from a numeric task ID.
    """

    row = row_keys[row_index]
    phase = str(row["phase"])
    margin = selectivity[row_index, feature_id]
    rank = selectivity_ranks[row_index, feature_id]
    instruction_status = (
        "not_comparable"
        if not np.isfinite(margin)
        else "top20_candidate"
        if margin > 0 and 0 < rank <= PHASE_FEATURE_CANDIDATE_TOP_K
        else "positive_margin"
        if margin > 0
        else "not_selective"
    )

    family = _row_family_key(row)
    current_cell = _row_cell_key(row)
    family_rows = [
        index
        for index, candidate in enumerate(row_keys)
        if (
            family is not None
            and _row_family_key(candidate) == family
            and str(candidate["phase"]) == phase
            and _row_cell_key(candidate) is not None
            and np.isfinite(selectivity[index, feature_id])
        )
    ]
    eligible_family_cells = {
        _row_cell_key(row_keys[index]) for index in family_rows
    }
    positive_family_cells = {
        _row_cell_key(row_keys[index])
        for index in family_rows
        if selectivity[index, feature_id] > 0
    }
    top20_family_cells = {
        _row_cell_key(row_keys[index])
        for index in family_rows
        if (
            selectivity[index, feature_id] > 0
            and 0 < selectivity_ranks[index, feature_id]
            <= PHASE_FEATURE_CANDIDATE_TOP_K
        )
    }
    if family is None or current_cell is None:
        family_status = "mapping_unavailable"
    elif not np.isfinite(margin):
        family_status = "instruction_not_comparable"
    elif len(eligible_family_cells) <= 1:
        family_status = "single_cell_only"
    elif len(top20_family_cells) >= 2:
        family_status = "repeated_top20"
    elif len(positive_family_cells) >= 2:
        family_status = "positive_only"
    else:
        family_status = "not_repeated"

    phase_rows_by_family: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(row_keys):
        candidate_family = _row_family_key(candidate)
        if (
            candidate_family is not None
            and _row_cell_key(candidate) is not None
            and str(candidate["phase"]) == phase
            and np.isfinite(selectivity[index, feature_id])
        ):
            phase_rows_by_family[candidate_family].append(index)
    positive_families = {
        candidate_family
        for candidate_family, indices in phase_rows_by_family.items()
        if any(selectivity[index, feature_id] > 0 for index in indices)
    }
    top20_families = {
        candidate_family
        for candidate_family, indices in phase_rows_by_family.items()
        if any(
            selectivity[index, feature_id] > 0
            and 0 < selectivity_ranks[index, feature_id]
            <= PHASE_FEATURE_CANDIDATE_TOP_K
            for index in indices
        )
    }
    eligible_family_count = len(phase_rows_by_family)
    cross_family_status = (
        "not_assessable"
        if eligible_family_count < 2
        else "repeated_top20"
        if len(top20_families) >= 2
        else "positive_only"
        if len(positive_families) >= 2
        else "not_repeated"
    )
    return {
        "instruction": {
            "status": instruction_status,
            "instruction_id": row.get("instruction_id"),
            "cell_id": row.get("cell_id"),
            "task_description": str(row["task_description"]),
            "raw_task_id": row.get("raw_task_id", row.get("task_id")),
            "selectivity_margin": (
                float(margin) if np.isfinite(margin) else None
            ),
            "selectivity_rank": int(rank) if rank > 0 else None,
        },
        "family": {
            "status": family_status,
            "task_family_id": family,
            "task_family_label": row.get("task_family_label"),
            "eligible_cell_count": len(eligible_family_cells),
            "positive_cell_count": len(positive_family_cells),
            "top20_cell_count": len(top20_family_cells),
            "supporting_cell_ids": sorted(
                cell[0] for cell in top20_family_cells if cell is not None
            ),
            "comparison_contract": (
                "same canonical task family + exact phase + distinct cells"
            ),
        },
        "cross_family": {
            "status": cross_family_status,
            "phase_label": phase,
            "eligible_family_count": eligible_family_count,
            "positive_family_count": len(positive_families),
            "top20_family_count": len(top20_families),
            "eligible_task_family_ids": sorted(phase_rows_by_family),
            "comparison_contract": "exact phase-label match across families only",
        },
    }


def _round_robin_feature_ids(
    ranks: np.ndarray,
    *,
    limit: int,
) -> list[int]:
    """Keep every visible row represented while filling the feature axis."""

    selected: list[int] = []
    selected_set: set[int] = set()
    row_orders = [
        np.flatnonzero(row > 0)[np.argsort(row[row > 0], kind="stable")]
        for row in ranks
    ]
    cursor = 0
    while len(selected) < limit:
        added = False
        for order in row_orders:
            if cursor >= len(order):
                continue
            feature_id = int(order[cursor])
            if feature_id not in selected_set:
                selected.append(feature_id)
                selected_set.add(feature_id)
                added = True
                if len(selected) == limit:
                    break
        if not added and all(cursor >= len(order) - 1 for order in row_orders):
            break
        cursor += 1
    return selected


def load_phase_feature_score_matrix(
    *,
    score_path: Path,
    ranking: str,
) -> dict:
    """Load one full score matrix using the checkpoint's audited row order."""

    if ranking not in PHASE_FEATURE_HEATMAP_RANKINGS:
        raise ValueError(
            f"Unsupported phase heatmap ranking: {ranking}"
        )
    score_path = Path(score_path).expanduser().resolve()
    if not score_path.is_file():
        raise FileNotFoundError(f"Feature score artifact not found: {score_path}")

    import torch

    artifact = torch.load(
        score_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(artifact, dict):
        raise ValueError("Feature score artifact must contain a dictionary")
    matrix_key = PHASE_FEATURE_HEATMAP_RANKINGS[ranking]
    matrix_value = artifact.get(matrix_key)
    if (
        not torch.is_tensor(matrix_value)
        or matrix_value.ndim != 2
        or matrix_value.shape[0] <= 0
        or matrix_value.shape[1] <= 0
    ):
        raise ValueError(
            f"Feature score artifact has invalid {matrix_key}"
        )
    matrix = (
        matrix_value.detach()
        .to(dtype=torch.float64, device="cpu")
        .numpy()
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Feature score matrix contains a non-finite value")

    raw_row_keys = artifact.get("row_keys")
    if (
        not isinstance(raw_row_keys, list)
        or len(raw_row_keys) != matrix.shape[0]
    ):
        raise ValueError("Feature score row_keys do not match the matrix")
    row_keys: list[dict] = []
    seen_rows: set[tuple[str, str]] = set()
    for raw_row in raw_row_keys:
        if not isinstance(raw_row, dict):
            raise ValueError("Feature score row key must be an object")
        raw_task_id = raw_row.get("task_id")
        row = {
            "cluster_id": str(raw_row.get("cluster_id", "")).strip(),
            "task_description": str(
                raw_row.get("task_description", "")
            ).strip(),
            "task_id": (
                int(raw_task_id) if raw_task_id is not None else None
            ),
            "raw_task_id": (
                int(raw_task_id) if raw_task_id is not None else None
            ),
            "instruction_id": (
                str(raw_row["instruction_id"]).strip()
                if raw_row.get("instruction_id") is not None
                else None
            ),
            "cell_id": (
                str(raw_row["cell_id"]).strip()
                if raw_row.get("cell_id") is not None
                else None
            ),
            "task_family_id": (
                str(raw_row["task_family_id"]).strip()
                if raw_row.get("task_family_id") is not None
                else None
            ),
            "task_family_label": (
                str(
                    raw_row.get("task_family_label")
                    or raw_row["task_family_id"]
                ).strip()
                if raw_row.get("task_family_id") is not None
                else None
            ),
            "phase_scheme": (
                str(raw_row["phase_scheme"]).strip()
                if raw_row.get("phase_scheme") is not None
                else None
            ),
            "phrase": str(raw_row.get("phrase", "")).strip(),
            "phase": str(raw_row.get("phase", "")).strip(),
            "episode_coverage": (
                float(raw_row["episode_coverage"])
                if raw_row.get("episode_coverage") is not None
                else None
            ),
        }
        if (
            not row["cluster_id"]
            or not row["task_description"]
            or not row["phase"]
        ):
            raise ValueError("Feature score row identity is incomplete")
        identity = (row["task_description"], row["cluster_id"])
        if identity in seen_rows:
            raise ValueError(
                f"Feature score artifact has duplicate row {identity}"
            )
        seen_rows.add(identity)
        row_keys.append(row)

    row_results_by_identity: dict[tuple[str, str], dict] = {}
    for result in artifact.get("row_results") or []:
        if not isinstance(result, dict):
            continue
        identity = (
            str(result.get("task_description", "")),
            str(result.get("cluster_id", "")),
        )
        row_results_by_identity[identity] = result

    selected_task_ids: dict[tuple[str, str], set[int]] = defaultdict(set)
    for event in artifact.get("selected_events") or []:
        if not isinstance(event, dict) or event.get("task_id") is None:
            continue
        identity = (
            str(event.get("task_description", "")),
            str(event.get("cluster_id", "")),
        )
        selected_task_ids[identity].add(int(event["task_id"]))
    for row in row_keys:
        identity = (row["task_description"], row["cluster_id"])
        task_ids = selected_task_ids.get(identity, set())
        if len(task_ids) > 1:
            raise ValueError(
                f"Feature score row spans multiple task IDs: {identity}"
            )
        selected_task_id = next(iter(task_ids), None)
        if (
            row["task_id"] is not None
            and selected_task_id is not None
            and row["task_id"] != selected_task_id
        ):
            raise ValueError(
                f"Feature score row task ID mismatch: {identity}"
            )
        if row["task_id"] is None:
            row["task_id"] = selected_task_id
    return {
        "score_path": score_path,
        "score_sha256": _sha256(score_path),
        "ranking": ranking,
        "matrix_key": matrix_key,
        "matrix": matrix,
        "row_keys": row_keys,
        "row_results_by_identity": row_results_by_identity,
        "window_size": (
            int(artifact["window_size"])
            if artifact.get("window_size") is not None
            else None
        ),
        "row_semantics": str(artifact.get("row_semantics", "")),
        "score_definition": str(
            (artifact.get("score_definitions") or {}).get(matrix_key, "")
        ),
    }


def build_phase_feature_heatmap(
    *,
    score_data: dict,
    dataset: str,
    context: dict,
    mode: str,
    limit: int,
    task_description: str | None = None,
    pinned_feature_id: int | None = None,
) -> dict:
    """Build one compact, exact phase×feature matrix for the browser."""

    if mode not in {"strength", "selectivity"}:
        raise ValueError(f"Unsupported phase heatmap mode: {mode}")
    if not 4 <= limit <= 48:
        raise ValueError("Phase heatmap feature limit must be in [4, 48]")

    full_matrix = np.asarray(score_data["matrix"], dtype=np.float64)
    full_row_keys = list(score_data["row_keys"])
    full_selectivity = _task_local_phase_selectivity(
        full_matrix,
        full_row_keys,
    )
    full_selectivity_ranks = _rank_feature_rows(full_selectivity)
    available_tasks = [
        {
            "task_description": description,
            "task_ids": sorted(
                {
                    int(row["task_id"])
                    for row in full_row_keys
                    if (
                        row["task_description"] == description
                        and row.get("task_id") is not None
                    )
                }
            ),
            "cell_ids": sorted(
                {
                    str(row["cell_id"])
                    for row in full_row_keys
                    if row["task_description"] == description and row.get("cell_id")
                }
            ),
            "task_family_ids": sorted(
                {
                    str(row["task_family_id"])
                    for row in full_row_keys
                    if (
                        row["task_description"] == description
                        and row.get("task_family_id")
                    )
                }
            ),
            "phase_count": len(
                {
                    str(row["phase"])
                    for row in full_row_keys
                    if row["task_description"] == description
                }
            ),
        }
        for description in sorted(
            {str(row["task_description"]) for row in full_row_keys}
        )
    ]
    matrix = full_matrix
    row_keys = full_row_keys
    selected_indices = list(range(len(full_row_keys)))
    if task_description:
        selected_indices = [
            index
            for index, row in enumerate(full_row_keys)
            if row["task_description"] == task_description
        ]
        if not selected_indices:
            raise KeyError(task_description)
        matrix = full_matrix[selected_indices]
        row_keys = [full_row_keys[index] for index in selected_indices]

    strength_ranks = _rank_feature_rows(matrix)
    selectivity = full_selectivity[selected_indices]
    selectivity_ranks = full_selectivity_ranks[selected_indices]
    candidate_ranks = selectivity_ranks.copy()
    candidate_ranks[
        (~np.isfinite(selectivity))
        | (selectivity <= 0)
        | (selectivity_ranks > PHASE_FEATURE_CANDIDATE_TOP_K)
    ] = -1
    # Column membership is always governed by event-aligned selectivity.
    # ``strength`` changes only the color/value view; it must never reintroduce
    # persistent high-score features that are not phase-vs-rest candidates.
    active_ranks = candidate_ranks
    selected_features = _round_robin_feature_ids(
        active_ranks,
        limit=limit,
    )
    if pinned_feature_id is not None:
        pinned_feature_id = int(pinned_feature_id)
        if not 0 <= pinned_feature_id < matrix.shape[1]:
            raise ValueError("Pinned feature ID is outside the dictionary")
        if not np.any(candidate_ranks[:, pinned_feature_id] > 0):
            raise ValueError(
                "Pinned feature is not an event-aligned Top-20 +margin candidate"
            )
        if pinned_feature_id not in selected_features:
            if len(selected_features) >= limit:
                selected_features.pop()
            selected_features.append(pinned_feature_id)
    if not selected_features:
        return {
            "format": PHASE_FEATURE_HEATMAP_FORMAT,
            "dataset": dataset,
            "mode": mode,
            "ranking": score_data["ranking"],
            "window_size": score_data["window_size"],
            "context": context,
            "task_filter": task_description,
            "feature_limit": limit,
            "pinned_feature_id": pinned_feature_id,
            "dict_size": int(matrix.shape[1]),
            "row_count": len(row_keys),
            "available_tasks": available_tasks,
            "features": [],
            "rows": [],
            "anchor_kind": (
                "oracle_phase_entry"
                if dataset == "oracle"
                else "automatic_awe_event_anchor"
            ),
            "strict_phase_entry": dataset == "oracle",
            "temporal_semantics": PHASE_FEATURE_TEMPORAL_SEMANTICS,
            "candidate_contract": (
                "event_aligned Δ>0 row-local Top-20 only; persistent controls excluded"
            ),
            "message": (
                "선택한 범위에는 비교 가능한 event-aligned Δ>0 "
                "Top-20 후보가 없습니다."
            ),
            "score_definition": score_data["score_definition"],
            "score_sha256": score_data["score_sha256"],
        }

    # Candidate ownership and column ordering stay phase-selective even when
    # the user switches the cell colors to raw event-aligned score.
    active_values = selectivity
    feature_winners: dict[int, int] = {}
    for feature_id in selected_features:
        finite_rows = np.flatnonzero(
            np.isfinite(active_values[:, feature_id])
        )
        if not len(finite_rows):
            continue
        winner = finite_rows[
            int(np.argmax(active_values[finite_rows, feature_id]))
        ]
        feature_winners[feature_id] = int(winner)
    selected_features.sort(
        key=lambda feature_id: (
            feature_winners.get(feature_id, len(row_keys)),
            int(
                active_ranks[
                    feature_winners.get(feature_id, 0),
                    feature_id,
                ]
            )
            if feature_id in feature_winners
            else matrix.shape[1] + 1,
            feature_id,
        )
    )

    row_min = np.min(matrix, axis=1)
    row_max = np.max(matrix, axis=1)
    row_ranges = row_max - row_min
    finite_selected_margins = selectivity[
        :, selected_features
    ][np.isfinite(selectivity[:, selected_features])]
    selectivity_scale = (
        float(np.quantile(np.abs(finite_selected_margins), 0.95))
        if finite_selected_margins.size
        else 1.0
    )
    if not math.isfinite(selectivity_scale) or selectivity_scale <= 0:
        selectivity_scale = 1.0

    rows: list[dict] = []
    result_lookup = score_data["row_results_by_identity"]
    for row_index, row_key in enumerate(row_keys):
        identity = (
            row_key["task_description"],
            row_key["cluster_id"],
        )
        row_result = result_lookup.get(identity, {})
        cells = []
        for feature_id in selected_features:
            margin = selectivity[row_index, feature_id]
            strength = (
                (matrix[row_index, feature_id] - row_min[row_index])
                / row_ranges[row_index]
                if row_ranges[row_index] > 0
                else 0.0
            )
            cells.append(
                {
                    "feature_id": feature_id,
                    "score": float(matrix[row_index, feature_id]),
                    "strength_rank": int(
                        strength_ranks[row_index, feature_id]
                    ),
                    "strength": float(np.clip(strength, 0.0, 1.0)),
                    "selectivity_margin": (
                        float(margin) if math.isfinite(margin) else None
                    ),
                    "selectivity_rank": (
                        int(selectivity_ranks[row_index, feature_id])
                        if selectivity_ranks[row_index, feature_id] > 0
                        else None
                    ),
                    "selectivity": (
                        float(
                            np.clip(
                                margin / selectivity_scale,
                                -1.0,
                                1.0,
                            )
                        )
                        if math.isfinite(margin)
                        else None
                    ),
                }
            )
        rows.append(
            {
                "id": f"row_{row_index:03d}",
                **row_key,
                "num_episode_groups": (
                    int(row_result["num_episode_groups"])
                    if row_result.get("num_episode_groups") is not None
                    else None
                ),
                "num_events": (
                    int(row_result["num_events"])
                    if row_result.get("num_events") is not None
                    else None
                ),
                "cells": cells,
            }
        )
        full_row_index = selected_indices[row_index]
        for cell in cells:
            cell["evidence_ladder"] = _event_aligned_evidence_ladder(
                row_index=full_row_index,
                feature_id=int(cell["feature_id"]),
                row_keys=full_row_keys,
                selectivity=full_selectivity,
                selectivity_ranks=full_selectivity_ranks,
            )

    features = []
    for feature_id in selected_features:
        winner_index = feature_winners[feature_id]
        features.append(
            {
                "feature_id": feature_id,
                "winner_row_id": f"row_{winner_index:03d}",
                "winner_phase": row_keys[winner_index]["phase"],
                "winner_task_description": row_keys[winner_index][
                    "task_description"
                ],
                "strength_support": int(
                    np.sum(
                        (strength_ranks[:, feature_id] > 0)
                        & (strength_ranks[:, feature_id] <= 20)
                    )
                ),
                "positive_selectivity_support": int(
                    np.sum(selectivity[:, feature_id] > 0)
                ),
            }
        )
    return {
        "format": PHASE_FEATURE_HEATMAP_FORMAT,
        "dataset": dataset,
        "mode": mode,
        "ranking": score_data["ranking"],
        "window_size": score_data["window_size"],
        "context": context,
        "task_filter": task_description,
        "feature_limit": limit,
        "pinned_feature_id": pinned_feature_id,
        "dict_size": int(matrix.shape[1]),
        "row_count": len(rows),
        "feature_count": len(features),
        "features": features,
        "rows": rows,
        "available_tasks": available_tasks,
        "score_definition": score_data["score_definition"],
        "row_semantics": score_data["row_semantics"],
        "score_sha256": score_data["score_sha256"],
        "anchor_kind": (
            "oracle_phase_entry"
            if dataset == "oracle"
            else "automatic_awe_event_anchor"
        ),
        "strict_phase_entry": dataset == "oracle",
        "temporal_semantics": PHASE_FEATURE_TEMPORAL_SEMANTICS,
        "selectivity_definition": (
            "score(instruction/cell, phase, feature) - "
            "max(score(same instruction/cell, other phase, feature))"
        ),
        "selectivity_scale": selectivity_scale,
        "message": (
            "Full score matrix에서 event-aligned 상위 후보만 선택했습니다."
        ),
        "candidate_contract": (
            "event_aligned Δ>0 row-local Top-20 only; window_mean and "
            "task_mean are persistent-signal controls"
        ),
    }


def _phase_feature_control_sets(run: dict) -> tuple[dict, dict]:
    window_features: dict[tuple[str, str], set[int]] = {}
    for row in run["rankings"].get("window_mean", []):
        key = (str(row.get("task_description", "")), str(row.get("cluster_id", "")))
        window_features[key] = {
            int(feature["feature_id"])
            for feature in row.get("top_features", [])[:PHASE_FEATURE_CANDIDATE_TOP_K]
        }
    task_features: dict[str, set[int]] = {}
    for row in run["rankings"].get("task_mean", []):
        task_features[str(row.get("task_description", ""))] = {
            int(feature["feature_id"])
            for feature in row.get("top_features", [])[:PHASE_FEATURE_CANDIDATE_TOP_K]
        }
    return window_features, task_features


def build_phase_feature_overview(
    *,
    experiment_root: Path,
    results_payload: dict,
    task_identity_registry: dict,
    stage4_dir_name: str = "stage4",
) -> dict:
    """Aggregate the existing 45 event-aligned score matrices for the UI."""

    root = resolve_groot_artifact_path(experiment_root).resolve()
    stage4_dir_name = str(stage4_dir_name).strip()
    if (
        not stage4_dir_name
        or Path(stage4_dir_name).is_absolute()
        or len(Path(stage4_dir_name).parts) != 1
        or stage4_dir_name in {".", ".."}
    ):
        raise ValueError("stage4_dir_name must be one relative directory name")
    run_summaries: list[dict] = []
    support_by_family_phase: dict[tuple[str, str], dict] = {}
    tuple_occurrences: dict[
        tuple[str, str, str, str, int],
        list[dict],
    ] = defaultdict(list)
    family_phase_universe: dict[str, set[str]] = defaultdict(set)

    for run in results_payload["runs"]:
        score_path = (
            root
            / stage4_dir_name
            / str(run["condition_id"])
            / str(run["coverage_id"])
            / str(run["sae_id"])
            / "event_feature_scores.pt"
        )
        score_data = decorate_score_task_identities(
            load_phase_feature_score_matrix(
                score_path=score_path,
                ranking="event_aligned",
            ),
            task_identity_registry,
        )
        matrix = np.asarray(score_data["matrix"], dtype=np.float64)
        row_keys = list(score_data["row_keys"])
        selectivity = _task_local_phase_selectivity(matrix, row_keys)
        selectivity_ranks = _rank_feature_rows(selectivity)
        comparable_mask = np.any(np.isfinite(selectivity), axis=1)
        eligible_cells: dict[tuple[str, str], set[str]] = defaultdict(set)
        candidate_members: dict[tuple[str, str, int], list[dict]] = defaultdict(list)

        for row_index, row in enumerate(row_keys):
            family = _row_family_key(row)
            cell = _row_cell_key(row)
            phase = str(row["phase"])
            if family is None or cell is None or not comparable_mask[row_index]:
                continue
            cell_id = cell[0]
            family_phase_universe[family].add(phase)
            eligible_cells[(family, phase)].add(cell_id)
            candidate_ids = np.flatnonzero(
                (selectivity[row_index] > 0)
                & (selectivity_ranks[row_index] > 0)
                & (selectivity_ranks[row_index] <= PHASE_FEATURE_CANDIDATE_TOP_K)
            )
            for feature_id_value in candidate_ids:
                feature_id = int(feature_id_value)
                candidate_members[(family, phase, feature_id)].append(
                    {
                        "cell_id": cell_id,
                        "task_description": str(row["task_description"]),
                        "cluster_id": str(row["cluster_id"]),
                        "feature_id": feature_id,
                        "score": float(matrix[row_index, feature_id]),
                        "delta": float(selectivity[row_index, feature_id]),
                        "selectivity_rank": int(
                            selectivity_ranks[row_index, feature_id]
                        ),
                    }
                )

        window_features, task_features = _phase_feature_control_sets(run)
        repeated_tuples: list[dict] = []
        for (family, phase, feature_id), members in sorted(candidate_members.items()):
            members_by_cell = {member["cell_id"]: member for member in members}
            if len(members_by_cell) < 2:
                continue
            supporting_members = list(members_by_cell.values())
            deltas = [member["delta"] for member in supporting_members]
            scores = [member["score"] for member in supporting_members]
            repeated = {
                "task_family_id": family,
                "task_family_label": family,
                "phase": phase,
                "feature_id": feature_id,
                "supporting_cell_ids": sorted(members_by_cell),
                "supporting_cell_count": len(members_by_cell),
                "eligible_cell_count": len(eligible_cells[(family, phase)]),
                "worst_case_delta": float(min(deltas)),
                "best_delta": float(max(deltas)),
                "minimum_score": float(min(scores)),
                "maximum_score": float(max(scores)),
                "members": sorted(
                    supporting_members,
                    key=lambda member: member["cell_id"],
                ),
                "window_mean_top20_overlap": any(
                    feature_id
                    in window_features.get(
                        (member["task_description"], member["cluster_id"]),
                        set(),
                    )
                    for member in supporting_members
                ),
                "task_mean_top20_overlap": any(
                    feature_id in task_features.get(member["task_description"], set())
                    for member in supporting_members
                ),
            }
            repeated_tuples.append(repeated)
            occurrence_key = (
                str(run["condition_id"]),
                str(run["sae_id"]),
                family,
                phase,
                feature_id,
            )
            tuple_occurrences[occurrence_key].append(
                {
                    **repeated,
                    "analysis_id": str(run["id"]),
                    "coverage_id": str(run["coverage_id"]),
                    "coverage": float(run["coverage"]),
                }
            )

        repeated_phase_keys = {
            (row["task_family_id"], row["phase"]) for row in repeated_tuples
        }
        for family_phase, cells in eligible_cells.items():
            if len(cells) < 2:
                continue
            support = support_by_family_phase.setdefault(
                family_phase,
                {
                    "task_family_id": family_phase[0],
                    "task_family_label": family_phase[0],
                    "phase": family_phase[1],
                    "eligible_run_count": 0,
                    "repeat_positive_run_count": 0,
                },
            )
            support["eligible_run_count"] += 1
            if family_phase in repeated_phase_keys:
                support["repeat_positive_run_count"] += 1

        run_summaries.append(
            {
                "analysis_id": str(run["id"]),
                "condition_id": str(run["condition_id"]),
                "condition_code": str(run["condition_code"]),
                "coverage_id": str(run["coverage_id"]),
                "coverage": float(run["coverage"]),
                "checkpoint_id": str(run["sae_id"]),
                "checkpoint_label": str(run["sae_label"]),
                "total_phase_rows": len(row_keys),
                "comparable_phase_rows": int(np.sum(comparable_mask)),
                "not_comparable_phase_rows": int(np.sum(~comparable_mask)),
                "repeated_phase_count": len(repeated_phase_keys),
                "repeated_tuple_count": len(repeated_tuples),
                "eligible_family_phases": [
                    {
                        "task_family_id": family,
                        "phase": phase,
                        "eligible_cell_count": len(cells),
                    }
                    for (family, phase), cells in sorted(eligible_cells.items())
                    if len(cells) >= 2
                ],
                "repeated_candidates": repeated_tuples,
            }
        )

    compact_cells: list[dict] = []
    for condition in results_payload["facets"]["conditions"]:
        for coverage in results_payload["facets"]["coverages"]:
            bundle_runs = [
                run
                for run in run_summaries
                if run["condition_id"] == condition["id"]
                and run["coverage_id"] == coverage["id"]
            ]
            if not bundle_runs:
                continue
            row_counts = {
                (run["comparable_phase_rows"], run["total_phase_rows"])
                for run in bundle_runs
            }
            if len(row_counts) != 1:
                raise ValueError(
                    "Checkpoint score rows disagree within condition/coverage"
                )
            comparable_rows, total_rows = next(iter(row_counts))
            compact_cells.append(
                {
                    "condition_id": str(condition["id"]),
                    "condition_code": str(condition["code"]),
                    "coverage_id": str(coverage["id"]),
                    "coverage": float(coverage["value"]),
                    "comparable_phase_rows": comparable_rows,
                    "total_phase_rows": total_rows,
                    "checkpoints": [
                        {
                            "checkpoint_id": run["checkpoint_id"],
                            "checkpoint_label": run["checkpoint_label"],
                            "analysis_id": run["analysis_id"],
                            "repeated_phase_count": run["repeated_phase_count"],
                            "repeated_tuple_count": run["repeated_tuple_count"],
                        }
                        for run in sorted(
                            bundle_runs,
                            key=lambda item: item["checkpoint_label"],
                        )
                    ],
                }
            )

    required_coverages = {
        str(coverage["id"]) for coverage in results_payload["facets"]["coverages"]
    }
    representative_candidates: list[dict] = []
    representatives_by_condition_checkpoint: dict[
        tuple[str, str],
        list[dict],
    ] = defaultdict(list)
    for occurrence_key, occurrences in tuple_occurrences.items():
        condition_id, checkpoint_id, family, phase, feature_id = occurrence_key
        if {row["coverage_id"] for row in occurrences} != required_coverages:
            continue
        representatives_by_condition_checkpoint[(condition_id, checkpoint_id)].append(
            {
                "condition_id": condition_id,
                "checkpoint_id": checkpoint_id,
                "task_family_id": family,
                "phase": phase,
                "feature_id": feature_id,
                "worst_case_delta": min(row["worst_case_delta"] for row in occurrences),
                "minimum_score": min(row["minimum_score"] for row in occurrences),
                "supporting_cell_ids": sorted(
                    {
                        cell_id
                        for row in occurrences
                        for cell_id in row["supporting_cell_ids"]
                    }
                ),
                "coverage_ids": sorted(required_coverages),
                "analysis_ids": [
                    row["analysis_id"]
                    for row in sorted(occurrences, key=lambda item: item["coverage"])
                ],
                "window_mean_top20_overlap": any(
                    row["window_mean_top20_overlap"] for row in occurrences
                ),
                "task_mean_top20_overlap": any(
                    row["task_mean_top20_overlap"] for row in occurrences
                ),
            }
        )
    checkpoint_labels = {
        str(checkpoint["id"]): str(checkpoint["label"])
        for checkpoint in results_payload["facets"]["checkpoints"]
    }
    condition_codes = {
        str(condition["id"]): str(condition["code"])
        for condition in results_payload["facets"]["conditions"]
    }
    for (condition_id, checkpoint_id), candidates in sorted(
        representatives_by_condition_checkpoint.items()
    ):
        winner = max(
            candidates,
            key=lambda row: (row["worst_case_delta"], -row["feature_id"]),
        )
        representative_candidates.append(
            {
                **winner,
                "condition_code": condition_codes[condition_id],
                "checkpoint_label": checkpoint_labels[checkpoint_id],
            }
        )

    known_family_phase_sets = [
        phases for _, phases in sorted(family_phase_universe.items())
    ]
    shared_exact_phases = (
        sorted(set.intersection(*(set(phases) for phases in known_family_phase_sets)))
        if len(known_family_phase_sets) >= 2
        else []
    )
    identity_meta = task_identity_registry["meta"]
    total_rows = sum(run["total_phase_rows"] for run in run_summaries)
    comparable_rows = sum(run["comparable_phase_rows"] for run in run_summaries)
    bundle_phase_rows = sum(
        int(phase_set["num_phase_groups"])
        for phase_set in results_payload.get("phase_sets", [])
    )
    return {
        "format": PHASE_FEATURE_OVERVIEW_FORMAT,
        "headline": {
            "runs": len(run_summaries),
            "bundle_phase_rows": bundle_phase_rows,
            "total_phase_rows": total_rows,
            "comparable_phase_rows": comparable_rows,
            "not_comparable_phase_rows": total_rows - comparable_rows,
            "source_episodes": identity_meta["num_source_episodes"],
            "instruction_cells": identity_meta["num_instruction_cells"],
            "task_families": identity_meta["num_task_families"],
            "checkpoints": len(results_payload["facets"]["checkpoints"]),
        },
        "task_identity": identity_meta,
        "compact_cells": compact_cells,
        "family_phase_support": sorted(
            support_by_family_phase.values(),
            key=lambda row: (row["task_family_id"], row["phase"]),
        ),
        "representative_candidates": representative_candidates,
        "cross_family": {
            "status": "assessable" if shared_exact_phases else "not_assessable",
            "shared_exact_phase_count": len(shared_exact_phases),
            "shared_exact_phases": shared_exact_phases,
            "task_family_count": len(known_family_phase_sets),
            "interpretation": (
                "No exact phase ontology is shared across task families; this is N/A, "
                "not negative evidence."
                if not shared_exact_phases
                else "Only exact shared phase labels are eligible."
            ),
        },
        "semantics": {
            "candidate_ranking": "event_aligned",
            "candidate_contract": "within-instruction Δ>0 row-local Top-20",
            "temporal_semantics": PHASE_FEATURE_TEMPORAL_SEMANTICS,
            "anchor_kind": "automatic_awe_event_anchor",
            "strict_phase_entry": False,
            "persistent_controls": ["window_mean", "task_mean"],
            "feature_identity_scope": "checkpoint-local",
            "independent_replications": False,
            "sensitivity_artifacts_share_source_episodes": True,
        },
        "audit": {
            "claim_strength": str(results_payload["meta"]["claim_strength"]),
            "human_review": "0/90 automatic provisional labels",
            "controlled_w4": "not_run",
            "bootstrap": "not_run",
            "max_t": "not_run",
            "direction": "not_identifiable_from_combined_score",
            "cross_family_generalization": "not_assessable",
        },
    }


_TASK_LOCAL_GRID_SUMMARY_RELATIVE = (
    Path("analysis")
    / "task_local_phase_feature_ranking"
    / "summary.json"
)


def _probability(value: Any, *, field: str) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} must be a finite probability")
    return probability


def _phase_grid_test_key(*parts: str) -> str:
    return json.dumps(
        list(parts),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _load_phase_grid_cell(
    *,
    stage4_root: Path,
    summary_path: Path,
) -> dict[str, Any]:
    relative = summary_path.relative_to(stage4_root)
    if (
        len(relative.parts) != 5
        or Path(*relative.parts[2:]) != _TASK_LOCAL_GRID_SUMMARY_RELATIVE
    ):
        raise ValueError(
            "Task-local summary must be at "
            "<stage4>/<condition>/<coverage>/analysis/"
            "task_local_phase_feature_ranking/summary.json"
        )
    condition_dir, coverage_id = relative.parts[:2]
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "task_local_phase_feature_ranking_v2":
        raise ValueError(
            f"{summary_path}: unsupported task-local summary schema"
        )
    scope = payload.get("scope")
    analysis_config = payload.get("analysis_config")
    runs = payload.get("runs")
    matched_results = payload.get("matched_task_phase_results")
    if not isinstance(scope, dict) or not isinstance(analysis_config, dict):
        raise ValueError(f"{summary_path}: missing scope or analysis_config")
    if not isinstance(runs, dict) or not isinstance(matched_results, list):
        raise ValueError(f"{summary_path}: missing runs or matched results")

    condition_id = str(scope.get("condition_id") or "").strip()
    if condition_id != condition_dir:
        raise ValueError(
            f"{summary_path}: condition directory and summary disagree"
        )
    run_labels = [
        str(label).strip()
        for label in analysis_config.get("run_labels") or []
    ]
    if (
        not run_labels
        or len(run_labels) != len(set(run_labels))
        or set(run_labels) != set(runs)
    ):
        raise ValueError(f"{summary_path}: run labels are incomplete")
    if int(scope.get("num_runs", -1)) != len(run_labels):
        raise ValueError(f"{summary_path}: scope run count disagrees")
    reference_label = str(
        analysis_config.get("reference_label") or ""
    ).strip()
    if reference_label not in run_labels:
        raise ValueError(f"{summary_path}: invalid reference checkpoint label")
    alpha = _probability(
        scope.get("alpha"),
        field=f"{summary_path}: scope.alpha",
    )

    matched_keys: list[tuple[str, str]] = []
    for row in matched_results:
        if not isinstance(row, dict):
            raise ValueError(f"{summary_path}: invalid matched result")
        key = (
            str(row.get("task_description") or "").strip(),
            str(row.get("phase") or "").strip(),
        )
        if not all(key):
            raise ValueError(f"{summary_path}: incomplete task-phase key")
        matched_keys.append(key)
        _probability(
            row.get("best_holm_p"),
            field=f"{summary_path}: matched best_holm_p",
        )
    if len(matched_keys) != len(set(matched_keys)):
        raise ValueError(f"{summary_path}: duplicate matched task-phase cell")
    reported_support = scope.get("matched_statistically_supported") or {}
    if (
        int(reported_support.get("total", -1)) != len(matched_results)
        or int(reported_support.get("count", -1))
        != sum(
            bool(row.get("statistically_supported"))
            for row in matched_results
        )
    ):
        raise ValueError(f"{summary_path}: matched support count disagrees")

    matched_key_set = set(matched_keys)
    checkpoint_hashes: dict[str, str] = {}
    for label in run_labels:
        run = runs[label]
        if not isinstance(run, dict):
            raise ValueError(f"{summary_path}: invalid run {label!r}")
        checkpoint_hash = str(run.get("checkpoint_sha256") or "").strip()
        if not checkpoint_hash:
            raise ValueError(
                f"{summary_path}: run {label!r} lacks checkpoint SHA"
            )
        checkpoint_hashes[label] = checkpoint_hash
        task_results = run.get("task_phase_results")
        if not isinstance(task_results, dict):
            raise ValueError(
                f"{summary_path}: run {label!r} lacks task-phase results"
            )
        run_keys: set[tuple[str, str]] = set()
        for task_description, phase_results in task_results.items():
            if not isinstance(phase_results, dict):
                raise ValueError(
                    f"{summary_path}: invalid phases for {task_description!r}"
                )
            for phase, row in phase_results.items():
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{summary_path}: invalid checkpoint result"
                    )
                key = (str(task_description).strip(), str(phase).strip())
                if not all(key) or key in run_keys:
                    raise ValueError(
                        f"{summary_path}: invalid checkpoint task-phase key"
                    )
                run_keys.add(key)
                _probability(
                    row.get(
                        "best_holm_p_across_all_run_task_phase_cells"
                    ),
                    field=f"{summary_path}: checkpoint Holm p",
                )
        if run_keys != matched_key_set:
            raise ValueError(
                f"{summary_path}: checkpoint and matched task-phase cells differ"
            )

    return {
        "condition_id": condition_id,
        "coverage_id": coverage_id,
        "summary_path": summary_path,
        "summary_sha256": _sha256(summary_path),
        "payload": payload,
        "alpha": alpha,
        "run_labels": run_labels,
        "reference_label": reference_label,
        "checkpoint_hashes": checkpoint_hashes,
        "task_phase_keys": matched_keys,
    }


def _support_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    test_count = len(rows)
    cell_local_count = sum(
        row["cell_holm_p"] <= row["alpha"] for row in rows
    )
    grid_count = sum(row["grid_holm_p"] <= row["alpha"] for row in rows)
    return {
        "test_count": test_count,
        "cell_local_corrected_support_count": int(cell_local_count),
        "grid_corrected_support_count": int(grid_count),
        "grid_corrected_support_rate": (
            float(grid_count / test_count) if test_count else None
        ),
        "best_raw_p": (
            min(row["raw_p"] for row in rows) if rows else None
        ),
        "best_cell_holm_p": (
            min(row["cell_holm_p"] for row in rows) if rows else None
        ),
        "best_grid_holm_p": (
            min(row["grid_holm_p"] for row in rows) if rows else None
        ),
    }


def _aggregate_source_audit(
    cells: list[dict[str, Any]],
    *,
    checkpoint_test_count: int,
    matched_test_count: int,
) -> list[dict[str, str]]:
    source_statuses: dict[str, list[str]] = defaultdict(list)
    for cell in cells:
        for row in cell["payload"].get("confound_audit") or []:
            if isinstance(row, dict) and row.get("gate") and row.get("status"):
                source_statuses[str(row["gate"])].append(str(row["status"]))

    def inherited(gate: str) -> tuple[str, str]:
        statuses = source_statuses.get(gate, [])
        if not statuses:
            return "N/A", "Source summaries do not report this gate."
        if "FAIL" in statuses:
            status = "FAIL"
        elif statuses and all(value == "PASS" for value in statuses):
            status = "PASS"
        elif statuses and all(value == "N/A" for value in statuses):
            status = "N/A"
        else:
            status = "N/A"
        counts = {
            value: statuses.count(value)
            for value in ("PASS", "FAIL", "N/A")
            if value in statuses
        }
        evidence = ", ".join(
            f"{label} {count}/{len(statuses)}"
            for label, count in counts.items()
        )
        return status, f"Inherited source-audit statuses: {evidence}."

    audit: list[dict[str, str]] = []
    for gate in (
        "Length",
        "Task identity",
        "Instruction balance",
        "In-sample rescue",
        "Rollout pooling",
        "Phase / dwell",
    ):
        status, evidence = inherited(gate)
        audit.append({"gate": gate, "status": status, "evidence": evidence})
    audit.append(
        {
            "gate": "Feature multiplicity",
            "status": "PASS",
            "evidence": (
                "Source p-values use phase-wise feature max-T; this grid adds "
                f"Holm correction across {matched_test_count} decoder-matched "
                "primary tests and, separately, "
                f"{checkpoint_test_count} checkpoint sensitivity tests."
            ),
        }
    )
    for gate in (
        "Label confidence",
        "Checkpoint independence",
        "Observation != causation",
        "Scene-local != general",
        "Exact phase entry",
    ):
        status, evidence = inherited(gate)
        audit.append({"gate": gate, "status": status, "evidence": evidence})
    return audit


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _support_fraction(stats: dict[str, Any]) -> str:
    return (
        f"{stats['grid_corrected_support_count']}/"
        f"{stats['test_count']}"
    )


def _render_phase_grid_markdown(summary: dict[str, Any]) -> str:
    scope = summary["scope"]
    comparison = summary["focus_checkpoint_comparison"]
    superiority_label = (
        "established"
        if comparison["superiority_established"]
        else "not established"
    )
    lines = [
        "# V12 Stage4 phase-feature grid summary",
        "",
        "## 결과 수치와 범위",
        "",
        (
            f"- 분석 cell: {scope['analysis_cell_count']} "
            f"({scope['condition_count']} conditions × "
            f"{scope['coverage_count']} coverages)"
        ),
        (
            f"- SAE checkpoints: {scope['checkpoint_count']}; "
            f"checkpoint sensitivity tests: "
            f"{scope['checkpoint_test_count']}"
        ),
        (
            "- Decoder-matched primary tests: "
            f"{scope['matched_test_count']}; grid-Holm support: "
            f"{summary['matched_support']['grid_corrected_support_count']}/"
            f"{summary['matched_support']['test_count']}"
        ),
        (
            "- 모든 cell에 공통인 exact task×phase: "
            f"{scope['common_exact_task_phase_count']}"
        ),
        (
            f"- Focus checkpoint `{comparison['focus_checkpoint_label']}`: "
            f"공통 cell grid-Holm support "
            f"{comparison['focus_grid_corrected_support_count']}/"
            f"{comparison['tests_per_checkpoint']}"
        ),
        "",
        "## Confound audit",
        "",
        "| Gate | Status | Evidence |",
        "|---|---|---|",
    ]
    for row in summary["confound_audit"]:
        lines.append(
            f"| {_markdown_cell(row['gate'])} | "
            f"**{row['status']}** | {_markdown_cell(row['evidence'])} |"
        )
    lines.extend(
        [
            "",
            "## Claim strength",
            "",
            "- **diagnostic evidence**",
            (
                "- Overall verdict: "
                f"**{summary['verdict']}**"
            ),
            (
                f"- SAE superiority: **{superiority_label}**"
                f" — {_markdown_cell(comparison['reason'])}"
            ),
            "",
            "## SAE comparison on common exact task×phase cells",
            "",
            "| Checkpoint | Grid-Holm support | Best grid-Holm p | Count leader |",
            "|---|---:|---:|---|",
        ]
    )
    for checkpoint in summary["by_checkpoint"]:
        common = checkpoint["common_exact_task_phase"]
        best_p = common["best_grid_holm_p"]
        best_p_text = f"{best_p:.4g}" if best_p is not None else "N/A"
        is_count_leader = str(
            checkpoint["checkpoint_label"] in comparison["count_leaders"]
        ).lower()
        lines.append(
            f"| `{_markdown_cell(checkpoint['checkpoint_label'])}` | "
            f"{_support_fraction(common)} | "
            f"{best_p_text} | "
            f"{is_count_leader} |"
        )
    lines.extend(
        [
            "",
            "이 표의 SAE별 값은 checkpoint-local feature에 대한 보정된 "
            "sensitivity count이다. SAE 간 성능 차이를 직접 검정한 값은 아니다.",
            "",
            "## Condition × coverage × SAE",
            "",
        ]
    )
    checkpoint_labels = scope["checkpoint_labels"]
    header = (
        "| Condition | Coverage | Matched all-3 | "
        + " | ".join(f"`{label}`" for label in checkpoint_labels)
        + " |"
    )
    lines.extend(
        [
            header,
            "|---|---|---:|" + "---:|" * len(checkpoint_labels),
        ]
    )
    for cell in summary["cells"]:
        by_label = {
            row["checkpoint_label"]: row
            for row in cell["checkpoints"]
        }
        lines.append(
            f"| `{_markdown_cell(cell['condition_id'])}` | "
            f"`{_markdown_cell(cell['coverage_id'])}` | "
            f"{_support_fraction(cell['matched'])} | "
            + " | ".join(
                _support_fraction(by_label[label])
                for label in checkpoint_labels
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "각 분자는 전체 grid Holm 보정 후 alpha 이하인 수이다. "
            "기존 cell-local Holm 수는 machine-readable `summary.json`에 "
            "별도로 보존한다.",
            "",
            "## Common exact task×phase",
            "",
        ]
    )
    if summary["common_exact_task_phases"]:
        common_header = (
            "| Task | Phase | Matched support | "
            + " | ".join(f"`{label}`" for label in checkpoint_labels)
            + " |"
        )
        lines.extend(
            [
                common_header,
                "|---|---|---:|" + "---:|" * len(checkpoint_labels),
            ]
        )
        for row in summary["common_exact_task_phases"]:
            checkpoint_support = {
                checkpoint["checkpoint_label"]: checkpoint
                for checkpoint in row["checkpoints"]
            }
            lines.append(
                f"| {_markdown_cell(row['task_description'])} | "
                f"`{_markdown_cell(row['phase'])}` | "
                f"{_support_fraction(row['matched'])} | "
                + " | ".join(
                    _support_fraction(checkpoint_support[label])
                    for label in checkpoint_labels
                )
                + " |"
            )
    else:
        lines.append(
            f"{scope['analysis_cell_count']}개 cell 모두에 공통인 "
            "exact task×phase가 없다."
        )
    lines.extend(
        [
            "",
            "## Supported decoder-matched features",
            "",
        ]
    )
    supported_matched_tests = [
        row
        for row in summary["evidence"]["matched_tests"]
        if row["grid_statistically_supported"]
    ]
    if supported_matched_tests:
        lines.extend(
            [
                "| Condition | Coverage | Task | Phase | "
                "Checkpoint-local feature IDs | Grid Holm |",
                "|---|---|---|---|---|---:|",
            ]
        )
        for row in supported_matched_tests:
            feature_ids = ", ".join(
                (
                    f"{identity['checkpoint_label']}:"
                    + (
                        f"F{identity['feature_id']}"
                        if identity["feature_id"] is not None
                        else "N/A"
                    )
                )
                for identity in row["feature_identities"]
            )
            lines.append(
                f"| `{_markdown_cell(row['condition_id'])}` | "
                f"`{_markdown_cell(row['coverage_id'])}` | "
                f"{_markdown_cell(row['task_description'])} | "
                f"`{_markdown_cell(row['phase'])}` | "
                f"{_markdown_cell(feature_ids)} | "
                f"{row['grid_holm_p']:.4g} |"
            )
    else:
        lines.append("Grid-Holm을 통과한 decoder-matched feature triplet이 없다.")
    lines.extend(
        [
            "",
            "## Supported checkpoint-local features",
            "",
            "Feature ID는 반드시 `(checkpoint label, checkpoint SHA, feature ID)` "
            "tuple로 해석한다. 서로 다른 checkpoint의 같은 정수 ID를 같은 "
            "feature로 합치지 않는다.",
            "",
        ]
    )
    supported_checkpoint_tests = [
        row
        for row in summary["evidence"]["checkpoint_tests"]
        if row["grid_statistically_supported"]
    ]
    if supported_checkpoint_tests:
        lines.extend(
            [
                "| Condition | Coverage | Checkpoint | Task | Phase | "
                "Feature | Grid Holm |",
                "|---|---|---|---|---|---:|---:|",
            ]
        )
        for row in supported_checkpoint_tests:
            feature_id = row["feature_identity"]["feature_id"]
            lines.append(
                f"| `{_markdown_cell(row['condition_id'])}` | "
                f"`{_markdown_cell(row['coverage_id'])}` | "
                f"`{_markdown_cell(row['checkpoint_label'])}` | "
                f"{_markdown_cell(row['task_description'])} | "
                f"`{_markdown_cell(row['phase'])}` | "
                f"{feature_id if feature_id is not None else 'N/A'} | "
                f"{row['grid_holm_p']:.4g} |"
            )
    else:
        lines.append("Grid-Holm을 통과한 checkpoint-local feature가 없다.")
    lines.extend(
        [
            "",
            "이 결과는 관찰적 phase association 진단이다. Length, phase/dwell, "
            "label confidence, checkpoint independence 및 held-out "
            "generalization gate가 해결되지 않아 causal feature 또는 최종 SAE "
            "선정 근거로 단독 사용할 수 없다.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_stage4_grid(
    *,
    stage4_root: Path,
    output_dir: Path,
    expected_conditions: int = 5,
    expected_coverages: int = 3,
    focus_checkpoint_label: str = "sae10k",
) -> dict[str, Any]:
    """Aggregate task-local phase evidence over a complete Stage4 grid.

    Feature-search p-values are already max-T adjusted inside each source
    analysis. This function applies an additional Holm correction across the
    full condition×coverage grid. Decoder-matched all-checkpoint tests are the
    primary diagnostic family; checkpoint-local tests remain a sensitivity
    comparison even after correction.
    """

    stage4_root = Path(stage4_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not stage4_root.is_dir():
        raise FileNotFoundError(f"Stage4 root not found: {stage4_root}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if expected_conditions <= 0 or expected_coverages <= 0:
        raise ValueError("Expected grid dimensions must be positive")
    focus_checkpoint_label = str(focus_checkpoint_label).strip()
    if not focus_checkpoint_label:
        raise ValueError("focus_checkpoint_label must be non-empty")

    summary_paths = sorted(
        stage4_root.glob(
            f"*/*/{_TASK_LOCAL_GRID_SUMMARY_RELATIVE.as_posix()}"
        )
    )
    cells = [
        _load_phase_grid_cell(
            stage4_root=stage4_root,
            summary_path=path.resolve(),
        )
        for path in summary_paths
    ]
    if not cells:
        raise FileNotFoundError(
            f"No task-local Stage4 summaries found under {stage4_root}"
        )

    condition_ids = sorted({cell["condition_id"] for cell in cells})
    if len(condition_ids) != expected_conditions:
        raise ValueError(
            f"Expected {expected_conditions} conditions, found "
            f"{len(condition_ids)}"
        )
    coverage_sets = {
        condition_id: {
            cell["coverage_id"]
            for cell in cells
            if cell["condition_id"] == condition_id
        }
        for condition_id in condition_ids
    }
    for condition_id, coverage_ids in coverage_sets.items():
        if len(coverage_ids) != expected_coverages:
            raise ValueError(
                f"{condition_id}: expected {expected_coverages} coverages, "
                f"found {len(coverage_ids)}"
            )
    distinct_coverage_sets = {
        tuple(sorted(coverage_ids))
        for coverage_ids in coverage_sets.values()
    }
    if len(distinct_coverage_sets) != 1:
        raise ValueError("Coverage IDs differ across conditions")
    coverage_ids = list(next(iter(distinct_coverage_sets)))
    expected_cell_count = expected_conditions * expected_coverages
    if len(cells) != expected_cell_count:
        raise ValueError(
            f"Expected {expected_cell_count} grid cells, found {len(cells)}"
        )

    run_labels = list(cells[0]["run_labels"])
    if focus_checkpoint_label not in run_labels:
        raise ValueError(
            f"Focus checkpoint {focus_checkpoint_label!r} is not in "
            f"{run_labels}"
        )
    alpha = cells[0]["alpha"]
    reference_label = cells[0]["reference_label"]
    checkpoint_hashes = dict(cells[0]["checkpoint_hashes"])
    for cell in cells[1:]:
        if set(cell["run_labels"]) != set(run_labels):
            raise ValueError("Checkpoint labels differ across grid cells")
        if cell["reference_label"] != reference_label:
            raise ValueError(
                "Reference checkpoint label differs across grid cells"
            )
        if not math.isclose(cell["alpha"], alpha, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("Alpha differs across grid cells")
        if cell["checkpoint_hashes"] != checkpoint_hashes:
            raise ValueError("Checkpoint identity differs across grid cells")

    pair_sets = [set(cell["task_phase_keys"]) for cell in cells]
    common_task_phase_keys = set.intersection(*pair_sets)
    checkpoint_tests: list[dict[str, Any]] = []
    matched_tests: list[dict[str, Any]] = []
    checkpoint_raw_p: dict[str, float] = {}
    matched_raw_p: dict[str, float] = {}

    for cell in cells:
        condition_id = cell["condition_id"]
        coverage_id = cell["coverage_id"]
        payload = cell["payload"]
        for label in run_labels:
            task_results = payload["runs"][label]["task_phase_results"]
            for task_description, phase_results in task_results.items():
                for phase, row in phase_results.items():
                    candidates = row.get("top_candidates") or []
                    if not isinstance(candidates, list):
                        raise ValueError(
                            f"{cell['summary_path']}: top_candidates is not a list"
                        )
                    best = candidates[0] if candidates else None
                    raw_p = _probability(
                        best["max_t_p"] if best is not None else 1.0,
                        field="checkpoint max-T p",
                    )
                    test_key = _phase_grid_test_key(
                        "checkpoint",
                        condition_id,
                        coverage_id,
                        label,
                        str(task_description),
                        str(phase),
                    )
                    if test_key in checkpoint_raw_p:
                        raise ValueError(f"Duplicate checkpoint test: {test_key}")
                    checkpoint_raw_p[test_key] = raw_p
                    checkpoint_tests.append(
                        {
                            "test_key": test_key,
                            "condition_id": condition_id,
                            "coverage_id": coverage_id,
                            "checkpoint_label": label,
                            "checkpoint_sha256": checkpoint_hashes[label],
                            "task_description": str(task_description),
                            "phase": str(phase),
                            "common_exact_task_phase": (
                                (str(task_description), str(phase))
                                in common_task_phase_keys
                            ),
                            "feature_identity": {
                                "checkpoint_label": label,
                                "checkpoint_sha256": checkpoint_hashes[label],
                                "feature_id": (
                                    int(best["feature_id"])
                                    if best is not None
                                    else None
                                ),
                            },
                            "robust_margin": (
                                float(best["robust_margin"])
                                if best is not None
                                else None
                            ),
                            "raw_p": raw_p,
                            "cell_holm_p": _probability(
                                row[
                                    "best_holm_p_across_all_run_task_phase_cells"
                                ],
                                field="checkpoint cell Holm p",
                            ),
                            "alpha": alpha,
                            "inference_scope": "descriptive_sensitivity_only",
                        }
                    )

        for row in payload["matched_task_phase_results"]:
            task_description = str(row["task_description"])
            phase = str(row["phase"])
            candidates = row.get("top_candidates") or []
            if not isinstance(candidates, list):
                raise ValueError(
                    f"{cell['summary_path']}: matched candidates is not a list"
                )
            best = candidates[0] if candidates else None
            raw_p = _probability(
                best["p_all3_conjunction"] if best is not None else 1.0,
                field="matched conjunction p",
            )
            test_key = _phase_grid_test_key(
                "matched",
                condition_id,
                coverage_id,
                task_description,
                phase,
            )
            if test_key in matched_raw_p:
                raise ValueError(f"Duplicate matched test: {test_key}")
            matched_raw_p[test_key] = raw_p
            feature_ids = (
                best.get("feature_ids") if best is not None else {}
            )
            if feature_ids and set(feature_ids) != set(run_labels):
                raise ValueError(
                    f"{cell['summary_path']}: matched feature IDs are incomplete"
                )
            matched_tests.append(
                {
                    "test_key": test_key,
                    "condition_id": condition_id,
                    "coverage_id": coverage_id,
                    "task_description": task_description,
                    "phase": phase,
                    "common_exact_task_phase": (
                        (task_description, phase) in common_task_phase_keys
                    ),
                    "feature_identities": [
                        {
                            "checkpoint_label": label,
                            "checkpoint_sha256": checkpoint_hashes[label],
                            "feature_id": (
                                int(feature_ids[label])
                                if label in feature_ids
                                else None
                            ),
                        }
                        for label in run_labels
                    ],
                    "raw_p": raw_p,
                    "cell_holm_p": _probability(
                        row["best_holm_p"],
                        field="matched cell Holm p",
                    ),
                    "alpha": alpha,
                    "inference_scope": (
                        "grid_wide_decoder_matched_diagnostic_family"
                    ),
                }
            )

    # Keep the heavy statistical module lazy for read-only browser imports.
    from event_sae.scoring.phase_selectivity import holm_adjusted_p_values

    checkpoint_grid_holm = holm_adjusted_p_values(checkpoint_raw_p)
    matched_grid_holm = holm_adjusted_p_values(matched_raw_p)
    for row in checkpoint_tests:
        row["grid_holm_p"] = checkpoint_grid_holm[row["test_key"]]
        row["grid_statistically_supported"] = (
            row["grid_holm_p"] <= alpha
        )
    for row in matched_tests:
        row["grid_holm_p"] = matched_grid_holm[row["test_key"]]
        row["grid_statistically_supported"] = (
            row["grid_holm_p"] <= alpha
        )

    cell_summaries: list[dict[str, Any]] = []
    for cell in cells:
        condition_id = cell["condition_id"]
        coverage_id = cell["coverage_id"]
        cell_checkpoint_tests = [
            row
            for row in checkpoint_tests
            if row["condition_id"] == condition_id
            and row["coverage_id"] == coverage_id
        ]
        cell_matched_tests = [
            row
            for row in matched_tests
            if row["condition_id"] == condition_id
            and row["coverage_id"] == coverage_id
        ]
        cell_summaries.append(
            {
                "condition_id": condition_id,
                "coverage_id": coverage_id,
                "source_summary": str(cell["summary_path"]),
                "source_summary_sha256": cell["summary_sha256"],
                "matched": _support_statistics(cell_matched_tests),
                "checkpoints": [
                    {
                        "checkpoint_label": label,
                        "checkpoint_sha256": checkpoint_hashes[label],
                        **_support_statistics(
                            [
                                row
                                for row in cell_checkpoint_tests
                                if row["checkpoint_label"] == label
                            ]
                        ),
                    }
                    for label in run_labels
                ],
            }
        )

    by_condition = []
    for condition_id in condition_ids:
        condition_checkpoint_tests = [
            row
            for row in checkpoint_tests
            if row["condition_id"] == condition_id
        ]
        by_condition.append(
            {
                "condition_id": condition_id,
                "matched": _support_statistics(
                    [
                        row
                        for row in matched_tests
                        if row["condition_id"] == condition_id
                    ]
                ),
                "checkpoints": [
                    {
                        "checkpoint_label": label,
                        **_support_statistics(
                            [
                                row
                                for row in condition_checkpoint_tests
                                if row["checkpoint_label"] == label
                            ]
                        ),
                    }
                    for label in run_labels
                ],
            }
        )

    by_coverage = []
    for coverage_id in coverage_ids:
        coverage_checkpoint_tests = [
            row
            for row in checkpoint_tests
            if row["coverage_id"] == coverage_id
        ]
        by_coverage.append(
            {
                "coverage_id": coverage_id,
                "matched": _support_statistics(
                    [
                        row
                        for row in matched_tests
                        if row["coverage_id"] == coverage_id
                    ]
                ),
                "checkpoints": [
                    {
                        "checkpoint_label": label,
                        **_support_statistics(
                            [
                                row
                                for row in coverage_checkpoint_tests
                                if row["checkpoint_label"] == label
                            ]
                        ),
                    }
                    for label in run_labels
                ],
            }
        )

    by_checkpoint: list[dict[str, Any]] = []
    for label in run_labels:
        label_rows = [
            row
            for row in checkpoint_tests
            if row["checkpoint_label"] == label
        ]
        by_checkpoint.append(
            {
                "checkpoint_label": label,
                "checkpoint_sha256": checkpoint_hashes[label],
                "all_task_phase": _support_statistics(label_rows),
                "common_exact_task_phase": _support_statistics(
                    [
                        row
                        for row in label_rows
                        if row["common_exact_task_phase"]
                    ]
                ),
            }
        )

    common_exact_task_phases = []
    for task_description, phase in sorted(common_task_phase_keys):
        common_exact_task_phases.append(
            {
                "task_description": task_description,
                "phase": phase,
                "matched": _support_statistics(
                    [
                        row
                        for row in matched_tests
                        if row["task_description"] == task_description
                        and row["phase"] == phase
                    ]
                ),
                "checkpoints": [
                    {
                        "checkpoint_label": label,
                        **_support_statistics(
                            [
                                row
                                for row in checkpoint_tests
                                if row["checkpoint_label"] == label
                                and row["task_description"] == task_description
                                and row["phase"] == phase
                            ]
                        ),
                    }
                    for label in run_labels
                ],
            }
        )

    comparison_counts = {
        row["checkpoint_label"]: row["common_exact_task_phase"][
            "grid_corrected_support_count"
        ]
        for row in by_checkpoint
    }
    tests_per_checkpoint = by_checkpoint[0]["common_exact_task_phase"][
        "test_count"
    ]
    if common_task_phase_keys:
        if len(
            {
                row["common_exact_task_phase"]["test_count"]
                for row in by_checkpoint
            }
        ) != 1:
            raise ValueError("Common task-phase denominator differs by checkpoint")
        largest_count = max(comparison_counts.values())
        count_leaders = sorted(
            label
            for label, count in comparison_counts.items()
            if count == largest_count
        )
        if largest_count == 0:
            count_leaders = []
            comparison_status = "no_grid_corrected_support"
            reason = (
                "No checkpoint has grid-Holm support on the shared exact "
                "task×phase cells; superiority is not supported."
            )
        elif count_leaders == [focus_checkpoint_label]:
            comparison_status = "focus_is_unique_corrected_count_leader"
            reason = (
                "The focus checkpoint has the largest corrected-support count, "
                "but hit counts are not a paired test of SAE performance and "
                "the checkpoints are not independent."
            )
        elif focus_checkpoint_label in count_leaders:
            comparison_status = "focus_tied_for_corrected_count_lead"
            reason = (
                "The focus checkpoint ties for the largest corrected-support "
                "count; SAE superiority is not identified."
            )
        else:
            comparison_status = "another_checkpoint_leads_corrected_count"
            reason = (
                "The focus checkpoint does not lead the corrected-support "
                "count, and no paired SAE superiority test was run."
            )
    else:
        largest_count = 0
        count_leaders = []
        comparison_status = "not_assessable_no_common_task_phase"
        reason = (
            "No exact task×phase cell is shared by the full grid, so the "
            "checkpoint comparison is not assessable."
        )

    confound_audit = _aggregate_source_audit(
        cells,
        checkpoint_test_count=len(checkpoint_tests),
        matched_test_count=len(matched_tests),
    )
    summary: dict[str, Any] = {
        "schema_version": STAGE4_GRID_SUMMARY_FORMAT,
        "scope": {
            "stage4_root": str(stage4_root),
            "condition_count": len(condition_ids),
            "coverage_count": len(coverage_ids),
            "analysis_cell_count": len(cells),
            "checkpoint_count": len(run_labels),
            "checkpoint_labels": run_labels,
            "reference_checkpoint_label": reference_label,
            "checkpoint_test_count": len(checkpoint_tests),
            "matched_test_count": len(matched_tests),
            "common_exact_task_phase_count": len(common_task_phase_keys),
            "alpha": alpha,
        },
        "correction_contract": {
            "source_feature_search": (
                "phase-wise max-T over every feature, shared across W4/W5"
            ),
            "primary_family": (
                "all decoder-matched task-phase conjunction tests over every "
                "condition and coverage"
            ),
            "primary_family_size": len(matched_tests),
            "primary_grid_correction": "Holm FWER",
            "checkpoint_sensitivity_family": (
                "all checkpoint×task-phase best-candidate tests over every "
                "condition and coverage"
            ),
            "checkpoint_sensitivity_family_size": len(checkpoint_tests),
            "checkpoint_grid_correction": "Holm FWER",
            "checkpoint_inference_scope": "descriptive_sensitivity_only",
        },
        "matched_support": _support_statistics(matched_tests),
        "cells": sorted(
            cell_summaries,
            key=lambda row: (row["condition_id"], row["coverage_id"]),
        ),
        "by_condition": by_condition,
        "by_coverage": by_coverage,
        "by_checkpoint": by_checkpoint,
        "common_exact_task_phases": common_exact_task_phases,
        "focus_checkpoint_comparison": {
            "focus_checkpoint_label": focus_checkpoint_label,
            "tests_per_checkpoint": tests_per_checkpoint,
            "corrected_support_counts": comparison_counts,
            "largest_corrected_support_count": largest_count,
            "count_leaders": count_leaders,
            "focus_grid_corrected_support_count": comparison_counts[
                focus_checkpoint_label
            ],
            "focus_is_count_leader": (
                focus_checkpoint_label in count_leaders
            ),
            "focus_is_unique_count_leader": (
                count_leaders == [focus_checkpoint_label]
            ),
            "superiority_established": False,
            "status": comparison_status,
            "reason": reason,
        },
        "feature_identity_contract": {
            "scope": "checkpoint-local",
            "identity_fields": [
                "checkpoint_label",
                "checkpoint_sha256",
                "feature_id",
            ],
            "integer_ids_comparable_across_checkpoints": False,
            "decoder_matched_ids": (
                "Only feature_identities inside one matched test are linked, "
                "using the source strict all-pair decoder MNN contract."
            ),
        },
        "inputs": {
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "source_summaries": [
                {
                    "condition_id": cell["condition_id"],
                    "coverage_id": cell["coverage_id"],
                    "path": str(cell["summary_path"]),
                    "sha256": cell["summary_sha256"],
                }
                for cell in cells
            ],
        },
        "evidence": {
            "checkpoint_tests": sorted(
                checkpoint_tests,
                key=lambda row: (
                    row["condition_id"],
                    row["coverage_id"],
                    row["checkpoint_label"],
                    row["task_description"],
                    row["phase"],
                ),
            ),
            "matched_tests": sorted(
                matched_tests,
                key=lambda row: (
                    row["condition_id"],
                    row["coverage_id"],
                    row["task_description"],
                    row["phase"],
                ),
            ),
        },
        "confound_audit": confound_audit,
        "claim_strength": "diagnostic_evidence",
        "verdict": (
            "confounded — 판정 보류"
            if any(row["status"] == "FAIL" for row in confound_audit)
            else "diagnostic evidence only"
        ),
        "claim_contract": {
            "supported_claims": [
                "grid-corrected observational phase association",
                "descriptive corrected-support count by checkpoint",
            ],
            "prohibited_claims": [
                "best SAE from support-count differences alone",
                "causal phase-control feature",
                "same feature from equal integer IDs across checkpoints",
                "task-independent phase feature without held-out validation",
            ],
        },
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "report": str(output_dir / "report.md"),
        },
    }

    report_text = _render_phase_grid_markdown(summary)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        report_text,
        encoding="utf-8",
    )
    return summary


__all__ = [
    "RESULT_RANKINGS",
    "build_phase_feature_heatmap",
    "build_phase_feature_overview",
    "compact_ranking_row",
    "decorate_score_task_identities",
    "format_checkpoint_label",
    "load_controlled_task_identity_registry",
    "load_phase_feature_score_matrix",
    "summarize_stage4_grid",
]
