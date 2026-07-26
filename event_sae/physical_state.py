"""Shared physical-state transforms for waypoint sampling and event features.

The RoboCasa action-phase extension deliberately computes gripper state in one
place so waypoint extraction and clustering cannot silently use different
definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, TypeVar

import numpy as np
from scipy.signal import find_peaks


KeyT = TypeVar("KeyT", bound=Hashable)


@dataclass(frozen=True)
class GripperStateSeries:
    """Task-normalized gripper aperture and its record-aligned derivative."""

    aperture: np.ndarray
    normalized_aperture: np.ndarray
    aperture_delta: np.ndarray
    normalization_min: float
    normalization_max: float


def gripper_aperture_from_qpos(gripper_qpos: np.ndarray) -> np.ndarray:
    """Return the sign-invariant two-finger aperture ``sum(abs(qpos))``."""

    qpos = np.asarray(gripper_qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[0] == 0 or qpos.shape[1] == 0:
        raise ValueError(
            "gripper_qpos must be a non-empty [T,D] array, "
            f"got shape {qpos.shape}"
        )
    if not np.isfinite(qpos).all():
        raise ValueError("gripper_qpos contains non-finite values")
    return np.abs(qpos).sum(axis=1)


def normalize_gripper_aperture(
    aperture: np.ndarray,
    *,
    normalization_min: float,
    normalization_max: float,
) -> GripperStateSeries:
    """Apply a fixed task-local min/max transform and a backward difference."""

    aperture = np.asarray(aperture, dtype=np.float64)
    if aperture.ndim != 1 or aperture.size == 0:
        raise ValueError(
            f"aperture must be a non-empty one-dimensional array, got {aperture.shape}"
        )
    if not np.isfinite(aperture).all():
        raise ValueError("aperture contains non-finite values")
    if not np.isfinite(normalization_min) or not np.isfinite(normalization_max):
        raise ValueError("gripper normalization bounds must be finite")
    if normalization_max < normalization_min:
        raise ValueError("gripper normalization max must be >= min")

    span = float(normalization_max - normalization_min)
    if span <= np.finfo(np.float64).eps:
        normalized = np.zeros_like(aperture)
    else:
        normalized = np.clip((aperture - normalization_min) / span, 0.0, 1.0)
    delta = np.empty_like(normalized)
    delta[0] = 0.0
    delta[1:] = normalized[1:] - normalized[:-1]
    return GripperStateSeries(
        aperture=aperture,
        normalized_aperture=normalized,
        aperture_delta=delta,
        normalization_min=float(normalization_min),
        normalization_max=float(normalization_max),
    )


def build_task_local_gripper_states(
    qpos_by_key: Mapping[KeyT, np.ndarray],
    task_by_key: Mapping[KeyT, str],
) -> tuple[dict[KeyT, GripperStateSeries], dict[str, tuple[float, float]]]:
    """Fit aperture bounds per task and transform every keyed trajectory.

    The normalization scope is all supplied trajectories for a task. Callers
    should therefore pass the full source trajectory set, even when a later
    filter selects only a subset of episodes or waypoints.
    """

    if set(qpos_by_key) != set(task_by_key):
        raise ValueError("qpos_by_key and task_by_key must contain the same keys")
    aperture_by_key = {
        key: gripper_aperture_from_qpos(qpos)
        for key, qpos in qpos_by_key.items()
    }
    aperture_by_task: dict[str, list[np.ndarray]] = {}
    for key, aperture in aperture_by_key.items():
        aperture_by_task.setdefault(str(task_by_key[key]), []).append(aperture)

    bounds_by_task: dict[str, tuple[float, float]] = {}
    for task, arrays in aperture_by_task.items():
        concatenated = np.concatenate(arrays)
        bounds_by_task[task] = (
            float(np.min(concatenated)),
            float(np.max(concatenated)),
        )

    states = {}
    for key, aperture in aperture_by_key.items():
        lower, upper = bounds_by_task[str(task_by_key[key])]
        states[key] = normalize_gripper_aperture(
            aperture,
            normalization_min=lower,
            normalization_max=upper,
        )
    return states, bounds_by_task


def gripper_closing_peak_indices(
    normalized_aperture: np.ndarray,
    *,
    min_height: float = 0.08,
    min_prominence: float = 0.04,
    min_distance: int = 3,
) -> list[int]:
    """Detect record indices with a prominent decrease in normalized aperture."""

    normalized = np.asarray(normalized_aperture, dtype=np.float64)
    if normalized.ndim != 1:
        raise ValueError(
            "normalized_aperture must be one-dimensional, "
            f"got shape {normalized.shape}"
        )
    if normalized.size < 2:
        return []
    if not np.isfinite(normalized).all():
        raise ValueError("normalized_aperture contains non-finite values")
    if not np.isfinite(min_height) or min_height <= 0.0:
        raise ValueError("min_height must be finite and positive")
    if not np.isfinite(min_prominence) or min_prominence < 0.0:
        raise ValueError("min_prominence must be finite and non-negative")
    if min_distance < 1:
        raise ValueError("min_distance must be at least 1")

    delta = np.empty_like(normalized)
    delta[0] = 0.0
    delta[1:] = normalized[1:] - normalized[:-1]
    peaks, _ = find_peaks(
        -delta,
        height=min_height,
        prominence=min_prominence,
        distance=min_distance,
    )
    return [int(index) for index in peaks]
