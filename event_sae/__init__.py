"""event_sae — lazy top-level package.

Subpackages have different dependency sets (e.g. `event_sae.sae` needs
`dictionary_learning`, `event_sae.openpi` needs the openpi fork). To
avoid forcing every consumer to install everything, the top-level
exports are loaded lazily on attribute access. Usage:

  from event_sae import SAE                  # loads event_sae.sae lazily
  from event_sae.openpi.eval import runner   # never touches event_sae.sae
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GROOT_LOG_ROOT = REPO_ROOT / "logs/groot_n15"
DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_EXPERIMENT_ROOT = (
    DEFAULT_GROOT_LOG_ROOT
    / "experiments/v9_abs_position_gripper_3view_action_phase_v1"
)
DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_PROFILE = (
    REPO_ROOT / "configs/groot/abs_position_gripper_multiview_phase.json"
)
LEGACY_GROOT_ARTIFACT_RELOCATIONS = {
    "pq3_l15_activation_cache": Path("stage1_sae/activation_cache"),
    "pq3_l15_stage1_selected": Path(
        "stage1_sae/checkpoints/step_sweep/bs4096_steps001200_seed0"
    ),
    "pq3_l15_candidate_4000": Path(
        "stage1_sae/checkpoints/step_sweep/bs4096_steps004000_seed0"
    ),
    "pq3_l15_production_10000": Path(
        "stage1_sae/checkpoints/step_sweep/bs4096_steps010000_seed0"
    ),
    "pq3_l15_candidate_20000": Path(
        "stage1_sae/checkpoints/step_sweep/bs4096_steps020000_seed0"
    ),
    "pq3_l15_bs8192_rows4915200_seed0": Path(
        "stage1_sae/checkpoints/batch_size_sweep/"
        "bs8192_steps000600_rows04915200_seed0"
    ),
    "pq3_l15_bs16384_rows4915200_seed0": Path(
        "stage1_sae/checkpoints/batch_size_sweep/"
        "bs16384_steps000300_rows04915200_seed0"
    ),
    "pq3_l15_bs8192_rows40960000_seed0": Path(
        "stage1_sae/checkpoints/batch_size_sweep/"
        "bs8192_steps005000_rows40960000_seed0"
    ),
    "pq3_l15_bs16384_rows40960000_seed0": Path(
        "stage1_sae/checkpoints/batch_size_sweep/"
        "bs16384_steps002500_rows40960000_seed0"
    ),
    "pq3_l15_batch_sweep_summary.json": Path(
        "stage1_sae/batch_size_sweep_summary.json"
    ),
    "pq3_stage2_keyframes": Path("stage2_waypoints/relative_position"),
    "pq3_stage2_keyframes_abs": Path("stage2_waypoints/absolute_position"),
    "pq3_stage3_events": Path(
        "stage3_event_descriptors/relative_position"
    ),
    "pq3_stage3_events_abs": Path(
        "stage3_event_descriptors/absolute_position"
    ),
    "pq3_stage4_event_sae": Path("stage4_feature_ranking"),
    "v9_abs_gripper_3view_actionphase_v1": Path(
        "experiments/v9_abs_position_gripper_3view_action_phase_v1"
    ),
}
SUPPORTED_PROFILE_FORMAT = "event_sae_pipeline_profile_v1"


def resolve_groot_artifact_path(path: str | Path) -> Path:
    """Map a pre-migration GR00T artifact path to its canonical location.

    Historical artifacts retain their original path strings so their bytes and
    provenance hashes stay immutable. This resolver replaces the removed
    top-level symlink layer for project loaders.
    """

    candidate = Path(path).expanduser()
    parts = candidate.parts
    for index in range(max(0, len(parts) - 2)):
        if parts[index : index + 2] != ("logs", "groot_n15"):
            continue
        legacy_name = parts[index + 2]
        relocated = LEGACY_GROOT_ARTIFACT_RELOCATIONS.get(legacy_name)
        if relocated is not None:
            return DEFAULT_GROOT_LOG_ROOT.joinpath(
                relocated,
                *parts[index + 3 :],
            )
        break
    if not candidate.is_absolute() and parts:
        relocated = LEGACY_GROOT_ARTIFACT_RELOCATIONS.get(parts[0])
        if relocated is not None:
            return DEFAULT_GROOT_LOG_ROOT.joinpath(relocated, *parts[1:])
    return candidate


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with resolve_groot_artifact_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PipelineProfile:
    """A validated pipeline profile with typed nested-value access."""

    path: Path
    data: dict[str, Any]

    @property
    def profile_id(self) -> str:
        return str(self.data["profile_id"])

    def require(self, *keys: str) -> Any:
        value: Any = self.data
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                dotted = ".".join(keys)
                raise ValueError(f"{self.path}: missing profile value {dotted}")
            value = value[key]
        return value

    def path_value(self, *keys: str) -> Path:
        value = Path(str(self.require(*keys)))
        resolved = value if value.is_absolute() else REPO_ROOT / value
        return resolve_groot_artifact_path(resolved)


def load_pipeline_profile(
    path: Path = DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_PROFILE,
) -> PipelineProfile:
    """Load and minimally validate an Event-SAE pipeline profile."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Pipeline profile not found: {resolved}")
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{resolved}: profile must be a JSON object")
    if data.get("format") != SUPPORTED_PROFILE_FORMAT:
        raise ValueError(
            f"{resolved}: unsupported profile format {data.get('format')!r}"
        )
    profile_id = data.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError(f"{resolved}: profile_id must be a non-empty string")
    clustering = data.get("clustering")
    if isinstance(clustering, dict) and "episode_coverage_sweep" in clustering:
        raw_thresholds = clustering["episode_coverage_sweep"]
        if not isinstance(raw_thresholds, list) or not raw_thresholds:
            raise ValueError(
                f"{resolved}: clustering.episode_coverage_sweep must be a non-empty list"
            )
        thresholds = [float(value) for value in raw_thresholds]
        if (
            thresholds != sorted(set(thresholds))
            or any(not 0.0 <= value <= 1.0 for value in thresholds)
        ):
            raise ValueError(
                f"{resolved}: coverage sweep must be unique, sorted, and within [0, 1]"
            )
        annotation_min = float(
            clustering.get("annotation_min_episode_coverage", thresholds[0])
        )
        if annotation_min != thresholds[0]:
            raise ValueError(
                f"{resolved}: annotation minimum must equal the lowest sweep threshold"
            )
    return PipelineProfile(path=resolved, data=data)


__all__ = [
    "SAE",
    "BatchTopKSAE",
    "SAETrainConfig",
    "train_sae",
    "DEFAULT_GROOT_LOG_ROOT",
    "DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_EXPERIMENT_ROOT",
    "DEFAULT_GROOT_ABS_GRIPPER_MULTIVIEW_PROFILE",
    "LEGACY_GROOT_ARTIFACT_RELOCATIONS",
    "PipelineProfile",
    "load_pipeline_profile",
    "resolve_groot_artifact_path",
    "sha256_file",
]


def __getattr__(name: str):
    if name in ("SAE", "BatchTopKSAE"):
        from event_sae import sae as _sae

        return getattr(_sae, name)
    if name in ("SAETrainConfig", "train_sae"):
        from event_sae import train as _train

        return getattr(_train, name)
    raise AttributeError(f"module 'event_sae' has no attribute {name!r}")
