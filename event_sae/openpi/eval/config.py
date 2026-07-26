"""Configuration dataclasses and YAML loader for openpi LIBERO eval.

Mirrors `event_sae.openvla.eval.config` where possible. openpi runs a
server-client split, so an extra ``ServerConfig`` block records the
host / port / replan that the openpi-client websocket attaches to.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5


@dataclass
class EnvConfig:
    task_suite_name: str = "libero_spatial"
    task_ids: list[int] | None = None   # None = every task in the suite
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    seed: int = 7
    resolution: int = 256
    max_steps: int | None = None


@dataclass
class LoggingConfig:
    root_dir: str = "logs/openpi"
    run_tag: str = ""
    save_video: bool = True
    save_actions: bool = True
    save_chunk_records: bool = True
    save_prompt_records: bool = True
    save_trajectory_records: bool = True


@dataclass
class LiberoConfig:
    config_path: str = "examples/libero/libero_config"
    mujoco_gl: str = "egl"


@dataclass
class SAECollectConfig:
    enabled: bool = False
    mode: str = "dense"               # "dense" or "topk"
    capture_target: str = "action_expert"  # "action_expert" or "paligemma"
    layer_idxs: str = "17"            # comma-separated for dense; single int for topk
    flush_every_rows: int = 50_000

    # episode lifecycle (matches openpi-mech SaeCollectConfig)
    finalize_episodes: bool = True    # send _sae_collection_control finalize per episode
    keep_failed_episodes: bool = True   # collection-time default: keep failed-rollout
                                        # activations (still valid training data); flip
                                        # to False only if downstream needs success-only.

    # topk-only
    sae_checkpoint: str = ""
    topk: int = 64
    rows_per_shard: int = 20_000


@dataclass
class RunConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    libero: LiberoConfig = field(default_factory=LiberoConfig)
    sae_collect: SAECollectConfig = field(default_factory=SAECollectConfig)


def validate_policy_override_pair(
    config_name: str | None,
    checkpoint_dir: str | None,
) -> None:
    """Require custom policy configuration and checkpoint as one contract."""

    if bool(config_name) != bool(checkpoint_dir):
        raise ValueError(
            "--config and --checkpoint-dir form one policy contract; "
            "provide both or neither."
        )


def validate_intervention_layer(
    capture_target: str,
    layer_idx: int,
    model_depth: int,
) -> None:
    """Reject out-of-range and structurally inert intervention layers."""

    if layer_idx < 0 or layer_idx >= model_depth:
        raise ValueError(
            f"layer_idx={layer_idx} outside model depth {model_depth} "
            f"for {capture_target}."
        )
    if capture_target == "paligemma" and layer_idx == model_depth - 1:
        raise ValueError(
            "The final PaliGemma layer is not a valid intervention target: "
            "no downstream action-expert block consumes the edited prefix state."
        )


def validate_sae_intervention_checkpoint(
    checkpoint_path: str | Path,
    *,
    capture_target: str,
    layer_idx: int,
    activation_dim: int,
    state_dict: Mapping[str, Any],
    feature_indices: Sequence[int] = (),
    active_feature_drop_count: int = 0,
) -> dict[str, Any]:
    """Validate an SAE and its sibling config before installing a policy hook."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")

    config_path = checkpoint_path.with_name("config.json")
    if not config_path.is_file():
        raise FileNotFoundError(
            "SAE checkpoint contract requires a sibling config.json: "
            f"{config_path}"
        )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        trainer = payload["trainer"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid SAE config contract: {config_path}") from exc

    expected_submodule = {
        "action_expert": "post_mlp_residual",
        "paligemma": "post_mlp_residual__paligemma",
    }.get(capture_target)
    if expected_submodule is None:
        raise ValueError(f"Unsupported capture_target={capture_target!r}")

    required_config_fields = (
        "dict_class",
        "layer",
        "activation_dim",
        "dict_size",
        "submodule_name",
    )
    missing_config = [key for key in required_config_fields if key not in trainer]
    if missing_config:
        raise ValueError(
            f"SAE config {config_path} missing trainer fields: {missing_config}"
        )

    if trainer["dict_class"] != "BatchTopKSAE":
        raise ValueError(
            "Expected BatchTopKSAE checkpoint, got "
            f"dict_class={trainer['dict_class']!r}"
        )

    configured_layer = int(trainer["layer"])
    configured_activation_dim = int(trainer["activation_dim"])
    dictionary_size = int(trainer["dict_size"])
    configured_submodule = str(trainer["submodule_name"])
    if configured_layer != int(layer_idx):
        raise ValueError(
            f"SAE layer mismatch: config has {configured_layer}, "
            f"but --layer-idx={layer_idx}."
        )
    if configured_activation_dim != int(activation_dim):
        raise ValueError(
            f"SAE activation width mismatch: config has "
            f"{configured_activation_dim}, model target has {activation_dim}."
        )
    if configured_submodule != expected_submodule:
        raise ValueError(
            f"SAE capture-target mismatch: config submodule is "
            f"{configured_submodule!r}, expected {expected_submodule!r} "
            f"for {capture_target}."
        )
    if dictionary_size <= 0:
        raise ValueError(f"SAE dict_size must be positive, got {dictionary_size}.")

    required_state = (
        "encoder.weight",
        "encoder.bias",
        "decoder.weight",
        "b_dec",
        "k",
        "threshold",
    )
    missing_state = [key for key in required_state if key not in state_dict]
    if missing_state:
        raise ValueError(
            f"SAE checkpoint {checkpoint_path} missing keys: {missing_state}"
        )

    expected_shapes = {
        "encoder.weight": (dictionary_size, configured_activation_dim),
        "encoder.bias": (dictionary_size,),
        "decoder.weight": (configured_activation_dim, dictionary_size),
        "b_dec": (configured_activation_dim,),
    }
    for key, expected_shape in expected_shapes.items():
        raw_shape = getattr(state_dict[key], "shape", None)
        actual_shape = (
            tuple(int(dimension) for dimension in raw_shape)
            if raw_shape is not None
            else None
        )
        if actual_shape != expected_shape:
            raise ValueError(
                f"SAE tensor shape mismatch for {key}: "
                f"got {actual_shape}, expected {expected_shape}."
            )

    normalized_features = tuple(int(feature_id) for feature_id in feature_indices)
    if len(set(normalized_features)) != len(normalized_features):
        raise ValueError(
            f"Duplicate --feature-indices are not allowed: {normalized_features}"
        )
    invalid_features = [
        feature_id
        for feature_id in normalized_features
        if feature_id < 0 or feature_id >= dictionary_size
    ]
    if invalid_features:
        raise ValueError(
            f"Feature ids outside [0, {dictionary_size}): {invalid_features}"
        )
    if not 0 <= int(active_feature_drop_count) <= dictionary_size:
        raise ValueError(
            "--active-feature-drop-count must be between 0 and "
            f"{dictionary_size}, got {active_feature_drop_count}."
        )

    return {
        "config_path": str(config_path),
        "layer_idx": configured_layer,
        "activation_dim": configured_activation_dim,
        "dict_size": dictionary_size,
        "submodule_name": configured_submodule,
    }


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path, overrides: Dict[str, Any] | None = None) -> RunConfig:
    import yaml

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data = _deep_update(data, overrides)
    return RunConfig(
        server=ServerConfig(**data.get("server", {})),
        env=EnvConfig(**data.get("env", {})),
        logging=LoggingConfig(**data.get("logging", {})),
        libero=LiberoConfig(**data.get("libero", {})),
        sae_collect=SAECollectConfig(**data.get("sae_collect", {})),
    )


def parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    """Parse ``--override key.path=value`` strings into nested dicts.

    Values are parsed via ``yaml.safe_load`` (matches openpi-mech
    `run_eval.py::_parse_override`), so list / null / bool / int / float
    literals all work, e.g.
    ``env.task_ids=[0,1,2]`` → ``[0, 1, 2]``.
    """
    import yaml

    overrides: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        target = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return overrides
