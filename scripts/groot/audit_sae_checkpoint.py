"""Audit a GR00T PQ3 BatchTopK SAE checkpoint on action-token residuals.

This is intentionally a thin adapter. It reuses the PQ3 loader from
train_sae_robocasa.py, Event-SAE's checkpoint loader, and
dictionary_learning.evaluation.evaluate. GR00T-specific code is limited
to source/checkpoint contract checks, deterministic sampling, per-feature
firing statistics, and durable JSON/NPZ output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dictionary_learning.evaluation import evaluate as evaluate_dictionary


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from event_sae.openvla.activations import load_batch_topk_sae
from scripts.groot.train_sae_robocasa import (
    DEFAULT_ACTION_HORIZON,
    DEFAULT_FEATURE_KIND,
    DEFAULT_FUTURE_TOKENS,
    DEFAULT_MODEL_TOKENS,
    DEFAULT_STATE_TOKENS,
    EXPECTED_PQ3_CELL_COUNTS,
    InMemoryBatchLoader,
    load_activation_cache,
    load_layer_activations,
)


EXPECTED_LM_NAME = "groot_n15"
EXPECTED_SUBMODULE = "dit_block_residual_action_tokens"
SOURCE_IDENTITY_KEYS = (
    "format",
    "source_root",
    "source_files",
    "num_files",
    "cell_counts",
    "num_records",
    "num_activation_rows",
    "feature_kind",
    "feature_axes",
    "capture_token_mode",
    "capture_layers",
    "physical_layer",
    "source_denoising_steps",
    "source_model_tokens",
    "token_scope",
    "action_horizon",
    "action_token_slice",
    "activation_dim",
    "source_dtype",
)


def _checkpoint_and_run_dirs(checkpoint: Path) -> tuple[Path, Path, Path]:
    checkpoint = checkpoint.resolve()
    if checkpoint.is_dir():
        trainer_dir = checkpoint
        checkpoint_path = trainer_dir / "ae.pt"
    else:
        checkpoint_path = checkpoint
        trainer_dir = (
            checkpoint.parent.parent
            if checkpoint.parent.name == "checkpoints"
            else checkpoint.parent
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing SAE checkpoint: {checkpoint_path}")
    run_dir = trainer_dir.parent if trainer_dir.name.startswith("trainer_") else trainer_dir
    return checkpoint_path, trainer_dir, run_dir


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _validate_checkpoint_contract(
    config: dict[str, Any], *, layer: int, dim: int
) -> dict[str, Any]:
    trainer = config.get("trainer")
    if not isinstance(trainer, dict):
        raise ValueError("Checkpoint config is missing the trainer object")
    expected = {
        "dict_class": "BatchTopKSAE",
        "activation_dim": dim,
        "layer": layer,
        "lm_name": EXPECTED_LM_NAME,
        "submodule_name": EXPECTED_SUBMODULE,
    }
    mismatches = {
        key: {"actual": trainer.get(key), "expected": value}
        for key, value in expected.items()
        if trainer.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint contract mismatch: {mismatches}")

    dict_size = int(trainer.get("dict_size", 0))
    k = int(trainer.get("k", 0))
    if dict_size <= 0 or not 1 <= k <= dict_size:
        raise ValueError(f"Invalid checkpoint dict_size/k: dict_size={dict_size}, k={k}")
    return trainer


def _validate_source_identity(current: dict[str, Any], stored: dict[str, Any]) -> None:
    mismatches = {
        key: {"current": current.get(key), "stored": stored.get(key)}
        for key in SOURCE_IDENTITY_KEYS
        if current.get(key) != stored.get(key)
    }
    if mismatches:
        first_key = next(iter(mismatches))
        raise ValueError(
            "Current PQ3 source does not match the checkpoint source manifest; "
            f"first mismatch {first_key}={mismatches[first_key]}"
        )


@torch.no_grad()
def _feature_statistics(
    sae,
    activations: torch.Tensor,
    *,
    batch_size: int,
    n_batches: int,
    device: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float | int | bool]]:
    loader = iter(InMemoryBatchLoader(activations, batch_size, device, seed))
    dict_size = int(sae.dict_size)
    firing_count = torch.zeros(dict_size, dtype=torch.int64, device=device)
    activation_sum = torch.zeros(dict_size, dtype=torch.float32, device=device)
    activation_max = torch.full(
        (dict_size,), -torch.inf, dtype=torch.float32, device=device
    )
    squared_error_sum = 0.0
    element_count = 0

    for _ in range(n_batches):
        batch = next(loader)
        features = sae.encode(batch)
        reconstruction = sae.decode(features)
        if not (
            torch.isfinite(batch).all()
            and torch.isfinite(features).all()
            and torch.isfinite(reconstruction).all()
        ):
            raise ValueError("Non-finite value encountered during SAE encode/decode audit")

        fired = features != 0
        firing_count += fired.sum(dim=0)
        activation_sum += features.float().sum(dim=0)
        activation_max = torch.maximum(activation_max, features.float().amax(dim=0))
        squared_error_sum += float((batch - reconstruction).float().square().sum().item())
        element_count += int(batch.numel())

    evaluated_rows = n_batches * batch_size
    dead = firing_count == 0
    activation_max[dead] = 0
    mean_when_fired = activation_sum / firing_count.clamp_min(1)
    frequency = firing_count.float() / evaluated_rows
    arrays = {
        "feature_id": np.arange(dict_size, dtype=np.int32),
        "firing_count": firing_count.cpu().numpy(),
        "firing_frequency": frequency.cpu().numpy(),
        "mean_activation_when_fired": mean_when_fired.cpu().numpy(),
        "max_activation": activation_max.cpu().numpy(),
    }
    summary: dict[str, float | int | bool] = {
        "reconstruction_mse": squared_error_sum / element_count,
        "dead_feature_count": int(dead.sum().item()),
        "dead_feature_fraction": float(dead.float().mean().item()),
        "input_finite": True,
        "encoded_finite": True,
        "reconstruction_finite": True,
    }
    return arrays, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path)
    source.add_argument("--activation-cache", type=Path)
    parser.add_argument("--sae-checkpoint", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--feature-output", type=Path, default=None)
    parser.add_argument("--trust-pkl", action="store_true")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--activation-dim", type=int, default=1536)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--model-tokens", type=int, default=DEFAULT_MODEL_TOKENS)
    parser.add_argument("--state-tokens", type=int, default=DEFAULT_STATE_TOKENS)
    parser.add_argument("--future-tokens", type=int, default=DEFAULT_FUTURE_TOKENS)
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument("--expected-feature-kind", default=DEFAULT_FEATURE_KIND)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--max-audit-rows",
        type=int,
        default=8192,
        help="Deterministic sampled rows; 0 uses all complete batches.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--allow-partial-inventory", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-ram-gib", type=float, default=32.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.input_dir is not None:
        if not args.trust_pkl:
            raise SystemExit("Refusing to load pickle files without --trust-pkl.")
        if not args.input_dir.is_dir():
            raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    elif args.activation_cache is None or not args.activation_cache.is_file():
        raise SystemExit(f"Activation cache does not exist: {args.activation_cache}")
    if args.batch_size <= 0 or args.max_audit_rows < 0:
        raise SystemExit("--batch-size must be positive and --max-audit-rows must be >= 0")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path, trainer_dir, run_dir = _checkpoint_and_run_dirs(
        args.sae_checkpoint
    )
    sae, checkpoint_config = load_batch_topk_sae(checkpoint_path, device=device)
    trainer_config = _validate_checkpoint_contract(
        checkpoint_config, layer=args.layer, dim=args.activation_dim
    )

    if args.activation_cache is not None:
        activations, current_source = load_activation_cache(
            args.activation_cache,
            layer_id=args.layer,
            activation_dim=args.activation_dim,
        )
    else:
        activations, current_source = load_layer_activations(
            args.input_dir,
            layer_id=args.layer,
            activation_dim=args.activation_dim,
            expected_denoise_steps=args.denoise_steps,
            expected_feature_kind=args.expected_feature_kind,
            max_files=args.max_files,
            progress_every=args.progress_every,
            expected_model_tokens=args.model_tokens,
            state_tokens=args.state_tokens,
            future_tokens=args.future_tokens,
            action_horizon=args.action_horizon,
            max_ram_gib=args.max_ram_gib,
            materialize=True,
            expected_cell_counts=(
                None
                if args.allow_partial_inventory or args.max_files > 0
                else EXPECTED_PQ3_CELL_COUNTS
            ),
        )
        assert activations is not None

    source_manifest_path = (
        args.source_manifest.resolve()
        if args.source_manifest is not None
        else run_dir / "groot_source_manifest.json"
    )
    stored_source = _load_json_object(source_manifest_path)
    _validate_source_identity(current_source, stored_source)

    requested_rows = len(activations) if args.max_audit_rows == 0 else args.max_audit_rows
    sampled_rows = min(len(activations), requested_rows)
    n_batches = sampled_rows // args.batch_size
    if n_batches < 1:
        raise ValueError(
            f"Need at least one full audit batch: rows={len(activations)}, "
            f"requested={requested_rows}, batch_size={args.batch_size}"
        )
    evaluated_rows = n_batches * args.batch_size

    metrics = evaluate_dictionary(
        sae,
        iter(InMemoryBatchLoader(activations, args.batch_size, device, args.seed)),
        normalize_batch=False,
        device=device,
        n_batches=n_batches,
    )
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError(f"Non-finite metric returned by dictionary_learning.evaluate: {metrics}")

    feature_arrays, custom_summary = _feature_statistics(
        sae,
        activations,
        batch_size=args.batch_size,
        n_batches=n_batches,
        device=device,
        seed=args.seed,
    )
    measured_frac_alive = 1.0 - float(custom_summary["dead_feature_fraction"])
    if not math.isclose(
        float(metrics["frac_alive"]), measured_frac_alive, rel_tol=0.0, abs_tol=1e-6
    ):
        raise AssertionError(
            "dictionary_learning.evaluate frac_alive disagrees with firing audit: "
            f"{metrics['frac_alive']} != {measured_frac_alive}"
        )

    output_path = (
        args.output.resolve() if args.output is not None else run_dir / "sae_quality.json"
    )
    feature_output_path = (
        args.feature_output.resolve()
        if args.feature_output is not None
        else output_path.with_name("sae_quality_by_feature.npz")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_output_path, **feature_arrays)

    quality = {
        "schema_version": "groot_n15_pq3_sae_quality_v1",
        "evaluation_backend": "dictionary_learning.evaluation.evaluate",
        "checkpoint": {
            "path": str(checkpoint_path),
            "trainer_dir": str(trainer_dir),
            "layer": int(trainer_config["layer"]),
            "activation_dim": int(trainer_config["activation_dim"]),
            "dict_size": int(trainer_config["dict_size"]),
            "k": int(trainer_config["k"]),
            "threshold": float(sae.threshold.item()),
            "submodule_name": trainer_config["submodule_name"],
        },
        "source": {
            "manifest_path": str(source_manifest_path),
            "source_root": current_source["source_root"],
            "num_files": current_source["num_files"],
            "num_records": current_source["num_records"],
            "num_activation_rows": current_source["num_activation_rows"],
            "physical_layer": current_source["physical_layer"],
            "action_token_slice": current_source["action_token_slice"],
        },
        "sample": {
            "seed": args.seed,
            "selection": "first_complete_batches_of_seeded_torch_randperm",
            "available_rows": len(activations),
            "requested_rows": requested_rows,
            "evaluated_rows": evaluated_rows,
            "batch_size": args.batch_size,
            "n_batches": n_batches,
        },
        "normalization": {
            "normalize_batch": False,
            "reason": "trainSAE rescales saved BatchTopKSAE biases for raw activations",
        },
        "metrics": {key: float(value) for key, value in metrics.items()} | custom_summary,
        "feature_stats_path": str(feature_output_path),
    }
    output_path.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[audit] rows={evaluated_rows}/{len(activations)} "
        f"fve={metrics['frac_variance_explained']:.6f} "
        f"cossim={metrics['cossim']:.6f} l0={metrics['l0']:.3f} "
        f"dead={custom_summary['dead_feature_count']}/{sae.dict_size} "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
