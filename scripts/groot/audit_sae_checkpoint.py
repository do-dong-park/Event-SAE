"""Audit a GR00T PQ3 BatchTopK SAE checkpoint on action-token residuals.

This cache-only adapter reuses Event-SAE's activation/checkpoint loaders and
``dictionary_learning.evaluation.evaluate``. Raw PQ3 pickle validation belongs
to ``export_pq3_activation_shards.py``; this audit is limited to
source/checkpoint contract checks, deterministic sampling, per-feature firing
statistics, and durable JSON/NPZ output.
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

from event_sae.sae import load_batch_topk_sae
from event_sae.groot.activations import (
    InMemoryActivationBatchLoader,
    load_activation_cache,
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
    loader = iter(
        InMemoryActivationBatchLoader(
            activations,
            batch_size,
            device,
            seed,
        )
    )
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


@torch.no_grad()
def _calibrate_inference_threshold(
    sae,
    activations: torch.Tensor,
    *,
    calibration_rows: int,
    device: str,
    seed: int,
) -> tuple[float, int]:
    evaluated_rows = min(len(activations), calibration_rows)
    if evaluated_rows < 1:
        raise ValueError("Threshold calibration requires at least one activation row")
    batch = next(
        iter(
            InMemoryActivationBatchLoader(
                activations,
                evaluated_rows,
                device,
                seed,
            )
        )
    )
    pre_activations = torch.relu(sae.encoder(batch - sae.b_dec)).flatten()
    target_active = min(int(sae.k.item()) * evaluated_rows, pre_activations.numel())
    selected_minimum = torch.topk(
        pre_activations, target_active, sorted=False
    ).values.amin()
    threshold = torch.nextafter(
        selected_minimum, torch.full_like(selected_minimum, -torch.inf)
    )
    sae.threshold.fill_(threshold)
    return float(threshold.item()), evaluated_rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-cache", type=Path, required=True)
    parser.add_argument("--sae-checkpoint", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--feature-output", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--activation-dim", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--max-audit-rows",
        type=int,
        default=8192,
        help="Deterministic sampled rows; 0 uses all complete batches.",
    )
    parser.add_argument(
        "--recalibrate-threshold-rows",
        type=int,
        default=0,
        help=(
            "Deterministic rows used to reset the inference threshold to average "
            "L0=k; 0 audits the threshold stored in the checkpoint."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.activation_cache.is_file():
        raise SystemExit(f"Activation cache does not exist: {args.activation_cache}")
    if args.batch_size <= 0 or args.max_audit_rows < 0:
        raise SystemExit("--batch-size must be positive and --max-audit-rows must be >= 0")
    if args.recalibrate_threshold_rows < 0:
        raise SystemExit("--recalibrate-threshold-rows must be >= 0")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path, trainer_dir, run_dir = _checkpoint_and_run_dirs(
        args.sae_checkpoint
    )
    output_path = (
        args.output.resolve() if args.output is not None else run_dir / "sae_quality.json"
    )
    feature_output_path = (
        args.feature_output.resolve()
        if args.feature_output is not None
        else output_path.with_name("sae_quality_by_feature.npz")
    )
    if output_path == feature_output_path:
        raise ValueError("Audit JSON and feature NPZ outputs must be different paths")
    existing_outputs = [
        path
        for path in (output_path, feature_output_path)
        if path.exists() or path.is_symlink()
    ]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing audit output(s): "
            + ", ".join(str(path) for path in existing_outputs)
        )

    sae, checkpoint_config = load_batch_topk_sae(checkpoint_path, device=device)
    trainer_config = _validate_checkpoint_contract(
        checkpoint_config, layer=args.layer, dim=args.activation_dim
    )

    activations, current_source = load_activation_cache(
        args.activation_cache,
        layer_id=args.layer,
        activation_dim=args.activation_dim,
    )

    source_manifest_path = (
        args.source_manifest.resolve()
        if args.source_manifest is not None
        else run_dir / "groot_source_manifest.json"
    )
    stored_source = _load_json_object(source_manifest_path)
    _validate_source_identity(current_source, stored_source)

    saved_threshold = float(sae.threshold.item())
    threshold_calibration: dict[str, Any] = {
        "mode": "checkpoint_saved_threshold",
        "saved_threshold": saved_threshold,
        "effective_threshold": saved_threshold,
        "requested_rows": 0,
        "evaluated_rows": 0,
        "seed": args.seed,
        "target_average_l0": int(sae.k.item()),
    }
    if args.recalibrate_threshold_rows > 0:
        effective_threshold, calibration_rows = _calibrate_inference_threshold(
            sae,
            activations,
            calibration_rows=args.recalibrate_threshold_rows,
            device=device,
            seed=args.seed,
        )
        threshold_calibration.update(
            mode="deterministic_global_topk_calibration",
            effective_threshold=effective_threshold,
            requested_rows=args.recalibrate_threshold_rows,
            evaluated_rows=calibration_rows,
        )

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
        iter(
            InMemoryActivationBatchLoader(
                activations,
                args.batch_size,
                device,
                args.seed,
            )
        ),
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
            "saved_threshold": saved_threshold,
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
        "threshold_calibration": threshold_calibration,
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
