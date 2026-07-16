"""Compact no-cache SAE training for all captured GR00T RoboCasa DiT layers.

Only the GR00T PKL adapter is local. SAE configuration and training reuse
Event-SAE's existing BatchTopK code, with one SAE per physical layer.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from dictionary_learning.training import trainSAE


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from event_sae.train import SAETrainConfig, build_batch_topk_trainer_config


DEFAULT_CAPTURE_LAYERS = (0, 2, 4, 8, 10, 12, 15)
EXPECTED_FEATURE_AXES = ("layer", "denoise_step", "feature_dim")


class InMemoryBatchLoader:
    """Infinite shuffled batches over one layer's CPU activations."""

    def __init__(self, data: torch.Tensor, batch_size: int, device: str, seed: int):
        if data.ndim != 2 or len(data) < batch_size:
            raise ValueError(f"Need [N,D] with N >= batch_size, got {tuple(data.shape)}")
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

    def __iter__(self) -> Iterator[torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed)
        full_rows = len(self.data) // self.batch_size * self.batch_size
        while True:
            order = torch.randperm(len(self.data), generator=generator)[:full_rows]
            for indices in order.split(self.batch_size):
                yield self.data.index_select(0, indices).to(
                    self.device, dtype=torch.float32, non_blocking=True
                )


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def load_layer_activations(
    root: Path,
    *,
    layer_id: int,
    activation_dim: int,
    expected_denoise_steps: int,
    expected_feature_kind: str,
    max_files: int,
    progress_every: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Read trusted PKLs and expand one layer's K denoise states into rows."""
    paths = sorted(root.rglob("*.pkl"))
    paths = paths[:max_files] if max_files > 0 else paths
    if not paths:
        raise FileNotFoundError(f"No PKLs under {root}")

    chunks: list[torch.Tensor] = []
    reference: tuple[Any, ...] | None = None
    num_records = 0
    for file_index, path in enumerate(paths, 1):
        with path.open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 -- gated by --trust-pkl.
        kind = str(payload.get("feature_kind") or "")
        axes = tuple(payload.get("feature_axes") or ())
        layers = tuple(int(x) for x in (payload.get("capture_layers") or ()))
        hidden_states = payload.get("hidden_states") or ()
        if not kind or axes[:3] != EXPECTED_FEATURE_AXES or not layers or not hidden_states:
            raise ValueError(f"{path}: invalid GR00T activation metadata")
        if expected_feature_kind and kind != expected_feature_kind:
            raise ValueError(f"{path}: feature_kind={kind!r}, expected {expected_feature_kind!r}")
        if layer_id not in layers:
            raise ValueError(f"{path}: layer {layer_id} not in {layers}")

        contract = (kind, axes, layers)
        if reference is None:
            reference = contract
        elif contract != reference:
            raise ValueError(f"{path}: activation contract differs from the first PKL")

        layer_position = layers.index(layer_id)
        for record_index, hidden in enumerate(hidden_states):
            array = _as_numpy(hidden)
            expected_shape = (len(layers), expected_denoise_steps, activation_dim)
            if array.shape != expected_shape:
                raise ValueError(
                    f"{path}: hidden_states[{record_index}]={array.shape}, expected {expected_shape}"
                )
            selected = np.ascontiguousarray(array[layer_position], dtype=np.float32)
            if not np.isfinite(selected).all():
                raise ValueError(f"{path}: non-finite layer {layer_id} activation")
            chunks.append(torch.from_numpy(selected))  # [K,D] -> K independent rows
        num_records += len(hidden_states)
        if progress_every > 0 and (file_index % progress_every == 0 or file_index == len(paths)):
            print(f"[load] layer={layer_id} files={file_index}/{len(paths)}", flush=True)

    activations = torch.cat(chunks).contiguous()
    assert reference is not None
    kind, axes, layers = reference
    return activations, {
        "format": "groot_n15_robocasa_in_memory_all_denoise_v1",
        "source_root": str(root.resolve()),
        "num_files": len(paths),
        "num_records": num_records,
        "num_activation_rows": len(activations),
        "feature_kind": kind,
        "feature_axes": list(axes),
        "capture_layers": list(layers),
        "physical_layer": layer_id,
        "source_denoising_steps": expected_denoise_steps,
        "denoise_mode": "all_as_independent_rows",
        "activation_dim": activation_dim,
        "dtype": str(activations.dtype),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--trust-pkl", action="store_true")
    parser.add_argument("--layer", type=int, default=None, help="Omit to train all layers")
    parser.add_argument("--activation-dim", type=int, default=1536)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--expected-feature-kind", default="")
    parser.add_argument("--dict-size", type=int, default=0, help="0 means activation_dim")
    parser.add_argument("--sae-k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--run-tag", default="groot_n15_robocasa_dit_allk_smoke")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1500)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--audit-only", action="store_true")
    return parser


def _train_layer(args, layer: int, save_dir: Path) -> None:
    activations, audit = load_layer_activations(
        args.input_dir,
        layer_id=layer,
        activation_dim=args.activation_dim,
        expected_denoise_steps=args.denoise_steps,
        expected_feature_kind=args.expected_feature_kind,
        max_files=args.max_files,
        progress_every=args.progress_every,
    )
    dict_size = args.dict_size or args.activation_dim
    cfg = SAETrainConfig(
        data_dir=str(args.input_dir),
        layer_idx=layer,
        activation_dim=args.activation_dim,
        dict_size=dict_size,
        k=args.sae_k,
        lr=args.lr,
        steps=args.steps,
        batch_size=args.batch_size,
        run_tag=args.run_tag,
        submodule_name="dit_block_residual",
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        num_workers=0,
        pin_memory=False,
    )
    trainer_cfg = build_batch_topk_trainer_config(cfg)
    trainer_cfg.update(
        warmup_steps=min(args.warmup_steps, args.steps),
        seed=args.seed,
        lm_name="groot_n15",
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "groot_source_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[layer {layer}] rows={len(activations)} shape={tuple(activations.shape)} "
        f"denoise=all/{args.denoise_steps} save_dir={save_dir}",
        flush=True,
    )
    if args.audit_only:
        return

    loader = InMemoryBatchLoader(
        activations,
        batch_size=args.batch_size,
        device=cfg.resolved_device(),
        seed=args.seed,
    )
    save_steps = (
        list(range(args.save_every, args.steps + 1, args.save_every))
        if args.save_every > 0
        else []
    )
    if args.wandb_project and args.wandb_group:
        os.environ["WANDB_RUN_GROUP"] = args.wandb_group
    trainSAE(
        data=loader,
        trainer_configs=[trainer_cfg],
        steps=args.steps,
        save_dir=str(save_dir),
        save_steps=save_steps,
        log_steps=args.log_steps,
        normalize_activations=True,
        use_wandb=bool(args.wandb_project),
        wandb_project=args.wandb_project,
        verbose=True,
        device=cfg.resolved_device(),
        autocast_dtype=torch.float32,
    )
    del loader, activations
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = _build_parser().parse_args()
    if not args.trust_pkl:
        raise SystemExit("Refusing to load pickle files without --trust-pkl.")
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    dict_size = args.dict_size or args.activation_dim
    if not 1 <= args.sae_k <= dict_size:
        raise SystemExit(f"--sae-k must be in [1, {dict_size}], got {args.sae_k}")

    layers = DEFAULT_CAPTURE_LAYERS if args.layer is None else (args.layer,)
    for layer in layers:
        save_dir = args.save_dir / f"layer_{layer:02d}" if args.layer is None else args.save_dir
        _train_layer(args, layer, save_dir)


if __name__ == "__main__":
    main()
