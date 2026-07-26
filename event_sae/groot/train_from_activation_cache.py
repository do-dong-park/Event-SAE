"""Train a BatchTopK SAE from an audited GR00T RoboCasa activation cache.

Raw PQ3 pickle validation and action-token extraction belong to
``scripts/groot/export_pq3_activation_shards.py``.  This module is the thin
GR00T adapter around the paper implementation in :mod:`event_sae.train`:
it loads one portable activation cache, records its source/training contracts,
and delegates the actual optimization to ``dictionary_learning.trainSAE``.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from event_sae.groot.activations import (
    InMemoryActivationBatchLoader,
    load_activation_cache,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-cache", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--activation-dim", type=int, default=1536)
    parser.add_argument("--dict-size", type=int, default=0, help="0 means activation_dim")
    parser.add_argument("--sae-k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--decay-start-steps",
        type=int,
        default=-1,
        help="LR decay start step; -1 uses floor(0.8 * --steps).",
    )
    parser.add_argument(
        "--threshold-start-steps",
        type=int,
        default=1000,
        help="Step at which BatchTopK begins updating its inference threshold.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--run-tag", default="groot_n15_pq3_dit_allk_action16_smoke")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1500)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.activation_cache.is_file():
        raise SystemExit(f"Activation cache does not exist: {args.activation_cache}")
    if args.activation_dim <= 0:
        raise SystemExit(
            f"--activation-dim must be > 0, got {args.activation_dim}"
        )
    if args.steps <= 0:
        raise SystemExit(f"--steps must be > 0, got {args.steps}")
    if args.batch_size <= 0:
        raise SystemExit(f"--batch-size must be > 0, got {args.batch_size}")
    if not 0 <= args.warmup_steps <= args.steps:
        raise SystemExit(
            f"--warmup-steps must be in [0, {args.steps}], got {args.warmup_steps}"
        )
    if args.decay_start_steps < -1 or args.decay_start_steps > args.steps:
        raise SystemExit(
            "--decay-start-steps must be -1 or in "
            f"[0, {args.steps}], got {args.decay_start_steps}"
        )
    if args.threshold_start_steps < 0:
        raise SystemExit(
            "--threshold-start-steps must be >= 0, "
            f"got {args.threshold_start_steps}"
        )

    dict_size = args.dict_size or args.activation_dim
    if not 1 <= args.sae_k <= dict_size:
        raise SystemExit(f"--sae-k must be in [1, {dict_size}], got {args.sae_k}")


def train_sae_from_activation_cache(args: argparse.Namespace) -> None:
    """Load one validated cache and delegate BatchTopK optimization."""

    activations, source_manifest = load_activation_cache(
        args.activation_cache,
        layer_id=args.layer,
        activation_dim=args.activation_dim,
    )

    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "groot_source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[layer {args.layer}] rows={source_manifest['num_activation_rows']} "
        f"tokens={source_manifest['token_scope']}/{source_manifest['action_horizon']} "
        f"denoise=all/{source_manifest['source_denoising_steps']} "
        f"source_dtype={source_manifest['source_dtype']} save_dir={save_dir}",
        flush=True,
    )

    from event_sae.train import SAETrainConfig, train_sae

    dict_size = args.dict_size or args.activation_dim
    cfg = SAETrainConfig(
        data_dir=str(args.activation_cache.resolve()),
        layer_idx=args.layer,
        activation_dim=args.activation_dim,
        dict_size=dict_size,
        k=args.sae_k,
        lr=args.lr,
        steps=args.steps,
        batch_size=args.batch_size,
        run_tag=args.run_tag,
        submodule_name="dit_block_residual_action_tokens",
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        num_workers=0,
        pin_memory=False,
        warmup_steps=args.warmup_steps,
        decay_start_step=(
            None if args.decay_start_steps < 0 else args.decay_start_steps
        ),
        threshold_start_step=args.threshold_start_steps,
        seed=args.seed,
        lm_name="groot_n15",
        save_every=args.save_every,
        log_steps=args.log_steps,
        use_wandb_env=False,
    )
    source_rows = int(source_manifest["num_activation_rows"])
    training_contract = {
        "format": "event_sae_training_contract_v1",
        "physical_layer": args.layer,
        "activation_dim": args.activation_dim,
        "dict_size": dict_size,
        "sae_k": args.sae_k,
        "learning_rate": args.lr,
        "source_activation_rows": source_rows,
        "batch_size": cfg.batch_size,
        "steps": cfg.steps,
        "row_presentations": cfg.batch_size * cfg.steps,
        "effective_dataset_passes": cfg.batch_size * cfg.steps / source_rows,
        "warmup_steps": cfg.warmup_steps,
        "decay_start_step": cfg.resolved_decay_start(),
        "threshold_start_step": cfg.threshold_start_step,
        "seed": cfg.seed,
        "normalize_activations": cfg.normalize_activations,
    }
    (save_dir / "training_contract.json").write_text(
        json.dumps(training_contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    loader = InMemoryActivationBatchLoader(
        activations,
        batch_size=cfg.batch_size,
        device=cfg.resolved_device(),
        seed=cfg.seed,
    )
    train_sae(cfg, save_dir=str(save_dir), data=loader)
    del loader, activations
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = _build_parser().parse_args()
    _validate_args(args)
    train_sae_from_activation_cache(args)


if __name__ == "__main__":
    main()
