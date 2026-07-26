"""SAE training entry point for offline activation shards.

Migrated from `mechanistic-steering-vlas/src/sae_train/train.py`. Differences:
- `device` is now a config field (auto-detected by default), instead of
  hardcoded `"cuda:0"`.
- Wandb logging is opt-in: set `wandb_project` to enable; empty string disables.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE, BatchTopKTrainer
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def _auto_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class SAETrainConfig:
    """Single-run config for training one BatchTopK SAE."""

    data_dir: str
    layer_idx: int             # Should match the layer number in shard filenames.
    activation_dim: int        # Activation width d_model (e.g., OpenVLA post-residual = 4096).
    dict_size: int             # Number of SAE features (dictionary width).
    k: int
    lr: float
    steps: int
    batch_size: int
    run_tag: str = "offline"
    submodule_name: str = "post_mlp_residual"
    device: str = ""           # Empty = auto (cuda if available else cpu).
    wandb_project: str = ""    # Empty = wandb disabled.
    wandb_group: str = ""
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    warmup_steps: int = 1000
    decay_start_step: int | None = None  # None = floor(0.8 * steps).
    threshold_start_step: int = 1000
    seed: int = 0
    save_every: int = 1500              # <= 0 disables intermediate saves.
    log_steps: int = 500
    normalize_activations: bool = True
    verbose: bool = True
    lm_name: str = "openvla_offline"
    use_wandb_env: bool = True

    def resolved_device(self) -> str:
        return self.device or _auto_device()

    def resolved_decay_start(self) -> int:
        if self.decay_start_step is None:
            return int(self.steps * 0.8)
        return self.decay_start_step

    def save_steps(self) -> list[int]:
        if self.save_every <= 0:
            return []
        return list(range(self.save_every, self.steps + 1, self.save_every))


class _ActivationShardDataset(IterableDataset):
    def __init__(self, cfg: SAETrainConfig, shard_paths: list[Path]):
        super().__init__()
        self.cfg = cfg
        self.shard_paths = shard_paths

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            local_shards = list(self.shard_paths)
        else:
            local_shards = list(self.shard_paths[worker.id :: worker.num_workers])
        # Skip shards that cannot yield a full batch: they waste I/O and,
        # if a worker is assigned ONLY such shards, would cause the
        # DataLoader's round-robin to wait forever. File-size heuristic:
        # rows ≈ shard_size / (activation_dim * 4 bytes-per-float32).
        threshold_bytes = self.cfg.batch_size * self.cfg.activation_dim * 4
        local_shards = [s for s in local_shards if s.stat().st_size >= threshold_bytes]
        if not local_shards:
            return  # worker has nothing to yield; exit cleanly
        while True:
            if len(local_shards) > 1:
                order = torch.randperm(len(local_shards)).tolist()
                shard_iter = [local_shards[i] for i in order]
            else:
                shard_iter = local_shards
            for shard_path in shard_iter:
                activations = torch.load(shard_path, map_location="cpu")
                if activations.ndim != 2:
                    raise ValueError(
                        f"Expected 2D activations in {shard_path}, got shape {tuple(activations.shape)}"
                    )
                order = torch.randperm(activations.shape[0])
                activations = activations[order].to(torch.float32)
                for start in range(0, activations.shape[0], self.cfg.batch_size):
                    batch = activations[start : start + self.cfg.batch_size]
                    if batch.shape[0] < self.cfg.batch_size:
                        continue
                    yield batch


class ActivationShardDataLoader:
    """Iterable loader for layer-specific activation shards."""

    def __init__(self, cfg: SAETrainConfig):
        self.cfg = cfg
        self.device = cfg.resolved_device()
        shard_paths = sorted(Path(cfg.data_dir).glob(f"layer_{cfg.layer_idx:02d}_shard_*.pt"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No activation shards found in {cfg.data_dir} for layer {cfg.layer_idx}."
            )
        self.shard_paths = shard_paths
        self._dataset = _ActivationShardDataset(cfg, shard_paths)
        # Clamp num_workers to the number of shards that can yield a full
        # batch. A worker assigned only "too small" shards yields nothing
        # and stalls the DataLoader's round-robin. File-size heuristic:
        # rows ≈ shard_size / (activation_dim * 4 bytes-per-float32).
        threshold_bytes = cfg.batch_size * cfg.activation_dim * 4
        viable_count = sum(1 for p in shard_paths if p.stat().st_size >= threshold_bytes)
        if viable_count == 0:
            raise RuntimeError(
                f"No shard in {cfg.data_dir} has >= {cfg.batch_size} rows "
                f"(activation_dim={cfg.activation_dim}). Reduce batch_size, or "
                f"recollect more activations."
            )
        num_workers = min(cfg.num_workers, viable_count)
        dataloader_kwargs: dict[str, Any] = {
            "dataset": self._dataset,
            "batch_size": None,
            "num_workers": num_workers,
            "pin_memory": cfg.pin_memory and self.device.startswith("cuda"),
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        self._dataloader = DataLoader(**dataloader_kwargs)

        self.epoch = 0
        self.global_step = 0
        self.batches_in_epoch = 0
        self.samples_seen = 0
        self.current_shard = ""

    def __iter__(self):
        self.batches_in_epoch = 0
        for batch in self._dataloader:
            batch = batch.to(self.device, non_blocking=True)
            self.global_step += 1
            self.batches_in_epoch += 1
            self.samples_seen += int(batch.shape[0])
            yield batch


def build_batch_topk_trainer_config(cfg: SAETrainConfig) -> dict[str, Any]:
    """Build a single trainer config dict for `dictionary_learning.trainSAE`."""
    return {
        "trainer": BatchTopKTrainer,
        "dict_class": BatchTopKSAE,
        "activation_dim": cfg.activation_dim,
        "dict_size": cfg.dict_size,
        "k": cfg.k,
        "lr": cfg.lr,
        "steps": cfg.steps,
        "warmup_steps": cfg.warmup_steps,
        "decay_start": cfg.resolved_decay_start(),
        "threshold_start_step": cfg.threshold_start_step,
        "seed": cfg.seed,
        "device": cfg.resolved_device(),
        "layer": cfg.layer_idx,
        "lm_name": cfg.lm_name,
        "submodule_name": cfg.submodule_name,
        "wandb_name": f"{cfg.run_tag}-l{cfg.layer_idx:02d}",
    }


def train_sae(
    cfg: SAETrainConfig,
    save_dir: str,
    *,
    data: Iterable[torch.Tensor] | None = None,
) -> None:
    """Train one BatchTopK SAE using the shared dictionary-learning runtime.

    By default activations come from :class:`ActivationShardDataLoader`.
    Model-specific integrations can inject any iterable of ``[batch, dim]``
    tensors, keeping source validation and batching outside the common
    training loop.
    """

    dataloader = data if data is not None else ActivationShardDataLoader(cfg)
    trainer_cfg = build_batch_topk_trainer_config(cfg)
    wandb_project = cfg.wandb_project
    if not wandb_project and cfg.use_wandb_env:
        wandb_project = os.environ.get("WANDB_PROJECT", "")
    use_wandb = bool(wandb_project)
    if use_wandb and cfg.wandb_group:
        # Upstream `trainSAE` does not accept a `wandb_group` kwarg; wandb
        # picks up grouping from the WANDB_RUN_GROUP env var instead.
        os.environ["WANDB_RUN_GROUP"] = cfg.wandb_group
    trainSAE(
        data=dataloader,
        trainer_configs=[trainer_cfg],
        steps=cfg.steps,
        save_dir=save_dir,
        save_steps=cfg.save_steps(),
        log_steps=cfg.log_steps,
        normalize_activations=cfg.normalize_activations,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        verbose=cfg.verbose,
        device=cfg.resolved_device(),
        autocast_dtype=torch.float32,
    )
