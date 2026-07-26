"""openVLA activation collection — register PyTorch forward hooks on decoder
layers and cache layer-wise post-MLP residual activations to disk as shards.

Two modes (paper has data from both):

- `apply_collect_hooks` (offline-friendly, default): dense residual shards
  + `activation_index.jsonl` metadata. Top-k SAE encoding is a separate
  offline pass via `scripts/extract_topk.py`.
- `apply_sae_topk_collect_hooks` (online): load a trained `BatchTopKSAE`
  and apply it inside the hook, writing sparse `token_topk_sparse_v1`
  shards directly. Matches the older paper run.

Per-step metadata is read from `model._sae_hook_context`, which
`event_sae.openvla.eval.runner.eval_libero` sets before every `get_action`
call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch

from event_sae.sae import load_batch_topk_sae


class ActivationCollectHandle:
    """Handle returned by collection helpers. Call `.remove()` at end of
    rollout to flush the last partial shard and unregister all hooks.
    """

    def __init__(self, hooks: List[object], flush_fn):
        self._hooks = hooks
        self._flush_fn = flush_fn

    def remove(self) -> None:
        self._flush_fn()
        for h in self._hooks:
            h.remove()


# ---------------------------------------------------------------------------
# Offline-friendly dense + metadata collection
# ---------------------------------------------------------------------------


def apply_collect_hooks(
    model,
    layer_idxs: List[int],
    output_dir: str | Path,
    flush_every: int = 10_000,
) -> ActivationCollectHandle:
    """Hook each `i` in `layer_idxs` on `model.language_model.model.layers[i]`
    and write dense shards + `activation_index.jsonl` into `output_dir`.

    Buffer flushes to a new shard once it reaches `flush_every` rows
    (forward boundaries are never split across shards).
    """
    decoder_layers = model.language_model.model.layers
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "activation_index.jsonl"
    index_file = index_path.open("w", encoding="utf-8")

    buffers = {idx: [] for idx in layer_idxs}
    pending_records = {idx: [] for idx in layer_idxs}
    buffered = {idx: 0 for idx in layer_idxs}
    shard_ids = {idx: 0 for idx in layer_idxs}
    global_forward_idx = {"value": 0}

    def flush_layer(idx: int) -> None:
        if not buffers[idx]:
            return
        acts = torch.cat(buffers[idx], dim=0)
        shard_name = f"layer_{idx:02d}_shard_{shard_ids[idx]:06d}.pt"
        torch.save(acts, output_dir / shard_name)
        for record in pending_records[idx]:
            record["shard_path"] = shard_name
            index_file.write(json.dumps(record) + "\n")
        index_file.flush()
        shard_ids[idx] += 1
        buffers[idx].clear()
        pending_records[idx].clear()
        buffered[idx] = 0

    def flush_all() -> None:
        for idx in layer_idxs:
            flush_layer(idx)
        index_file.close()

    def make_hook(idx: int):
        def hook_fn(module, inputs, output):
            hidden = output[0]
            sample = hidden.reshape(-1, hidden.shape[-1]).detach().to(torch.float32).cpu()
            n_rows = int(sample.shape[0])

            ctx = getattr(model, "_sae_hook_context", {}) or {}
            if idx == layer_idxs[0]:
                global_forward_idx["value"] += 1
            pending_records[idx].append(
                {
                    "layer_idx": int(idx),
                    "row_start": int(buffered[idx]),
                    "row_end": int(buffered[idx] + n_rows),
                    "episode_num": ctx.get("episode_num"),
                    "step_in_episode": ctx.get("step_in_episode"),
                    "task_id": ctx.get("task_id"),
                    "task_episode_idx": ctx.get("task_episode_idx"),
                    "global_forward_idx": int(global_forward_idx["value"]),
                }
            )
            buffers[idx].append(sample)
            buffered[idx] += n_rows
            if buffered[idx] >= flush_every:
                flush_layer(idx)

        return hook_fn

    hooks = [decoder_layers[idx].register_forward_hook(make_hook(idx)) for idx in layer_idxs]
    return ActivationCollectHandle(hooks=hooks, flush_fn=flush_all)


# ---------------------------------------------------------------------------
# Online SAE-applied top-k collection
# ---------------------------------------------------------------------------


def apply_sae_topk_collect_hooks(
    model,
    layer_idx: int,
    sae_checkpoint_path: str | Path,
    output_dir: str | Path,
    topk: int = 64,
    rows_per_shard: int = 20_000,
) -> ActivationCollectHandle:
    """Online mode — apply a trained `BatchTopKSAE` inside the hook and write
    sparse top-k shards (`token_topk_sparse_v1`) directly to `output_dir`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = str(model.language_model.lm_head.weight.device)
    sae, config = load_batch_topk_sae(Path(sae_checkpoint_path), device=device)
    trainer_cfg = config["trainer"]
    activation_dim = int(trainer_cfg["activation_dim"])
    dict_size = int(trainer_cfg["dict_size"])
    if not (1 <= topk <= dict_size):
        raise ValueError(f"topk must be in [1, {dict_size}], got {topk}")

    decoder_layers = model.language_model.model.layers
    if not (0 <= layer_idx < len(decoder_layers)):
        raise IndexError(f"Layer index out of range: {layer_idx}")

    buffer: dict[str, list[torch.Tensor]] = {
        "episode_num": [],
        "step_in_episode": [],
        "global_forward_idx": [],
        "batch_idx": [],
        "token_idx": [],
        "top_feature_ids": [],
        "top_feature_vals": [],
    }
    state = {"buffer_rows": 0, "shard_idx": 0, "total_rows": 0, "global_forward_idx": 0}
    manifest = {
        "format": "token_topk_sparse_v1",
        "layer": layer_idx,
        "sae_path": str(sae_checkpoint_path),
        "dict_size": dict_size,
        "activation_dim": activation_dim,
        "topk": topk,
        "rows_per_shard": rows_per_shard,
        "num_shards": 0,
        "total_rows": 0,
        "shards": [],
    }

    def _take(key: str, n: int) -> torch.Tensor:
        parts = []
        remaining = n
        while remaining > 0:
            head = buffer[key][0]
            take = min(remaining, int(head.shape[0]))
            parts.append(head[:take])
            if take == int(head.shape[0]):
                buffer[key].pop(0)
            else:
                buffer[key][0] = head[take:]
            remaining -= take
        return torch.cat(parts, dim=0)

    def _flush(n: int) -> None:
        payload = {key: _take(key, n) for key in buffer}
        shard_name = f"shard_{state['shard_idx']:06d}.pt"
        torch.save(payload, output_dir / shard_name)
        start = state["total_rows"]
        end = start + n
        manifest["shards"].append(
            {
                "shard_idx": state["shard_idx"],
                "path": shard_name,
                "num_rows": n,
                "row_start": start,
                "row_end": end,
            }
        )
        state["shard_idx"] += 1
        state["total_rows"] = end
        state["buffer_rows"] -= n
        manifest["num_shards"] = state["shard_idx"]
        manifest["total_rows"] = state["total_rows"]

    def hook_fn(module, inputs, output):
        ctx = getattr(model, "_sae_hook_context", {}) or {}
        episode_num = ctx.get("episode_num")
        step_in_episode = ctx.get("step_in_episode")
        if episode_num is None or step_in_episode is None:
            return output
        hidden = output[0]
        if hidden.ndim != 3 or hidden.shape[-1] != activation_dim:
            raise ValueError(
                f"Unexpected hidden shape {tuple(hidden.shape)}; expected (batch, seq, {activation_dim})"
            )
        flat = hidden.reshape(-1, hidden.shape[-1]).to(device=hidden.device, dtype=torch.float32)
        encoded = sae.encode(flat)
        values, indices = torch.topk(encoded, k=topk, dim=-1)
        state["global_forward_idx"] += 1
        batch_size, seq_len = int(hidden.shape[0]), int(hidden.shape[1])
        n_rows = batch_size * seq_len
        buffer["episode_num"].append(torch.full((n_rows,), int(episode_num), dtype=torch.int64))
        buffer["step_in_episode"].append(torch.full((n_rows,), int(step_in_episode), dtype=torch.int64))
        buffer["global_forward_idx"].append(
            torch.full((n_rows,), int(state["global_forward_idx"]), dtype=torch.int64)
        )
        buffer["batch_idx"].append(
            torch.arange(batch_size, dtype=torch.int64).unsqueeze(1).expand(batch_size, seq_len).reshape(-1)
        )
        buffer["token_idx"].append(
            torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).expand(batch_size, seq_len).reshape(-1)
        )
        buffer["top_feature_ids"].append(indices.reshape(n_rows, topk).to(dtype=torch.int32).cpu())
        buffer["top_feature_vals"].append(values.reshape(n_rows, topk).to(dtype=torch.float32).cpu())
        state["buffer_rows"] += n_rows
        while state["buffer_rows"] >= rows_per_shard:
            _flush(rows_per_shard)
        return output

    hook = decoder_layers[layer_idx].register_forward_hook(hook_fn)

    def flush_all() -> None:
        if state["buffer_rows"] > 0:
            _flush(state["buffer_rows"])
        with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return ActivationCollectHandle(hooks=[hook], flush_fn=flush_all)
