"""Online top-k extension to openpi's activation collector.

openpi-mech (and our fork) ship dense activation collection via
`openpi.sae_collection.ActivationCollector`. This module adds **online
top-k** mode: in the io_callback path, hidden states are encoded
through a trained PyTorch SAE and written as
`token_topk_sparse_v1` shards (the same format that `event_sae.scoring`
already consumes for the openvla half).

Wire-up happens in `scripts/openpi/serve_policy.py`: that script
constructs either the stock `ActivationCollector` (mode=dense) or this
`TopKActivationCollector` (mode=topk), registers it via
`openpi.sae_collection.runtime.set_active_collector`, then starts the
policy server. There is no event-sae-side wrapper API; the collector
constructor takes the same arguments as openpi's stock collector plus
the SAE checkpoint and top-k budget.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from openpi.sae_collection.collector import (
    FORWARD_TYPE_NAMES,
    ActivationCollector,
    SAECollectionConfig,
)

from event_sae.sae import load_batch_topk_sae


class TopKActivationCollector(ActivationCollector):
    """ActivationCollector subclass that replaces the dense data path with
    an online top-k path:

        hidden_state  --io_callback-->  PyTorch SAE encode  -->  top-k  -->  token_topk_sparse_v1 shard

    Bookkeeping (per-episode buffers, request context, control-channel
    handling, finalize/close hooks) is inherited from the parent. Only
    ``record_activation`` and the shard-writing path are replaced.

    Constrained to a single layer per call (multi-layer paper sweeps run
    one server per layer).
    """

    def __init__(
        self,
        config: SAECollectionConfig,
        *,
        policy_config_name: str,
        checkpoint_dir: str,
        model_depth: int,
        d_model: int,
        sae_checkpoint: str,
        topk: int = 64,
        rows_per_shard: int = 20_000,
        device: str | None = None,
    ) -> None:
        super().__init__(
            config,
            policy_config_name=policy_config_name,
            checkpoint_dir=checkpoint_dir,
            model_depth=model_depth,
            d_model=d_model,
        )
        if len(self.selected_layers) != 1:
            raise ValueError(
                f"Online top-k mode supports a single layer per server; got {list(self.selected_layers)}"
            )
        self._layer_idx = int(self.selected_layers[0])

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._sae, sae_config = load_batch_topk_sae(Path(sae_checkpoint), device=device)
        self._sae_device = device
        self._dict_size = int(sae_config["trainer"]["dict_size"])
        self._topk = int(topk)
        self._rows_per_shard = int(rows_per_shard)
        self._sae_checkpoint_path = str(sae_checkpoint)

        # token_topk_sparse_v1 column buffer. ``chunk_start_step`` and
        # ``executed_chunk_len`` are OpenPI-specific (sentinel -1 for OpenVLA
        # legacy) so the downstream score can compute
        # ``_effective_steps(step_mapping)`` without re-reading the dense
        # activation_index.
        self._buffer: dict[str, list[torch.Tensor]] = {
            "episode_num": [],
            "step_in_episode": [],
            "global_forward_idx": [],
            "forward_type": [],
            "forward_step_idx": [],
            "expert_idx": [],
            "batch_idx": [],
            "token_idx": [],
            "chunk_start_step": [],
            "executed_chunk_len": [],
            "top_feature_ids": [],
            "top_feature_vals": [],
        }
        self._buffer_rows = 0
        self._shard_idx = 0
        self._total_rows = 0
        self._executed_chunk_lens_seen: set[int] = set()
        # Per-episode counters so `finalize_episode` can report meaningful
        # row counts to the client. The parent (dense) collector returns
        # them from the staged dense buffers, which the online TopK path
        # bypasses — see `record_activation` below.
        self._episode_activation_records: dict[int, int] = {}
        self._episode_num_rows: dict[int, int] = {}
        self._manifest: dict = {
            "format": "token_topk_sparse_v1",
            "backbone": "openpi",
            "capture_target": self.capture_target,
            "layer": self._layer_idx,
            "dict_size": self._dict_size,
            "topk": self._topk,
            "sae_path": self._sae_checkpoint_path,
            "shards": [],
            "num_shards": 0,
            "total_rows": 0,
        }

    def record_activation(
        self,
        *,
        layer_idx: int,
        expert_idx: int,
        forward_type: int,
        forward_step_idx: int,
        hidden_states: np.ndarray,
    ) -> None:
        if self._request_state is None:
            return
        if expert_idx != self.expert_idx:
            return
        if layer_idx != self._layer_idx:
            return
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected hidden_states ndim=3, got shape={tuple(hidden_states.shape)}")

        forward_name = FORWARD_TYPE_NAMES.get(forward_type, f"forward_type_{forward_type}")
        forward_key = (forward_name, int(forward_step_idx))
        if forward_key not in self._request_state.forward_ids:
            self._global_forward_idx += 1
            self._request_state.forward_ids[forward_key] = self._global_forward_idx
        global_forward_idx = self._request_state.forward_ids[forward_key]

        batch_size, seq_len, _ = hidden_states.shape
        # Gemma's internal compute is bfloat16; `torch.from_numpy` does not
        # accept that dtype (jax exposes it via ml_dtypes), so cast in numpy
        # first. Float32 is what `_sae.encode` expects anyway.
        hidden_np = np.ascontiguousarray(np.asarray(hidden_states, dtype=np.float32))
        flat = (
            torch.from_numpy(hidden_np)
            .reshape(-1, hidden_states.shape[-1])
            .to(device=self._sae_device, dtype=torch.float32)
        )
        with torch.no_grad():
            encoded = self._sae.encode(flat)
            values, indices = torch.topk(encoded, k=self._topk, dim=-1)
        n_rows = int(flat.shape[0])

        context = self._request_state.context
        episode_num = int(context.get("episode_num", -1))
        # Per-token env-step mapping: OpenPI records each forward as one
        # chunked inference over `seq_len` future tokens; the env step a
        # token corresponds to is `chunk_start + token_idx`. Falls back to
        # broadcasting the request-context `step_in_episode` for non-chunked
        # backends. Matches the offline extract_topk.py fix + openpi-mech's
        # `step_mapping="action_executed"` semantics.
        chunk_start = context.get("chunk_start_step")
        if chunk_start is None:
            chunk_start = context.get("action_chunk_start_step")
        if chunk_start is None:
            chunk_start = int(context.get("step_in_episode", -1))
        chunk_start = int(chunk_start)
        executed_chunk_len = int(context.get("executed_chunk_len", -1))
        if executed_chunk_len >= 0:
            self._executed_chunk_lens_seen.add(executed_chunk_len)
        token_local = torch.arange(seq_len, dtype=torch.int64)
        step_per_token = (chunk_start + token_local).unsqueeze(0).expand(batch_size, seq_len).reshape(-1)

        buf = self._buffer
        buf["episode_num"].append(torch.full((n_rows,), episode_num, dtype=torch.int64))
        buf["step_in_episode"].append(step_per_token)
        buf["global_forward_idx"].append(torch.full((n_rows,), int(global_forward_idx), dtype=torch.int64))
        buf["forward_type"].append(torch.full((n_rows,), int(forward_type), dtype=torch.int64))
        buf["forward_step_idx"].append(torch.full((n_rows,), int(forward_step_idx), dtype=torch.int64))
        buf["expert_idx"].append(torch.full((n_rows,), int(expert_idx), dtype=torch.int64))
        buf["batch_idx"].append(
            torch.arange(batch_size, dtype=torch.int64).unsqueeze(1).expand(batch_size, seq_len).reshape(-1)
        )
        buf["token_idx"].append(
            token_local.unsqueeze(0).expand(batch_size, seq_len).reshape(-1)
        )
        buf["chunk_start_step"].append(torch.full((n_rows,), chunk_start, dtype=torch.int64))
        buf["executed_chunk_len"].append(torch.full((n_rows,), executed_chunk_len, dtype=torch.int64))
        buf["top_feature_ids"].append(indices.to(dtype=torch.int32).cpu())
        buf["top_feature_vals"].append(values.to(dtype=torch.float32).cpu())
        self._buffer_rows += n_rows

        # Lifecycle bookkeeping for the override of `finalize_episode`:
        # the parent (dense) collector tallies row counts from its own
        # staged buffers, which the TopK path bypasses entirely.
        if episode_num >= 0:
            self._episode_activation_records[episode_num] = (
                self._episode_activation_records.get(episode_num, 0) + 1
            )
            self._episode_num_rows[episode_num] = (
                self._episode_num_rows.get(episode_num, 0) + n_rows
            )

        while self._buffer_rows >= self._rows_per_shard:
            self._flush_topk_shard(self._rows_per_shard)

    def _flush_topk_shard(self, n_rows: int) -> None:
        cat = {k: torch.cat(v, dim=0) for k, v in self._buffer.items()}
        payload = {k: cat[k][:n_rows] for k in cat}
        for k in self._buffer:
            remainder = cat[k][n_rows:]
            self._buffer[k] = [remainder] if remainder.shape[0] > 0 else []

        shard_name = f"shard_{self._shard_idx:06d}.pt"
        torch.save(payload, self._activations_dir / shard_name)

        self._manifest["shards"].append(
            {
                "shard_idx": self._shard_idx,
                "path": shard_name,
                "num_rows": n_rows,
                "row_start": self._total_rows,
                "row_end": self._total_rows + n_rows,
            }
        )
        self._manifest["num_shards"] = self._shard_idx + 1
        self._manifest["total_rows"] += n_rows
        self._total_rows += n_rows
        self._buffer_rows -= n_rows
        self._shard_idx += 1

    def server_metadata(self) -> dict:
        """Same as the parent dense collector's metadata, plus an explicit
        ``mode = "topk"`` flag so clients can tell the two collection
        modes apart (e.g. to warn about the online-TopK keep=False
        no-retroactive-drop caveat)."""
        meta = super().server_metadata()
        meta["mode"] = "topk"
        meta["topk"] = self._topk
        return meta

    def finalize_episode(self, episode_num: int, keep: bool) -> dict:
        """Override the parent (dense) finalize so the client sees real
        TopK row counts. The parent looks at staged dense buffers, which
        the TopK path bypasses — so without this override the client
        sees ``activation_records=0 num_rows=0`` per episode even though
        shards have data.

        Limitation: online TopK **does not actually drop on keep=False**.
        Rows enter the global buffer at ``record_activation`` time and
        may have already been flushed to a shard before ``finalize_episode``
        arrives, so we cannot retroactively remove them. We log a warning
        and return ``kept=False`` for protocol completeness, but the data
        is still on disk. The paper-faithful default
        (``keep_failed_episodes=True``) sidesteps this; users who need
        success-only TopK shards should filter at the score / ranking
        step using ``success.csv``.
        """
        records = int(self._episode_activation_records.pop(int(episode_num), 0))
        n_rows = int(self._episode_num_rows.pop(int(episode_num), 0))
        if not keep:
            self._write_log(
                f"WARNING: Online TopK does not support drop-on-keep=False. "
                f"Rows for episode={episode_num} (records={records} num_rows={n_rows}) "
                f"already entered the global buffer / shard and cannot be retracted. "
                f"Reporting kept=False but data is on disk; filter downstream by success.csv."
            )
        else:
            self._write_log(
                f"Online TopK: kept episode={episode_num} activation_records={records} num_rows={n_rows}"
            )
        return {
            "status": "ok",
            "episode_num": int(episode_num),
            "kept": bool(keep),
            "activation_records": records,
            "num_rows": n_rows,
        }

    def close(self) -> None:
        if self._buffer_rows > 0:
            self._flush_topk_shard(self._buffer_rows)
        self._manifest["executed_chunk_lens_seen"] = sorted(self._executed_chunk_lens_seen)
        self._manifest["finalize_status_source"] = "TopKActivationCollector"
        # Write manifest to the run dir root so the layout matches the offline
        # `extract_topk.py` output (``<output-dir>/manifest.json`` + shards).
        # The shards themselves still live under the capture-target subdir.
        run_root_manifest = self.run_dir / "manifest.json"
        with run_root_manifest.open("w", encoding="utf-8") as f:
            # Rebase shard paths to be relative to run_dir so downstream
            # readers can resolve them whether they pass the run root or the
            # subdir as topk-run-dir.
            rebased = dict(self._manifest)
            rebased["shards"] = [
                {
                    **shard,
                    "path": str(
                        (self._activations_dir / shard["path"]).relative_to(self.run_dir)
                    ),
                }
                for shard in self._manifest["shards"]
            ]
            json.dump(rebased, f, indent=2)
        # Keep the legacy subdir copy too (with subdir-relative paths) so older
        # callers that point at ``<run_root>/sae_activations/...`` still work.
        subdir_manifest_path = self._activations_dir / "manifest.json"
        with subdir_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)
        super().close()
        # Patch the parent's run_metadata.json: in online TopK mode the
        # dense `layer_state` counters are always zero / empty (we never
        # write to the parent's dense buffers). Mark the file as
        # authoritative-from-manifest for downstream readers.
        run_metadata_path = self.run_dir / "run_metadata.json"
        if run_metadata_path.is_file():
            try:
                meta = json.loads(run_metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
            meta["online_topk"] = {
                "is_online_topk_run": True,
                "authoritative_metadata_file": "manifest.json",
                "note": (
                    "Online TopKActivationCollector bypasses the parent dense buffer, "
                    "so layer_state.total_rows_written / sample_hidden_shapes in this "
                    "run_metadata.json are NOT updated. For online TopK runs, read the "
                    "top-level manifest.json (token_topk_sparse_v1) for shard / row "
                    "counts."
                ),
                "topk_total_rows": int(self._total_rows),
                "topk_num_shards": int(self._shard_idx),
            }
            run_metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
