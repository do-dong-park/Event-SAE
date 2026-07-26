"""Shared GR00T activation cache and in-memory training adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import torch

from event_sae import resolve_groot_artifact_path
from event_sae import sha256_file as _sha256
from event_sae.sae import load_batch_topk_sae


REQUIRED_FEATURE_KIND = "groot_n15_dit_block_residual_full_tokens_denoise"
REQUIRED_CAPTURE_TOKEN_MODE = "all_token_full"
ACTIVATION_CACHE_FORMAT = "groot_n15_pq3_action_activation_cache_v1"
SPARSE_TOPK_FORMAT = "token_topk_sparse_v1"
SPARSE_ENCODING_AUDIT_FORMAT = "groot_n15_pq3_stage4_topk_audit_v1"
ACTIVATION_ROW_ORDER = [
    "source_file",
    "record",
    "denoise_step",
    "action_token_offset",
]


class InMemoryActivationBatchLoader:
    """Yield infinite shuffled batches from resident CPU activations."""

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
                # Keep the resident CPU copy in its source dtype (PQ3 is
                # fp16) and cast only the current batch for SAE training.
                yield self.data.index_select(0, indices).to(
                    self.device, dtype=torch.float32, non_blocking=True
                )


def save_activation_cache(
    path: Path,
    activations: torch.Tensor,
    source_manifest: dict[str, Any],
) -> None:
    """Persist audited action-token rows for transfer to a training host."""

    if activations.ndim != 2:
        raise ValueError(f"Activation cache requires [N,D], got {tuple(activations.shape)}")
    if int(source_manifest.get("num_activation_rows", -1)) != len(activations):
        raise ValueError("Activation rows do not match source manifest")
    if int(source_manifest.get("activation_dim", -1)) != activations.shape[1]:
        raise ValueError("Activation dimension does not match source manifest")
    if not torch.isfinite(activations).all():
        raise ValueError("Activation cache contains non-finite values")

    path = resolve_groot_artifact_path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": ACTIVATION_CACHE_FORMAT,
            "activations": activations.detach().cpu().contiguous(),
            "source_manifest": source_manifest,
        },
        path,
    )


def load_activation_cache(
    path: Path,
    *,
    layer_id: int,
    activation_dim: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load and validate a cache produced by :func:`save_activation_cache`."""

    path = resolve_groot_artifact_path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing activation cache: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != ACTIVATION_CACHE_FORMAT:
        raise ValueError(f"Unsupported activation cache format in {path}")

    activations = payload.get("activations")
    source_manifest = payload.get("source_manifest")
    if not torch.is_tensor(activations) or activations.ndim != 2:
        raise ValueError(f"{path}: expected activation tensor [N,D]")
    if not isinstance(source_manifest, dict):
        raise ValueError(f"{path}: missing source_manifest")
    if activations.shape[1] != activation_dim:
        raise ValueError(
            f"{path}: activation_dim={activations.shape[1]}, expected {activation_dim}"
        )

    expected_identity = {
        "physical_layer": layer_id,
        "activation_dim": activation_dim,
        "token_scope": "action",
        "capture_token_mode": REQUIRED_CAPTURE_TOKEN_MODE,
        "feature_kind": REQUIRED_FEATURE_KIND,
    }
    mismatches = {
        key: {"actual": source_manifest.get(key), "expected": value}
        for key, value in expected_identity.items()
        if source_manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{path}: activation cache identity mismatch: {mismatches}")
    if int(source_manifest.get("num_activation_rows", -1)) != len(activations):
        raise ValueError(f"{path}: activation row count does not match source manifest")
    if not torch.isfinite(activations).all():
        raise ValueError(f"{path}: non-finite activation cache")

    return activations.contiguous(), dict(source_manifest)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    """Serialize a JSON object for content identity, independent of formatting."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _find_checkpoint_source_manifest(checkpoint_path: Path) -> Path:
    """Find the nearest run-level source manifest above a checkpoint."""

    checkpoint_path = resolve_groot_artifact_path(checkpoint_path).resolve()
    start = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    searched: list[Path] = []
    for directory in (start, *start.parents):
        candidate = directory / "groot_source_manifest.json"
        searched.append(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find groot_source_manifest.json above checkpoint "
        f"{checkpoint_path}; searched from {searched[0]} to {searched[-1]}"
    )


def _validate_checkpoint_source_manifest(
    *,
    checkpoint_path: Path,
    activation_source_manifest: dict[str, Any],
) -> tuple[Path, str]:
    """Require canonical source-manifest equality for checkpoint and cache."""

    manifest_path = _find_checkpoint_source_manifest(checkpoint_path)
    checkpoint_source_manifest = _load_json(manifest_path)
    activation_content = _canonical_json_bytes(activation_source_manifest)
    checkpoint_content = _canonical_json_bytes(checkpoint_source_manifest)
    if activation_content != checkpoint_content:
        differing_keys = sorted(
            key
            for key in (
                set(activation_source_manifest) | set(checkpoint_source_manifest)
            )
            if activation_source_manifest.get(key)
            != checkpoint_source_manifest.get(key)
        )
        raise ValueError(
            "Activation cache source_manifest does not exactly match the "
            "checkpoint run source manifest"
            + (
                f"; differing top-level keys={differing_keys[:10]}"
                if differing_keys
                else ""
            )
        )
    return manifest_path, hashlib.sha256(activation_content).hexdigest()


def reconstruct_action_token_row_metadata(
    *,
    num_records: int,
    episode_num: int,
    global_record_start: int,
    denoise_steps: int,
    action_horizon: int,
    executed_action_steps: int,
) -> dict[str, torch.Tensor]:
    """Restore one source's row coordinates without reading raw PKLs."""

    if num_records <= 0:
        raise ValueError("num_records must be positive")
    if denoise_steps <= 0 or action_horizon <= 0:
        raise ValueError("denoise_steps and action_horizon must be positive")
    if not 1 <= executed_action_steps <= action_horizon:
        raise ValueError("executed_action_steps must be within the action horizon")

    rows_per_record = denoise_steps * action_horizon
    local_row = torch.arange(num_records * rows_per_record, dtype=torch.int64)
    record_idx = torch.div(local_row, rows_per_record, rounding_mode="floor")
    within_record = torch.remainder(local_row, rows_per_record)
    denoise_step = torch.div(within_record, action_horizon, rounding_mode="floor")
    token_idx = torch.remainder(within_record, action_horizon)
    chunk_start_step = record_idx * executed_action_steps

    return {
        "episode_num": torch.full_like(local_row, int(episode_num)),
        "step_in_episode": chunk_start_step + token_idx,
        "global_forward_idx": record_idx + int(global_record_start),
        "batch_idx": torch.zeros_like(local_row),
        "record_idx": record_idx,
        "denoise_step": denoise_step,
        "token_idx": token_idx,
        "chunk_start_step": chunk_start_step,
        "executed_chunk_len": torch.full_like(local_row, int(executed_action_steps)),
    }


def join_activation_trajectory_inventories(
    *,
    source_manifest: dict[str, Any],
    trajectory_manifest: dict[str, Any],
    denoise_steps: int,
    action_horizon: int,
    executed_action_steps: int,
) -> list[dict[str, Any]]:
    """Exact-join Stage 1 source rows to Stage 2 episode metadata."""

    if list(source_manifest.get("row_order") or []) != ACTIVATION_ROW_ORDER:
        raise ValueError(
            f"Unexpected activation row order: {source_manifest.get('row_order')!r}"
        )
    if int(source_manifest.get("source_denoising_steps", -1)) != denoise_steps:
        raise ValueError("Source denoise-step count does not match the requested contract")
    if int(source_manifest.get("action_horizon", -1)) != action_horizon:
        raise ValueError("Source action horizon does not match the requested contract")

    source_rows = list(source_manifest.get("source_inventory") or [])
    trajectory_rows = list(trajectory_manifest.get("episodes") or [])
    if not source_rows or not trajectory_rows:
        raise ValueError("Source or trajectory manifest has no episode inventory")

    trajectory_by_source: dict[str, dict[str, Any]] = {}
    episode_nums: set[int] = set()
    for row in trajectory_rows:
        source_file = str(row["source_file"])
        episode_num = int(row["episode_num"])
        if source_file in trajectory_by_source:
            raise ValueError(f"Duplicate trajectory source_file: {source_file}")
        if episode_num in episode_nums:
            raise ValueError(f"Duplicate trajectory episode_num: {episode_num}")
        trajectory_by_source[source_file] = row
        episode_nums.add(episode_num)

    joined: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    expected_row_start = 0
    global_record_start = 0
    for source in source_rows:
        source_file = str(source["path"])
        if source_file in seen_sources:
            raise ValueError(f"Duplicate activation source_file: {source_file}")
        seen_sources.add(source_file)
        trajectory = trajectory_by_source.get(source_file)
        if trajectory is None:
            raise ValueError(
                f"Activation source missing from trajectory manifest: {source_file}"
            )

        num_records = int(source["num_records"])
        row_start = int(source["row_start"])
        row_stop = int(source["row_stop"])
        expected_rows = num_records * denoise_steps * action_horizon
        if row_start != expected_row_start or row_stop - row_start != expected_rows:
            raise ValueError(f"Activation row span mismatch for {source_file}")
        if int(trajectory["num_records"]) != num_records:
            raise ValueError(f"Trajectory record count mismatch for {source_file}")
        if int(trajectory["n_action_steps"]) != executed_action_steps:
            raise ValueError(f"Executed action-step count mismatch for {source_file}")

        joined.append(
            {
                "source_file": source_file,
                "episode_num": int(trajectory["episode_num"]),
                "task_id": int(trajectory["task_id"]),
                "task_description": str(trajectory["task_description"]),
                "num_records": num_records,
                "row_start": row_start,
                "row_stop": row_stop,
                "global_record_start": global_record_start,
            }
        )
        expected_row_start = row_stop
        global_record_start += num_records

    extra_sources = sorted(set(trajectory_by_source) - seen_sources)
    if extra_sources:
        raise ValueError(
            f"Trajectory sources missing from activation cache: {extra_sources[0]}"
        )
    if expected_row_start != int(source_manifest.get("num_activation_rows", -1)):
        raise ValueError("Joined activation rows do not match source manifest total")
    if global_record_start != int(source_manifest.get("num_records", -1)):
        raise ValueError("Joined records do not match source manifest total")
    return joined


def encode_activation_cache_to_sparse_topk(args: Any) -> None:
    """Encode a GR00T activation cache into provenance-aware sparse SAE shards."""

    activation_cache = args.activation_cache.resolve()
    trajectory_manifest_path = args.trajectory_manifest.resolve()
    sae_checkpoint = args.sae_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if args.topk <= 0:
        raise ValueError("topk must be positive")

    activations, source_manifest = load_activation_cache(
        activation_cache,
        layer_id=args.layer,
        activation_dim=args.activation_dim,
    )
    (
        checkpoint_source_manifest_path,
        source_manifest_sha256,
    ) = _validate_checkpoint_source_manifest(
        checkpoint_path=sae_checkpoint,
        activation_source_manifest=source_manifest,
    )
    activation_cache_sha256 = _sha256(activation_cache)
    trajectory_manifest = _load_json(trajectory_manifest_path)
    joined = join_activation_trajectory_inventories(
        source_manifest=source_manifest,
        trajectory_manifest=trajectory_manifest,
        denoise_steps=args.denoise_steps,
        action_horizon=args.action_horizon,
        executed_action_steps=args.executed_action_steps,
    )

    if args.max_sources > 0:
        if not args.allow_partial:
            raise ValueError("--max-sources requires --allow-partial")
        joined = joined[: args.max_sources]
    if not args.allow_partial:
        if len(joined) != args.expected_files:
            raise ValueError(f"Expected {args.expected_files} files, got {len(joined)}")
        if sum(row["num_records"] for row in joined) != args.expected_records:
            raise ValueError("Full-run record count mismatch")
        if sum(row["row_stop"] - row["row_start"] for row in joined) != args.expected_rows:
            raise ValueError("Full-run activation row count mismatch")

    device = args.device
    sae, config = load_batch_topk_sae(sae_checkpoint, device=device)
    trainer_config = config["trainer"]
    dict_size = int(trainer_config["dict_size"])
    activation_dim = int(trainer_config["activation_dim"])
    if activation_dim != args.activation_dim:
        raise ValueError("SAE activation dimension does not match the cache")
    if args.topk > dict_size:
        raise ValueError("topk exceeds SAE dictionary size")

    output_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "format": SPARSE_TOPK_FORMAT,
        "backend": "groot_n15_pq3",
        "capture_target": "action_expert",
        "layer": args.layer,
        "sae_path": str(sae_checkpoint),
        "sae_sha256": _sha256(sae_checkpoint),
        "activation_cache": str(activation_cache),
        "activation_cache_bytes": activation_cache.stat().st_size,
        "activation_cache_sha256": activation_cache_sha256,
        "activation_source_manifest_sha256": source_manifest_sha256,
        "checkpoint_source_manifest": str(checkpoint_source_manifest_path),
        "checkpoint_source_manifest_sha256": source_manifest_sha256,
        "source_manifest_comparison": "canonical_json_exact",
        "trajectory_manifest": str(trajectory_manifest_path),
        "trajectory_manifest_sha256": _sha256(trajectory_manifest_path),
        "dict_size": dict_size,
        "activation_dim": activation_dim,
        "topk": args.topk,
        "source_denoising_steps": args.denoise_steps,
        "action_horizon": args.action_horizon,
        "executed_action_steps": args.executed_action_steps,
        "event_step_scale": args.executed_action_steps,
        "row_order": ACTIVATION_ROW_ORDER,
        "executed_chunk_lens_seen": [args.executed_action_steps],
        "partial": bool(args.allow_partial),
        "num_shards": 0,
        "total_rows": 0,
        "total_records": 0,
        "shards": [],
    }

    total_positive = 0
    rows_with_more_than_topk = 0
    max_positive = 0
    total_output_rows = 0
    for shard_idx, source in enumerate(joined):
        dense = activations[source["row_start"] : source["row_stop"]]
        metadata = reconstruct_action_token_row_metadata(
            num_records=source["num_records"],
            episode_num=source["episode_num"],
            global_record_start=source["global_record_start"],
            denoise_steps=args.denoise_steps,
            action_horizon=args.action_horizon,
            executed_action_steps=args.executed_action_steps,
        )
        if len(dense) != len(metadata["episode_num"]):
            raise AssertionError("Dense rows and restored metadata diverged")

        value_chunks: list[torch.Tensor] = []
        id_chunks: list[torch.Tensor] = []
        for row_start in range(0, len(dense), args.batch_size):
            batch = dense[row_start : row_start + args.batch_size]
            with torch.no_grad():
                encoded = sae.encode(batch.to(device=device, dtype=torch.float32))
                if not torch.isfinite(encoded).all():
                    raise ValueError("SAE produced non-finite codes")
                positive_counts = (encoded > 0).sum(dim=1)
                values, indices = torch.topk(encoded, k=args.topk, dim=1)
            total_positive += int(positive_counts.sum().item())
            rows_with_more_than_topk += int((positive_counts > args.topk).sum().item())
            max_positive = max(max_positive, int(positive_counts.max().item()))
            value_chunks.append(values.to(dtype=torch.float32, device="cpu"))
            id_chunks.append(indices.to(dtype=torch.int32, device="cpu"))
            del encoded, positive_counts, values, indices

        if args.require_lossless_topk and rows_with_more_than_topk:
            raise RuntimeError(
                f"topk={args.topk} truncates positive SAE codes; "
                f"rows_over_topk={rows_with_more_than_topk}, max_positive={max_positive}"
            )

        payload = {
            **metadata,
            "top_feature_ids": torch.cat(id_chunks, dim=0),
            "top_feature_vals": torch.cat(value_chunks, dim=0),
        }
        shard_name = f"shard_{shard_idx:06d}.pt"
        torch.save(payload, output_dir / shard_name)
        num_rows = len(dense)
        manifest["shards"].append(
            {
                "shard_idx": shard_idx,
                "path": shard_name,
                "source_path": source["source_file"],
                "episode_num": source["episode_num"],
                "num_records": source["num_records"],
                "num_rows": num_rows,
                "row_start": total_output_rows,
                "row_end": total_output_rows + num_rows,
            }
        )
        total_output_rows += num_rows
        manifest["num_shards"] += 1
        manifest["total_rows"] += num_rows
        manifest["total_records"] += source["num_records"]
        if args.progress_every > 0 and (
            (shard_idx + 1) % args.progress_every == 0
            or shard_idx + 1 == len(joined)
        ):
            print(
                f"[topk] sources={shard_idx + 1}/{len(joined)} "
                f"records={manifest['total_records']} rows={manifest['total_rows']}",
                flush=True,
            )

    manifest["encoding_stats"] = {
        "positive_entries": total_positive,
        "mean_positive_features_per_row": total_positive / max(total_output_rows, 1),
        "max_positive_features_per_row": max_positive,
        "rows_with_more_positive_features_than_topk": rows_with_more_than_topk,
        "lossless_topk": rows_with_more_than_topk == 0,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    audit = {
        "format": SPARSE_ENCODING_AUDIT_FORMAT,
        "source_join_count": len(joined),
        "unique_source_count": len({row["source_file"] for row in joined}),
        "unique_episode_count": len({row["episode_num"] for row in joined}),
        "total_records": manifest["total_records"],
        "total_rows": manifest["total_rows"],
        "expected_executed_rows": (
            manifest["total_records"]
            * args.denoise_steps
            * args.executed_action_steps
        ),
        "expected_environment_steps": (
            manifest["total_records"] * args.executed_action_steps
        ),
        "encoding_stats": manifest["encoding_stats"],
    }
    (output_dir / "provenance_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2), flush=True)
