"""Stream foreign GR00T activation shards through an existing SAE checkpoint.

The canonical Stage-4 encoder intentionally requires the activation source to
equal the SAE training source.  That guard is correct for checkpoint audits,
but it prevents applying a frozen SAE to a held-out rollout collection.  This
module provides a separate, explicit transfer-encoding path: training and
evaluation source manifests are both hashed, their equality is recorded, and
no equality is claimed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from event_sae import sha256_file
from event_sae.groot.activations import (
    ACTIVATION_ROW_ORDER,
    SPARSE_ENCODING_AUDIT_FORMAT,
    SPARSE_TOPK_FORMAT,
    join_activation_trajectory_inventories,
    reconstruct_action_token_row_metadata,
)
from event_sae.sae import load_batch_topk_sae


SOURCE_FORMAT = "groot_n15_robocasa_pq3_action_tokens_v2"
SOURCE_SHARD_FORMAT = "groot_n15_pq3_action_activation_shard_v1"
TRANSFER_AUDIT_FORMAT = "groot_n15_transfer_topk_audit_v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _checkpoint_source_manifest(checkpoint_path: Path) -> Path | None:
    start = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    for directory in (start, *start.parents):
        candidate = directory / "groot_source_manifest.json"
        if candidate.is_file():
            return candidate
    return None


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def encode_transfer_activation_shards(
    *,
    activation_shard_dir: Path,
    trajectory_manifest_path: Path,
    sae_checkpoint: Path,
    output_dir: Path,
    layer: int = 15,
    activation_dim: int = 1536,
    denoise_steps: int = 4,
    action_horizon: int = 16,
    executed_action_steps: int = 5,
    topk: int = 96,
    batch_size: int = 4096,
    device: str = "cuda:0",
    expected_sources: int | None = None,
    require_lossless_topk: bool = False,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Encode evaluation shards without pretending they trained the SAE."""

    activation_shard_dir = Path(activation_shard_dir).resolve()
    trajectory_manifest_path = Path(trajectory_manifest_path).resolve()
    sae_checkpoint = Path(sae_checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    source_manifest_path = activation_shard_dir / "groot_source_manifest.json"
    for label, path in (
        ("activation source manifest", source_manifest_path),
        ("trajectory manifest", trajectory_manifest_path),
        ("SAE checkpoint", sae_checkpoint),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    if topk <= 0 or batch_size <= 0:
        raise ValueError("topk and batch_size must be positive")

    source_manifest = _load_json(source_manifest_path)
    if source_manifest.get("format") != SOURCE_FORMAT:
        raise ValueError(
            f"Unsupported activation source format: {source_manifest.get('format')!r}"
        )
    expected_identity = {
        "physical_layer": layer,
        "activation_dim": activation_dim,
        "source_denoising_steps": denoise_steps,
        "action_horizon": action_horizon,
        "token_scope": "action",
    }
    mismatches = {
        key: {"actual": source_manifest.get(key), "expected": expected}
        for key, expected in expected_identity.items()
        if source_manifest.get(key) != expected
    }
    if list(source_manifest.get("row_order") or []) != ACTIVATION_ROW_ORDER:
        mismatches["row_order"] = {
            "actual": source_manifest.get("row_order"),
            "expected": ACTIVATION_ROW_ORDER,
        }
    if mismatches:
        raise ValueError(f"Activation source identity mismatch: {mismatches}")

    trajectory_manifest = _load_json(trajectory_manifest_path)
    joined = join_activation_trajectory_inventories(
        source_manifest=source_manifest,
        trajectory_manifest=trajectory_manifest,
        denoise_steps=denoise_steps,
        action_horizon=action_horizon,
        executed_action_steps=executed_action_steps,
    )
    if expected_sources is not None and len(joined) != expected_sources:
        raise ValueError(
            f"Expected {expected_sources} sources, found {len(joined)}"
        )

    shard_entries = source_manifest.get("cache_layout", {}).get("shards")
    if not isinstance(shard_entries, list) or len(shard_entries) != len(joined):
        raise ValueError("Activation shard inventory does not match source join")

    sae, config = load_batch_topk_sae(sae_checkpoint, device=device)
    trainer_config = config["trainer"]
    dict_size = int(trainer_config["dict_size"])
    checkpoint_dim = int(trainer_config["activation_dim"])
    if checkpoint_dim != activation_dim:
        raise ValueError(
            f"SAE activation_dim={checkpoint_dim} != source={activation_dim}"
        )
    if topk > dict_size:
        raise ValueError(f"topk={topk} exceeds SAE dict_size={dict_size}")

    checkpoint_manifest_path = _checkpoint_source_manifest(sae_checkpoint)
    checkpoint_manifest = (
        _load_json(checkpoint_manifest_path)
        if checkpoint_manifest_path is not None
        else None
    )
    checkpoint_source_match = bool(
        checkpoint_manifest is not None
        and _canonical_json(checkpoint_manifest)
        == _canonical_json(source_manifest)
    )

    output_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "format": SPARSE_TOPK_FORMAT,
        "backend": "groot_n15_transfer_rollout",
        "capture_target": "action_expert",
        "layer": layer,
        "sae_path": str(sae_checkpoint),
        "sae_sha256": sha256_file(sae_checkpoint),
        "activation_source_manifest": str(source_manifest_path),
        "activation_source_manifest_sha256": sha256_file(
            source_manifest_path
        ),
        "checkpoint_source_manifest": (
            None
            if checkpoint_manifest_path is None
            else str(checkpoint_manifest_path)
        ),
        "checkpoint_source_manifest_sha256": (
            None
            if checkpoint_manifest_path is None
            else sha256_file(checkpoint_manifest_path)
        ),
        "checkpoint_source_match": checkpoint_source_match,
        "source_manifest_comparison": (
            "canonical_json_equal"
            if checkpoint_source_match
            else "transfer_encoding_intentionally_distinct"
        ),
        "transfer_encoding": True,
        "trajectory_manifest": str(trajectory_manifest_path),
        "trajectory_manifest_sha256": sha256_file(
            trajectory_manifest_path
        ),
        "dict_size": dict_size,
        "activation_dim": activation_dim,
        "topk": topk,
        "source_denoising_steps": denoise_steps,
        "action_horizon": action_horizon,
        "executed_action_steps": executed_action_steps,
        "event_step_scale": 1,
        "event_time_unit": "environment_action_step",
        "row_order": ACTIVATION_ROW_ORDER,
        "executed_chunk_lens_seen": [executed_action_steps],
        "partial": not bool(source_manifest.get("inventory_verified")),
        "source_model_tokens": int(source_manifest["source_model_tokens"]),
        "source_action_token_slice": dict(
            source_manifest["action_token_slice"]
        ),
        "num_shards": 0,
        "total_rows": 0,
        "total_records": 0,
        "shards": [],
    }

    total_positive = 0
    rows_over_topk = 0
    max_positive = 0
    total_output_rows = 0
    for shard_index, (source, shard_entry) in enumerate(
        zip(joined, shard_entries, strict=True)
    ):
        if str(shard_entry["source_path"]) != str(source["source_file"]):
            raise ValueError(
                f"Shard/source mismatch at {shard_index}: "
                f"{shard_entry['source_path']!r} != {source['source_file']!r}"
            )
        source_shard_path = activation_shard_dir / str(shard_entry["path"])
        payload = torch.load(source_shard_path, map_location="cpu")
        if (
            not isinstance(payload, dict)
            or payload.get("format") != SOURCE_SHARD_FORMAT
        ):
            raise ValueError(f"Unsupported activation shard: {source_shard_path}")
        dense = payload.get("activations")
        expected_rows = int(source["row_stop"]) - int(source["row_start"])
        if (
            not torch.is_tensor(dense)
            or dense.ndim != 2
            or tuple(dense.shape) != (expected_rows, activation_dim)
        ):
            raise ValueError(
                f"{source_shard_path}: activation shape mismatch "
                f"{getattr(dense, 'shape', None)}"
            )
        if not torch.isfinite(dense).all():
            raise ValueError(f"{source_shard_path}: non-finite activations")

        metadata = reconstruct_action_token_row_metadata(
            num_records=int(source["num_records"]),
            episode_num=int(source["episode_num"]),
            global_record_start=int(source["global_record_start"]),
            denoise_steps=denoise_steps,
            action_horizon=action_horizon,
            executed_action_steps=executed_action_steps,
        )
        value_chunks: list[torch.Tensor] = []
        id_chunks: list[torch.Tensor] = []
        for row_start in range(0, len(dense), batch_size):
            batch = dense[row_start : row_start + batch_size]
            with torch.no_grad():
                encoded = sae.encode(
                    batch.to(device=device, dtype=torch.float32)
                )
                if not torch.isfinite(encoded).all():
                    raise ValueError("SAE produced non-finite codes")
                positive_counts = (encoded > 0).sum(dim=1)
                values, indices = torch.topk(encoded, k=topk, dim=1)
            total_positive += int(positive_counts.sum().item())
            rows_over_topk += int((positive_counts > topk).sum().item())
            max_positive = max(
                max_positive,
                int(positive_counts.max().item()),
            )
            value_chunks.append(
                values.to(dtype=torch.float32, device="cpu")
            )
            id_chunks.append(
                indices.to(dtype=torch.int32, device="cpu")
            )

        if require_lossless_topk and rows_over_topk:
            raise RuntimeError(
                f"topk={topk} truncates positive codes; "
                f"rows_over_topk={rows_over_topk}"
            )
        output_payload = {
            **metadata,
            "top_feature_ids": torch.cat(id_chunks, dim=0),
            "top_feature_vals": torch.cat(value_chunks, dim=0),
        }
        shard_name = f"shard_{shard_index:06d}.pt"
        torch.save(output_payload, output_dir / shard_name)
        num_rows = len(dense)
        manifest["shards"].append(
            {
                "shard_idx": shard_index,
                "path": shard_name,
                "source_path": source["source_file"],
                "source_activation_shard": str(source_shard_path),
                "source_activation_shard_sha256": sha256_file(
                    source_shard_path
                ),
                "episode_num": int(source["episode_num"]),
                "num_records": int(source["num_records"]),
                "num_rows": num_rows,
                "row_start": total_output_rows,
                "row_end": total_output_rows + num_rows,
            }
        )
        total_output_rows += num_rows
        manifest["num_shards"] += 1
        manifest["total_rows"] += num_rows
        manifest["total_records"] += int(source["num_records"])
        if progress_every > 0 and (
            (shard_index + 1) % progress_every == 0
            or shard_index + 1 == len(joined)
        ):
            print(
                f"[transfer-topk] sources={shard_index + 1}/{len(joined)} "
                f"records={manifest['total_records']} "
                f"rows={manifest['total_rows']}",
                flush=True,
            )

    manifest["encoding_stats"] = {
        "positive_entries": total_positive,
        "mean_positive_features_per_row": (
            total_positive / max(total_output_rows, 1)
        ),
        "max_positive_features_per_row": max_positive,
        "rows_with_more_positive_features_than_topk": rows_over_topk,
        "lossless_topk": rows_over_topk == 0,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    audit = {
        "format": TRANSFER_AUDIT_FORMAT,
        "compatible_sparse_format": SPARSE_ENCODING_AUDIT_FORMAT,
        "transfer_encoding": True,
        "checkpoint_source_match": checkpoint_source_match,
        "source_join_count": len(joined),
        "unique_episode_count": len(
            {int(row["episode_num"]) for row in joined}
        ),
        "total_records": int(manifest["total_records"]),
        "total_rows": int(manifest["total_rows"]),
        "expected_executed_rows": (
            int(manifest["total_records"])
            * denoise_steps
            * executed_action_steps
        ),
        "expected_environment_steps": (
            int(manifest["total_records"]) * executed_action_steps
        ),
        "encoding_stats": manifest["encoding_stats"],
    }
    (output_dir / "provenance_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return audit
