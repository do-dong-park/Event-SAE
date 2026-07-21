"""Stream GR00T N1.5 PQ3 L15 action-token activations into portable shards.

This standalone script is intentionally independent of Event-SAE and
dictionary_learning so that only this file needs to be copied to the host
where the trusted rollout PKLs live. Export keeps memory bounded to one PKL
plus that file's selected action-token rows. Merge is intended for the local
SAE training host.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


CAPTURE_LAYERS = (0, 2, 4, 8, 10, 12, 15)
FEATURE_KIND = "groot_n15_dit_block_residual_full_tokens_denoise"
FEATURE_AXES = ("layer", "denoise_step", "model_token", "feature_dim")
CAPTURE_TOKEN_MODE = "all_token_full"
EXPECTED_CELL_COUNTS = {
    "OpenDrawer/pq3_drawer_left": 30,
    "OpenDrawer/pq3_drawer_right": 30,
    "PickPlaceCounterToCabinet/pq3_ppcc_beer": 30,
    "PickPlaceCounterToCabinet/pq3_ppcc_bread": 30,
    "PickPlaceCounterToCabinet/pq3_ppcc_pizza_cutter": 30,
}
SOURCE_FORMAT = "groot_n15_robocasa_pq3_action_tokens_v2"
SHARD_FORMAT = "groot_n15_pq3_action_activation_shard_v1"
CACHE_FORMAT = "groot_n15_pq3_action_activation_cache_v1"
MANIFEST_NAME = "groot_source_manifest.json"


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def _dtype_name(value: Any) -> str:
    if torch.is_tensor(value):
        return str(value.dtype).removeprefix("torch.")
    return str(np.asarray(value).dtype)


def _inventory(root: Path, allow_partial: bool) -> tuple[list[Path], dict[str, int]]:
    paths = sorted(path for path in root.rglob("*.pkl") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No PKLs under {root}")
    symlinks = [path for path in paths if path.is_symlink()]
    if symlinks:
        raise ValueError(f"PQ3 source inventory contains symlink PKL: {symlinks[0]}")
    counts = dict(
        sorted(Counter(path.relative_to(root).parent.as_posix() for path in paths).items())
    )
    if not allow_partial and counts != EXPECTED_CELL_COUNTS:
        raise ValueError(
            f"PQ3 source inventory mismatch: actual={counts}, "
            f"expected={EXPECTED_CELL_COUNTS}"
        )
    return paths, counts


def export_shards(args: argparse.Namespace) -> None:
    if not args.trust_pkl:
        raise ValueError("Refusing pickle.load without explicit --trust-pkl")
    root = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if args.state_tokens + args.future_tokens + args.action_horizon != args.model_tokens:
        raise ValueError("state + future + action token counts must equal model tokens")
    if args.layer not in CAPTURE_LAYERS:
        raise ValueError(f"layer {args.layer} not in {CAPTURE_LAYERS}")

    paths, cell_counts = _inventory(root, args.allow_partial_inventory)
    output.mkdir(parents=True)
    action_start = args.model_tokens - args.action_horizon
    layer_position = CAPTURE_LAYERS.index(args.layer)
    expected_shape = (
        len(CAPTURE_LAYERS),
        args.denoise_steps,
        args.model_tokens,
        args.activation_dim,
    )
    source_files: list[str] = []
    source_inventory: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    source_dtype: str | None = None
    num_records = 0
    num_rows = 0
    selected_nbytes = 0

    for file_index, path in enumerate(paths):
        relative_name = path.relative_to(root).as_posix()
        row_start = num_rows
        with path.open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 -- explicitly trusted by caller.

        kind = str(payload.get("feature_kind") or "")
        axes = tuple(payload.get("feature_axes") or ())
        capture_mode = str(payload.get("capture_token_mode") or "")
        layers = tuple(int(x) for x in (payload.get("capture_layers") or ()))
        hidden_states = payload.get("hidden_states") or ()
        horizon = payload.get("model_action_horizon")
        payload_steps = payload.get("num_inference_timesteps")
        if kind != args.expected_feature_kind:
            raise ValueError(f"{path}: feature_kind={kind!r}")
        if axes != FEATURE_AXES:
            raise ValueError(f"{path}: feature_axes={axes}, expected={FEATURE_AXES}")
        if capture_mode != CAPTURE_TOKEN_MODE:
            raise ValueError(f"{path}: capture_token_mode={capture_mode!r}")
        if layers != CAPTURE_LAYERS:
            raise ValueError(f"{path}: capture_layers={layers}, expected={CAPTURE_LAYERS}")
        if horizon is None or int(horizon) != args.action_horizon:
            raise ValueError(f"{path}: model_action_horizon={horizon!r}")
        if payload_steps is not None and int(payload_steps) != args.denoise_steps:
            raise ValueError(f"{path}: num_inference_timesteps={payload_steps!r}")
        if not hidden_states:
            raise ValueError(f"{path}: missing hidden_states")

        file_rows: list[torch.Tensor] = []
        for record_index, hidden in enumerate(hidden_states):
            current_dtype = _dtype_name(hidden)
            if source_dtype is None:
                source_dtype = current_dtype
            elif current_dtype != source_dtype:
                raise ValueError(
                    f"{path}: hidden_states[{record_index}] dtype={current_dtype}, "
                    f"expected={source_dtype}"
                )
            array = _as_numpy(hidden)
            if array.shape != expected_shape:
                raise ValueError(
                    f"{path}: hidden_states[{record_index}]={array.shape}, "
                    f"expected={expected_shape}"
                )
            selected = np.ascontiguousarray(
                array[layer_position, :, action_start:, :]
            )
            if selected.shape != (
                args.denoise_steps,
                args.action_horizon,
                args.activation_dim,
            ):
                raise AssertionError(f"Internal action slice error: {selected.shape}")
            if not np.isfinite(selected).all():
                raise ValueError(
                    f"{path}: non-finite layer {args.layer} action-token activation"
                )
            file_rows.append(torch.from_numpy(selected.reshape(-1, args.activation_dim)))

        shard_tensor = torch.cat(file_rows, dim=0).contiguous()
        shard_name = f"layer_{args.layer}_shard_{file_index:06d}.pt"
        shard_path = output / shard_name
        torch.save(
            {"format": SHARD_FORMAT, "activations": shard_tensor},
            shard_path,
        )
        shard_rows = len(shard_tensor)
        shard_nbytes = shard_tensor.numel() * shard_tensor.element_size()
        num_records += len(hidden_states)
        num_rows += shard_rows
        selected_nbytes += shard_nbytes
        source_files.append(relative_name)
        source_inventory.append(
            {
                "path": relative_name,
                "num_records": len(hidden_states),
                "row_start": row_start,
                "row_stop": num_rows,
            }
        )
        shards.append(
            {
                "path": shard_name,
                "source_path": relative_name,
                "num_rows": shard_rows,
                "nbytes": shard_nbytes,
            }
        )
        del shard_tensor, file_rows, payload, hidden_states
        gc.collect()
        if args.progress_every > 0 and (
            (file_index + 1) % args.progress_every == 0
            or file_index + 1 == len(paths)
        ):
            print(
                f"[export] files={file_index + 1}/{len(paths)} "
                f"records={num_records} rows={num_rows} "
                f"selected_gib={selected_nbytes / 1024**3:.2f}",
                flush=True,
            )

    manifest = {
        "format": SOURCE_FORMAT,
        "source_root": str(root),
        "source_files": source_files,
        "source_inventory": source_inventory,
        "num_files": len(paths),
        "cell_counts": cell_counts,
        "expected_cell_counts": None if args.allow_partial_inventory else EXPECTED_CELL_COUNTS,
        "inventory_verified": not args.allow_partial_inventory,
        "num_records": num_records,
        "num_activation_rows": num_rows,
        "feature_kind": args.expected_feature_kind,
        "feature_axes": list(FEATURE_AXES),
        "capture_token_mode": CAPTURE_TOKEN_MODE,
        "capture_layers": list(CAPTURE_LAYERS),
        "physical_layer": args.layer,
        "source_denoising_steps": args.denoise_steps,
        "denoise_mode": "all_as_independent_rows",
        "source_model_tokens": args.model_tokens,
        "token_layout": {
            "state_tokens": args.state_tokens,
            "future_tokens": args.future_tokens,
            "action_tokens": args.action_horizon,
        },
        "token_scope": "action",
        "action_horizon": args.action_horizon,
        "action_token_slice": {
            "start": action_start,
            "stop": args.model_tokens,
            "semantics": "half_open_model_token_indices",
        },
        "row_order": ["source_file", "record", "denoise_step", "action_token_offset"],
        "activation_dim": args.activation_dim,
        "source_dtype": source_dtype,
        "resident_dtype": f"torch.{source_dtype}",
        "selected_activation_bytes": selected_nbytes,
        "estimated_materialization_peak_bytes": selected_nbytes * 2,
        "materialized": True,
        "cache_layout": {
            "format": SHARD_FORMAT,
            "manifest": MANIFEST_NAME,
            "num_shards": len(shards),
            "shards": shards,
        },
    }
    manifest_path = output / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[export] complete manifest={manifest_path}", flush=True)


def merge_shards(args: argparse.Namespace) -> None:
    shard_dir = args.shard_dir.resolve()
    output_cache = args.output_cache.resolve()
    if output_cache.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache: {output_cache}")
    manifest_path = shard_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != SOURCE_FORMAT:
        raise ValueError(f"Unsupported source manifest: {manifest.get('format')!r}")
    shard_entries = manifest.get("cache_layout", {}).get("shards")
    if not isinstance(shard_entries, list) or not shard_entries:
        raise ValueError(f"{manifest_path}: missing shard inventory")
    selected_nbytes = int(manifest.get("selected_activation_bytes", -1))
    estimated_peak = selected_nbytes * 2
    max_ram_bytes = int(args.max_ram_gib * 1024**3) if args.max_ram_gib > 0 else 0
    if max_ram_bytes and estimated_peak > max_ram_bytes:
        raise MemoryError(
            f"Merge estimated peak={estimated_peak / 1024**3:.2f} GiB exceeds "
            f"limit={args.max_ram_gib:.2f} GiB"
        )

    chunks: list[torch.Tensor] = []
    rows = 0
    for shard_index, entry in enumerate(shard_entries, 1):
        shard_path = shard_dir / entry["path"]
        payload = torch.load(shard_path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("format") != SHARD_FORMAT:
            raise ValueError(f"Unsupported shard format: {shard_path}")
        tensor = payload.get("activations")
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            raise ValueError(f"{shard_path}: expected activation tensor [N,D]")
        if tensor.shape[1] != int(manifest["activation_dim"]):
            raise ValueError(f"{shard_path}: activation dimension mismatch")
        if len(tensor) != int(entry["num_rows"]):
            raise ValueError(f"{shard_path}: row count mismatch")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{shard_path}: non-finite activation")
        chunks.append(tensor.contiguous())
        rows += len(tensor)
        if args.progress_every > 0 and (
            shard_index % args.progress_every == 0 or shard_index == len(shard_entries)
        ):
            print(
                f"[merge] shards={shard_index}/{len(shard_entries)} rows={rows}",
                flush=True,
            )

    activations = torch.cat(chunks, dim=0).contiguous()
    if len(activations) != int(manifest["num_activation_rows"]):
        raise ValueError("Merged row count does not match source manifest")
    if activations.numel() * activations.element_size() != selected_nbytes:
        raise ValueError("Merged byte count does not match source manifest")
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_cache.with_suffix(output_cache.suffix + ".tmp")
    torch.save(
        {
            "format": CACHE_FORMAT,
            "activations": activations,
            "source_manifest": manifest,
        },
        temporary,
    )
    temporary.replace(output_cache)
    print(
        f"[merge] complete cache={output_cache} "
        f"shape={tuple(activations.shape)} dtype={activations.dtype}",
        flush=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Audit PKLs and write bounded shards")
    export.add_argument("--input-dir", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--trust-pkl", action="store_true")
    export.add_argument("--layer", type=int, default=15)
    export.add_argument("--activation-dim", type=int, default=1536)
    export.add_argument("--denoise-steps", type=int, default=4)
    export.add_argument("--model-tokens", type=int, default=49)
    export.add_argument("--state-tokens", type=int, default=1)
    export.add_argument("--future-tokens", type=int, default=32)
    export.add_argument("--action-horizon", type=int, default=16)
    export.add_argument("--expected-feature-kind", default=FEATURE_KIND)
    export.add_argument("--progress-every", type=int, default=10)
    export.add_argument("--allow-partial-inventory", action="store_true")
    export.set_defaults(func=export_shards)

    merge = subparsers.add_parser("merge", help="Merge transferred shards locally")
    merge.add_argument("--shard-dir", type=Path, required=True)
    merge.add_argument("--output-cache", type=Path, required=True)
    merge.add_argument("--max-ram-gib", type=float, default=32.0)
    merge.add_argument("--progress-every", type=int, default=10)
    merge.set_defaults(func=merge_shards)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
