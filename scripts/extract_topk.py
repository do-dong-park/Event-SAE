"""CLI: offline extract top-k SAE activations from dense shards.

``--dense-dir`` is **the directory that contains ``activation_index.jsonl``**;
each record's ``shard_path`` is resolved relative to that directory.

Both layouts therefore work without code changes:

* **OpenPI**: pass the eval run root, e.g.
  ``logs/openpi/sae_collection/<run>/``. Index lives at the root and
  ``shard_path`` is ``sae_activations/post_mlp_residual/layer_NN_shard_*.pt``.
* **OpenVLA**: pass the per-target subdir, e.g.
  ``logs/openvla/<run>/sae_activations/post_mlp_residual/``. Index and
  shards are siblings inside it.

Reads:
  - Dense residual shards resolved from ``shard_path`` in the index.
  - Trained ``BatchTopKSAE`` checkpoint (``ae.pt`` with sibling ``config.json``).

Writes (under ``--output-dir``, default ``{dense_dir}/topk_activations``):
  - ``shard_NNNNNN.pt`` with sparse top-k rows + metadata
  - ``manifest.json`` in ``token_topk_sparse_v1`` format (same as online mode)

The output is byte-format-compatible with
``event_sae.openvla.activations.apply_sae_topk_collect_hooks``, so
downstream scoring can consume either online or offline shards uniformly.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sae.sae import load_batch_topk_sae


def _load_index(index_path: Path, layer_idx: int) -> dict[str, list[dict]]:
    by_shard: dict[str, list[dict]] = defaultdict(list)
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if int(record["layer_idx"]) != layer_idx:
                continue
            by_shard[str(record["shard_path"])].append(record)
    for records in by_shard.values():
        records.sort(key=lambda r: int(r["row_start"]))
    return by_shard


def _validate_row_spans(
    records: list[dict],
    *,
    num_rows: int,
    shard_path: Path,
) -> None:
    """Require index rows to partition a dense shard exactly once."""

    expected_start = 0
    for record in records:
        row_start = int(record["row_start"])
        row_end = int(record["row_end"])
        if row_start != expected_start:
            relation = "overlap" if row_start < expected_start else "gap"
            raise ValueError(
                f"{shard_path}: activation index {relation} at row {expected_start}"
            )
        if row_end <= row_start or row_end > num_rows:
            raise ValueError(
                f"{shard_path}: invalid activation index span "
                f"[{row_start}, {row_end}) for {num_rows} rows"
            )
        expected_start = row_end
    if expected_start != num_rows:
        raise ValueError(
            f"{shard_path}: activation index covers [0, {expected_start}), "
            f"expected [0, {num_rows})"
        )


def _load_dense_shard(path: Path, activation_dim: int) -> torch.Tensor:
    dense = torch.load(path, map_location="cpu").to(torch.float32)
    if dense.ndim != 2 or int(dense.shape[1]) != activation_dim:
        raise ValueError(
            f"Unexpected dense shard shape {tuple(dense.shape)} in {path}; "
            f"expected (N, {activation_dim})"
        )
    return dense


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline SAE encode of dense activation shards.")
    parser.add_argument(
        "--dense-dir",
        required=True,
        help=(
            "Directory that contains activation_index.jsonl; shard paths in the "
            "index are resolved relative to it. For OpenPI eval runs pass the run "
            "root (the index sits at the root). For OpenVLA collection runs pass "
            "the per-target subdir (e.g. sae_activations/post_mlp_residual)."
        ),
    )
    parser.add_argument("--sae-checkpoint", required=True, help="Path to trained ae.pt")
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dir (default: {dense_dir}/topk_activations).",
    )
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available else cpu).")
    args = parser.parse_args()

    dense_dir = Path(args.dense_dir).resolve()
    index_path = dense_dir / "activation_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (dense_dir / "topk_activations").resolve()
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sae, config = load_batch_topk_sae(Path(args.sae_checkpoint), device=device)
    trainer_cfg = config["trainer"]
    activation_dim = int(trainer_cfg["activation_dim"])
    dict_size = int(trainer_cfg["dict_size"])
    if not (1 <= args.topk <= dict_size):
        raise ValueError(f"topk must be in [1, {dict_size}], got {args.topk}")

    index = _load_index(index_path, layer_idx=args.layer_idx)
    if not index:
        raise RuntimeError(f"No index records found for layer {args.layer_idx} in {index_path}")

    # Validate every dense shard and its exact index partition before creating
    # output, so a late gap/overlap cannot leave a mixed partial run.
    for src_shard_name, records in sorted(index.items()):
        src_shard_path = dense_dir / src_shard_name
        if not src_shard_path.is_file():
            raise FileNotFoundError(f"Missing dense shard: {src_shard_path}")
        dense = _load_dense_shard(src_shard_path, activation_dim)
        _validate_row_spans(
            records,
            num_rows=int(dense.shape[0]),
            shard_path=src_shard_path,
        )
        del dense
    output_dir.mkdir(parents=True)

    # Probe a record for OpenPI-specific fields. ``capture_target`` and
    # ``executed_chunk_len`` propagate to the manifest so the scorer can
    # auto-detect step_mapping (action_executed for AE, chunk_executed for
    # PG, inference_step for OpenVLA legacy).
    probe_record = next(iter(index.values()))[0]
    capture_target = probe_record.get("capture_target")
    executed_chunk_len_seen = sorted({
        int(r["executed_chunk_len"])
        for records in index.values()
        for r in records
        if r.get("executed_chunk_len") is not None
    })

    manifest = {
        "format": "token_topk_sparse_v1",
        "layer": args.layer_idx,
        "sae_path": str(args.sae_checkpoint),
        "dict_size": dict_size,
        "activation_dim": activation_dim,
        "topk": args.topk,
        "capture_target": capture_target,
        "executed_chunk_lens_seen": executed_chunk_len_seen,
        "num_shards": 0,
        "total_rows": 0,
        "shards": [],
    }

    shard_names = sorted(index.keys())
    total_rows = 0
    for src_shard_name in shard_names:
        src_shard_path = dense_dir / src_shard_name
        if not src_shard_path.is_file():
            raise FileNotFoundError(f"Missing dense shard: {src_shard_path}")
        dense = _load_dense_shard(src_shard_path, activation_dim)
        records = index[src_shard_name]
        n_rows = int(dense.shape[0])

        episode_num = torch.zeros((n_rows,), dtype=torch.int64)
        step_in_episode = torch.zeros((n_rows,), dtype=torch.int64)
        global_forward_idx = torch.zeros((n_rows,), dtype=torch.int64)
        token_idx = torch.zeros((n_rows,), dtype=torch.int64)
        batch_idx = torch.zeros((n_rows,), dtype=torch.int64)
        # OpenPI-only: per-row chunk_start_step + executed_chunk_len so
        # `score_cluster_features` can compute `_effective_steps` under the
        # chosen step_mapping. OpenVLA records leave these at -1 (sentinel).
        chunk_start_step = torch.full((n_rows,), -1, dtype=torch.int64)
        executed_chunk_len = torch.full((n_rows,), -1, dtype=torch.int64)
        for record in records:
            row_start = int(record["row_start"])
            row_end = int(record["row_end"])
            episode_num[row_start:row_end] = int(record.get("episode_num") or 0)
            global_forward_idx[row_start:row_end] = int(
                record.get("global_forward_idx") or 0
            )
            # Per-token env-step mapping. OpenPI records each forward as one
            # chunked inference covering `seq_len` future tokens; the env step
            # a token corresponds to is `chunk_start + token_idx`. OpenVLA
            # records each forward as one env-step (no chunking), and all
            # rows of a record share the same `step_in_episode`. Matches
            # openpi-mech's `step_mapping="action_executed"` semantics.
            tokens_local = torch.arange(row_end - row_start, dtype=torch.int64)
            token_idx[row_start:row_end] = tokens_local
            chunk_start = record.get("chunk_start_step")
            if chunk_start is None:
                chunk_start = record.get("action_chunk_start_step")
            if chunk_start is not None:
                chunk_start_step[row_start:row_end] = int(chunk_start)
                step_in_episode[row_start:row_end] = int(chunk_start) + tokens_local
            else:
                step_in_episode[row_start:row_end] = int(
                    record.get("step_in_episode") or 0
                )
            if record.get("executed_chunk_len") is not None:
                executed_chunk_len[row_start:row_end] = int(
                    record["executed_chunk_len"]
                )

        with torch.no_grad():
            encoded = sae.encode(dense.to(device))
            values, indices = torch.topk(encoded, k=args.topk, dim=-1)
            values = values.float().cpu()
            indices = indices.to(torch.int32).cpu()

        out_shard_name = f"shard_{manifest['num_shards']:06d}.pt"
        torch.save(
            {
                "episode_num": episode_num,
                "step_in_episode": step_in_episode,
                "global_forward_idx": global_forward_idx,
                "batch_idx": batch_idx,
                "token_idx": token_idx,
                "chunk_start_step": chunk_start_step,
                "executed_chunk_len": executed_chunk_len,
                "top_feature_ids": indices,
                "top_feature_vals": values,
            },
            output_dir / out_shard_name,
        )
        manifest["shards"].append(
            {
                "shard_idx": manifest["num_shards"],
                "path": out_shard_name,
                "num_rows": n_rows,
                "row_start": total_rows,
                "row_end": total_rows + n_rows,
            }
        )
        manifest["num_shards"] += 1
        total_rows += n_rows
        print(f"Encoded {src_shard_name} → {out_shard_name}  rows={n_rows}")

    manifest["total_rows"] = total_rows
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {output_dir / 'manifest.json'}")
    print(f"Total rows: {total_rows}")


if __name__ == "__main__":
    main()
