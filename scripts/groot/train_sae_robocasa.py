"""Train BatchTopK SAEs on GR00T RoboCasa PQ3 DiT residual streams.

PQ3 stores full-token block residuals with axes layer, denoise step,
model token, and feature dimension. The paper-aligned primary contract
trains one SAE per physical layer on the final action-token slice only:
every denoise step and action-token offset is an independent row.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CAPTURE_LAYERS = (0, 2, 4, 8, 10, 12, 15)
DEFAULT_FEATURE_KIND = "groot_n15_dit_block_residual_full_tokens_denoise"
EXPECTED_FEATURE_AXES = ("layer", "denoise_step", "model_token", "feature_dim")
EXPECTED_CAPTURE_TOKEN_MODE = "all_token_full"
DEFAULT_MODEL_TOKENS = 49
DEFAULT_STATE_TOKENS = 1
DEFAULT_FUTURE_TOKENS = 32
DEFAULT_ACTION_HORIZON = 16
ACTIVATION_CACHE_FORMAT = "groot_n15_pq3_action_activation_cache_v1"
EXPECTED_PQ3_CELL_COUNTS = {
    "OpenDrawer/pq3_drawer_left": 30,
    "OpenDrawer/pq3_drawer_right": 30,
    "PickPlaceCounterToCabinet/pq3_ppcc_beer": 30,
    "PickPlaceCounterToCabinet/pq3_ppcc_bread": 30,
    "PickPlaceCounterToCabinet/pq3_ppcc_pizza_cutter": 30,
}


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
                # Keep the resident CPU copy in its source dtype (PQ3 is
                # fp16) and cast only the current batch for SAE training.
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


def _source_dtype_name(value: Any) -> str:
    if torch.is_tensor(value):
        return str(value.dtype).removeprefix("torch.")
    return str(np.asarray(value).dtype)


def _validate_token_contract(
    *,
    token_scope: str,
    expected_model_tokens: int,
    state_tokens: int,
    future_tokens: int,
    action_horizon: int,
) -> int:
    if token_scope != "action":
        raise ValueError(
            f"Unsupported token_scope={token_scope!r}; the primary PQ3 contract is 'action'"
        )
    values = {
        "expected_model_tokens": expected_model_tokens,
        "state_tokens": state_tokens,
        "future_tokens": future_tokens,
        "action_horizon": action_horizon,
    }
    if (
        any(value < 0 for value in values.values())
        or expected_model_tokens == 0
        or action_horizon == 0
    ):
        raise ValueError(f"Token counts must be positive where applicable, got {values}")
    if state_tokens + future_tokens + action_horizon != expected_model_tokens:
        raise ValueError(
            "Token layout mismatch: "
            f"state({state_tokens}) + future({future_tokens}) + action({action_horizon}) "
            f"!= model_tokens({expected_model_tokens})"
        )
    return expected_model_tokens - action_horizon


def load_layer_activations(
    root: Path,
    *,
    layer_id: int,
    activation_dim: int,
    expected_denoise_steps: int,
    expected_feature_kind: str,
    max_files: int,
    progress_every: int,
    token_scope: str = "action",
    expected_model_tokens: int = DEFAULT_MODEL_TOKENS,
    state_tokens: int = DEFAULT_STATE_TOKENS,
    future_tokens: int = DEFAULT_FUTURE_TOKENS,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    max_ram_gib: float = 32.0,
    materialize: bool = True,
    expected_cell_counts: dict[str, int] | None = None,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Audit PQ3 PKLs and optionally materialize one layer's action rows.

    Each source record is [L,K,T,D]. The selected physical layer is
    sliced to the final action_horizon token positions and flattened
    without pooling from [K,A,D] to [K*A,D].

    materialize=False performs the full contract and finite audit without
    retaining activation rows, keeping audit-only memory bounded.
    """

    action_start = _validate_token_contract(
        token_scope=token_scope,
        expected_model_tokens=expected_model_tokens,
        state_tokens=state_tokens,
        future_tokens=future_tokens,
        action_horizon=action_horizon,
    )
    if max_ram_gib < 0:
        raise ValueError(f"max_ram_gib must be >= 0, got {max_ram_gib}")

    paths = sorted(path for path in root.rglob("*.pkl") if path.is_file())
    paths = paths[:max_files] if max_files > 0 else paths
    if not paths:
        raise FileNotFoundError(f"No PKLs under {root}")
    symlinks = [path for path in paths if path.is_symlink()]
    if symlinks:
        raise ValueError(f"PQ3 source inventory must not contain symlink PKLs: {symlinks[0]}")

    cell_counts = Counter(path.relative_to(root).parent.as_posix() for path in paths)
    actual_cell_counts = dict(sorted(cell_counts.items()))
    expected_inventory = (
        dict(sorted(expected_cell_counts.items()))
        if expected_cell_counts is not None
        else None
    )
    if expected_inventory is not None and actual_cell_counts != expected_inventory:
        raise ValueError(
            "PQ3 source inventory mismatch: "
            f"actual={actual_cell_counts}, expected={expected_inventory}"
        )

    chunks: list[torch.Tensor] = []
    reference: tuple[Any, ...] | None = None
    source_dtype: str | None = None
    num_records = 0
    num_rows = 0
    selected_nbytes = 0
    source_files: list[str] = []
    source_inventory: list[dict[str, Any]] = []
    max_ram_bytes = int(max_ram_gib * 1024**3) if max_ram_gib > 0 else 0

    for file_index, path in enumerate(paths, 1):
        relative_path = path.relative_to(root)
        relative_name = relative_path.as_posix()
        source_files.append(relative_name)
        file_row_start = num_rows

        with path.open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 -- gated by trust-pkl.

        kind = str(payload.get("feature_kind") or "")
        axes = tuple(payload.get("feature_axes") or ())
        capture_mode = str(payload.get("capture_token_mode") or "")
        layers = tuple(int(x) for x in (payload.get("capture_layers") or ()))
        hidden_states = payload.get("hidden_states") or ()
        payload_horizon = payload.get("model_action_horizon")
        if not kind or axes != EXPECTED_FEATURE_AXES or not layers or not hidden_states:
            raise ValueError(
                f"{path}: invalid PQ3 metadata; expected axes={EXPECTED_FEATURE_AXES}"
            )
        if expected_feature_kind and kind != expected_feature_kind:
            raise ValueError(f"{path}: feature_kind={kind!r}, expected {expected_feature_kind!r}")
        if layers != DEFAULT_CAPTURE_LAYERS:
            raise ValueError(
                f"{path}: capture_layers={layers}, expected {DEFAULT_CAPTURE_LAYERS}"
            )
        if capture_mode != EXPECTED_CAPTURE_TOKEN_MODE:
            raise ValueError(
                f"{path}: capture_token_mode={capture_mode!r}, "
                f"expected {EXPECTED_CAPTURE_TOKEN_MODE!r}"
            )
        if payload_horizon is None or int(payload_horizon) != action_horizon:
            raise ValueError(
                f"{path}: model_action_horizon={payload_horizon!r}, expected {action_horizon}"
            )
        payload_denoise_steps = payload.get("num_inference_timesteps")
        if (
            payload_denoise_steps is not None
            and int(payload_denoise_steps) != expected_denoise_steps
        ):
            raise ValueError(
                f"{path}: num_inference_timesteps={payload_denoise_steps}, "
                f"expected {expected_denoise_steps}"
            )
        if layer_id not in layers:
            raise ValueError(f"{path}: layer {layer_id} not in {layers}")

        contract = (kind, axes, capture_mode, layers, int(payload_horizon))
        if reference is None:
            reference = contract
        elif contract != reference:
            raise ValueError(f"{path}: activation contract differs from the first PKL")

        layer_position = layers.index(layer_id)
        file_chunks: list[torch.Tensor] = []
        expected_shape = (
            len(layers),
            expected_denoise_steps,
            expected_model_tokens,
            activation_dim,
        )
        for record_index, hidden in enumerate(hidden_states):
            current_dtype = _source_dtype_name(hidden)
            if source_dtype is None:
                source_dtype = current_dtype
            elif current_dtype != source_dtype:
                raise ValueError(
                    f"{path}: hidden_states[{record_index}] dtype={current_dtype}, "
                    f"expected consistent dtype={source_dtype}"
                )

            array = _as_numpy(hidden)
            if array.shape != expected_shape:
                raise ValueError(
                    f"{path}: hidden_states[{record_index}]={array.shape}, expected {expected_shape}"
                )
            action_tokens = np.ascontiguousarray(
                array[layer_position, :, action_start:, :]
            )
            if action_tokens.shape != (
                expected_denoise_steps,
                action_horizon,
                activation_dim,
            ):
                raise AssertionError(f"Internal action slice error: got {action_tokens.shape}")
            if not np.isfinite(action_tokens).all():
                raise ValueError(
                    f"{path}: non-finite layer {layer_id} action-token activation"
                )

            rows = action_tokens.reshape(-1, activation_dim)
            num_rows += int(rows.shape[0])
            selected_nbytes += int(rows.nbytes)
            if materialize:
                # The contiguous action slice is detached from the much
                # larger full-token payload before the payload is released.
                file_chunks.append(torch.from_numpy(rows))

        if materialize:
            chunks.append(torch.cat(file_chunks, dim=0))
            estimated_peak_bytes = selected_nbytes * 2
            if max_ram_bytes and estimated_peak_bytes > max_ram_bytes:
                raise MemoryError(
                    "Selected action-token activations exceed the configured in-memory "
                    f"peak guard: estimated_peak={estimated_peak_bytes / 1024**3:.2f} GiB, "
                    f"limit={max_ram_gib:.2f} GiB. Raise --max-ram-gib only after "
                    "checking host RAM, or implement a bounded shard cache."
                )

        num_records += len(hidden_states)
        source_inventory.append(
            {
                "path": relative_name,
                "num_records": len(hidden_states),
                "row_start": file_row_start,
                "row_stop": num_rows,
            }
        )
        if progress_every > 0 and (
            file_index % progress_every == 0 or file_index == len(paths)
        ):
            mode = "load" if materialize else "audit"
            print(
                f"[{mode}] layer={layer_id} files={file_index}/{len(paths)} "
                f"records={num_records} rows={num_rows}",
                flush=True,
            )

    activations: torch.Tensor | None
    if materialize:
        activations = torch.cat(chunks, dim=0).contiguous()
        if len(activations) != num_rows:
            raise AssertionError(
                f"Materialized rows={len(activations)} but audit counted {num_rows}"
            )
    else:
        activations = None

    assert reference is not None
    kind, axes, capture_mode, layers, payload_horizon = reference
    audit = {
        "format": "groot_n15_robocasa_pq3_action_tokens_v2",
        "source_root": str(root.resolve()),
        "source_files": source_files,
        "source_inventory": source_inventory,
        "num_files": len(paths),
        "cell_counts": actual_cell_counts,
        "expected_cell_counts": expected_inventory,
        "inventory_verified": expected_inventory is not None,
        "num_records": num_records,
        "num_activation_rows": num_rows,
        "feature_kind": kind,
        "feature_axes": list(axes),
        "capture_token_mode": capture_mode,
        "capture_layers": list(layers),
        "physical_layer": layer_id,
        "source_denoising_steps": expected_denoise_steps,
        "denoise_mode": "all_as_independent_rows",
        "source_model_tokens": expected_model_tokens,
        "token_layout": {
            "state_tokens": state_tokens,
            "future_tokens": future_tokens,
            "action_tokens": action_horizon,
        },
        "token_scope": token_scope,
        "action_horizon": payload_horizon,
        "action_token_slice": {
            "start": action_start,
            "stop": expected_model_tokens,
            "semantics": "half_open_model_token_indices",
        },
        "row_order": ["source_file", "record", "denoise_step", "action_token_offset"],
        "activation_dim": activation_dim,
        "source_dtype": source_dtype,
        "resident_dtype": str(activations.dtype) if activations is not None else None,
        "selected_activation_bytes": selected_nbytes,
        "estimated_materialization_peak_bytes": selected_nbytes * 2,
        "materialized": materialize,
    }
    return activations, audit


def save_activation_cache(
    path: Path,
    activations: torch.Tensor,
    source_manifest: dict[str, Any],
) -> None:
    """Save audited action-token rows for transfer to the training host."""

    if activations.ndim != 2:
        raise ValueError(f"Activation cache requires [N,D], got {tuple(activations.shape)}")
    if int(source_manifest.get("num_activation_rows", -1)) != len(activations):
        raise ValueError("Activation rows do not match source manifest")
    if int(source_manifest.get("activation_dim", -1)) != activations.shape[1]:
        raise ValueError("Activation dimension does not match source manifest")
    if not torch.isfinite(activations).all():
        raise ValueError("Activation cache contains non-finite values")

    path = path.resolve()
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
    """Load and validate a cache produced by save_activation_cache()."""

    path = path.resolve()
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
    if tuple(activations.shape)[1] != activation_dim:
        raise ValueError(
            f"{path}: activation_dim={activations.shape[1]}, expected {activation_dim}"
        )

    expected_identity = {
        "physical_layer": layer_id,
        "activation_dim": activation_dim,
        "token_scope": "action",
        "capture_token_mode": EXPECTED_CAPTURE_TOKEN_MODE,
        "feature_kind": DEFAULT_FEATURE_KIND,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path)
    source.add_argument("--activation-cache", type=Path)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument(
        "--export-activation-cache",
        type=Path,
        default=None,
        help="Audit raw PKLs, save selected [N,D] rows, and exit without training.",
    )
    parser.add_argument("--trust-pkl", action="store_true")
    parser.add_argument("--layer", type=int, default=None, help="Omit to train all layers")
    parser.add_argument("--activation-dim", type=int, default=1536)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument(
        "--token-scope",
        choices=("action",),
        default="action",
        help="Primary contract: train on action tokens only.",
    )
    parser.add_argument("--model-tokens", type=int, default=DEFAULT_MODEL_TOKENS)
    parser.add_argument("--state-tokens", type=int, default=DEFAULT_STATE_TOKENS)
    parser.add_argument("--future-tokens", type=int, default=DEFAULT_FUTURE_TOKENS)
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument("--expected-feature-kind", default=DEFAULT_FEATURE_KIND)
    parser.add_argument("--dict-size", type=int, default=0, help="0 means activation_dim")
    parser.add_argument("--sae-k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--run-tag", default="groot_n15_pq3_dit_allk_action16_smoke")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1500)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--allow-partial-inventory",
        action="store_true",
        help=(
            "Skip the exact 5-cell/150-file PQ3 inventory gate. "
            "--max-files also implies a partial inventory."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--max-ram-gib",
        type=float,
        default=32.0,
        help="Guard on estimated peak selected-activation RAM; 0 disables.",
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser


def _train_layer(args, layer: int, save_dir: Path) -> None:
    if args.activation_cache is not None:
        activations, audit = load_activation_cache(
            args.activation_cache,
            layer_id=layer,
            activation_dim=args.activation_dim,
        )
    else:
        materialize = not args.audit_only or args.export_activation_cache is not None
        activations, audit = load_layer_activations(
            args.input_dir,
            layer_id=layer,
            activation_dim=args.activation_dim,
            expected_denoise_steps=args.denoise_steps,
            expected_feature_kind=args.expected_feature_kind,
            max_files=args.max_files,
            progress_every=args.progress_every,
            token_scope=args.token_scope,
            expected_model_tokens=args.model_tokens,
            state_tokens=args.state_tokens,
            future_tokens=args.future_tokens,
            action_horizon=args.action_horizon,
            max_ram_gib=args.max_ram_gib,
            materialize=materialize,
            expected_cell_counts=(
                None
                if args.allow_partial_inventory or args.max_files > 0
                else EXPECTED_PQ3_CELL_COUNTS
            ),
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "groot_source_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[layer {layer}] rows={audit['num_activation_rows']} "
        f"tokens={audit['token_scope']}/{audit['action_horizon']} "
        f"denoise=all/{args.denoise_steps} source_dtype={audit['source_dtype']} "
        f"save_dir={save_dir}",
        flush=True,
    )

    if args.export_activation_cache is not None:
        assert activations is not None
        save_activation_cache(args.export_activation_cache, activations, audit)
        print(
            f"[cache] path={args.export_activation_cache.resolve()} "
            f"bytes={args.export_activation_cache.resolve().stat().st_size}",
            flush=True,
        )
        return
    if args.audit_only:
        return
    assert activations is not None

    from dictionary_learning.training import trainSAE
    from event_sae.train import SAETrainConfig, build_batch_topk_trainer_config

    dict_size = args.dict_size or args.activation_dim
    cfg = SAETrainConfig(
        data_dir=str(args.activation_cache or args.input_dir),
        layer_idx=layer,
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
    )
    trainer_cfg = build_batch_topk_trainer_config(cfg)
    trainer_cfg.update(
        warmup_steps=min(args.warmup_steps, args.steps),
        seed=args.seed,
        lm_name="groot_n15",
    )

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
    if args.input_dir is not None:
        if not args.trust_pkl:
            raise SystemExit("Refusing to load pickle files without --trust-pkl.")
        if not args.input_dir.is_dir():
            raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    else:
        if args.activation_cache is None or not args.activation_cache.is_file():
            raise SystemExit(f"Activation cache does not exist: {args.activation_cache}")
        if args.layer is None:
            raise SystemExit("--activation-cache requires an explicit --layer")

    if args.export_activation_cache is not None:
        if args.input_dir is None:
            raise SystemExit("--export-activation-cache requires --input-dir")
        if args.layer is None:
            raise SystemExit("--export-activation-cache requires an explicit --layer")
        if args.audit_only:
            raise SystemExit("--export-activation-cache already audits; omit --audit-only")

    try:
        _validate_token_contract(
            token_scope=args.token_scope,
            expected_model_tokens=args.model_tokens,
            state_tokens=args.state_tokens,
            future_tokens=args.future_tokens,
            action_horizon=args.action_horizon,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.max_ram_gib < 0:
        raise SystemExit(f"--max-ram-gib must be >= 0, got {args.max_ram_gib}")

    dict_size = args.dict_size or args.activation_dim
    if not 1 <= args.sae_k <= dict_size:
        raise SystemExit(f"--sae-k must be in [1, {dict_size}], got {args.sae_k}")

    layers = DEFAULT_CAPTURE_LAYERS if args.layer is None else (args.layer,)
    for layer in layers:
        save_dir = args.save_dir / f"layer_{layer:02d}" if args.layer is None else args.save_dir
        _train_layer(args, layer, save_dir)


if __name__ == "__main__":
    main()
