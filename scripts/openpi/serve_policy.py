"""Serve an openpi (π₀.₅) policy with event-grounded SAE collection
or single-feature intervention.

Direct port of `openpi-event-sae/scripts/serve_policy.py`, slimmed to
four modes:
  * ``--mode baseline`` — serve the unmodified policy with no SAE hook.
  * ``--mode dense`` — dense activation collection (paper-default).
  * ``--mode topk``  — online top-k via ``event_sae.openpi.TopKActivationCollector``.
  * ``--mode intervene`` — install the fork's ``SAEReconstruction``
    hook on a single (capture_target, layer): the activation is run
    through the trained SAE, the listed ``--feature-indices`` are
    scaled by ``--feature-alpha`` (0.0 = hard zero-out, 0.5 = soft
    half-strength, 1.0 = no feature edit). The decoded feature-edit
    delta is mixed into the original residual with ``--recon-alpha``;
    the unedited SAE reconstruction is never substituted for the
    residual. Matches openpi-mech's
    ``rollout_eval_openpi_sae_feature_{zeroout,soft_intervention}_*``
    sweeps.
"""

from __future__ import annotations

import argparse
import atexit
import datetime as dt
import logging
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openpi.models import gemma as _gemma
from openpi.models import pi0_config as _pi0_config
from openpi.policies import policy_config as _policy_config
from openpi.sae_collection import collector as _sae_collector
from openpi.sae_collection import reconstruction as _sae_reconstruction
from openpi.sae_collection import runtime as _sae_runtime
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

from event_sae.openpi.activations import TopKActivationCollector
from event_sae.openpi.eval.config import (
    validate_intervention_layer,
    validate_policy_override_pair,
    validate_sae_intervention_checkpoint,
)


_DEFAULT_POLICY_BY_ENV = {
    "libero": ("pi05_libero", "gs://openpi-assets/checkpoints/pi05_libero"),
}


def _resolve_policy_spec(args) -> tuple[_config.TrainConfig, str]:
    """Mirrors openpi-mech _resolve_policy_spec: returns (train_config, checkpoint_dir)."""
    validate_policy_override_pair(args.config, args.checkpoint_dir)
    if args.config and args.checkpoint_dir:
        train_config = _config.get_config(args.config)
        return train_config, args.checkpoint_dir
    if args.env in _DEFAULT_POLICY_BY_ENV:
        name, default_ckpt = _DEFAULT_POLICY_BY_ENV[args.env]
        train_config = _config.get_config(name)
        return train_config, default_ckpt
    raise ValueError(
        f"No default policy for env={args.env!r}; pass --config and --checkpoint-dir explicitly."
    )


def _target_variant_and_spec(train_config, capture_target: str):
    if capture_target == "action_expert":
        variant = train_config.model.action_expert_variant
    elif capture_target == "paligemma":
        variant = train_config.model.paligemma_variant
    else:
        raise ValueError(f"Unsupported capture_target={capture_target!r}")
    return variant, _sae_collector.CAPTURE_TARGETS[capture_target]


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(p.strip()) for p in raw.split(",") if p.strip())


def _make_collector(args, train_config, checkpoint_dir: str):
    """Build either the stock dense ActivationCollector (mode=dense) or our
    TopKActivationCollector (mode=topk). Returns (collector, sample_kwargs).
    """
    if not isinstance(train_config.model, _pi0_config.Pi0Config):
        raise ValueError("SAE collection currently supports Pi0/Pi0.5 JAX checkpoints only.")

    target_variant, _spec = _target_variant_and_spec(train_config, args.capture_target)
    target_config = _gemma.get_config(target_variant)
    state_token_offset = 0 if train_config.model.pi05 else 1
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{train_config.name}_{args.capture_target}_{timestamp}"
    layer_indices = _parse_int_list(args.layer_indices)
    if args.mode == "topk" and len(layer_indices) != 1:
        raise ValueError(f"mode=topk requires a single --layer-indices entry; got {layer_indices}")

    config = _sae_collector.SAECollectionConfig(
        output_root=args.output_root,
        run_name=run_name,
        append_to_existing=False,
        capture_target=args.capture_target,
        layer_indices=layer_indices,
        flush_every_rows=int(args.flush_every_rows),
        model_type=train_config.model.model_type.value,
        pi05=bool(train_config.model.pi05),
        action_horizon=int(train_config.model.action_horizon),
        action_dim=int(train_config.model.action_dim),
        state_token_offset=state_token_offset,
        action_token_offset=state_token_offset,
        replan_steps=None,
    )
    base_kwargs = dict(
        policy_config_name=train_config.name,
        checkpoint_dir=checkpoint_dir,
        model_depth=target_config.depth,
        d_model=target_config.width,
    )

    if args.mode == "dense":
        collector = _sae_collector.ActivationCollector(config, **base_kwargs)
    elif args.mode == "topk":
        collector = TopKActivationCollector(
            config,
            sae_checkpoint=args.sae_checkpoint,
            topk=int(args.topk),
            rows_per_shard=int(args.rows_per_shard),
            **base_kwargs,
        )
    else:
        raise ValueError(f"Unknown --mode={args.mode!r} (expected dense | topk)")

    # Enable the JAX-side capture hook in openpi.models.gemma. Without
    # this flag, ``io_capture_activation`` is dead code inside ``jax.lax.cond``
    # and the JAX trace never wires up the hook → collector buffer stays
    # empty → ``activation_records=0`` at finalize. (Matches the openpi-mech
    # / upstream openpi-event-sae serve_policy.py.)
    os.environ["OPENPI_ENABLE_SAE_COLLECTION"] = "1"
    _sae_runtime.set_active_collector(collector)
    sample_kwargs = {
        "sae_capture_layer_mask": collector.layer_mask,
        "sae_capture_expert_idx": collector.expert_idx,
    }
    return collector, sample_kwargs


def _make_intervention(args, train_config) -> dict:
    """Install the JAX-side SAEReconstruction hook for ``--mode intervene``.

    Mirrors ``_make_sae_reconstruction`` in the openpi fork's
    ``scripts/serve_policy.py``: load the SAE checkpoint, hand its
    parameters to ``_sae_reconstruction.make_state_from_arrays``,
    register it via ``set_active_reconstruction``, and flip
    ``OPENPI_ENABLE_SAE_RECONSTRUCTION=1`` so the JAX trace wires the
    hook into ``gemma.py``. Returns the server-metadata dict the
    client sees as ``policy.metadata['sae_reconstruction']``.
    """
    import torch

    if not isinstance(train_config.model, _pi0_config.Pi0Config):
        raise ValueError("--mode intervene supports Pi0/Pi0.5 JAX checkpoints only.")

    target_variant, target_spec = _target_variant_and_spec(train_config, args.capture_target)
    target_config = _gemma.get_config(target_variant)
    validate_intervention_layer(
        args.capture_target,
        int(args.layer_idx),
        int(target_config.depth),
    )

    state_dict = torch.load(args.sae_checkpoint, map_location="cpu")
    feature_indices = _parse_int_list(args.feature_indices)
    if not feature_indices and int(args.active_feature_drop_count) <= 0:
        raise ValueError(
            "--mode intervene requires --feature-indices (or --active-feature-drop-count > 0)."
        )
    checkpoint_contract = validate_sae_intervention_checkpoint(
        args.sae_checkpoint,
        capture_target=args.capture_target,
        layer_idx=int(args.layer_idx),
        activation_dim=int(target_config.width),
        state_dict=state_dict,
        feature_indices=feature_indices,
        active_feature_drop_count=int(args.active_feature_drop_count),
    )

    state = _sae_reconstruction.make_state_from_arrays(
        ae_path=args.sae_checkpoint,
        capture_target=args.capture_target,
        expert_idx=target_spec.expert_idx,
        layer_idx=int(args.layer_idx),
        encoder_weight=state_dict["encoder.weight"].detach().cpu().numpy(),
        encoder_bias=state_dict["encoder.bias"].detach().cpu().numpy(),
        decoder_weight=state_dict["decoder.weight"].detach().cpu().numpy(),
        b_dec=state_dict["b_dec"].detach().cpu().numpy(),
        k=int(state_dict["k"].item()),
        threshold=float(state_dict["threshold"].item()),
        alpha=float(args.recon_alpha),
        feature_indices=feature_indices,
        feature_alpha=float(args.feature_alpha),
        active_feature_drop_count=int(args.active_feature_drop_count),
        active_feature_drop_seed=int(args.active_feature_drop_seed),
    )
    _sae_reconstruction.set_active_reconstruction(state)
    os.environ["OPENPI_ENABLE_SAE_RECONSTRUCTION"] = "1"

    return {
        "enabled": True,
        "schema_version": "openpi_sae_reconstruction_v1",
        "ae_path": args.sae_checkpoint,
        "checkpoint_contract": checkpoint_contract,
        "capture_target": args.capture_target,
        "layer_idx": int(args.layer_idx),
        "alpha": float(args.recon_alpha),
        "feature_indices": list(feature_indices),
        "feature_alpha": float(args.feature_alpha),
        "active_feature_drop_count": int(args.active_feature_drop_count),
        "active_feature_drop_seed": int(args.active_feature_drop_seed),
        "model_type": train_config.model.model_type.value,
        "pi05": bool(train_config.model.pi05),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Serve openpi policy with event-grounded SAE collection."
    )
    # Policy
    p.add_argument(
        "--env",
        default="libero",
        choices=sorted(_DEFAULT_POLICY_BY_ENV),
        help="Pick a default policy for this env.",
    )
    p.add_argument("--config", default=None, help="Override default: openpi train-config name.")
    p.add_argument("--checkpoint-dir", default=None, help="Override default: checkpoint dir.")
    p.add_argument("--default-prompt", default=None)
    p.add_argument("--port", type=int, default=8000)
    # SAE collection / intervention
    p.add_argument(
        "--mode",
        default="dense",
        choices=("baseline", "dense", "topk", "intervene"),
    )
    p.add_argument("--output-root", default="logs/openpi/sae_collection")
    p.add_argument("--run-name", default=None)
    p.add_argument(
        "--capture-target",
        default="action_expert",
        choices=("action_expert", "paligemma"),
    )
    p.add_argument(
        "--layer-indices",
        default="0,5,11,17",
        help="Collection: comma-separated layers. For mode=topk, exactly one layer.",
    )
    p.add_argument("--flush-every-rows", type=int, default=50_000)
    # topk + intervene shared
    p.add_argument(
        "--sae-checkpoint",
        default="",
        help="Path to trained SAE ae.pt. Required for mode={topk,intervene}.",
    )
    # topk-only
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--rows-per-shard", type=int, default=20_000)
    # intervene-only
    p.add_argument(
        "--layer-idx",
        type=int,
        default=None,
        help="mode=intervene: which single layer to install the SAE recon hook on.",
    )
    p.add_argument(
        "--feature-indices",
        default="",
        help=(
            "mode=intervene: comma-list of in-range SAE feature ids from "
            "candidates.jsonl (e.g. '12' or '12,37')."
        ),
    )
    p.add_argument(
        "--feature-alpha",
        type=float,
        default=0.0,
        help=(
            "mode=intervene: target-feature scale. 0.0 = hard zero-out, "
            "0.5 = soft half-strength, 1.0 = no feature edit. "
            "Paper uses 0.0 + soft grid {0.25, 0.5, 0.75}."
        ),
    )
    p.add_argument(
        "--recon-alpha",
        type=float,
        default=1.0,
        help=(
            "mode=intervene: intervention-delta strength, "
            "h' = h + alpha * (Dec(z_edited) - Dec(z)). "
            "1.0 applies the full edit; 0.0 disables it."
        ),
    )
    p.add_argument(
        "--active-feature-drop-count",
        type=int,
        default=0,
        help=(
            "mode=intervene: zero out the top-N active features per token "
            "(instead of fixed --feature-indices)."
        ),
    )
    p.add_argument("--active-feature-drop-seed", type=int, default=0)
    return p


def _install_shutdown_handlers(collector) -> None:
    """Register robust shutdown so ``collector.close()`` runs no matter how
    the process exits (normal return, SIGINT, SIGTERM, SystemExit). Why
    we can't just rely on ``try/finally`` around ``serve_forever()``:

    * sbatch's ``kill ${PID}`` sends SIGTERM, which Python does not raise
      as an exception by default — process dies without unwinding.
    * Even ``kill -INT`` (SIGINT) does not reliably propagate through
      Python 3.11's ``asyncio.run`` + websockets stack; we observed it
      hang in cleanup with no ``Collector closed.`` log line, no shards
      on disk, and the asyncio loop stuck after handler dispatch.

    Two layers of belt-and-braces:

    * ``signal.signal(SIGTERM, sys.exit)`` — converts SIGTERM into
      ``SystemExit``, which DOES unwind through ``try/finally`` (and
      fires registered ``atexit`` handlers).
    * ``atexit.register(collector.close)`` — runs on any normal Python
      shutdown (return, SystemExit, KeyboardInterrupt that bubbles to
      the top, even ``sys.exit``). Only SIGKILL bypasses atexit.

    ``collector=None`` covers the ``--mode intervene`` path: no rows to
    flush, but we still clear the reconstruction runtime + env vars.
    """
    def _close_once():
        if collector is not None and not getattr(collector, "_event_sae_closed", False):
            try:
                collector.close()
            finally:
                setattr(collector, "_event_sae_closed", True)
        _sae_runtime.clear_active_collector()
        _sae_reconstruction.clear_active_reconstruction()
        os.environ.pop("OPENPI_ENABLE_SAE_COLLECTION", None)
        os.environ.pop("OPENPI_ENABLE_SAE_RECONSTRUCTION", None)

    atexit.register(_close_once)
    signal.signal(signal.SIGTERM, lambda _s, _f: sys.exit(0))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    args = _build_parser().parse_args()
    if args.mode == "topk" and not args.sae_checkpoint:
        raise SystemExit("--mode=topk requires --sae-checkpoint")
    if args.mode == "intervene":
        if not args.sae_checkpoint:
            raise SystemExit("--mode=intervene requires --sae-checkpoint")
        if args.layer_idx is None:
            raise SystemExit("--mode=intervene requires --layer-idx")

    train_config, checkpoint_dir = _resolve_policy_spec(args)

    collector = None
    recon_metadata = None
    if args.mode == "intervene":
        recon_metadata = _make_intervention(args, train_config)
        sample_kwargs = None
    elif args.mode == "baseline":
        sample_kwargs = None
    else:
        collector, sample_kwargs = _make_collector(args, train_config, checkpoint_dir)

    _install_shutdown_handlers(collector)
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
        sample_kwargs=sample_kwargs,
    )
    metadata = dict(policy.metadata)
    if recon_metadata is not None:
        metadata["sae_reconstruction"] = recon_metadata
        logging.info("Enabled SAE reconstruction: %s", recon_metadata)
    elif collector is not None:
        metadata["sae_collection"] = collector.server_metadata()
        logging.info("Enabled SAE collection: %s", metadata["sae_collection"])
    else:
        logging.info("Serving no-hook baseline policy.")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
