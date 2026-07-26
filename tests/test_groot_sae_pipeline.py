from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

from event_sae.groot.activations import (
    InMemoryActivationBatchLoader,
    REQUIRED_FEATURE_KIND,
    load_activation_cache,
    save_activation_cache,
)
from scripts.groot.audit_sae_checkpoint import (
    _calibrate_inference_threshold,
    main as audit_main,
)
from scripts.groot.export_pq3_activation_shards import (
    CAPTURE_LAYERS,
    main as export_shards_main,
)
from event_sae.groot.train_from_activation_cache import _build_parser, main
from event_sae.openvla.activations import (
    load_batch_topk_sae as legacy_load_batch_topk_sae,
)
from event_sae.sae import load_batch_topk_sae
from event_sae.train import SAETrainConfig, build_batch_topk_trainer_config, train_sae


def test_openvla_checkpoint_loader_is_a_compatibility_reexport() -> None:
    from event_sae.sae import BatchTopKSAE as ReexportedBatchTopKSAE

    assert legacy_load_batch_topk_sae is load_batch_topk_sae
    assert ReexportedBatchTopKSAE is BatchTopKSAE


def test_recalibrated_threshold_matches_target_average_l0() -> None:
    torch.manual_seed(0)
    sae = BatchTopKSAE(activation_dim=3, dict_size=6, k=2)
    activations = torch.randn(32, 3)

    threshold, evaluated_rows = _calibrate_inference_threshold(
        sae,
        activations,
        calibration_rows=32,
        device="cpu",
        seed=0,
    )

    assert evaluated_rows == 32
    assert threshold == pytest.approx(float(sae.threshold.item()))
    assert int((sae.encode(activations) != 0).sum().item()) == 32 * 2


def test_parser_exposes_batch_topk_schedule_controls() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--activation-cache",
            "cache.pt",
            "--save-dir",
            "out",
            "--layer",
            "15",
            "--decay-start-steps",
            "240",
            "--threshold-start-steps",
            "25",
        ]
    )
    assert args.decay_start_steps == 240
    assert args.threshold_start_steps == 25


def test_shared_train_runtime_accepts_injected_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    injected_loader = [torch.zeros(8, 3)]

    def fake_train_sae(**kwargs) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("event_sae.train.trainSAE", fake_train_sae)
    cfg = SAETrainConfig(
        data_dir="unused-for-injected-loader",
        layer_idx=15,
        activation_dim=3,
        dict_size=6,
        k=2,
        lr=1e-4,
        steps=1200,
        batch_size=8,
        run_tag="groot-test",
        submodule_name="dit_block_residual_action_tokens",
        device="cpu",
        warmup_steps=100,
        decay_start_step=960,
        threshold_start_step=1000,
        seed=7,
        save_every=1200,
        log_steps=100,
        lm_name="groot_n15",
        use_wandb_env=False,
    )

    trainer_cfg = build_batch_topk_trainer_config(cfg)
    train_sae(cfg, str(tmp_path), data=injected_loader)

    assert trainer_cfg["warmup_steps"] == 100
    assert trainer_cfg["decay_start"] == 960
    assert trainer_cfg["threshold_start_step"] == 1000
    assert trainer_cfg["seed"] == 7
    assert trainer_cfg["lm_name"] == "groot_n15"
    assert calls[0]["data"] is injected_loader
    assert calls[0]["trainer_configs"] == [trainer_cfg]
    assert calls[0]["save_steps"] == [1200]
    assert calls[0]["log_steps"] == 100
    assert calls[0]["normalize_activations"] is True
    assert calls[0]["device"] == "cpu"


def _write_pq3(
    root: Path,
    *,
    capture_token_mode: str = "all_token_full",
    model_action_horizon: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    cell = root / "OpenDrawer" / "pq3_drawer_left"
    cell.mkdir(parents=True)
    base = torch.arange(
        len(CAPTURE_LAYERS) * 4 * 49 * 3, dtype=torch.float32
    ).reshape(len(CAPTURE_LAYERS), 4, 49, 3)
    hidden0 = (base.remainder(997) / 100).to(torch.float16)
    hidden1 = ((base + 101).remainder(997) / 100).to(torch.float16)
    payload = {
        "feature_kind": REQUIRED_FEATURE_KIND,
        "feature_axes": ["layer", "denoise_step", "model_token", "feature_dim"],
        "capture_token_mode": capture_token_mode,
        "capture_layers": list(CAPTURE_LAYERS),
        "model_action_horizon": model_action_horizon,
        "num_inference_timesteps": 4,
        "hidden_states": [hidden0, hidden1],
    }
    with (cell / "task8--ep0--succ0.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    return hidden0, hidden1


def _write_activation_cache(path: Path) -> tuple[torch.Tensor, dict]:
    activations = (
        torch.arange(128 * 3, dtype=torch.float32).reshape(128, 3) / 100
    ).to(torch.float16)
    source_manifest = {
        "format": "groot_n15_robocasa_pq3_action_tokens_v2",
        "source_root": "/synthetic/pq3",
        "source_files": ["OpenDrawer/pq3_drawer_left/task8--ep0--succ0.pkl"],
        "source_inventory": [
            {
                "path": "OpenDrawer/pq3_drawer_left/task8--ep0--succ0.pkl",
                "num_records": 2,
                "row_start": 0,
                "row_stop": len(activations),
            }
        ],
        "num_files": 1,
        "cell_counts": {"OpenDrawer/pq3_drawer_left": 1},
        "num_records": 2,
        "num_activation_rows": len(activations),
        "feature_kind": REQUIRED_FEATURE_KIND,
        "feature_axes": [
            "layer",
            "denoise_step",
            "model_token",
            "feature_dim",
        ],
        "capture_token_mode": "all_token_full",
        "capture_layers": list(CAPTURE_LAYERS),
        "physical_layer": 15,
        "source_denoising_steps": 4,
        "source_model_tokens": 49,
        "token_scope": "action",
        "action_horizon": 16,
        "action_token_slice": {
            "start": 33,
            "stop": 49,
            "semantics": "half_open_model_token_indices",
        },
        "activation_dim": 3,
        "source_dtype": "float16",
        "resident_dtype": "torch.float16",
    }
    save_activation_cache(path, activations, source_manifest)
    return activations, source_manifest


def test_groot_cli_delegates_schedule_to_shared_training_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "l15_action_tokens.pt"
    save_dir = tmp_path / "run"
    _write_activation_cache(cache_path)
    calls: list[tuple[SAETrainConfig, str, object]] = []

    def fake_train_sae(cfg, save_dir, *, data) -> None:
        calls.append((cfg, save_dir, data))

    monkeypatch.setattr("event_sae.train.train_sae", fake_train_sae)
    monkeypatch.setattr(
        sys,
        "argv",
        [
        "event_sae.groot.train_from_activation_cache",
            "--activation-cache",
            str(cache_path),
            "--save-dir",
            str(save_dir),
            "--layer",
            "15",
            "--activation-dim",
            "3",
            "--dict-size",
            "6",
            "--sae-k",
            "2",
            "--steps",
            "12",
            "--batch-size",
            "32",
            "--warmup-steps",
            "2",
            "--decay-start-steps",
            "9",
            "--threshold-start-steps",
            "3",
            "--seed",
            "7",
            "--save-every",
            "12",
            "--log-steps",
            "4",
            "--device",
            "cpu",
        ],
    )

    main()

    cfg, called_save_dir, loader = calls[0]
    assert called_save_dir == str(save_dir)
    assert isinstance(loader, InMemoryActivationBatchLoader)
    assert cfg.warmup_steps == 2
    assert cfg.decay_start_step == 9
    assert cfg.threshold_start_step == 3
    assert cfg.seed == 7
    assert cfg.save_every == 12
    assert cfg.log_steps == 4
    assert cfg.lm_name == "groot_n15"
    assert cfg.use_wandb_env is False
    contract = json.loads((save_dir / "training_contract.json").read_text())
    assert contract["warmup_steps"] == 2
    assert contract["decay_start_step"] == 9
    assert contract["threshold_start_step"] == 3
    assert contract["seed"] == 7
    source_manifest = json.loads(
        (save_dir / "groot_source_manifest.json").read_text()
    )
    assert source_manifest["num_activation_rows"] == 128


def test_streaming_shard_export_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    shard_dir = tmp_path / "shards"
    cache_path = tmp_path / "l15_action_tokens.pt"
    hidden0, hidden1 = _write_pq3(input_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_pq3_activation_shards.py",
            "export",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(shard_dir),
            "--trust-pkl",
            "--activation-dim",
            "3",
            "--allow-partial-inventory",
            "--progress-every",
            "0",
        ],
    )
    export_shards_main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_pq3_activation_shards.py",
            "merge",
            "--shard-dir",
            str(shard_dir),
            "--output-cache",
            str(cache_path),
            "--progress-every",
            "0",
        ],
    )
    export_shards_main()

    activations, manifest = load_activation_cache(
        cache_path,
        layer_id=15,
        activation_dim=3,
    )
    expected = torch.cat(
        [
            hidden0[-1, :, 33:, :].reshape(-1, 3),
            hidden1[-1, :, 33:, :].reshape(-1, 3),
        ]
    )
    torch.testing.assert_close(activations, expected)
    assert manifest["num_files"] == 1
    assert manifest["cache_layout"]["num_shards"] == 1
    assert manifest["inventory_verified"] is False


def test_strict_inventory_rejects_partial_pq3_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    _write_pq3(input_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_pq3_activation_shards.py",
            "export",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(tmp_path / "shards"),
            "--trust-pkl",
            "--activation-dim",
            "3",
            "--progress-every",
            "0",
        ],
    )
    with pytest.raises(ValueError, match="source inventory mismatch"):
        export_shards_main()


def test_checkpoint_audit_reuses_existing_evaluation_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "l15_action_tokens.pt"
    run_dir = tmp_path / "run"
    trainer_dir = run_dir / "trainer_0"
    _activations, source_manifest = _write_activation_cache(cache_path)
    trainer_dir.mkdir(parents=True)
    (run_dir / "groot_source_manifest.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )

    sae = BatchTopKSAE(activation_dim=3, dict_size=6, k=2)
    sae.threshold.fill_(0.0)
    torch.save(sae.state_dict(), trainer_dir / "ae.pt")
    (trainer_dir / "config.json").write_text(
        json.dumps(
            {
                "trainer": {
                    "dict_class": "BatchTopKSAE",
                    "activation_dim": 3,
                    "dict_size": 6,
                    "k": 2,
                    "layer": 15,
                    "lm_name": "groot_n15",
                    "submodule_name": "dit_block_residual_action_tokens",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_sae_checkpoint.py",
            "--activation-cache",
            str(cache_path),
            "--sae-checkpoint",
            str(trainer_dir / "ae.pt"),
            "--activation-dim",
            "3",
            "--batch-size",
            "32",
            "--max-audit-rows",
            "64",
            "--device",
            "cpu",
        ],
    )

    audit_main()

    quality = json.loads((run_dir / "sae_quality.json").read_text())
    assert quality["evaluation_backend"] == "dictionary_learning.evaluation.evaluate"
    assert quality["sample"]["evaluated_rows"] == 64
    assert quality["source"]["action_token_slice"]["start"] == 33
    assert quality["metrics"]["input_finite"] is True
    assert "frac_variance_explained" in quality["metrics"]
    assert quality["threshold_calibration"]["mode"] == (
        "checkpoint_saved_threshold"
    )
    assert quality["checkpoint"]["saved_threshold"] == 0.0
    with np.load(run_dir / "sae_quality_by_feature.npz") as feature_stats:
        assert tuple(feature_stats["firing_count"].shape) == (6,)


@pytest.mark.parametrize("existing_kind", ["json", "npz"])
def test_checkpoint_audit_preflights_paired_outputs_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_kind: str,
) -> None:
    cache_path = tmp_path / "activation_cache.pt"
    checkpoint_path = tmp_path / "run" / "trainer_0" / "ae.pt"
    output_path = tmp_path / "outputs" / "quality.json"
    feature_output_path = tmp_path / "outputs" / "quality_features.npz"
    cache_path.touch()
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.touch()
    output_path.parent.mkdir(parents=True)
    existing_path = output_path if existing_kind == "json" else feature_output_path
    missing_path = feature_output_path if existing_kind == "json" else output_path
    sentinel = f"existing-{existing_kind}".encode()
    existing_path.write_bytes(sentinel)

    def unexpected_checkpoint_load(*args, **kwargs):
        raise AssertionError("checkpoint load must not run after output preflight fails")

    monkeypatch.setattr(
        "scripts.groot.audit_sae_checkpoint.load_batch_topk_sae",
        unexpected_checkpoint_load,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_sae_checkpoint.py",
            "--activation-cache",
            str(cache_path),
            "--sae-checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_path),
            "--feature-output",
            str(feature_output_path),
            "--device",
            "cpu",
        ],
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit_main()

    assert existing_path.read_bytes() == sentinel
    assert not missing_path.exists()


@pytest.mark.parametrize(
    ("capture_token_mode", "model_action_horizon", "message"),
    [
        ("action_token_mean", 16, "capture_token_mode"),
        ("all_token_full", 8, "model_action_horizon"),
    ],
)
def test_rejects_non_pq3_or_wrong_action_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_token_mode: str,
    model_action_horizon: int,
    message: str,
) -> None:
    input_dir = tmp_path / "input"
    _write_pq3(
        input_dir,
        capture_token_mode=capture_token_mode,
        model_action_horizon=model_action_horizon,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_pq3_activation_shards.py",
            "export",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(tmp_path / "shards"),
            "--trust-pkl",
            "--activation-dim",
            "3",
            "--allow-partial-inventory",
            "--progress-every",
            "0",
        ],
    )
    with pytest.raises(ValueError, match=message):
        export_shards_main()
