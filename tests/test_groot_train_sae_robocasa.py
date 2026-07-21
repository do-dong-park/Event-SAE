from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

from scripts.groot.audit_sae_checkpoint import main as audit_main
from scripts.groot.train_sae_robocasa import (
    DEFAULT_CAPTURE_LAYERS,
    DEFAULT_FEATURE_KIND,
    EXPECTED_PQ3_CELL_COUNTS,
    InMemoryBatchLoader,
    load_activation_cache,
    load_layer_activations,
    main,
    save_activation_cache,
)


def _write_pq3(
    root: Path,
    *,
    capture_token_mode: str = "all_token_full",
    model_action_horizon: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    cell = root / "OpenDrawer" / "pq3_drawer_left"
    cell.mkdir(parents=True)
    base = torch.arange(
        len(DEFAULT_CAPTURE_LAYERS) * 4 * 49 * 3, dtype=torch.float32
    ).reshape(len(DEFAULT_CAPTURE_LAYERS), 4, 49, 3)
    hidden0 = (base.remainder(997) / 100).to(torch.float16)
    hidden1 = ((base + 101).remainder(997) / 100).to(torch.float16)
    payload = {
        "feature_kind": DEFAULT_FEATURE_KIND,
        "feature_axes": ["layer", "denoise_step", "model_token", "feature_dim"],
        "capture_token_mode": capture_token_mode,
        "capture_layers": list(DEFAULT_CAPTURE_LAYERS),
        "model_action_horizon": model_action_horizon,
        "num_inference_timesteps": 4,
        "hidden_states": [hidden0, hidden1],
    }
    with (cell / "task8--ep0--succ0.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    return hidden0, hidden1


def _load(
    root: Path,
    *,
    materialize: bool = True,
    expected_cell_counts: dict[str, int] | None = None,
):
    return load_layer_activations(
        root,
        layer_id=15,
        activation_dim=3,
        expected_denoise_steps=4,
        expected_feature_kind=DEFAULT_FEATURE_KIND,
        max_files=0,
        progress_every=0,
        token_scope="action",
        expected_model_tokens=49,
        state_tokens=1,
        future_tokens=32,
        action_horizon=16,
        max_ram_gib=1,
        materialize=materialize,
        expected_cell_counts=expected_cell_counts,
    )


def test_loads_last_16_action_tokens_without_pooling(tmp_path: Path) -> None:
    hidden0, hidden1 = _write_pq3(tmp_path)
    activations, audit = _load(
        tmp_path,
        expected_cell_counts={"OpenDrawer/pq3_drawer_left": 1},
    )

    expected = torch.cat(
        [
            hidden0[-1, :, 33:, :].reshape(-1, 3),
            hidden1[-1, :, 33:, :].reshape(-1, 3),
        ]
    )
    assert activations is not None
    torch.testing.assert_close(activations, expected)
    assert tuple(activations.shape) == (2 * 4 * 16, 3)
    assert audit["num_activation_rows"] == 2 * 4 * 16
    assert audit["token_scope"] == "action"
    assert audit["action_token_slice"] == {
        "start": 33,
        "stop": 49,
        "semantics": "half_open_model_token_indices",
    }
    assert audit["cell_counts"] == {"OpenDrawer/pq3_drawer_left": 1}
    assert audit["inventory_verified"] is True
    assert audit["source_dtype"] == "float16"
    assert audit["resident_dtype"] == "torch.float16"
    assert audit["source_inventory"][0]["row_stop"] == 128

    batch = next(iter(InMemoryBatchLoader(activations, 32, "cpu", seed=0)))
    assert batch.dtype == torch.float32
    assert tuple(batch.shape) == (32, 3)


def test_audit_only_counts_rows_without_materializing(tmp_path: Path) -> None:
    _write_pq3(tmp_path)
    activations, audit = _load(tmp_path, materialize=False)

    assert activations is None
    assert audit["materialized"] is False
    assert audit["resident_dtype"] is None
    assert audit["num_records"] == 2
    assert audit["num_activation_rows"] == 128


def test_audit_only_cli_writes_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    save_dir = tmp_path / "output"
    _write_pq3(input_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_sae_robocasa.py",
            "--input-dir",
            str(input_dir),
            "--save-dir",
            str(save_dir),
            "--trust-pkl",
            "--layer",
            "15",
            "--activation-dim",
            "3",
            "--sae-k",
            "2",
            "--progress-every",
            "0",
            "--allow-partial-inventory",
            "--audit-only",
        ],
    )

    main()

    manifest = json.loads((save_dir / "groot_source_manifest.json").read_text())
    assert manifest["materialized"] is False
    assert manifest["capture_layers"] == list(DEFAULT_CAPTURE_LAYERS)
    assert manifest["physical_layer"] == 15
    assert manifest["num_activation_rows"] == 128
    assert manifest["inventory_verified"] is False


def test_export_activation_cache_cli_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    save_dir = tmp_path / "export"
    cache_path = save_dir / "l15_action_tokens.pt"
    _write_pq3(input_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_sae_robocasa.py",
            "--input-dir",
            str(input_dir),
            "--save-dir",
            str(save_dir),
            "--export-activation-cache",
            str(cache_path),
            "--trust-pkl",
            "--layer",
            "15",
            "--activation-dim",
            "3",
            "--sae-k",
            "2",
            "--progress-every",
            "0",
            "--allow-partial-inventory",
        ],
    )

    main()

    activations, manifest = load_activation_cache(
        cache_path,
        layer_id=15,
        activation_dim=3,
    )
    assert tuple(activations.shape) == (128, 3)
    assert activations.dtype == torch.float16
    assert manifest["materialized"] is True
    assert not (save_dir / "trainer_0").exists()


def test_strict_inventory_rejects_partial_pq3_source(tmp_path: Path) -> None:
    _write_pq3(tmp_path)
    with pytest.raises(ValueError, match="source inventory mismatch"):
        _load(
            tmp_path,
            materialize=False,
            expected_cell_counts=EXPECTED_PQ3_CELL_COUNTS,
        )


def test_checkpoint_audit_reuses_existing_evaluation_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    cache_path = tmp_path / "l15_action_tokens.pt"
    run_dir = tmp_path / "run"
    trainer_dir = run_dir / "trainer_0"
    _write_pq3(input_dir)
    activations, source_manifest = _load(input_dir)
    assert activations is not None
    save_activation_cache(cache_path, activations, source_manifest)
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
    with np.load(run_dir / "sae_quality_by_feature.npz") as feature_stats:
        assert tuple(feature_stats["firing_count"].shape) == (6,)


@pytest.mark.parametrize(
    ("capture_token_mode", "model_action_horizon", "message"),
    [
        ("action_token_mean", 16, "capture_token_mode"),
        ("all_token_full", 8, "model_action_horizon"),
    ],
)
def test_rejects_non_pq3_or_wrong_action_contract(
    tmp_path: Path,
    capture_token_mode: str,
    model_action_horizon: int,
    message: str,
) -> None:
    _write_pq3(
        tmp_path,
        capture_token_mode=capture_token_mode,
        model_action_horizon=model_action_horizon,
    )
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, materialize=False)
