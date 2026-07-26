from __future__ import annotations

import json
from pathlib import Path

import torch

import event_sae.groot.transfer_topk as transfer_module
from event_sae.groot.transfer_topk import encode_transfer_activation_shards
from event_sae.scoring.score_matrix import open_sparse_topk_artifact


class _FakeSae:
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros((len(batch), 2), device=batch.device)
        return torch.cat((torch.relu(batch), zeros), dim=1)


def test_transfer_topk_records_distinct_training_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    shard_dir = tmp_path / "source"
    shard_dir.mkdir()
    dense = torch.tensor([[1.0, -1.0], [2.0, 3.0]])
    torch.save(
        {
            "format": "groot_n15_pq3_action_activation_shard_v1",
            "activations": dense,
        },
        shard_dir / "layer_15_shard_000000.pt",
    )
    source_manifest = {
        "format": "groot_n15_robocasa_pq3_action_tokens_v2",
        "source_inventory": [
            {
                "path": "task/task5--ep0--succ1.pkl",
                "num_records": 1,
                "row_start": 0,
                "row_stop": 2,
            }
        ],
        "num_records": 1,
        "num_activation_rows": 2,
        "physical_layer": 15,
        "activation_dim": 2,
        "source_denoising_steps": 1,
        "action_horizon": 2,
        "token_scope": "action",
        "row_order": [
            "source_file",
            "record",
            "denoise_step",
            "action_token_offset",
        ],
        "inventory_verified": False,
        "source_model_tokens": 4,
        "action_token_slice": {"start": 2, "stop": 4},
        "cache_layout": {
            "shards": [
                {
                    "path": "layer_15_shard_000000.pt",
                    "source_path": "task/task5--ep0--succ1.pkl",
                    "num_rows": 2,
                }
            ]
        },
    }
    (shard_dir / "groot_source_manifest.json").write_text(
        json.dumps(source_manifest),
        encoding="utf-8",
    )
    trajectory_manifest = tmp_path / "trajectory.json"
    trajectory_manifest.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "source_file": "task/task5--ep0--succ1.pkl",
                        "episode_num": 0,
                        "task_id": 5,
                        "task_description": "task",
                        "num_records": 1,
                        "n_action_steps": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "ae.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    (checkpoint_dir / "groot_source_manifest.json").write_text(
        json.dumps({"different": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        transfer_module,
        "load_batch_topk_sae",
        lambda *_args, **_kwargs: (
            _FakeSae(),
            {"trainer": {"dict_size": 4, "activation_dim": 2}},
        ),
    )

    output = tmp_path / "topk"
    result = encode_transfer_activation_shards(
        activation_shard_dir=shard_dir,
        trajectory_manifest_path=trajectory_manifest,
        sae_checkpoint=checkpoint,
        output_dir=output,
        activation_dim=2,
        denoise_steps=1,
        action_horizon=2,
        executed_action_steps=2,
        topk=2,
        batch_size=2,
        device="cpu",
        expected_sources=1,
        progress_every=0,
    )

    assert result["transfer_encoding"] is True
    assert result["checkpoint_source_match"] is False
    artifact = open_sparse_topk_artifact(output)
    assert artifact.manifest["source_manifest_comparison"] == (
        "transfer_encoding_intentionally_distinct"
    )
    _, payload = next(artifact.iter_shards())
    assert payload["episode_num"].tolist() == [0, 0]
    assert payload["step_in_episode"].tolist() == [0, 1]
    assert payload["top_feature_ids"].shape == (2, 2)
