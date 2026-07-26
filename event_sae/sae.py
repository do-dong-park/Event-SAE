"""Backend-neutral SAE classes and checkpoint loading for Event-SAE."""

import json
from pathlib import Path

from event_sae import resolve_groot_artifact_path


def load_batch_topk_sae(checkpoint_path: Path, device: str):
    """Load a ``BatchTopKSAE`` from a ``trainSAE`` checkpoint directory."""

    from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

    checkpoint_path = resolve_groot_artifact_path(checkpoint_path)
    trainer_dir = (
        checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path
    )
    config_path = trainer_dir / "config.json"
    if not config_path.is_file():
        config_path = trainer_dir.parent / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    trainer_cfg = config["trainer"]
    if trainer_cfg.get("dict_class") != "BatchTopKSAE":
        raise ValueError(
            "Expected BatchTopKSAE checkpoint, got "
            f"dict_class={trainer_cfg.get('dict_class')!r}"
        )
    ae_path = (
        checkpoint_path
        if checkpoint_path.is_file()
        else trainer_dir / "ae.pt"
    )
    sae = BatchTopKSAE.from_pretrained(
        str(ae_path),
        k=int(trainer_cfg["k"]),
        device=device,
    )
    sae.eval()
    return sae, config


__all__ = ["SAE", "BatchTopKSAE", "load_batch_topk_sae"]


def __getattr__(name: str):
    """Load dictionary-learning classes only when callers request them."""

    if name == "SAE":
        from dictionary_learning import AutoEncoder

        return AutoEncoder
    if name == "BatchTopKSAE":
        from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

        return BatchTopKSAE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
