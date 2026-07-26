"""OpenVLA activation collection with dependency-light package imports."""

__all__ = [
    "ActivationCollectHandle",
    "apply_collect_hooks",
    "apply_sae_topk_collect_hooks",
]


def __getattr__(name: str):
    if name in __all__:
        from event_sae.openvla import activations as _activations

        return getattr(_activations, name)
    raise AttributeError(f"module 'event_sae.openvla' has no attribute {name!r}")
