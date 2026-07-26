"""OpenVLA evaluation API with lazy config and runner exports."""

__all__ = [
    "EnvConfig",
    "EvalResult",
    "LoggingConfig",
    "ModelConfig",
    "RunConfig",
    "SAECollectConfig",
    "eval_libero",
    "load_config",
    "parse_overrides",
    "resolve_task_ids",
]

_CONFIG_EXPORTS = {
    "EnvConfig",
    "LoggingConfig",
    "ModelConfig",
    "RunConfig",
    "SAECollectConfig",
    "load_config",
    "parse_overrides",
    "resolve_task_ids",
}
_RUNNER_EXPORTS = {"EvalResult", "eval_libero"}


def __getattr__(name: str):
    if name in _CONFIG_EXPORTS:
        from event_sae.openvla.eval import config as _config

        return getattr(_config, name)
    if name in _RUNNER_EXPORTS:
        from event_sae.openvla.eval import runner as _runner

        return getattr(_runner, name)
    raise AttributeError(
        f"module 'event_sae.openvla.eval' has no attribute {name!r}"
    )
