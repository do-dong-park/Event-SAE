"""OpenPI evaluation API with lazy config and runner exports."""

__all__ = [
    "EnvConfig",
    "LiberoConfig",
    "LoggingConfig",
    "RunConfig",
    "SAECollectConfig",
    "ServerConfig",
    "eval_libero",
    "load_config",
    "parse_overrides",
]

_CONFIG_EXPORTS = {
    "EnvConfig",
    "LiberoConfig",
    "LoggingConfig",
    "RunConfig",
    "SAECollectConfig",
    "ServerConfig",
    "load_config",
    "parse_overrides",
}


def __getattr__(name: str):
    if name in _CONFIG_EXPORTS:
        from event_sae.openpi.eval import config as _config

        return getattr(_config, name)
    if name == "eval_libero":
        from event_sae.openpi.eval import runner as _runner

        return _runner.eval_libero
    raise AttributeError(
        f"module 'event_sae.openpi.eval' has no attribute {name!r}"
    )
