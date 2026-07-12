"""Fail-closed production configuration for fl_v3."""

from .resolved import (
    ConfigError, ResolvedConfig, load_resolved_config, resolve_config,
    verify_physical_data_identities,
)

__all__ = [
    "ConfigError", "ResolvedConfig", "load_resolved_config", "resolve_config",
    "verify_physical_data_identities",
]
