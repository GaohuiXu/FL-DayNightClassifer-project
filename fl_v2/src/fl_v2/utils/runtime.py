"""Runtime helpers shared by server, client, and analysis code.

Two tiny utilities, kept together because they have no dependencies and
are used at the same call sites (config parsing + per-call seeding).
"""
from __future__ import annotations


_TRUTHY = frozenset({"true", "1", "yes", "y", "on"})
_FALSY = frozenset({"false", "0", "no", "n", "off", ""})


def truthy(value, default: bool = False) -> bool:
    """Robust boolean parsing for YAML/Flower config values.

    `bool("false") == True` in Python — and `flwr run --run-config`
    quotes YAML overrides as Python strings — so the natural
    `bool(config.get("canonical-conv1", False))` pattern silently
    misreads any YAML that explicitly writes `false`. This helper
    handles bool, int/float, and the canonical string spellings.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    return default


def derive_seed(run_seed: int, client_id: int = 0, server_round: int = 0) -> int:
    """Deterministic 32-bit seed from (run_seed, client_id, server_round).

    Multiplicative mixing — three coprime multipliers ensure that swaps
    (1,2,0)↔(2,1,0) collide rarely. Caller picks which arguments to vary.
    """
    return (
        int(run_seed) * 100003
        + int(client_id) * 13
        + int(server_round) * 7
    ) & 0xFFFFFFFF
