"""Emit a `flwr run --run-config` string from a JSON config (T3 launcher helper).

`flwr run --run-config` takes TOML-ish `key='str' key2=42 key3=true` pairs. Building that
by hand in bash is error-prone (quoting, bool spelling), so the launcher drives it from
the SAME JSON the local_runner cross-check reads — one source of truth, no drift.

    python fl_v3/scripts/runconfig.py CONFIG.json [extra-key=value ...]

`extra-key=value` overrides append/replace JSON keys (value re-parsed as int/float/bool/str);
used by the gate to set per-run `experiment-name`.
"""
from __future__ import annotations

import json
import sys


def _fmt(key: str, value) -> str:
    if isinstance(value, bool):
        return f"{key}={'true' if value else 'false'}"
    if isinstance(value, (int, float)):
        return f"{key}={value}"
    if isinstance(value, list):
        # TOML array (e.g. det-class-weights). json.dumps emits valid TOML for numeric/bool/string
        # elements; flwr --run-config parses it as the typed array (NOT a quoted string).
        return f"{key}={json.dumps(value)}"
    return f"{key}='{value}'"


def _parse_scalar(s: str):
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def to_runconfig(cfg: dict) -> str:
    return " ".join(_fmt(k, v) for k, v in cfg.items())


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit("usage: runconfig.py CONFIG.json [key=value ...]")
    with open(argv[0], encoding="utf-8") as f:
        cfg = json.load(f)
    for override in argv[1:]:
        k, _, v = override.partition("=")
        cfg[k] = _parse_scalar(v)
    print(to_runconfig(cfg))


if __name__ == "__main__":
    main(sys.argv[1:])
