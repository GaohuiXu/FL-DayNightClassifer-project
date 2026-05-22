"""Wandb integration for federated training runs.

Server-side only — never invoked from client / Ray actor code. The single
choke point is :class:`WandbLogger`, instantiated by :class:`ExperimentLogger`
once at server startup. All wandb interactions go through that instance, so
strategy classes never import wandb directly.

Design choices:
- ``wandb-enabled = false`` (or ``wandb-mode = "disabled"``) → every method
  becomes a no-op; the rest of the codebase doesn't need to branch on it.
- ``WANDB_DIR`` is forced to ``<exp_dir>/wandb/`` so wandb's own files live
  with the experiment artifacts and can be deleted along with the rest.
- ``WANDB_MODE`` from the environment overrides the config value, so SLURM
  jobs can be flipped between online/offline without editing YAML.
- Tags + project + group are derived from existing config keys
  (``cycle``, ``phase``, ``attack-type``, ``defense-type``,
  ``num-malicious-nodes``, ``model-type``, ``seed``) so YAMLs don't need to
  duplicate that information.
"""
from __future__ import annotations

import os
import re
from typing import Any

from fl_v2.utils.runtime import truthy

# Strategy / defense names that appear as suffixes in experiment names.
# Used to strip them from the auto-derived wandb group name so e.g.
# ``phaseD-modelrep-15mal-nodefense`` and ``phaseD-modelrep-15mal-fedmedian``
# both map to the group ``phaseD-modelrep``.
_GROUP_SUFFIX_TOKENS = {
    "nodefense",
    "fedmedian",
    "fedtrimmedavg",
    "krum",
    "multikrum",
    "bulyan",
    "normclipped",
}
_MAL_RE = re.compile(r"^\d+mal$")


def _is_disabled(run_config) -> bool:
    enabled = truthy(run_config.get("wandb-enabled", False))
    mode = str(run_config.get("wandb-mode", "online")).strip().lower()
    return (not enabled) or mode == "disabled"


def _resolve_mode(run_config) -> str:
    """Env wins; then config; default online."""
    env = os.environ.get("WANDB_MODE", "").strip().lower()
    if env in ("online", "offline", "disabled"):
        return env
    cfg = str(run_config.get("wandb-mode", "online")).strip().lower()
    return cfg if cfg in ("online", "offline", "disabled") else "online"


def _derive_project(run_config) -> str:
    explicit = str(run_config.get("wandb-project", "")).strip()
    if explicit:
        return explicit
    cycle = str(run_config.get("cycle", "")).strip()
    return f"gtsrb-{cycle.replace('_', '-')}" if cycle else "gtsrb"


def _derive_group(run_config) -> str:
    explicit = str(run_config.get("wandb-group", "")).strip()
    if explicit:
        return explicit
    name = str(run_config.get("experiment-name", "exp"))
    parts = name.split("-")
    while parts and (parts[-1].lower() in _GROUP_SUFFIX_TOKENS or _MAL_RE.match(parts[-1].lower())):
        parts.pop()
    return "-".join(parts) if parts else name


def _derive_tags(run_config) -> list[str]:
    tags: list[str] = []

    def add(value: Any) -> None:
        s = str(value).strip()
        if s and s.lower() not in ("none", "0", "false"):
            tags.append(s)
        elif s.lower() == "none":
            tags.append("attack-none" if "attack" in tags[-1:] else s)

    cycle = str(run_config.get("cycle", "")).strip()
    phase = str(run_config.get("phase", "")).strip()
    if cycle:
        tags.append(cycle)
    if phase:
        tags.append(phase)

    model_type = str(run_config.get("model-type", "")).strip()
    if model_type:
        tags.append(model_type)

    attack_type = str(run_config.get("attack-type", "none")).strip()
    tags.append(f"attack:{attack_type}")

    defense_type = str(run_config.get("defense-type", "none")).strip()
    tags.append(f"defense:{defense_type}")

    num_mal = int(run_config.get("num-malicious-nodes", 0))
    if num_mal > 0:
        tags.append(f"{num_mal}mal")

    seed = run_config.get("seed", None)
    if seed is not None:
        tags.append(f"seed{int(seed)}")

    # Cycle 02 onward — surface fine-tuning regime so the wandb dashboard
    # can filter `pretrained AND trainable:head_only AND 5mal` to one cell.
    if truthy(run_config.get("pretrained-init", False)):
        tags.append("pretrained")
    trainable_layers = str(run_config.get("trainable-layers", "full_ft")).strip()
    if trainable_layers and trainable_layers != "full_ft":
        tags.append(f"trainable:{trainable_layers}")
    if truthy(run_config.get("canonical-conv1", False)):
        tags.append("canonical_conv1")

    extra = str(run_config.get("wandb-tags", "")).strip()
    if extra:
        tags.extend([t.strip() for t in extra.split(",") if t.strip()])

    # Dedupe while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in tags:
        if t not in seen:
            unique.append(t)
            seen.add(t)
    return unique


def _config_to_dict(run_config) -> dict[str, Any]:
    """Best-effort conversion of Flower's ConfigRecord-ish object to a plain dict.

    wandb's ``config=`` accepts dict-like objects but rejects nested non-JSON
    values. Cast everything we can to str/int/float/bool; drop the rest.
    """
    out: dict[str, Any] = {}
    try:
        items = dict(run_config).items()
    except Exception:
        return out
    for k, v in items:
        key = str(k)
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
        else:
            try:
                out[key] = str(v)
            except Exception:
                continue
    return out


class WandbLogger:
    """Central wandb sink. No-op when disabled."""

    def __init__(self, run_config, exp_dir: str) -> None:
        self.disabled = _is_disabled(run_config)
        self.run = None
        if self.disabled:
            print("[wandb] disabled (wandb-enabled=false or wandb-mode=disabled)", flush=True)
            return

        # Lazy import so disabled runs don't pay the import cost.
        import wandb

        mode = _resolve_mode(run_config)
        if mode == "disabled":
            self.disabled = True
            print("[wandb] disabled via WANDB_MODE=disabled", flush=True)
            return

        # Keep wandb's filesystem state inside the experiment dir so it
        # travels with the rest of the artifacts.
        wandb_dir = os.path.join(exp_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir

        seed = int(run_config.get("seed", 0))
        exp_name = str(run_config.get("experiment-name", "exp"))
        run_name = f"{exp_name}_seed{seed}"

        # wandb is a monitoring side-channel — its service failing to
        # start (e.g. ServicePollForTokenError when the wandb-core
        # subprocess cannot write its port file within the timeout) must
        # never kill a multi-hour FL run. Degrade gracefully on any
        # failure: continue with wandb disabled. The experiment's primary
        # outputs (norm_log.json, summary.json, rounds.csv, checkpoints)
        # do not depend on wandb.
        #
        # x_service_wait: wandb's 30 s default deadline for the wandb-core
        # service subprocess to come up is too tight on a contended FL
        # simulation node (50 Ray actors + SuperLink + SuperExec all
        # spawning at once) — the freshly-forked wandb-core loses the CPU
        # race and the port-file poll times out (observed: job 6665514).
        # Raise it to 120 s. It is a deadline, not a fixed wait: a fast
        # start still returns immediately, so the only cost is a longer
        # ceiling before the graceful-degrade path above triggers.
        try:
            self.run = wandb.init(
                project=_derive_project(run_config),
                group=_derive_group(run_config),
                name=run_name,
                tags=_derive_tags(run_config),
                mode=mode,
                dir=wandb_dir,
                config=_config_to_dict(run_config),
                reinit=True,
                settings=wandb.Settings(x_service_wait=120.0),
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.disabled = True
            self.run = None
            print(
                f"[wandb] init FAILED ({e!r}) — continuing with wandb "
                f"disabled; experiment outputs are unaffected.",
                flush=True,
            )
            return

        # Use server-round as the x-axis for all per-round metrics.
        try:
            self.run.define_metric("server-round")
            self.run.define_metric("server/*", step_metric="server-round")
            self.run.define_metric("client/*", step_metric="server-round")
            self.run.define_metric("updates/*", step_metric="server-round")
        except Exception as e:
            print(f"[wandb] define_metric warning: {e!r}", flush=True)

        print(
            f"[wandb] init mode={mode} project={self.run.project} "
            f"group={self.run.group} name={self.run.name} "
            f"id={self.run.id}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_round(
        self,
        server_round: int,
        server_metrics: dict | None,
        client_metrics: dict | None = None,
        update_metrics: dict | None = None,
    ) -> None:
        if self.disabled or self.run is None:
            return
        payload: dict[str, Any] = {"server-round": int(server_round)}
        if server_metrics:
            for k, v in server_metrics.items():
                if isinstance(v, (int, float)) and k != "server-round":
                    payload[f"server/{k}"] = float(v)
        if client_metrics:
            for k, v in client_metrics.items():
                if isinstance(v, (int, float)):
                    payload[f"client/{k}"] = float(v)
        # Cycle-02 gradient-space metrics: malicious-vs-honest update
        # summaries computed server-side from the strategy's norm log.
        if update_metrics:
            for k, v in update_metrics.items():
                if isinstance(v, (int, float)):
                    payload[f"updates/{k}"] = float(v)
        try:
            self.run.log(payload)
        except Exception as e:
            print(f"[wandb] log_round warning: {e!r}", flush=True)

    def log_image(self, name: str, path: str) -> None:
        if self.disabled or self.run is None:
            return
        if not os.path.exists(path):
            print(f"[wandb] log_image skipped (missing): {path}", flush=True)
            return
        try:
            import wandb
            self.run.log({name: wandb.Image(path)})
        except Exception as e:
            print(f"[wandb] log_image warning: {e!r}", flush=True)

    def log_summary(self, summary_dict: dict | None) -> None:
        if self.disabled or self.run is None or not summary_dict:
            return
        try:
            self.run.summary.update(_flatten_summary(summary_dict))
        except Exception as e:
            print(f"[wandb] log_summary warning: {e!r}", flush=True)

    def finalize(self) -> None:
        if self.disabled or self.run is None:
            return
        try:
            self.run.finish()
        except Exception as e:
            print(f"[wandb] finalize warning: {e!r}", flush=True)
        self.run = None


def _flatten_summary(summary: dict, prefix: str = "summary") -> dict[str, Any]:
    """Flatten a nested summary dict into wandb-friendly scalar keys.

    Example:
        {"final": {"test_accuracy": 0.9}, "best_test_accuracy": {"round": 50, "test_accuracy": 0.92}}
        → {"summary/final/test_accuracy": 0.9,
           "summary/best_test_accuracy/round": 50,
           "summary/best_test_accuracy/test_accuracy": 0.92}
    """
    flat: dict[str, Any] = {}
    for k, v in summary.items():
        key = f"{prefix}/{k}"
        if isinstance(v, dict):
            flat.update(_flatten_summary(v, key))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            flat[key] = v
        else:
            flat[key] = str(v)
    return flat
