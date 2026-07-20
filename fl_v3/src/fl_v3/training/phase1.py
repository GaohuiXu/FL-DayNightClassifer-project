"""Exact optimizer/schedule construction for S10 Phase I.

The scheduler mirrors MMCV 1.4's iteration-based CyclicLr/CyclicMomentum hooks,
but advances only after an accepted accumulated optimizer update.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import torch

from fl_v3.config import ResolvedConfig


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rule_matches(name: str, rule: Mapping[str, Any]) -> bool:
    prefix = str(rule["prefix"])
    contains = tuple(str(value) for value in rule["contains_any"])
    return name.startswith(prefix) and (
        not contains or any(fragment in name for fragment in contains)
    )


def build_phase1_optimizer(
    model: torch.nn.Module, config: ResolvedConfig
) -> torch.optim.AdamW:
    """Create explicit complete/disjoint AdamW groups from the resolved rules."""
    if not config.is_phase1:
        raise ValueError("build_phase1_optimizer requires a Phase-I config")
    spec = config.as_dict()["optimizer"]
    if spec["name"] != "AdamW":
        raise ValueError("Phase-I optimizer must remain AdamW")
    rules = list(spec["parameter_group_rules"])
    assigned: set[int] = set()
    groups: list[dict[str, Any]] = []
    group_identity: list[dict[str, Any]] = []
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise RuntimeError("Phase-I model has no trainable parameters")
    for rule in rules:
        selected = [
            (name, parameter)
            for name, parameter in named_parameters
            if id(parameter) not in assigned and _rule_matches(name, rule)
        ]
        if not selected:
            continue
        names = [name for name, _ in selected]
        parameters = [parameter for _, parameter in selected]
        if any(id(parameter) in assigned for parameter in parameters):
            raise RuntimeError("Phase-I optimizer parameter was assigned twice")
        assigned.update(id(parameter) for parameter in parameters)
        group_name = str(rule["name"])
        lr = float(spec["learning_rate"]) * float(rule["lr_mult"])
        weight_decay = float(spec["weight_decay"]) * float(rule["decay_mult"])
        names_sha256 = _canonical_sha256(names)
        groups.append({
            "params": parameters,
            "lr": lr,
            "weight_decay": weight_decay,
            "phase1_group_name": group_name,
            "phase1_parameter_names_sha256": names_sha256,
        })
        group_identity.append({
            "name": group_name,
            "parameter_count": len(parameters),
            "parameter_names": names,
            "parameter_names_sha256": names_sha256,
            "lr": lr,
            "weight_decay": weight_decay,
        })
    missing = [
        name for name, parameter in named_parameters if id(parameter) not in assigned
    ]
    if missing:
        raise RuntimeError(f"Phase-I optimizer rules leave parameters unassigned: {missing}")
    all_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != len(named_parameters):
        raise RuntimeError("Phase-I optimizer groups are not complete and disjoint")
    optimizer = torch.optim.AdamW(
        groups,
        lr=float(spec["learning_rate"]),
        betas=tuple(float(value) for value in spec["betas"]),
        eps=float(spec["eps"]),
        weight_decay=float(spec["weight_decay"]),
        amsgrad=bool(spec["amsgrad"]),
        fused=bool(spec["fused"]),
    )
    optimizer._phase1_group_identity = group_identity  # type: ignore[attr-defined]
    optimizer._phase1_config_sha256 = config.sha256  # type: ignore[attr-defined]
    validate_phase1_optimizer_identity(optimizer, model, config)
    return optimizer


def validate_phase1_optimizer_identity(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module | None,
    config: ResolvedConfig,
) -> None:
    if type(optimizer) is not torch.optim.AdamW:
        raise RuntimeError("Phase-I runtime optimizer is not exact AdamW")
    spec = config.as_dict()["optimizer"]
    expected_rules = {rule["name"]: rule for rule in spec["parameter_group_rules"]}
    seen_ids: list[int] = []
    for group in optimizer.param_groups:
        name = group.get("phase1_group_name")
        if name not in expected_rules:
            raise RuntimeError(f"Phase-I optimizer has unknown group {name!r}")
        rule = expected_rules[name]
        expected_lr = float(spec["learning_rate"]) * float(rule["lr_mult"])
        initial_lr = float(group.get("initial_lr", group["lr"]))
        if initial_lr != expected_lr:
            raise RuntimeError(f"Phase-I optimizer group {name!r} initial LR drift")
        expected_decay = float(spec["weight_decay"]) * float(rule["decay_mult"])
        if float(group["weight_decay"]) != expected_decay:
            raise RuntimeError(f"Phase-I optimizer group {name!r} weight decay drift")
        if float(group.get("initial_momentum", spec["betas"][0])) != float(
            spec["betas"][0]
        ) or float(group["betas"][1]) != float(spec["betas"][1]):
            raise RuntimeError(f"Phase-I optimizer group {name!r} beta identity drift")
        if float(group["eps"]) != float(spec["eps"]):
            raise RuntimeError(f"Phase-I optimizer group {name!r} epsilon drift")
        seen_ids.extend(id(parameter) for parameter in group["params"])
    if len(seen_ids) != len(set(seen_ids)):
        raise RuntimeError("Phase-I optimizer parameter groups overlap")
    if model is not None:
        expected_ids = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        if set(seen_ids) != expected_ids:
            raise RuntimeError("Phase-I optimizer parameter coverage is incomplete")


def _annealing_cos(start: float, end: float, factor: float) -> float:
    return float(end + 0.5 * (start - end) * (math.cos(math.pi * factor) + 1.0))


class Phase1CyclicScheduler:
    """Stateful accepted-update scheduler matching pinned MMCV hook order."""

    def __init__(self, optimizer: torch.optim.Optimizer, config: ResolvedConfig):
        if not config.is_phase1:
            raise ValueError("Phase1CyclicScheduler requires a Phase-I config")
        self.optimizer = optimizer
        self.spec = config.as_dict()["scheduler"]
        self.max_updates = int(config.data["training"]["max_optimizer_updates"])
        self.accepted_updates = 0
        self.last_epoch = 0
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.base_momenta = [float(group["betas"][0]) for group in optimizer.param_groups]
        for group, lr, momentum in zip(
            optimizer.param_groups, self.base_lrs, self.base_momenta
        ):
            group.setdefault("initial_lr", lr)
            group.setdefault("initial_momentum", momentum)
        self._spec_sha256 = _canonical_sha256(self.spec)
        self._apply(0)

    @staticmethod
    def _cyclic_value(
        base: float,
        update: int,
        max_updates: int,
        target_ratio: tuple[float, float],
        step_ratio_up: float,
    ) -> float:
        up_end = int(float(step_ratio_up) * int(max_updates))
        if up_end <= 0 or up_end >= max_updates:
            raise ValueError("invalid Phase-I cyclic schedule phase split")
        position = int(update) % int(max_updates)
        if position < up_end:
            return _annealing_cos(
                base, base * target_ratio[0], position / float(up_end)
            )
        return _annealing_cos(
            base * target_ratio[0],
            base * target_ratio[1],
            (position - up_end) / float(max_updates - up_end),
        )

    def _apply(self, update: int) -> None:
        lr_spec = self.spec["lr"]
        momentum_spec = self.spec["momentum"]
        lr_target = tuple(float(value) for value in lr_spec["target_ratio"])
        momentum_target = tuple(
            float(value) for value in momentum_spec["target_ratio"]
        )
        for group, base_lr, base_momentum in zip(
            self.optimizer.param_groups, self.base_lrs, self.base_momenta
        ):
            lr = self._cyclic_value(
                base_lr,
                update,
                self.max_updates,
                lr_target,
                float(lr_spec["step_ratio_up"]),
            )
            warmup = self.spec["warmup"]
            if warmup is not None and update < int(warmup["updates"]):
                ratio = float(warmup["ratio"])
                k = (1.0 - update / float(warmup["updates"])) * (1.0 - ratio)
                lr *= 1.0 - k
            momentum = self._cyclic_value(
                base_momentum,
                update,
                self.max_updates,
                momentum_target,
                float(momentum_spec["step_ratio_up"]),
            )
            group["lr"] = lr
            group["betas"] = (momentum, group["betas"][1])

    def step(self) -> None:
        if self.accepted_updates >= self.max_updates:
            raise RuntimeError("Phase-I scheduler advanced past its frozen update budget")
        self.accepted_updates += 1
        self.last_epoch = self.accepted_updates
        if self.accepted_updates < self.max_updates:
            self._apply(self.accepted_updates)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "s10.phase1.cyclic-scheduler.v1",
            "accepted_updates": self.accepted_updates,
            "last_epoch": self.last_epoch,
            "max_updates": self.max_updates,
            "base_lrs": list(self.base_lrs),
            "base_momenta": list(self.base_momenta),
            "spec_sha256": self._spec_sha256,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected_keys = {
            "schema", "accepted_updates", "last_epoch", "max_updates",
            "base_lrs", "base_momenta", "spec_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected_keys:
            raise RuntimeError("Phase-I scheduler checkpoint fields drift")
        if state["schema"] != "s10.phase1.cyclic-scheduler.v1":
            raise RuntimeError("Phase-I scheduler checkpoint schema drift")
        if (
            int(state["max_updates"]) != self.max_updates
            or list(state["base_lrs"]) != self.base_lrs
            or list(state["base_momenta"]) != self.base_momenta
            or state["spec_sha256"] != self._spec_sha256
        ):
            raise RuntimeError("Phase-I scheduler identity drift")
        updates = int(state["accepted_updates"])
        if int(state["last_epoch"]) != updates or not 0 <= updates <= self.max_updates:
            raise RuntimeError("Phase-I scheduler accepted-update state is invalid")
        self.accepted_updates = updates
        self.last_epoch = updates
        self._apply(updates if updates < self.max_updates else self.max_updates - 1)


def phase1_training_components(
    model: torch.nn.Module, config: ResolvedConfig
) -> tuple[torch.optim.AdamW, Phase1CyclicScheduler]:
    optimizer = build_phase1_optimizer(model, config)
    return optimizer, Phase1CyclicScheduler(optimizer, config)
