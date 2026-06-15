"""The FL skeleton carries NO classification assumption (a T0 invariant).

Positive proof: the dummy task uses an MSE criterion and a regression target,
and a full FL round runs end-to-end with it. Negative proof: the skeleton source
files do not hardcode a loss or a num-classes / single-logits assumption.
"""
from __future__ import annotations

import ast
import inspect

import torch.nn as nn

import fl_v3.client_app as client_app
import fl_v3.server_app as server_app
import fl_v3.training.loop as loop
from fl_v3.engine.local_runner import run_clean_round
from fl_v3.training.tasks import build_criterion, get_task


def test_dummy_task_uses_mse_not_crossentropy():
    task = get_task("dummy_regression")
    crit = task.build_criterion({"loss": "mse"})
    assert isinstance(crit, nn.MSELoss)
    assert not isinstance(crit, nn.CrossEntropyLoss)


def test_criterion_is_config_injected():
    assert isinstance(build_criterion("mse"), nn.MSELoss)
    assert isinstance(build_criterion("l1"), nn.L1Loss)
    assert isinstance(build_criterion("cross_entropy"), nn.CrossEntropyLoss)


_BANNED_CONFIG_KEYS = {"num-classes", "num_classes"}


def _coupling_findings(src: str) -> list[str]:
    """AST-level classification-coupling detector (ignores prose/docstrings).

    Flags: a ``CrossEntropyLoss`` identifier/attribute (constructing a loss in
    the skeleton instead of taking it from the Task), and an exact ``num-classes``
    / ``num_classes`` string used as a real value (e.g. ``run_config["num-classes"]``
    or ``.get("num_classes")``). Docstring mentions are whole-string Constants
    that never equal a banned key, so they are not flagged.
    """
    findings = []
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute,)) and getattr(node, "attr", None) == "CrossEntropyLoss":
            findings.append("CrossEntropyLoss (attribute)")
        if isinstance(node, ast.Name) and node.id == "CrossEntropyLoss":
            findings.append("CrossEntropyLoss (name)")
        if isinstance(node, ast.Constant) and node.value in _BANNED_CONFIG_KEYS:
            findings.append(f"config key {node.value!r}")
    return findings


def test_skeleton_sources_do_not_hardcode_loss_or_numclasses():
    # The skeleton must obtain the loss/model from the Task, never hardcode them.
    for mod in (client_app, server_app, loop):
        src = inspect.getsource(mod)
        found = _coupling_findings(src)
        assert not found, f"{mod.__name__} has classification coupling: {found}"


def test_regression_round_runs_end_to_end():
    cfg = {
        "task-type": "dummy_regression",
        "seed": 42,
        "device": "cpu",
        "num-clients": 4,
        "num-local-epochs": 1,
        "batch-size": 8,
        "learning-rate": 0.01,
        "num-workers": 0,
        "loss": "mse",
    }
    out = run_clean_round(cfg, defense="none", server_round=1)
    assert out["decision_valid"]
    assert out["eval"] is not None
    assert "eval_loss" in out["eval"]
    # No accuracy / class concept anywhere in the eval payload.
    assert all("acc" not in k.lower() for k in out["eval"])
