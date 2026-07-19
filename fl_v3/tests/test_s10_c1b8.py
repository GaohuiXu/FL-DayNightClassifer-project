from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "s10_c1b1_bn_b8.py"
    spec = importlib.util.spec_from_file_location("s10_c1b1_bn_b8", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bn_b8_envelope_is_exact_and_fail_fast():
    module = _module()
    assert module.PHYSICAL_BATCH == 8
    assert module.EVAL_PHYSICAL_BATCH == 4
    assert module.HORIZON == 769
    assert module.EXPECTED_D_LOW_SAMPLES == 6155
    assert module.EXPECTED_D_SELECT_SAMPLES == 4626
    assert module.BOUNDARIES == (1, 4, 16, 64, 256, 512, 769)
    spec = module._spec()
    assert spec["normalization"] == "batch_norm_1d"
    assert spec["physical_microbatch"] == 8
    assert spec["eval_physical_microbatch"] == 4
    assert spec["grad_scaler_init_scale"] == 8.0
    assert spec["fail_fast_numerical"] is True
    assert spec["operator_profile"] is False


def test_bn_b8_config_changes_only_declared_operational_fields():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / "configs" / "s10_c1b1_cur_a1_gn.json").read_text())
    new = json.loads((root / "configs" / "s10_c1b1_bn_b8.json").read_text())
    old["model"]["second_normalization"] = "batch_norm_1d"
    old["training"].update({
        "max_optimizer_steps": 769,
        "micro_batch_size": 8,
        "effective_global_batch": 8,
        "grad_scaler_init_scale": 8,
    })
    assert old == new


def test_bn_b8_reuses_exact_b4_token_order_and_trainable_w0():
    module = _module()
    assert module.EXPECTED_W0_SHA256 == (
        "87be0d2416b3ed06e2d1e9214e11ad3ac25bc275993b0865d918af6f332829d1"
    )
    assert module.EXPECTED_TOKEN_ORDER_SHA256 == (
        "947dc9bc8441267587df6b0b88d16efc84ab3c7ff0a1a152481ac2697f0a2eb1"
    )
    assert module.EXPECTED_REMAINDER_SHA256 == (
        "7495cdbec472ce49f29e8f19abe08fc9431a258b437a5db05ab89fae0db60443"
    )
    assert 6155 // module.PHYSICAL_BATCH == module.HORIZON
    assert 6155 % module.PHYSICAL_BATCH == 3

