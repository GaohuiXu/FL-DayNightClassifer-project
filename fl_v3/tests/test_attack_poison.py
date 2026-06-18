"""T5 poison operators: per-box field hygiene + relocation/delete/trigger_only/label_only semantics."""
from __future__ import annotations

import torch

from fl_v3.attacks import trigger as TR, poison as PO
from _attack_fixtures import attack_sample

SPEC = TR.TriggerSpec(min_points=4)


def test_relocation_keeps_count_shifts_center_and_triggers():
    s = attack_sample(0)
    cfg = PO.PoisonConfig(mode="relocation", delta_reloc=6.0, tau_pts=10)
    out = PO.poison_sample(s, cfg, SPEC)
    PO.assert_box_fields_consistent(out)
    assert out["num_boxes"] == s["num_boxes"]                       # relocation keeps the box
    assert float(out["gt_boxes"][0, 0]) == float(s["gt_boxes"][0, 0]) + 6.0  # centre shifted +Δx
    assert not torch.equal(out["images"], s["images"])             # trigger applied


def test_delete_removes_all_eligible_targets_with_field_hygiene():
    s = attack_sample(1)
    cfg = PO.PoisonConfig(mode="delete", tau_pts=10)
    out = PO.poison_sample(s, cfg, SPEC)
    PO.assert_box_fields_consistent(out)
    assert out["num_boxes"] == 0                                   # both eligible cars deleted
    for f in PO.ALL_RAGGED_FIELDS:
        n = out[f].shape[0] if hasattr(out[f], "shape") else len(out[f])
        assert int(n) == 0


def test_trigger_only_changes_image_not_labels():
    s = attack_sample(2)
    out = PO.poison_sample(s, PO.PoisonConfig(mode="trigger_only", tau_pts=10), SPEC)
    assert torch.equal(out["gt_boxes"], s["gt_boxes"])             # labels untouched
    assert not torch.equal(out["images"], s["images"])            # trigger applied


def test_label_only_changes_labels_not_image():
    s = attack_sample(3)
    out = PO.poison_sample(s, PO.PoisonConfig(mode="label_only", delta_reloc=6.0, tau_pts=10), SPEC)
    assert torch.equal(out["images"], s["images"])                # no trigger
    assert float(out["gt_boxes"][0, 0]) == float(s["gt_boxes"][0, 0]) + 6.0


def test_none_mode_is_identity():
    s = attack_sample(4)
    out = PO.poison_sample(s, PO.PoisonConfig(mode="none"), SPEC)
    assert out is s  # literal no-op


def test_field_hygiene_append_and_delete_roundtrip():
    s = attack_sample(5)
    base = int(s["num_boxes"])
    out = dict(s)
    PO._append_box(out, box7=[1, 2, 3, 4, 2, 1.6, 0.0], velocity=[0, 0], label=0, num_lidar_pts=5,
                   visibility=4, in_range=True, name="car", attribute="", instance_token="x", ann_token="z")
    PO.assert_box_fields_consistent(out)
    assert out["num_boxes"] == base + 1 and out["gt_ann_tokens"][-1] == "z"
    PO._delete_box(out, base)  # remove the just-appended one
    PO.assert_box_fields_consistent(out)
    assert out["num_boxes"] == base


def test_poison_config_from_run_config():
    rc = {"attack-mode": "delete", "attack-delta-reloc": 8.0, "asr-tau-pts": 12}
    cfg = PO.poison_config_from_run_config(rc)
    assert cfg.mode == "delete" and cfg.delta_reloc == 8.0 and cfg.tau_pts == 12


def test_poisonable_requires_lidar_support_and_range():
    s = attack_sample(6)
    s["gt_num_lidar_pts"] = torch.tensor([40, 2], dtype=torch.int64)  # box1 below τ_pts
    s["gt_in_range"] = torch.tensor([True, False])                    # box0 ok, box1 out
    idx = PO.poisonable_target_indices(s, PO.PoisonConfig(tau_pts=10), SPEC)
    assert idx == [0]
