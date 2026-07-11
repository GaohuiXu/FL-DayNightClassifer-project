"""S05 O-018 CenterHead candidate, mapping, and round-trip fixtures."""
from __future__ import annotations

import math

import torch

from fl_v3.models.fusion.bev_grid import BEVConfig
from fl_v3.models.fusion.centerhead_decode import (
    CenterHeadDecodeConfig,
    NUSCENES_TASK_SPECS,
    decode_centerhead,
    encode_canonical_boxes,
    select_task_candidates,
    task_local_to_global_ids,
)
from fl_v3.models.fusion.head import REG_CHANNELS


BEV = BEVConfig(
    point_cloud_range=(0.0, 0.0, -5.0, 23.0, 23.0, 5.0),
    bev_voxel=(1.0, 1.0),
    out_size_factor=1,
)


def _blank_outputs(batch_size: int = 1):
    outputs = []
    for spec in NUSCENES_TASK_SPECS:
        out = {
            "heatmap": torch.full(
                (batch_size, len(spec.class_names), BEV.head_ny, BEV.head_nx),
                -20.0,
            )
        }
        for name, channels in REG_CHANNELS.items():
            out[name] = torch.zeros((batch_size, channels, BEV.head_ny, BEV.head_nx))
        outputs.append(out)
    return outputs


def _set_encoded_box(outputs, task_id, local_class, batch_id, box, velocity, logit=8.0):
    encoded = encode_canonical_boxes(box.reshape(1, 7), velocity.reshape(1, 2), BEV)
    flat = int(encoded["spatial_indices"][0])
    row, col = divmod(flat, BEV.head_nx)
    outputs[task_id]["heatmap"][batch_id, local_class, row, col] = logit
    for field in REG_CHANNELS:
        outputs[task_id][field][batch_id, :, row, col] = encoded[field][0]
    return flat


def test_task_local_mapping_uses_names_not_cumulative_offsets():
    # Canonical devkit IDs: construction=4, bus=2, barrier=9, pedestrian=5, cone=8.
    assert task_local_to_global_ids() == (
        (0,),
        (1, 4),
        (2, 3),
        (9,),
        (6, 7),
        (5, 8),
    )


def test_tail_candidate_survives_500_higher_scoring_common_candidates():
    outputs = _blank_outputs()
    task = outputs[1]  # truck + construction_vehicle
    flat = task["heatmap"][0, 0].reshape(-1)
    # Exactly 500 common-class candidates outrank the rare candidate.  The official
    # second task-wide K=500 would drop it; O-018 must keep it for NMS.
    flat[:500] = torch.linspace(12.0, 6.0, 500)
    rare_flat = 520
    task["heatmap"][0, 1].reshape(-1)[rare_flat] = 1.0
    candidates = select_task_candidates(outputs, BEV)[0][1]
    assert candidates["scores"].numel() == 501
    rare = candidates["labels"] == 4
    assert int(rare.sum()) == 1
    assert int(candidates["spatial_indices"][rare][0]) == rare_flat


def test_equal_score_tie_is_global_class_then_flat_spatial_index():
    outputs = _blank_outputs()
    heat = outputs[2]["heatmap"]  # bus(global 2) + trailer(global 3)
    heat[0, 0].reshape(-1)[9] = 2.0
    heat[0, 0].reshape(-1)[3] = 2.0
    heat[0, 1].reshape(-1)[1] = 2.0
    candidates = select_task_candidates(outputs, BEV)[0][2]
    assert candidates["labels"].tolist() == [2, 2, 3]
    assert candidates["spatial_indices"].tolist() == [3, 9, 1]


def test_single_class_candidate_selection_matches_official_two_stage_topk():
    outputs = _blank_outputs()
    heat = outputs[0]["heatmap"]
    heat[0, 0].reshape(-1)[:510] = torch.linspace(10.0, 0.2, 510)
    ours = select_task_candidates(
        (outputs[0],), BEV, task_specs=(NUSCENES_TASK_SPECS[0],)
    )[0][0]

    # Official coder: per-class K=500 then task-wide K=500.  For one class the
    # second selection is an identity.  Stable sort supplies the deterministic tie rule.
    scores = heat[0, 0].sigmoid().reshape(-1)
    ref_scores, ref_flat = torch.sort(scores, descending=True, stable=True)
    ref_scores, ref_flat = ref_scores[:500], ref_flat[:500]
    keep = ref_scores > 0.1
    assert torch.equal(ours["scores"], ref_scores[keep])
    assert torch.equal(ours["spatial_indices"], ref_flat[keep])
    assert torch.equal(ours["labels"], torch.zeros_like(ref_flat[keep]))


def test_fp16_adjacent_logits_are_force_fp32_before_sigmoid_and_topk():
    outputs = _blank_outputs()
    outputs = [
        {name: value.to(torch.float16) for name, value in task.items()}
        for task in outputs
    ]
    heat = outputs[0]["heatmap"][0, 0].reshape(-1)
    # Adjacent binary16 logits whose FP32 sigmoid values are distinct but whose
    # rounded binary16 sigmoid values collide at 0.1251220703125.
    heat[0] = -1.9453125
    heat[1] = -1.9443359375

    candidates = select_task_candidates(
        (outputs[0],), BEV, task_specs=(NUSCENES_TASK_SPECS[0],)
    )[0][0]
    reference = heat.float().sigmoid()
    assert candidates["scores"].dtype == torch.float32
    assert candidates["velocity"].dtype == torch.float32
    assert reference[1] > reference[0]
    assert candidates["spatial_indices"][:2].tolist() == [1, 0]
    assert torch.equal(candidates["scores"][:2], reference[[1, 0]])


def test_fp16_logits_use_force_fp32_strict_threshold_neighbourhood():
    outputs = _blank_outputs()
    outputs = [
        {name: value.to(torch.float16) for name, value in task.items()}
        for task in outputs
    ]
    heat = outputs[0]["heatmap"][0, 0].reshape(-1)
    # Adjacent representable values bracketing logit(0.1) in binary16.
    heat[7] = -2.197265625
    heat[9] = -2.1953125
    reference = heat.float().sigmoid()
    assert reference[7] < 0.1 < reference[9]

    candidates = select_task_candidates(
        (outputs[0],), BEV, task_specs=(NUSCENES_TASK_SPECS[0],)
    )[0][0]
    assert 7 not in candidates["spatial_indices"].tolist()
    assert 9 in candidates["spatial_indices"].tolist()
    selected = candidates["spatial_indices"] == 9
    assert torch.equal(candidates["scores"][selected], reference[9:10])


def test_encode_decode_roundtrip_preserves_box_yaw_velocity_and_global_class():
    outputs = _blank_outputs()
    box = torch.tensor([7.25, 11.75, 0.8, 4.2, 1.7, 1.6, -2.7])
    velocity = torch.tensor([3.25, -0.75])
    _set_encoded_box(outputs, 1, 1, 0, box, velocity)  # construction_vehicle
    decoded = decode_centerhead(outputs, BEV)[0]
    assert decoded["labels"].tolist() == [4]
    assert torch.allclose(decoded["boxes"][0], box, atol=1e-5, rtol=1e-5)
    assert torch.allclose(decoded["velocity"][0], velocity, atol=1e-6, rtol=1e-6)
    assert abs(math.remainder(float(decoded["boxes"][0, 6] - box[6]), 2 * math.pi)) < 1e-6


def test_batched_decode_only_reorders_when_batch_is_permuted():
    outputs = _blank_outputs(batch_size=2)
    box_a = torch.tensor([2.2, 3.4, 0.5, 4.0, 2.0, 1.5, 0.25])
    box_b = torch.tensor([18.1, 17.6, 1.2, 3.0, 1.4, 1.8, -1.1])
    vel_a, vel_b = torch.tensor([1.0, 2.0]), torch.tensor([-2.0, 0.5])
    _set_encoded_box(outputs, 0, 0, 0, box_a, vel_a)
    _set_encoded_box(outputs, 0, 0, 1, box_b, vel_b)
    decoded = decode_centerhead(outputs, BEV)

    permuted = [{name: value[[1, 0]] for name, value in task.items()} for task in outputs]
    decoded_perm = decode_centerhead(permuted, BEV)
    for left, right in zip(decoded, reversed(decoded_perm)):
        for key in ("boxes", "scores", "labels", "velocity"):
            assert torch.equal(left[key], right[key])


def test_decode_config_is_exact_o018_budget_and_threshold_contract():
    cfg = CenterHeadDecodeConfig()
    assert cfg.score_threshold == 0.1
    assert cfg.per_class_pre_max == 500
    assert cfg.nms_pre_max == 1000
    assert cfg.nms_post_max == 83
    assert cfg.rotate_iou_threshold == 0.2
    assert cfg.post_center_range == (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)
    assert tuple(spec.nms_type for spec in NUSCENES_TASK_SPECS) == (
        "circle", "rotate", "rotate", "circle", "rotate", "rotate"
    )
    assert tuple(spec.circle_threshold_sq_m for spec in NUSCENES_TASK_SPECS) == (
        4.0, 12.0, 10.0, 1.0, 0.85, 0.175
    )
    assert NUSCENES_TASK_SPECS[-1].nms_scale == (2.5, 4.0)
    assert len(NUSCENES_TASK_SPECS) * cfg.nms_post_max == 498
    assert 498 <= 500  # official nuScenes max_boxes_per_sample
