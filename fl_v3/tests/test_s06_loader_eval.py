from __future__ import annotations

import numpy as np
import pytest
import torch

from fl_v3.eval.detection_eval import build_results_dict, decode_eval_set, submission_meta
from fl_v3.training.runtime_state import EpochPermutationSampler, PersistentEpochIterator


class EpochSampler:
    def __init__(self): self.epochs = []
    def set_epoch(self, epoch): self.epochs.append(epoch)


class Loader:
    def __init__(self): self.sampler = EpochSampler(); self.iterations = 0
    def __iter__(self):
        self.iterations += 1
        return iter([0, 1, 2])


def test_persistent_loader_and_sampler_across_epoch_resume():
    loader = Loader(); stream = PersistentEpochIterator(loader)
    assert list(stream.batches(0)) == [0, 1, 2]
    assert list(stream.batches(1)) == [0, 1, 2]
    # A resumed runtime keeps the same persistent object and starts at declared epoch.
    assert list(stream.batches(2)) == [0, 1, 2]
    assert loader.iterations == 3 and loader.sampler.epochs == [0, 1, 2]


def test_epoch_sampler_has_no_duplicate_or_omitted_indices_and_resume_matches():
    sampler = EpochPermutationSampler(list(range(17)), seed=9)
    sampler.set_epoch(3); continuous = list(sampler)
    resumed = EpochPermutationSampler(list(range(17)), seed=9)
    resumed.set_epoch(3)
    assert list(resumed) == continuous
    assert sorted(continuous) == list(range(17)) and len(set(continuous)) == 17


class EvalModel(torch.nn.Module):
    def __init__(self): super().__init__(); self.calls = 0
    def forward(self, batch):
        self.calls += 1
        assert "lidar_points" not in batch
        return {"x": batch["images"].float().sum().reshape(1)}
    def decode(self, head, score_threshold):
        return [{"boxes": torch.zeros(0, 7), "scores": torch.zeros(0),
                 "labels": torch.zeros(0, dtype=torch.long), "velocity": torch.zeros(0, 2)}]


def _batch():
    return {
        "images": torch.ones(1, 1),
        "lidar2img": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 6, 1, 1),
        "cam_intrinsics": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 6, 1, 1),
        "lidar_points": torch.ones(2, 6),
        "sample_token": ["tok"], "gt_boxes": [torch.zeros(0, 7)],
        "gt_labels": [torch.zeros(0, dtype=torch.long)], "gt_num_lidar_pts": [torch.zeros(0)],
        "gt_in_range": [torch.zeros(0, dtype=torch.bool)], "gt_velocity": [torch.zeros(0, 2)],
        "gt_ann_tokens": [[]], "gt_names": [[]], "gt_attribute": [[]],
        "lidar2ego": torch.eye(4).reshape(1, 4, 4),
        "ego2global_lidar": torch.eye(4).reshape(1, 4, 4),
    }


def test_eval_is_single_pass_autocast_policy_and_timing_output_neutral():
    cfg = {"model-mode": "camera_only", "precision": "fp32", "det-score-threshold": .1}
    a = EvalModel(); plain = decode_eval_set(a, [_batch()], torch.device("cpu"), cfg)
    timing = {}; b = EvalModel(); timed = decode_eval_set(b, [_batch()], torch.device("cpu"), cfg, timing)
    assert a.calls == b.calls == 1 and timing["batches"] == 1
    assert plain[0].sample_token == timed[0].sample_token
    assert np.array_equal(plain[0].boxes, timed[0].boxes)


def test_eval_rejects_non_six_camera_calibration():
    batch = _batch()
    batch["lidar2img"] = torch.eye(4).reshape(1, 1, 4, 4)
    with pytest.raises(ValueError, match="six cameras"):
        decode_eval_set(
            EvalModel(), [batch], torch.device("cpu"),
            {"model-mode": "camera_only", "precision": "fp32", "det-score-threshold": .1},
        )


def test_submission_metadata_records_actual_mode():
    assert submission_meta("camera_only")["use_camera"] is True
    assert submission_meta("camera_only")["use_lidar"] is False
    assert submission_meta("lidar_only")["use_camera"] is False
    assert submission_meta("lidar_only")["use_lidar"] is True


def test_submission_binds_config_checkpoint_and_data_identities():
    keys = ("resolved-config-sha256", "checkpoint-sha256", "runtime-dependencies-sha256",
            "nuscenes-train-cache-logical-sha256", "nuscenes-train-cache-pickle-sha256",
            "nuscenes-train-cache-sidecar-sha256", "nuscenes-val-cache-logical-sha256",
            "nuscenes-val-cache-pickle-sha256", "nuscenes-val-cache-sidecar-sha256",
            "nuscenes-zip-manifest-logical-sha256",
            "nuscenes-zip-manifest-file-sha256")
    cfg = {"model-mode": "camera_only", "checkpoint-weights": "raw",
           **{k: "a" * 64 for k in keys}}
    submission = build_results_dict([], [], ["tok"], run_config=cfg)
    assert submission["meta"]["use_camera"] and not submission["meta"]["use_lidar"]
    assert submission["fl_v3_provenance"]["model_mode"] == "camera_only"
    assert set(keys) <= set(submission["fl_v3_provenance"])
    assert submission["fl_v3_provenance"]["checkpoint_weights"] == "raw"
