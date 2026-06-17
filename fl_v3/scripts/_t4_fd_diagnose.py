"""Diagnose the T4 false-disappearance (9.4%): batch-invariance vs run-to-run determinism.

Decodes the SAME frozen-subset samples three ways on the trainval checkpoint and compares the
clean-detected car set per sample:
  passA  = batch 16   (the subset-construction batching)
  passA2 = batch 16   (re-run, same batching → tests run-to-run determinism)
  passB  = batch 1    (isolated per-sample → tests batch-invariance)
Reports the fraction of passA-detected cars that DISAPPEAR in A2 (determinism) and in B (batching).
"""
from __future__ import annotations
import json, sys
import numpy as np
import torch

CFG = "fl_v3/configs/t4_reference.json"
CKPT = "fl_outputs/nuscenes/experiments/cycle_04/t4_reference/t4_reference/final_model.pt"
SUBSET = "fl_outputs/nuscenes/experiments/cycle_04/t4_reference/readiness/frozen_asr_subset.json"
N_SAMPLES = 60  # subset samples to test

cfg = json.load(open(CFG)); cfg["det-eval-limit"] = 0
from fl_v3.training.tasks import get_task
from fl_v3.utils.runtime import enforce_determinism, seed_everything
from fl_v3.data.nuscenes import info_cache as IC, paths as P
from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset, make_loader
from fl_v3.models.fusion.collate import detection_collate_fn
from fl_v3.eval.detection_eval import decode_eval_set
from fl_v3.eval import asr as ASR

seed_everything(int(cfg.get("seed", 42))); enforce_determinism(strict=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", dev, flush=True)

task = get_task("nuscenes_detection")
model = task.build_model(cfg).to(dev)
model.load_state_dict(torch.load(CKPT, map_location=dev), strict=True); model.eval()

sub = json.load(open(SUBSET))
thr = ASR.thresholds_from_subset(sub)
sub_tokens = sorted({s for s, _ in sub["targets"]})[:N_SAMPLES]
val_info, _ = IC.load_cache(cfg["nuscenes-cache-dir"], cfg["nuscenes-version"], cfg["nuscenes-val-split"])
print(f"testing {len(sub_tokens)} subset samples", flush=True)

def decode_set(batch_size):
    ds = NuScenesMultimodalDataset(val_info, P.get_dataroot(cfg), sample_tokens=sub_tokens)
    loader = make_loader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                         seed=int(cfg.get("seed", 42)), collate_fn=detection_collate_fn)
    decs = decode_eval_set(model, loader, dev, cfg)
    return {d.sample_token: ASR.detected_target_anns(d, thr) for d in decs}

A  = decode_set(16)
A2 = decode_set(16)
B  = decode_set(1)

# the subset cars in these samples
subset_by_s = {}
for s, a in sub["targets"]:
    if s in sub_tokens:
        subset_by_s.setdefault(s, set()).add(a)

def disappeared(ref_set_map):
    tot = miss = 0
    for s, cars in subset_by_s.items():
        det = ref_set_map.get(s, set())
        for a in cars:
            tot += 1
            if a not in det:
                miss += 1
    return miss, tot

mA, t = disappeared(A)
mA2, _ = disappeared(A2)
mB, _ = disappeared(B)
print(f"subset cars in tested samples: {t}", flush=True)
print(f"passA  (batch16)  disappeared: {mA}/{t} = {mA/t:.4f}", flush=True)
print(f"passA2 (batch16 rerun, SAME batching) disappeared: {mA2}/{t} = {mA2/t:.4f}  <- run-to-run determinism", flush=True)
print(f"passB  (batch1, isolated) disappeared: {mB}/{t} = {mB/t:.4f}  <- batch-invariance", flush=True)
# A vs A2 per-sample detection-set identity
diffAA2 = sum(1 for s in sub_tokens if A.get(s, set()) != A2.get(s, set()))
diffAB = sum(1 for s in sub_tokens if A.get(s, set()) != B.get(s, set()))
print(f"samples with A != A2 (same batching, rerun): {diffAA2}/{len(sub_tokens)}", flush=True)
print(f"samples with A != B  (batch16 vs batch1)   : {diffAB}/{len(sub_tokens)}", flush=True)
