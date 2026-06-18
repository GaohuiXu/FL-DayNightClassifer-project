"""T5 mini code-path smoke (login-node, in-process) — NOT a scientific result (scale=mini-smoke).

Validates, on v1.0-mini with the resnet18 fallback, that:
  1. the malicious roster is derived (m≥1) at the mini-derived N, and the poison routing FIRES;
  2. the ``poison_rate=0`` null FedAvg checksum is BYTE-IDENTICAL to the attack-disabled clean run;
  3. two same-seed poisoned FedAvg runs are byte-identical (determinism);
  4. the 5-condition ablation runs end-to-end on the real (untrained) model + the cond-5a guards
     (LiDAR-invariance byte-identity + the camera-only readout) execute.

Heavy science is the trainval SLURM path; this only proves the code path + the byte-identity invariants.
"""
from __future__ import annotations

import json
import sys


def _cfg(path, **ov):
    c = json.load(open(path))
    c.update(ov)
    return c


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "fl_v3/configs/t5_mini_smoke.json"
    base = _cfg(cfg_path)
    base["nuscenes-cache-dir"] = base.get("nuscenes-cache-dir", "./fl_outputs/nuscenes/info_cache")

    from fl_v3.training.tasks import get_task
    from fl_v3.engine.local_runner import run_clean_rounds
    from fl_v3.attacks.poisoned_client import malicious_roster
    from fl_v3.utils.runtime import seed_everything, enforce_determinism

    task = get_task("nuscenes_detection")
    N = task.num_clients(base)
    roster = malicious_roster(int(base["seed"]), N, float(base["attack-rho"]))
    print(f"[smoke] derived N={N} roster={roster} m_r={len(roster)} (mini)")
    assert len(roster) >= 1 and len(roster) < N / 2, "roster must be a non-empty honest minority"

    # 1+3. poisoned FedAvg (1 round) — fires the routing; run twice for determinism.
    print("[smoke] poisoned run A …", flush=True)
    a = run_clean_rounds(dict(base), defense="none", num_rounds=1, fraction_train=1.0)
    print("[smoke] poisoned run B …", flush=True)
    b = run_clean_rounds(dict(base), defense="none", num_rounds=1, fraction_train=1.0)
    assert a["final_checksum"] == b["final_checksum"], "poisoned runs DIVERGED (non-deterministic!)"
    print(f"[smoke] determinism OK — poisoned checksum {a['final_checksum'][:16]}…")

    # 2. null (poison_rate=0) vs clean (attack-disabled) — BYTE-IDENTICAL.
    null = run_clean_rounds(_cfg(cfg_path, **{"attack-poison-rate": 0.0}), defense="none", num_rounds=1, fraction_train=1.0)
    clean = run_clean_rounds(_cfg(cfg_path, **{"attack-enabled": False}), defense="none", num_rounds=1, fraction_train=1.0)
    assert null["final_checksum"] == clean["final_checksum"], "null (rate=0) NOT byte-identical to clean!"
    print(f"[smoke] null≡clean BYTE-IDENTICAL — {null['final_checksum'][:16]}…")
    assert a["final_checksum"] != clean["final_checksum"], "poisoned == clean? the attack did NOT perturb training"
    print("[smoke] poisoned ≠ clean (the attack perturbs the trajectory) ✓")

    # 4. the 5-condition ablation + cond-5a guards on the real (untrained) model + a mini val sample.
    import torch
    from fl_v3.data.nuscenes import info_cache as IC, paths as P
    from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset
    from fl_v3.eval.asr import asr_thresholds_from_run_config
    from fl_v3.attacks import trigger as TR, fusion_ablation as FA
    from fl_v3.attacks.poison import _lidar_xyz

    seed_everything(0); enforce_determinism(strict=False)
    val_info, _ = IC.load_cache(base["nuscenes-cache-dir"], base["nuscenes-version"], base["nuscenes-val-split"])
    ds = NuScenesMultimodalDataset(val_info, P.get_dataroot(base))
    model = task.build_model(base).eval()
    thr = asr_thresholds_from_run_config(base)
    spec = TR.trigger_spec_from_run_config(base)
    # find a mini sample with a car GT
    target = None
    for i in range(len(ds)):
        s = ds[i]
        labs = s["gt_labels"].numpy()
        car = [m for m in range(len(labs)) if int(labs[m]) == thr.target_id]
        if car:
            pl = TR.compute_aligned_placement(s["gt_boxes"][car[0]].numpy(), _lidar_xyz(s),
                                              s["lidar2img"].numpy(), spec)
            if pl is not None:
                target = (s, s["gt_ann_tokens"][car[0]]); break
    assert target is not None, "no car target with a valid aligned placement in mini val"
    s, ann = target
    dev = torch.device("cpu")
    v = FA.evaluate_target(s, ann, model, model, dev, thr, int(base.get("det-max-objects", 200)), spec)
    print(f"[smoke] ablation evaluate_target OK — disappeared={v.disappeared} occlusion={v.occlusion_disappeared}")
    inv = FA.lidar_invariance_check(model, s, dev)
    print(f"[smoke] cond-5a LiDAR-invariance: {inv}")
    assert inv["lidar_invariant"], "cond-5a hook is NOT LiDAR-invariant (zeroing failed)!"
    rec = FA.camera_only_clean_recall(model, [(s, [ann])], dev, thr, int(base.get("det-max-objects", 200)))
    print(f"[smoke] camera-only clean-recall (untrained model, indicative only): {rec}")

    # phantom code-path
    from fl_v3.attacks import poison as PO
    pph = PO.poison_sample(s, PO.poison_config_from_run_config({**base, "attack-mode": "phantom"}), spec)
    print(f"[smoke] phantom inserts a box: {pph['num_boxes'] >= s['num_boxes']} ({s['num_boxes']}→{pph['num_boxes']})")
    print("[smoke] ALL MINI CODE-PATH CHECKS PASSED (scale=mini-smoke; NOT a scientific result).")


if __name__ == "__main__":
    main()
