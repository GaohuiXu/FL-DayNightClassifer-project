"""MCR Phase-3 STEP 2 — per-client gradient/update conflict analysis (DIAGNOSTIC, reuses train_local).

From a fixed global snapshot, run each of the 25 log_group clients' ONE local epoch (the SAME update FedAvg
averages), capture the per-client trainable update Δ_i = w_local − w_global, and quantify NON-IID conflict at
the GRADIENT level (the owner's Q2/Q3): per-MODULE pairwise cosine between clients + frac-negative ("conflict
ratio"), and PER-CLASS heatmap-head sign-agreement (the tail-collapse test). NO platform code change — this
script only CALLS the existing task/loop. Read-only w.r.t. the platform; writes one JSON.

Hypothesis under test (from the teardown + literature): the shared feature modules align across clients
(positive cosine) but the per-class CenterPoint HEATMAP head diverges, and the TAIL classes show ~random
sign-agreement (signal cancellation from clients lacking those classes) ⇒ FedAvg averages a mis-calibrated
tail detector (high recall, ~0 AP).

Run:  CONFIG=... CKPT=<round_N>/final_model.pt CACHE=<msweep10> sbatch run_p3_grad_conflict.sh
"""
from __future__ import annotations

import argparse, json, os
from collections import OrderedDict, defaultdict

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="global snapshot final_model.pt to start every client from")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full local epoch (what FedAvg averages); >0 caps for speed")
    ap.add_argument("--out", default="fl_v3/collab/fl_baseline/p3_grad_conflict.json")
    ap.add_argument("overrides", nargs="*", default=[])
    a = ap.parse_args()

    from fl_v3.utils.runtime import enforce_determinism, seed_everything, derive_seed
    from fl_v3.training.tasks import get_task, trainable_state_dict, load_trainable_state_dict
    from fl_v3.training.loop import train_local
    from fl_v3.data.nuscenes.class_map import DETECTION_NAMES

    cfg = json.load(open(a.config))
    for ov in a.overrides:
        k, _, v = ov.partition("=")
        try: v = int(v)
        except ValueError:
            try: v = float(v)
            except ValueError:
                if v in ("true", "false"): v = (v == "true")
        cfg[k] = v
    seed = int(cfg.get("seed", 42))
    enforce_determinism(strict=False, precision=str(cfg.get("precision", "bf16")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = get_task(str(cfg["task-type"]))
    n_clients = task.num_clients(cfg)
    print(f"[grad-conflict] N={n_clients} ckpt={a.checkpoint} max_steps={a.max_steps} device={device}", flush=True)

    # the fixed global trainable vector (what every client starts from)
    g = task.build_model(cfg).to(device)
    full = torch.load(a.checkpoint, map_location="cpu")
    g.load_state_dict(full, strict=True)
    global_tsd = OrderedDict((k, v.detach().cpu().clone()) for k, v in trainable_state_dict(g).items())
    keys = list(global_tsd.keys())
    del g

    criterion = task.build_criterion(cfg)
    lr = float(cfg.get("learning-rate", 1e-3)); wd = float(cfg.get("weight-decay", 0.0))
    rkw = dict(grad_clip_norm=float(cfg.get("grad-clip-norm", 0.0)),
               backbone_lr_mult=float(cfg.get("det-backbone-lr-mult", 1.0)),
               optimizer_name=str(cfg.get("det-optimizer", "adam")))

    deltas = []  # per client: dict key -> np.ndarray (the update)
    for cid in range(n_clients):
        seed_everything(derive_seed(seed, cid, 1))   # mirror the FL per-client seeding
        m = task.build_model(cfg).to(device)
        load_trainable_state_dict(m, {k: v.clone() for k, v in global_tsd.items()})
        cdata = task.client_data(cid, cfg)
        train_local(m, cdata.trainloader, criterion, device, num_epochs=1,
                    learning_rate=lr, weight_decay=wd, valloader=None, **rkw)
        tsd = trainable_state_dict(m)
        d = {k: (tsd[k].detach().cpu().float().numpy() - global_tsd[k].float().numpy()) for k in keys}
        deltas.append(d)
        nrm = float(np.sqrt(sum(float((d[k]**2).sum()) for k in keys)))
        print(f"[grad-conflict] client {cid:>2}/{n_clients} n={cdata.num_train} ||Δ||={nrm:.3f}", flush=True)
        del m
        if device.type == "cuda": torch.cuda.empty_cache()

    N = len(deltas)
    def topmod(k): return k.split(".")[0]
    mod_keys = defaultdict(list)
    for k in keys: mod_keys[topmod(k)].append(k)

    def stack_module(mk):
        return np.stack([np.concatenate([deltas[i][k].ravel() for k in mk]) for i in range(N)])  # (N,d)
    def cos_stats(V):
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        C = Vn @ Vn.T
        od = C[~np.eye(N, dtype=bool)]
        norms = np.linalg.norm(V, axis=1)
        return {"mean_pairwise_cos": float(od.mean()), "min_pairwise_cos": float(od.min()),
                "frac_negative_pairs": float((od < 0).mean()),
                "norm_mean": float(norms.mean()), "norm_cov": float(norms.std() / (norms.mean() + 1e-12))}

    print("\n" + "=" * 78); print("PER-MODULE cross-client update cosine (positive=agree, <0=conflict)"); print("=" * 78)
    order = ["camera_backbone", "camera_neck", "view_transform", "lidar_encoder", "lidar_backbone",
             "fusion", "bev_neck", "head"]
    per_module = {}
    print(f"{'module':<16}{'mean_cos':>10}{'min_cos':>9}{'%neg':>7}{'normCoV':>9}")
    for mk in order:
        if mk not in mod_keys: continue
        s = cos_stats(stack_module(mod_keys[mk])); per_module[mk] = s
        print(f"{mk:<16}{s['mean_pairwise_cos']:>10.3f}{s['min_pairwise_cos']:>9.3f}{100*s['frac_negative_pairs']:>7.1f}{s['norm_cov']:>9.3f}")

    # PER-CLASS heatmap head: each class = one output channel of head.heatmap.weight (10,64,1,1)
    print("\n" + "=" * 78); print("PER-CLASS heatmap-head update: cross-client cosine + sign-agreement"); print("=" * 78)
    hk = "head.heatmap.weight"
    per_class = {}
    if hk in keys:
        # per client, per class channel c: flatten (64,) ; W[i] shape (10,64,1,1)
        W = np.stack([deltas[i][hk].reshape(deltas[i][hk].shape[0], -1) for i in range(N)])  # (N,10,64)
        cenAP = {'car':.85,'truck':.48,'construction_vehicle':.23,'bus':.53,'trailer':.22,'barrier':.65,
                 'motorcycle':.68,'bicycle':.42,'pedestrian':.80,'traffic_cone':.72}
        print(f"{'class':<22}{'mean_cos':>10}{'sign_agree':>11}{'cenAP':>7}")
        for c, name in enumerate(DETECTION_NAMES):
            Vc = W[:, c, :]                              # (N,64)
            sc = cos_stats(Vc)
            agg = Vc.mean(0)                              # aggregated update direction for this class
            sign_agree = float((np.sign(Vc) == np.sign(agg)[None, :]).mean())  # frac coords agreeing with agg sign
            per_class[name] = {"mean_pairwise_cos": sc["mean_pairwise_cos"],
                               "frac_negative_pairs": sc["frac_negative_pairs"], "sign_agreement": sign_agree}
            print(f"{name:<22}{sc['mean_pairwise_cos']:>10.3f}{sign_agree:>11.3f}{cenAP.get(name,0):>7.2f}")

    out = {"checkpoint": a.checkpoint, "n_clients": N, "max_steps": a.max_steps,
           "per_module": per_module, "per_class_heatmap": per_class}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n[grad-conflict] wrote {a.out}")


if __name__ == "__main__":
    main()
