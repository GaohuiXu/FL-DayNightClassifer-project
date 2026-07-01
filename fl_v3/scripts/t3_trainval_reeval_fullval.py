"""Re-evaluate the trainval final checkpoints on the FULL val split (T3 Codex F1 fix).

The trainval gap run capped server eval at det-eval-limit=256 (a fixed
`sorted(sample_token)[:256]` subset) for speed, so the committed +0.2235 gap was a
256-sample-val proxy. This loads the two saved FULL-model checkpoints (Swin-T headline,
strict=True) and re-evaluates them on the ENTIRE v1.0-trainval val split (6019 samples,
det-eval-limit=0) to report the accurate IID-vs-log-group recall@2m gap. Eval-only (no
training); historical results were A40-comparable, while new validation should record the Arrhenius regime.
"""
from __future__ import annotations

import json
import os

import torch


def main() -> None:
    from fl_v3.training.tasks import get_task
    from fl_v3.utils.runtime import enforce_determinism, seed_everything

    cfg = json.load(open("fl_v3/configs/t3_trainval.json", encoding="utf-8"))
    cfg["det-eval-limit"] = 0  # FULL val split (declared scope: all val keyframes)
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    enforce_determinism(strict=True)
    seed_everything(int(cfg.get("seed", 42)))

    task = get_task("nuscenes_detection")
    device = torch.device(cfg["device"])
    crit = task.build_criterion(cfg)
    out_root = cfg["output-dir"]
    results = {}
    for mode in ("iid", "log_group"):
        ckpt = os.path.join(out_root, f"t3_trainval_{mode}", "final_model.pt")
        model = task.build_model(cfg).to(device)
        sd = torch.load(ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, f"{mode}: checkpoint not a full model"
        loader = task.eval_loader(cfg)  # full val (det-eval-limit=0)
        res = task.evaluate(model, loader, crit, device, cfg)
        results[mode] = res
        print(f"[reeval] {mode}: {res}", flush=True)

    ri = float(results["iid"]["proxy_recall_at_2m"])
    rl = float(results["log_group"]["proxy_recall_at_2m"])
    n_eval = float(results["iid"]["num-eval-examples"])
    summary = {
        "scope": "FULL v1.0-trainval val split (det-eval-limit=0)",
        "n_eval_examples": n_eval,
        "recall_at_2m_iid": ri,
        "recall_at_2m_log_group": rl,
        "non_iid_gap": ri - rl,
        "results": results,
    }
    print(f"[reeval] FULL-VAL (n={n_eval:.0f}) recall@2m  IID={ri:.4f}  log_group={rl:.4f}  "
          f"non-IID gap={ri - rl:+.4f}", flush=True)
    with open("fl_v3/collab/T3/trainval_fullval_reeval.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
