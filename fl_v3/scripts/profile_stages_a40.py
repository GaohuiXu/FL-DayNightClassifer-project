"""Per-stage runtime profiler for the AD detector training step (Cycle-04 D14 Phase-1 A).

Builds the REAL nuScenes detection Task model + ONE real client's trainloader (log-group trainval,
frozen Swin-T, batch 16 — the headline config), runs N real training steps under the
determinism-neutral StepProfiler, in BOTH numeric regimes (fp32 then tf32), and reports:
  * per-stage wall-clock (dataloader wait · the 8 forward submodules · loss · backward · optimizer);
  * the MEASURED frozen-backbone share of the step (settles D12's inferred 80–90%);
  * the per-stage TF32 speedup (fp32 mean / tf32 mean) + the end-to-end TF32 speedup;
  * peak GPU memory + sampled GPU/CPU utilization.

Run via run_profile_a40.sh (A40:1). Instrumentation only — never feeds a scientific checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, "fl_v3/src")
import torch

from fl_v3.training.tasks import get_task
from fl_v3.utils.profiling import StepProfiler, max_mem_mb, reset_peak_mem
from fl_v3.utils.runtime import (
    derive_seed,
    enforce_determinism,
    precision_state,
    seed_everything,
)

# The detector's top-level forward submodules, in execution order (detector.forward).
FORWARD_STAGES = [
    "preprocess", "camera_backbone", "camera_neck", "view_transform",
    "lidar_encoder", "fusion", "bev_neck", "head",
]
LOOP_STAGES = ["dataloader_wait", "forward_total", "loss", "backward", "optimizer_step"]


class UtilSampler:
    """Background sampler of GPU util/mem + CPU util (nvidia-smi + /proc/stat). Best-effort."""

    def __init__(self, period_s: float = 0.5):
        self.period_s = period_s
        self._stop = threading.Event()
        self._gpu_util, self._gpu_mem, self._cpu_util = [], [], []
        self._t = None

    def _cpu_times(self):
        with open("/proc/stat") as f:
            parts = f.readline().split()[1:]
        vals = list(map(int, parts))
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        return sum(vals), idle

    def _run(self):
        prev = self._cpu_times()
        while not self._stop.wait(self.period_s):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"], text=True, timeout=5
                ).strip().splitlines()[0]
                gu, gm = (x.strip() for x in out.split(","))
                self._gpu_util.append(float(gu)); self._gpu_mem.append(float(gm))
            except Exception:
                pass
            try:
                tot, idle = self._cpu_times()
                dtot, didle = tot - prev[0], idle - prev[1]
                if dtot > 0:
                    self._cpu_util.append(100.0 * (1.0 - didle / dtot))
                prev = (tot, idle)
            except Exception:
                pass

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start(); return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)

    def report(self):
        m = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
        return {
            "gpu_util_mean_pct": m(self._gpu_util), "gpu_util_max_pct": max(self._gpu_util, default=0.0),
            "gpu_mem_used_mean_mb": m(self._gpu_mem), "gpu_mem_used_max_mb": max(self._gpu_mem, default=0.0),
            "cpu_util_mean_pct": m(self._cpu_util), "n_samples": len(self._gpu_util),
        }


def profile_regime(numeric_mode, run_config, cid, steps, warmup, device, level="strict", compile_model=False,
                   fixed_batch=False):
    """Run (warmup+steps) real training steps under one numeric regime + determinism level.

    fixed_batch=True (E1, overcommit-ceiling diag): fetch ONE batch, move it on-device, and loop the
    SAME tensors — the dataloader is bypassed so the measurement isolates pure GPU compute + per-step
    launch/CPU overhead under concurrency (no data pipeline, no host->device transfer)."""
    enforce_determinism(strict=True, numeric_mode=numeric_mode, level=level)
    use_amp = (level == "relaxed") and device.type == "cuda"
    seed = int(run_config.get("seed", 42))
    # Mirror the real per-client seeding so the trajectory matches a real round-1 client.
    seed_everything(derive_seed(seed, cid, 1))

    task = get_task("nuscenes_detection")
    model = task.build_model(run_config).to(device)
    if compile_model:  # L8 (D15): torch.compile the frozen backbone (compile cost amortized over steps)
        model.camera_backbone = torch.compile(model.camera_backbone)
    criterion = task.build_criterion(run_config)
    cdata = task.client_data(cid, run_config)
    loader = cdata.trainloader
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=float(run_config.get("learning-rate", 3e-3)))
    model.train()

    prof = StepProfiler(use_cuda=(device.type == "cuda"))
    prof.attach_module_timers(model, FORWARD_STAGES)
    reset_peak_mem()

    from fl_v3.training.loop import _unpack_batch

    def _step(inputs, targets):
        optimizer.zero_grad()
        with prof.section("forward_total"):
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(inputs)
            else:
                out = model(inputs)
        with prof.section("loss"):
            loss = criterion(out, targets)
        with prof.section("backward"):
            loss.backward()
        with prof.section("optimizer_step"):
            optimizer.step()

    total = warmup + steps
    done = 0
    with UtilSampler() as sampler:
        if fixed_batch:
            # E1: bypass the dataloader entirely — one batch on-device, looped.
            fixed_inputs, fixed_targets = _unpack_batch(next(iter(loader)), device)
            while done < total:
                prof.records.setdefault("dataloader_wait", []).append(0.0)
                _step(fixed_inputs, fixed_targets)
                done += 1
        else:
            while done < total:
                it = iter(loader)
                while done < total:
                    t0 = time.perf_counter()
                    try:
                        batch = next(it)
                    except StopIteration:
                        break
                    # dataloader wait recorded manually (the fetch happened above)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    prof.records.setdefault("dataloader_wait", []).append(time.perf_counter() - t0)

                    inputs, targets = _unpack_batch(batch, device)
                    _step(inputs, targets)
                    done += 1
    prof.remove_hooks()
    util = sampler.report()

    summ = prof.summary(drop_warmup=warmup)
    fwd_total = sum(summ[s]["total_ms"] for s in FORWARD_STAGES if s in summ)
    # Per-step wall = the loop stages (dataloader + forward_total + loss + backward + optim).
    step_total = sum(summ[s]["total_ms"] for s in LOOP_STAGES if s in summ)
    backbone = summ.get("camera_backbone", {}).get("total_ms", 0.0)
    return {
        "numeric_mode": numeric_mode,
        "precision_state": precision_state(),
        "steps_measured": steps,
        "warmup_dropped": warmup,
        "per_stage": summ,
        "forward_total_ms": fwd_total,
        "step_total_ms": step_total,
        "backbone_share_of_forward": (backbone / fwd_total) if fwd_total else 0.0,
        "backbone_share_of_step": (backbone / step_total) if step_total else 0.0,
        "peak_gpu_mem_mb": max_mem_mb(),
        "utilization": util,
        "mean_step_ms": step_total / steps if steps else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="JSON run-config (the headline trainval config)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--client-id", type=int, default=0)
    ap.add_argument("--out", default="fl_v3/collab/speedup/profile_stages_a40.json")
    ap.add_argument("--modes", default="fp32,tf32")
    ap.add_argument("--level", default="strict", choices=["strict", "relaxed"],
                    help="D15 determinism level (relaxed: cudnn.benchmark + atomics + bf16 autocast)")
    ap.add_argument("--compile", action="store_true", help="L8: torch.compile the frozen backbone")
    ap.add_argument("--fixed-batch", action="store_true",
                    help="E1 overcommit-ceiling: loop one on-device batch (bypass dataloader)")
    ap.add_argument("--cache-dir", default="", help="override nuscenes-cache-dir (abs path)")
    ap.add_argument("--dataroot", default="", help="override nuscenes-dataroot (abs path)")
    ap.add_argument("--num-workers", type=int, default=-1,
                    help="override dataloader num-workers (-1 = config default); E3 host-RAM lever")
    a = ap.parse_args()

    with open(a.config, encoding="utf-8") as f:
        run_config = json.load(f)
    if a.cache_dir:
        run_config["nuscenes-cache-dir"] = a.cache_dir
    if a.dataroot:
        run_config["nuscenes-dataroot"] = a.dataroot
    if a.num_workers >= 0:
        run_config["num-workers"] = a.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[profile] WARNING: no CUDA — TF32 will not engage; numbers are login-node only.")

    results = {}
    for mode in a.modes.split(","):
        mode = mode.strip()
        print(f"\n===== profiling regime: {mode} level={a.level} (client {a.client_id}, "
              f"{a.warmup} warmup + {a.steps} steps) =====", flush=True)
        r = profile_regime(mode, run_config, a.client_id, a.steps, a.warmup, device, level=a.level,
                           compile_model=a.compile, fixed_batch=a.fixed_batch)
        results[mode] = r
        print(f"[profile] {mode}: mean_step={r['mean_step_ms']:.1f}ms  "
              f"forward={r['forward_total_ms']/r['steps_measured']:.1f}ms/step  "
              f"backbone_share_of_step={r['backbone_share_of_step']:.1%}  "
              f"peak_mem={r['peak_gpu_mem_mb']:.0f}MB  gpu_util={r['utilization']['gpu_util_mean_pct']:.0f}%")
        print(f"  per-stage (ms/step, share-of-step):")
        for s in LOOP_STAGES:
            if s == "forward_total":
                for fs in FORWARD_STAGES:
                    st = r["per_stage"].get(fs)
                    if st:
                        print(f"    forward.{fs:<16} {st['mean_ms']:8.2f}  "
                              f"{st['total_ms']/r['step_total_ms']:6.1%}")
            st = r["per_stage"].get(s)
            if st:
                print(f"    {s:<24} {st['mean_ms']:8.2f}  {st['total_ms']/r['step_total_ms']:6.1%}")

    # TF32 vs FP32 per-stage speedup + end-to-end.
    speedup = {}
    if "fp32" in results and "tf32" in results:
        f, t = results["fp32"]["per_stage"], results["tf32"]["per_stage"]
        for s in FORWARD_STAGES + LOOP_STAGES:
            if s in f and s in t and t[s]["mean_ms"] > 0:
                speedup[s] = f[s]["mean_ms"] / t[s]["mean_ms"]
        speedup["end_to_end_step"] = (
            results["fp32"]["mean_step_ms"] / results["tf32"]["mean_step_ms"]
            if results["tf32"]["mean_step_ms"] else 0.0
        )
        print("\n===== TF32 speedup (fp32/tf32, >1 = tf32 faster) =====")
        for s, v in speedup.items():
            print(f"  {s:<28} {v:.2f}x")

    out = {"config": a.config, "client_id": a.client_id, "regimes": results, "tf32_speedup": speedup}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"\n[profile] wrote {a.out}")


if __name__ == "__main__":
    main()
