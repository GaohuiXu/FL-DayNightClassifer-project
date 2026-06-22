"""MCR Phase-1 per-component profiler for the UNFROZEN model (single-GPU, A100).

Answers, on ONE A100, before any heavy run (orchestrator's prerequisite):
  1. Per-component teardown — per-top-level-module FORWARD (sync-bracketed hooks) + loop phases
     (dataloader-wait / forward / loss / backward / optimizer) + a torch.profiler CUDA-kernel
     breakdown that attributes the BACKWARD cost (kernel names map to components: cudnn conv =
     LSS/PointPillars/neck/head; the windowed-attention matmul+softmax = Swin; layer_norm = Swin).
  2. Is the GPU actually used? — live nvidia-smi util sampling + total CUDA-time vs wall-time
     (compute-bound vs launch/dataloader-bound).
  3. activation-checkpoint overhead — step time + peak mem with ckpt ON vs OFF.
  4. batch-size effect on utilisation — a small sweep (step time + util + peak mem; OOM-tolerant).

HOST-RAM DISCIPLINE: the (heavy) dataset is built ONCE and reused; each regime builds + FREES its own
loader/model so only one is ever alive (a 1-GPU SLURM alloc caps host RAM at ~1/4 the node). Results are
written incrementally so a late OOM still leaves a usable JSON. SMOKE/diagnostic (Rule #3). Determinism-
neutral (StepProfiler only sync()s + reads the clock). SHORT SLURM job (<<3h): run_p1_profile_a100.sh.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, "fl_v3/src")
import torch

from fl_v3.utils.runtime import enforce_determinism, precision_state, seed_everything, truthy
from fl_v3.utils.profiling import StepProfiler, max_mem_mb, reset_peak_mem
from fl_v3.training.tasks import get_task
from fl_v3.training.loop import _unpack_batch

FORWARD_STAGES = ["preprocess", "camera_backbone", "camera_neck", "view_transform",
                  "lidar_encoder", "fusion", "bev_neck", "head"]


class UtilSampler:
    """GPU-util sampler via a STREAMING nvidia-smi subprocess read through a pipe.

    The polling runs INSIDE nvidia-smi (``--loop-ms``); python only reads the pipe in a thread — so it
    does NOT hang the way ``subprocess.check_output`` from a thread during CUDA activity did (the bug that
    made the first profile report util=None). One sampler per regime → per-regime util."""

    def __init__(self, period_ms: int = 200):
        self.period_ms = period_ms
        self.samples = []          # (util_pct, mem_used_mib)
        self.proc = None
        self.thread = None
        self._stop_evt = threading.Event()

    def start(self):
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits", f"--loop-ms={self.period_ms}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        except Exception:
            self.proc = None

    def _read(self):
        if self.proc is None:
            return
        for line in self.proc.stdout:                # blocks on the pipe (no CUDA interaction → no hang)
            if self._stop_evt.is_set():
                break
            try:
                u, m = line.strip().split(",")       # 1 line/interval (job has 1 exclusive GPU)
                self.samples.append((float(u), float(m)))
            except Exception:
                pass

    def stop(self):
        self._stop_evt.set()
        if self.proc is not None:
            try:
                self.proc.terminate(); self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def report(self) -> dict:
        if not self.samples:
            return {"util_mean": None, "util_p50": None, "util_p90": None, "util_max": None,
                    "mem_used_max_mib": None, "n": 0}
        us = sorted(s[0] for s in self.samples)
        ms = [s[1] for s in self.samples]
        pct = lambda p: us[min(len(us) - 1, int(p * len(us)))]
        return {"util_mean": round(sum(us) / len(us), 1), "util_p50": pct(0.5), "util_p90": pct(0.9),
                "util_max": max(us), "mem_used_max_mib": max(ms), "n": len(us)}


def _make_loader(dataset, run_config):
    from fl_v3.data.nuscenes.dataset import make_loader
    from fl_v3.models.fusion.collate import detection_collate_fn
    return make_loader(dataset, batch_size=int(run_config.get("batch-size", 16)), shuffle=True,
                       num_workers=int(run_config.get("num-workers", 4)),
                       seed=int(run_config.get("seed", 42)), collate_fn=detection_collate_fn)


def _model_opt(task, run_config, device):
    seed_everything(int(run_config.get("seed", 42)))
    model = task.build_model(run_config).to(device)
    # --- optimization toggles (measured one-at-a-time) ---
    if truthy(run_config.get("det-channels-last", False)):
        # Only Conv2d weights are 4D — a global model.to(channels_last) RAISES on Swin's non-4D params
        # (LN/linear/rel-pos-bias). Convert just the conv modules → tensor-core NHWC convs on the BEV stack.
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.to(memory_format=torch.channels_last)
    if truthy(run_config.get("compile-backbone", False)):
        model.camera_backbone = torch.compile(model.camera_backbone)   # static-shape Swin subgraph only
    if truthy(run_config.get("det-compile-bev", False)):
        # static-shape BEV conv stack (A5): camera_neck + fusion + bev_neck + head. NOT view_transform
        # (ragged LSS splat) or lidar_encoder (variable pillar count) — those stay eager.
        model.camera_neck = torch.compile(model.camera_neck)
        model.fusion = torch.compile(model.fusion)
        model.bev_neck = torch.compile(model.bev_neck)
        model.head = torch.compile(model.head)
    crit = task.build_criterion(run_config)
    base_lr = float(run_config.get("learning-rate", 3e-3))
    bb_mult = float(run_config.get("det-backbone-lr-mult", 0.1))
    bb = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("camera_backbone.")]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("camera_backbone.")]
    groups = ([{"params": rest, "lr": base_lr}, {"params": bb, "lr": base_lr * bb_mult}] if bb
              else [{"params": rest, "lr": base_lr}])
    _fused = (device.type == "cuda" and not torch.backends.cudnn.deterministic)
    return model, crit, torch.optim.Adam(groups, lr=base_lr, fused=_fused), len(bb), len(rest)


def _free_loader(loader, it):
    try:
        if it is not None and hasattr(it, "_shutdown_workers"):
            it._shutdown_workers()
    except Exception:
        pass
    del it, loader
    gc.collect()


def _step(model, crit, batch, opt, device, use_amp, grad_clip, prof=None):
    inputs, targets = _unpack_batch(batch, device)
    opt.zero_grad()
    if prof is not None:
        with prof.section("forward_total"):
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(inputs)
                out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
            else:
                out = model(inputs)
        with prof.section("loss"):
            loss = crit(out, targets)
        with prof.section("backward"):
            loss.backward()
        with prof.section("optimizer"):
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            opt.step()
    else:
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(inputs)
            out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
        else:
            out = model(inputs)
        loss = crit(out, targets)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        opt.step()
    return float(loss.detach())


def profile_regime(task, dataset, run_config, device, steps, warmup, label, instrument=True):
    """One regime: per-module fwd + loop phases + util + peak mem + wall step time. Frees its loader/model."""
    model, crit, opt, n_bb, n_rest = _model_opt(task, run_config, device)
    loader = _make_loader(dataset, run_config)
    use_amp = (not torch.backends.cudnn.deterministic) and device.type == "cuda"
    grad_clip = float(run_config.get("grad-clip-norm", 0.0))
    model.train()
    prof = StepProfiler(use_cuda=(device.type == "cuda")) if instrument else None
    if prof is not None:
        prof.attach_module_timers(model, FORWARD_STAGES)
    reset_peak_mem()
    sampler = UtilSampler()      # started AFTER warmup so it samples only the timed steps (excludes the
    it = iter(loader)            # GPU-idle torch.compile Inductor compilation that happens during warmup).
    wall, oom = [], None
    try:
        for s in range(warmup + steps):
            if s == warmup:
                sampler.start()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if prof is not None:
                with prof.section("dataloader"):
                    batch = next(it)
            else:
                batch = next(it)
            _step(model, crit, batch, opt, device, use_amp, grad_clip, prof=prof)
            if device.type == "cuda":
                torch.cuda.synchronize()
            wall.append(time.perf_counter() - t0)
    except StopIteration:
        pass
    except RuntimeError as e:
        oom = str(e).splitlines()[0][:160]
    finally:
        sampler.stop()
        if prof is not None:
            prof.remove_hooks()
    wall = wall[warmup:] if len(wall) > warmup else wall
    res = {
        "label": label, "batch_size": int(run_config.get("batch-size")),
        "num_workers": int(run_config.get("num-workers", 4)),
        "activation_ckpt": truthy(run_config.get("det-activation-checkpoint", False)),
        "use_amp": use_amp, "n_backbone_tensors": n_bb, "n_rest_tensors": n_rest,
        "mean_step_ms": round(1e3 * sum(wall) / len(wall), 1) if wall else None,
        "util": sampler.report(), "peak_mem_mib": round(max_mem_mb(), 1), "oom": oom,
    }
    if prof is not None:
        res["per_stage"] = prof.summary(drop_warmup=warmup)
    _free_loader(loader, it)
    del model, opt, crit
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def kernel_breakdown(task, dataset, run_config, device, steps=6, warmup=3):
    """torch.profiler CUDA-kernel breakdown → attributes BACKWARD by kernel + compute-vs-launch."""
    from torch.profiler import ProfilerActivity, profile
    model, crit, opt, _, _ = _model_opt(task, run_config, device)
    loader = _make_loader(dataset, run_config)
    use_amp = (not torch.backends.cudnn.deterministic) and device.type == "cuda"
    grad_clip = float(run_config.get("grad-clip-norm", 0.0))
    model.train()
    it = iter(loader)
    try:
        for _ in range(warmup):
            _step(model, crit, next(it), opt, device, use_amp, grad_clip)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(steps):
                _step(model, crit, next(it), opt, device, use_amp, grad_clip)
            torch.cuda.synchronize()
        ka = list(prof.key_averages())
        # torch 2.7 renamed self_cuda_time_total → self_device_time_total (device-agnostic refactor).
        def _cuda_us(e):
            return float(getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0) or 0)
        total_cuda = sum(_cuda_us(e) for e in ka) / 1e3
        total_cpu = sum(e.self_cpu_time_total for e in ka) / 1e3
        top = sorted(ka, key=_cuda_us, reverse=True)[:25]
        rows = [{"kernel": e.key[:70], "self_cuda_ms_total": round(_cuda_us(e) / 1e3, 2),
                 "count": int(e.count)} for e in top]
        out = {"steps": steps, "total_self_cuda_ms": round(total_cuda, 1),
               "total_self_cpu_ms": round(total_cpu, 1),
               "cuda_ms_per_step": round(total_cuda / steps, 1), "top_kernels": rows}
    except Exception as e:
        out = {"error": str(e).splitlines()[0][:160]}
    finally:
        _free_loader(loader, it)
        del model, opt, crit
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _load_cfg(path, overrides):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for ov in overrides:
        k, _, v = ov.partition("=")
        for cast in (int, float):
            try:
                v = cast(v); break
            except ValueError:
                pass
        if v in ("true", "false"):
            v = (v == "true")
        cfg[k] = v
    return cfg


def _write(out, results):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fl_v3/configs/p1_unfrozen.json")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--num-workers", type=int, default=4, help="profiler workers (RAM-bounded; real run can use more)")
    ap.add_argument("--batch-sizes", default="16,24,32")
    ap.add_argument("--opt-compare", action="store_true",
                    help="measure the optimization sequence (baseline vs +compile vs +channels_last vs both) "
                         "at batch 16 ckpt-OFF instead of the standard regimes")
    ap.add_argument("--out", default="fl_v3/collab/model_capability/p1_profile_a100.json")
    ap.add_argument("overrides", nargs="*", default=[])
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("[p1-profile] no CUDA — this profile is meaningless off-GPU."); return 2
    enforce_determinism(strict=True, precision="bf16")
    device = torch.device("cuda")
    base = _load_cfg(a.config, a.overrides)
    base["det-freeze-backbone"] = False               # this profiler is for the UNFROZEN model
    base["num-workers"] = a.num_workers               # RAM-bounded loader
    print(f"[p1-profile] device={torch.cuda.get_device_name(0)} precision={precision_state()['precision']} "
          f"config={a.config} workers={a.num_workers}", flush=True)

    # Build the (heavy) dataset ONCE; reuse it for every regime.
    task = get_task("nuscenes_detection")
    cdata = task.client_data(0, base)
    dataset = cdata.trainloader.dataset
    del cdata; gc.collect()
    print(f"[p1-profile] dataset built once: {len(dataset)} samples", flush=True)

    results = {"device": torch.cuda.get_device_name(0), "precision": precision_state()["precision"],
               "steps": a.steps, "warmup": a.warmup, "num_workers": a.num_workers,
               "regimes": [], "kernel_breakdown": None}

    if a.opt_compare:
        # Optimization sequence — all at batch 16, ckpt OFF (the chosen baseline). Each combo: step time +
        # util + peak mem; the baseline also gets the per-component teardown. compile-time is absorbed by warmup.
        base0 = dict(base); base0["det-activation-checkpoint"] = False
        base0["batch-size"] = int(base.get("batch-size", 16))
        # Code-wins re-profile: baseline (now incl. loss-fix + non_blocking + fused-Adam), the current
        # recipe (compile+sdpa), and + BEV-stack compile coverage (A5).
        combos = [("baseline", {}),
                  ("compile+sdpa", {"compile-backbone": True, "det-swin-sdpa": True}),
                  ("compile+sdpa+bev", {"compile-backbone": True, "det-swin-sdpa": True, "det-compile-bev": True})]
        for i, (name, extra) in enumerate(combos):
            cfg = dict(base0); cfg.update(extra)
            print(f"[p1-profile] opt[{name}] (batch={cfg['batch-size']}, ckpt OFF)…", flush=True)
            # compile traces on first call → give it extra warmup so timed steps are post-compile.
            wu = a.warmup + (8 if extra.get("compile-backbone") else 0)
            try:
                results["regimes"].append(
                    profile_regime(task, dataset, cfg, device, a.steps, wu, name, instrument=(i == 0)))
            except Exception as e:
                results["regimes"].append({"label": name, "error": str(e).splitlines()[0][:160],
                                           "mean_step_ms": None, "util": {"util_mean": None}, "peak_mem_mib": None, "oom": "exception"})
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            _write(a.out, results)
        # kernel breakdown of the BEST combo (compile+sdpa) — confirm the attention bmm is gone.
        print("[p1-profile] kernel breakdown of compile+sdpa…", flush=True)
        cfgc = dict(base0); cfgc["compile-backbone"] = True; cfgc["det-swin-sdpa"] = True
        results["kernel_breakdown"] = kernel_breakdown(task, dataset, cfgc, device, steps=6, warmup=14)
        _write(a.out, results)
        print("\n===== OPT-COMPARE SUMMARY (batch 16, ckpt OFF) =====")
        b = results["regimes"][0].get("mean_step_ms")
        for r in results["regimes"]:
            u = r.get("util", {}); s = r.get("mean_step_ms")
            spd = f"{b / s:.2f}x" if (s and b) else "?"
            print(f"[{r['label']:>14}] step={s}ms ({spd})  util_mean={u.get('util_mean')}% p90={u.get('util_p90')}% "
                  f"max={u.get('util_max')}%  peak_mem={r.get('peak_mem_mib')}MiB  oom={r.get('oom')} "
                  f"{('ERR='+r['error']) if r.get('error') else ''}")
        kb = results.get("kernel_breakdown")
        if kb and "top_kernels" in kb:
            print(f"\n  COMPILED-model kernels: {kb['cuda_ms_per_step']} ms CUDA/step "
                  f"(self CUDA {kb['total_self_cuda_ms']}ms vs self CPU {kb['total_self_cpu_ms']}ms)")
            for k in kb["top_kernels"][:15]:
                print(f"      {k['self_cuda_ms_total']:8.2f} ms  x{k['count']:<5} {k['kernel']}")
        elif kb:
            print(f"\n  compiled kernel breakdown ERROR: {kb.get('error')}")
        print(f"\n[p1-profile] wrote {a.out}")
        return 0

    # R1: main teardown — batch 16, ckpt ON, full instrumentation.
    cfg1 = dict(base); cfg1["det-activation-checkpoint"] = True; cfg1["batch-size"] = int(base.get("batch-size", 16))
    print("[p1-profile] R1: per-component teardown (batch=%s, ckpt=ON)…" % cfg1["batch-size"], flush=True)
    results["regimes"].append(profile_regime(task, dataset, cfg1, device, a.steps, a.warmup, "ckpt_on", instrument=True))
    _write(a.out, results)

    # R1b: kernel breakdown (attributes backward).
    print("[p1-profile] R1b: torch.profiler kernel breakdown…", flush=True)
    results["kernel_breakdown"] = kernel_breakdown(task, dataset, cfg1, device, steps=6, warmup=3)
    _write(a.out, results)

    # R2: activation-ckpt OFF (overhead + does it fit on 40GB?).
    cfg2 = dict(base); cfg2["det-activation-checkpoint"] = False
    print("[p1-profile] R2: ckpt OFF (overhead / OOM check)…", flush=True)
    results["regimes"].append(profile_regime(task, dataset, cfg2, device, max(8, a.steps // 2), a.warmup, "ckpt_off", instrument=False))
    _write(a.out, results)

    # R3: batch-size sweep (ckpt ON) — utilisation vs batch.
    for bs in [int(x) for x in a.batch_sizes.split(",")]:
        if bs == int(base.get("batch-size", 16)):
            continue
        cfgb = dict(base); cfgb["det-activation-checkpoint"] = True; cfgb["batch-size"] = bs
        print(f"[p1-profile] R3: batch={bs} (ckpt ON)…", flush=True)
        results["regimes"].append(profile_regime(task, dataset, cfgb, device, max(8, a.steps // 2), a.warmup, f"batch_{bs}", instrument=False))
        _write(a.out, results)

    # readable summary
    print("\n===== P1 PROFILE SUMMARY =====")
    for r in results["regimes"]:
        u = r["util"]
        print(f"[{r['label']:>9}] batch={r['batch_size']} ckpt={r['activation_ckpt']} "
              f"step={r['mean_step_ms']}ms util_mean={u['util_mean']}% p50={u.get('util_p50')}% "
              f"p90={u.get('util_p90')}% max={u['util_max']}% peak_mem={r['peak_mem_mib']}MiB oom={r['oom']}")
        if "per_stage" in r:
            for s in FORWARD_STAGES + ["dataloader", "forward_total", "loss", "backward", "optimizer"]:
                st = r["per_stage"].get(s)
                if st:
                    print(f"            {s:>16}: {st['mean_ms']:8.2f} ms  ({100*st['share_of_summed']:4.1f}% of summed)")
    kb = results["kernel_breakdown"]
    if kb and "top_kernels" in kb:
        print(f"\n  kernel breakdown: {kb['cuda_ms_per_step']} ms CUDA/step over {kb['steps']} steps "
              f"(total self CUDA {kb['total_self_cuda_ms']}ms vs self CPU {kb['total_self_cpu_ms']}ms)")
        for k in kb["top_kernels"][:15]:
            print(f"      {k['self_cuda_ms_total']:8.2f} ms  x{k['count']:<5} {k['kernel']}")
    elif kb:
        print(f"\n  kernel breakdown ERROR: {kb.get('error')}")
    print(f"\n[p1-profile] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
