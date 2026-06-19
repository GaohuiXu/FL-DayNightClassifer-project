"""Per-stage runtime profiling — instrumentation ONLY, determinism-neutral (Cycle-04 D14 Phase-1 A).

Settles D12's *inferred* "80–90% backbone" with a MEASURED per-stage breakdown. Two timing scopes:

  * **Loop-level** sections (``with prof.section("backward"): ...``) — dataloader wait, loss,
    backward, optimizer step — bracketed by ``torch.cuda.synchronize()`` so each stage's GPU work
    is fully attributed to it (the charter's "synchronize() around each").
  * **Module-level** forward hooks (``prof.attach_module_timers(model, [...])``) — one timer per
    named top-level submodule (preprocess · camera_backbone · camera_neck · view_transform ·
    lidar_encoder · fusion · bev_neck · head), so the forward is decomposed without editing the
    model's ``forward``.

**Determinism guarantee (the load-bearing property — D14):** the hooks and sections only call
``torch.cuda.synchronize()`` (a stream wait — changes timing, never values) + ``perf_counter`` +
dict appends. They draw NO RNG and run NO tensor math, so a same-seed run with profiling ON is
BYTE-IDENTICAL to profiling OFF (asserted by ``tests/test_profiling_neutral.py``). Profiling must
NEVER feed a scientific checkpoint regardless, but this keeps it safe to attach during a real step.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch


class StepProfiler:
    """Accumulate per-stage wall-clock (seconds) over many steps; CUDA-sync-bracketed."""

    def __init__(self, use_cuda: Optional[bool] = None) -> None:
        self.use_cuda = torch.cuda.is_available() if use_cuda is None else bool(use_cuda)
        self.records: "Dict[str, List[float]]" = {}
        self._hook_handles: list = []

    def _sync(self) -> None:
        if self.use_cuda:
            torch.cuda.synchronize()

    def _record(self, name: str, dt: float) -> None:
        self.records.setdefault(name, []).append(float(dt))

    @contextmanager
    def section(self, name: str):
        """Time a loop-level stage; both ends ``synchronize()``-d so GPU work is attributed here."""
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self._record(name, time.perf_counter() - t0)

    def attach_module_timers(self, model: torch.nn.Module, names: List[str]) -> None:
        """Register sync-bracketed fwd pre/post hooks on each named top-level submodule.

        Hooks only sync + read the clock + append — no RNG, no math — so attaching them does NOT
        change the model output or RNG state (determinism-neutral). Attach only the TOP-LEVEL
        modules (no nesting) so per-module times sum to the forward without double-counting.
        """
        scratch: Dict[str, float] = {}
        for name in names:
            module = getattr(model, name, None)
            if module is None:
                continue

            def pre_hook(_m, _args, _name=name):
                self._sync()
                scratch[_name] = time.perf_counter()

            def post_hook(_m, _args, _out, _name=name):
                self._sync()
                if _name in scratch:
                    self._record(_name, time.perf_counter() - scratch[_name])

            self._hook_handles.append(module.register_forward_pre_hook(pre_hook))
            self._hook_handles.append(module.register_forward_hook(post_hook))

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def summary(self, drop_warmup: int = 0) -> dict:
        """Per-stage {mean_ms, median_ms, total_ms, n, share_of_summed}, dropping the first
        ``drop_warmup`` samples of each stage (CUDA autotune / allocator warm-up).

        ``share_of_summed`` is each stage's total over the sum of ALL recorded stage totals — a
        relative attribution (so the module-level forward stages + the loop-level stages compare
        on one denominator). The caller decides which keys are forward-internal vs loop-level.
        """
        import statistics

        out: Dict[str, dict] = {}
        for name, vals in self.records.items():
            v = vals[drop_warmup:] if drop_warmup and len(vals) > drop_warmup else vals
            if not v:
                continue
            out[name] = {
                "mean_ms": 1e3 * statistics.fmean(v),
                "median_ms": 1e3 * statistics.median(v),
                "total_ms": 1e3 * sum(v),
                "n": len(v),
            }
        grand = sum(s["total_ms"] for s in out.values()) or 1.0
        for s in out.values():
            s["share_of_summed"] = s["total_ms"] / grand
        return out


def max_mem_mb() -> float:
    """Peak CUDA memory allocated since last reset (MB); 0.0 on CPU."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_peak_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
