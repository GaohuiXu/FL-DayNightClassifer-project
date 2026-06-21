"""Profiling is determinism-NEUTRAL + the precision regime is wired (Cycle-04 D14 Phase-1 A/C; D16 knob).

The load-bearing claim: attaching the StepProfiler (sync-bracketed sections + forward hooks) to a
real train step changes TIMING only, never values or RNG — so a same-seed run with profiling ON is
byte-identical to profiling OFF. If this failed, profiling could not be trusted near a scientific
trajectory. Runs on the login node (CPU) — no GPU needed (the neutrality argument is GPU-agnostic).
"""
from __future__ import annotations

import hashlib

import pytest
import torch
import torch.nn as nn

from fl_v3.utils.profiling import StepProfiler
from fl_v3.utils.runtime import enforce_determinism, precision_state, seed_everything


class _ProfNet(nn.Module):
    """Tiny model with NAMED top-level submodules (mirrors the detector's stage layout)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def _param_checksum(model: nn.Module) -> str:
    h = hashlib.sha256()
    for _, p in sorted(model.state_dict().items()):
        a = p.detach().cpu().contiguous().numpy()
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _train_micro(profile: bool, seed: int = 7) -> tuple[str, bytes]:
    """One deterministic 3-step train micro-loop; return (param_checksum, torch_rng_state)."""
    enforce_determinism(strict=True, precision="fp32")
    seed_everything(seed)
    model = _ProfNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    prof = StepProfiler(use_cuda=False) if profile else None
    if prof:
        prof.attach_module_timers(model, ["fc1", "act", "fc2"])
    for _ in range(3):
        x = torch.randn(8, 16, generator=g)
        y = torch.randn(8, 4, generator=g)
        opt.zero_grad()
        if prof:
            with prof.section("forward_total"):
                out = model(x)
            with prof.section("loss"):
                loss = crit(out, y)
            with prof.section("backward"):
                loss.backward()
            with prof.section("optimizer_step"):
                opt.step()
        else:
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
    if prof:
        prof.remove_hooks()
        # the profiler recorded SOMETHING for every attached/sectioned stage
        for k in ("fc1", "fc2", "forward_total", "loss", "backward", "optimizer_step"):
            assert k in prof.records and len(prof.records[k]) == 3, f"missing stage {k}"
    return _param_checksum(model), torch.random.get_rng_state().clone().numpy().tobytes()


def test_profiling_is_byte_identical():
    """Profiling ON must produce the IDENTICAL trained params + RNG state as profiling OFF."""
    chk_off, rng_off = _train_micro(profile=False)
    chk_on, rng_on = _train_micro(profile=True)
    assert chk_on == chk_off, "profiling changed the trained parameters (NOT determinism-neutral)"
    assert rng_on == rng_off, "profiling advanced/changed the RNG state (NOT determinism-neutral)"


def test_profiling_summary_shape():
    chk, _ = _train_micro(profile=True)
    assert isinstance(chk, str) and len(chk) == 64


# --- C: precision regime wiring (regression; D16 single knob) ---
@pytest.mark.parametrize("precision,deterministic,benchmark,det_algos", [
    ("fp32", True, False, True),    # dev/determinism tool: byte-identical, autotuner off, atomics RAISE
    ("bf16", False, True, False),   # science: autotuner on, atomic scatter + AMP allowed (not byte-identical)
])
def test_precision_sets_flags(precision, deterministic, benchmark, det_algos):
    enforce_determinism(strict=True, precision=precision)
    # TF32 is retired (D16): BOTH regimes run residual fp32 ops at true IEEE FP32.
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"
    # the precision regime drives the determinism / autotuner / atomics flags:
    assert torch.backends.cudnn.deterministic is deterministic
    assert torch.backends.cudnn.benchmark is benchmark
    assert torch.are_deterministic_algorithms_enabled() is det_algos
    ps = precision_state()
    assert ps["precision"] == precision
    assert ps["determinism_level"] == ("strict" if deterministic else "relaxed")
    assert ps["float32_matmul_precision"] == "highest"
    # reset to the strict fp32 dev default so test order can't leak the bf16/relaxed regime.
    enforce_determinism(strict=True, precision="fp32")


def test_precision_bad_raises():
    # "tf32" is RETIRED under D16 (was a valid numeric-mode); only {bf16, fp32} are valid now.
    with pytest.raises(ValueError):
        enforce_determinism(strict=True, precision="tf32")
    enforce_determinism(strict=True, precision="fp32")
