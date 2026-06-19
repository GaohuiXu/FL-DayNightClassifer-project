"""Profiling is determinism-NEUTRAL + the numeric-mode regime is wired (Cycle-04 D14 Phase-1 A/C).

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
    enforce_determinism(strict=True, numeric_mode="fp32")
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


# --- C: numeric-mode regime wiring (regression) ---
@pytest.mark.parametrize("mode,matmul,cudnn,prec", [
    ("fp32", False, False, "highest"),
    ("tf32", True, True, "high"),
])
def test_numeric_mode_sets_flags(mode, matmul, cudnn, prec):
    enforce_determinism(strict=True, numeric_mode=mode)
    assert torch.backends.cuda.matmul.allow_tf32 is matmul
    assert torch.backends.cudnn.allow_tf32 is cudnn
    assert torch.get_float32_matmul_precision() == prec
    ps = precision_state()
    assert ps["cuda_matmul_allow_tf32"] is matmul
    assert ps["float32_matmul_precision"] == prec
    # reset to the strict fp32 default so test order can't leak the tf32 regime.
    enforce_determinism(strict=True, numeric_mode="fp32")


def test_numeric_mode_bad_raises():
    with pytest.raises(ValueError):
        enforce_determinism(strict=True, numeric_mode="bf16")
    enforce_determinism(strict=True, numeric_mode="fp32")
