"""Server-side adaptive optimizer for the clean FedAvg path.

This framework-free FedOpt implementation applies an adaptive step to the
FedAvg pseudo-gradient ``Δ = aggregated − global`` (Reddi et al. 2021,
*Adaptive Federated Optimization*, ICLR).

Update (per parameter tensor), with η = ``server_lr``::

    Δ_t = aggregated − global                     # the average client delta (FedAvg already computed it)
    fedavg   : x_{t+1} = aggregated               # IDENTITY when server_lr == 1 (byte-identical baseline)
    fedavgm  : m_t = β1 m_{t-1} + (1−β1) Δ_t ;     x_{t+1} = x_t + η · m̂_t
    fedadam  : m_t = β1 m_{t-1} + (1−β1) Δ_t ;
               v_t = β2 v_{t-1} + (1−β2) Δ_t² ;     x_{t+1} = x_t + η · m̂_t / (√v̂_t + τ)

``m̂``/``v̂`` are bias-corrected (Adam convention; toggle with ``bias_correction``). State (m, v, t) lives
on the optimizer instance, which the strategy and local runner carry across rounds.

**Determinism (D16):** fp64 accumulation, no RNG, deterministic for a fixed input order — so it adds no new
determinism obligation. **Default-off byte-identity:** ``kind='fedavg'`` + ``server_lr=1.0`` returns the
aggregate UNCHANGED, so an existing run (no server-optimizer config) is bit-for-bit the old FedAvg.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

VALID_KINDS = ("fedavg", "fedavgm", "fedadam")


@dataclass
class ServerOptState:
    """Carried across rounds: first-moment ``m``, second-moment ``v``, step count ``t``."""

    m: Optional[List[np.ndarray]] = None
    v: Optional[List[np.ndarray]] = None
    t: int = 0


@dataclass
class ServerOptimizer:
    """FedOpt server optimizer. ``fedavg`` (η=1) is the identity (byte-identical to plain FedAvg).

    ``warmup_rounds`` linearly ramps the effective server LR 0→``server_lr`` over the first ``warmup_rounds``
    rounds — the judge-mandated guard against the FedAdam early-round blow-up (before ``v̂`` stabilizes, an
    un-warmed adaptive step ≈ ``server_lr·sign(Δ)`` on every one of the 33M weights, far larger than the
    trained weight scale). 0 ⇒ no warmup.
    """

    kind: str = "fedavg"
    server_lr: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.99
    tau: float = 1e-3
    bias_correction: bool = True
    warmup_rounds: int = 0
    state: ServerOptState = field(default_factory=ServerOptState)

    def __post_init__(self) -> None:
        self.kind = str(self.kind).lower()
        if self.kind not in VALID_KINDS:
            raise ValueError(f"unknown server-optimizer {self.kind!r}; expected one of {VALID_KINDS}")
        self.server_lr = float(self.server_lr)
        self.beta1 = float(self.beta1)
        self.beta2 = float(self.beta2)
        self.tau = float(self.tau)
        self.warmup_rounds = int(self.warmup_rounds)

    @property
    def is_identity(self) -> bool:
        """True when this optimizer leaves the aggregate UNCHANGED (the byte-identical baseline)."""
        return self.kind == "fedavg" and self.server_lr == 1.0 and self.warmup_rounds <= 0

    def _eff_lr(self) -> float:
        """server_lr with the linear warmup ramp applied at the current step (``state.t``)."""
        if self.warmup_rounds > 0:
            return self.server_lr * min(1.0, float(self.state.t) / float(self.warmup_rounds))
        return self.server_lr

    def step(
        self,
        global_params: List[np.ndarray],
        aggregated_params: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Apply the server-optimizer step. ``global_params`` = the global at round start;
        ``aggregated_params`` = the clean FedAvg aggregate (the target). Returns the
        new global (same length / dtypes as ``aggregated_params``)."""
        if self.is_identity:
            return aggregated_params  # plain FedAvg — no copy, no change (crown-jewel byte-identity)

        delta = [
            np.asarray(a, dtype=np.float64) - np.asarray(g, dtype=np.float64)
            for a, g in zip(aggregated_params, global_params)
        ]
        if self.state.m is None:
            self.state.m = [np.zeros_like(d) for d in delta]
            self.state.v = [np.zeros_like(d) for d in delta]
        self.state.t += 1
        t = self.state.t
        bc1 = (1.0 - self.beta1 ** t) if self.bias_correction else 1.0
        bc2 = (1.0 - self.beta2 ** t) if self.bias_correction else 1.0
        eff_lr = self._eff_lr()

        out: List[np.ndarray] = []
        for i, (g, d) in enumerate(zip(global_params, delta)):
            g64 = np.asarray(g, dtype=np.float64)
            if self.kind == "fedavg":  # scaled FedAvg step (η ≠ 1), no momentum
                update = d
            else:  # fedavgm / fedadam — EMA of the pseudo-gradient
                self.state.m[i] = self.beta1 * self.state.m[i] + (1.0 - self.beta1) * d
                m_hat = self.state.m[i] / bc1
                if self.kind == "fedadam":
                    self.state.v[i] = self.beta2 * self.state.v[i] + (1.0 - self.beta2) * (d * d)
                    v_hat = self.state.v[i] / bc2
                    update = m_hat / (np.sqrt(v_hat) + self.tau)
                else:  # fedavgm
                    update = m_hat
            new_k = g64 + eff_lr * update
            out.append(np.asarray(new_k, dtype=np.asarray(aggregated_params[i]).dtype))
        return out


def build_server_optimizer(run_config: dict) -> ServerOptimizer:
    """Construct the server optimizer from the flat run-config (default = identity FedAvg)."""
    return ServerOptimizer(
        kind=str(run_config.get("server-optimizer", "fedavg")),
        server_lr=float(run_config.get("server-lr", 1.0)),
        beta1=float(run_config.get("server-beta1", 0.9)),
        beta2=float(run_config.get("server-beta2", 0.99)),
        tau=float(run_config.get("server-tau", 1e-3)),
        bias_correction=bool(run_config.get("server-opt-bias-correction", True)),
        warmup_rounds=int(run_config.get("server-lr-warmup-rounds", 0)),
    )
