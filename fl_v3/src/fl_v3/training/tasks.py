"""Task-agnostic FL interface (fl_v3 T0).

The FL skeleton must NOT assume classification: no hardcoded ``CrossEntropyLoss``,
no ``num-classes``, no single-logits readout. Everything task-specific — the
model, the **criterion (loss)**, the data, and the eval metrics — is supplied by
a :class:`Task` selected from :data:`TASK_REGISTRY` via the ``task-type`` config
key. The AD perception task (BEVFusion model, detection loss, mAP/NDS + ASR eval)
registers here in T2/T4 without touching the skeleton.

The T0 ``dummy_regression`` task deliberately uses an **MSE** criterion (not
CrossEntropy) on synthetic continuous targets — this is the positive proof that
the skeleton carries no classification assumption: a non-classification loss
flows end-to-end through train → aggregate → eval.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl_v3.data.partition import iid_partition
from fl_v3.models.dummy import TinyMLP
from fl_v3.utils.runtime import derive_seed, seeded_worker_init

# A criterion maps (model_output, target) -> scalar loss tensor. NOT assumed to
# be classification — MSE, L1, a detection loss, etc. all satisfy this.
Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# Loss factory registry — the loss is config-injected, never hardcoded in the
# skeleton. The AD detection loss registers here in T2.
_CRITERION_FACTORIES: Dict[str, Callable[[], Criterion]] = {
    "mse": lambda: nn.MSELoss(),
    "l1": lambda: nn.L1Loss(),
    "cross_entropy": lambda: nn.CrossEntropyLoss(),
}


def build_criterion(name: str) -> Criterion:
    """Construct a criterion by name (config-injected)."""
    if name not in _CRITERION_FACTORIES:
        raise ValueError(
            f"Unknown criterion {name!r}. Available: {sorted(_CRITERION_FACTORIES)}"
        )
    return _CRITERION_FACTORIES[name]()


@dataclass
class ClientData:
    """Per-client dataloaders + counts (task-agnostic)."""

    trainloader: DataLoader
    valloader: Optional[DataLoader]
    num_train: int
    num_val: int


class Task(abc.ABC):
    """A federated task: model + criterion + data + eval metrics.

    Concrete tasks implement these so the FL skeleton stays generic. ``run_config``
    is the flat Flower run-config dict (or any mapping).
    """

    name: str

    @abc.abstractmethod
    def num_clients(self, run_config: dict) -> int: ...

    @abc.abstractmethod
    def build_model(self, run_config: dict) -> nn.Module: ...

    @abc.abstractmethod
    def build_criterion(self, run_config: dict) -> Criterion: ...

    @abc.abstractmethod
    def client_data(self, client_id: int, run_config: dict) -> ClientData: ...

    @abc.abstractmethod
    def eval_loader(self, run_config: dict) -> DataLoader: ...

    @abc.abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        eval_loader: DataLoader,
        criterion: Criterion,
        device: torch.device,
        run_config: dict,
    ) -> dict:
        """Task-specific eval metrics. MUST NOT assume classification — return
        whatever the task measures (e.g. ``eval_loss``; later mAP/NDS/ASR)."""


TASK_REGISTRY: Dict[str, Task] = {}


def register_task(task: Task) -> Task:
    """Register a Task instance under ``task.name``."""
    if task.name in TASK_REGISTRY:
        raise ValueError(f"Task {task.name!r} already registered")
    TASK_REGISTRY[task.name] = task
    return task


def get_task(name: str) -> Task:
    if name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task-type {name!r}. Available: {sorted(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[name]


def available_tasks() -> list[str]:
    return sorted(TASK_REGISTRY)


# ---------------------------------------------------------------------------
# Dummy regression task — the T0 smoke (MSE loss; NO classification anywhere).
# ---------------------------------------------------------------------------


class DummyRegressionTask(Task):
    """Synthetic linear-regression-with-noise task for the T0 FL smoke.

    Each client gets a deterministic synthetic ``(X, y)`` slice seeded off
    ``derive_seed(run_seed, client_id)`` so partitions are reproducible and
    label-free of any class concept. The loss is **MSE** (config ``loss=mse``),
    proving the skeleton is not CrossEntropy-coupled.
    """

    name = "dummy_regression"

    def __init__(self, in_dim: int = 4, n_per_client: int = 64, val_ratio: float = 0.25):
        self.in_dim = in_dim
        self.n_per_client = n_per_client
        self.val_ratio = val_ratio
        # A fixed "ground-truth" linear map shared across clients (seeded once).
        gen = np.random.default_rng(0)
        self._w_true = gen.standard_normal(in_dim).astype(np.float32)
        self._b_true = np.float32(0.5)

    def num_clients(self, run_config: dict) -> int:
        return int(run_config.get("num-clients", 4))

    def build_model(self, run_config: dict) -> nn.Module:
        return TinyMLP(in_dim=self.in_dim, hidden=8, out_dim=1)

    def build_criterion(self, run_config: dict) -> Criterion:
        # Loss is config-injected; default MSE for this regression task.
        return build_criterion(str(run_config.get("loss", "mse")))

    def _synth(self, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        gen = np.random.default_rng(seed)
        X = gen.standard_normal((n, self.in_dim)).astype(np.float32)
        noise = (0.01 * gen.standard_normal(n)).astype(np.float32)
        y = (X @ self._w_true + self._b_true + noise).astype(np.float32).reshape(-1, 1)
        return torch.from_numpy(X), torch.from_numpy(y)

    def _loader(self, X, y, run_config, shuffle, seed) -> DataLoader:
        batch_size = int(run_config.get("batch-size", 8))
        num_workers = int(run_config.get("num-workers", 0))
        g = torch.Generator()
        g.manual_seed(int(seed))
        return DataLoader(
            TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=seeded_worker_init,
            generator=g,
            drop_last=False,
        )

    def client_data(self, client_id: int, run_config: dict) -> ClientData:
        run_seed = int(run_config.get("seed", 42))
        seed = derive_seed(run_seed, client_id)
        X, y = self._synth(self.n_per_client, seed)
        n_val = max(1, int(self.n_per_client * self.val_ratio))
        Xtr, ytr = X[n_val:], y[n_val:]
        Xva, yva = X[:n_val], y[:n_val]
        trainloader = self._loader(Xtr, ytr, run_config, shuffle=True, seed=seed)
        valloader = self._loader(Xva, yva, run_config, shuffle=False, seed=seed)
        return ClientData(
            trainloader=trainloader,
            valloader=valloader,
            num_train=int(len(Xtr)),
            num_val=int(len(Xva)),
        )

    def eval_loader(self, run_config: dict) -> DataLoader:
        # A fixed held-out eval set (seed distinct from any client).
        X, y = self._synth(256, seed=999_001)
        return self._loader(X, y, run_config, shuffle=False, seed=999_001)

    def evaluate(self, model, eval_loader, criterion, device, run_config) -> dict:
        # Delegate the mean-loss averaging to the single shared eval loop
        # (no duplicated loop to drift); only the metric naming is task-specific.
        from fl_v3.training.loop import evaluate as _evaluate

        m = _evaluate(model, eval_loader, criterion, device)
        return {"eval_loss": m["loss"], "num-eval-examples": m["num_samples"]}


register_task(DummyRegressionTask())
