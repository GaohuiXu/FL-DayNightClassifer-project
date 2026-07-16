"""Task-agnostic FL interface (fl_v3 T0).

The FL skeleton must NOT assume classification: no hardcoded ``CrossEntropyLoss``,
no ``num-classes``, no single-logits readout. Everything task-specific — the
model, the **criterion (loss)**, the data, and the eval metrics — is supplied by
a :class:`Task` selected from :data:`TASK_REGISTRY` via the ``task-type`` config
key. The AD perception task supplies BEVFusion construction, detection loss, and
clean evaluation without changing the skeleton.

The T0 ``dummy_regression`` task deliberately uses an **MSE** criterion (not
CrossEntropy) on synthetic continuous targets — this is the positive proof that
the skeleton carries no classification assumption: a non-classification loss
flows end-to-end through train → aggregate → eval.
"""
from __future__ import annotations

import abc
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl_v3.data.partition import iid_partition
from fl_v3.models.dummy import TinyMLP
from fl_v3.utils.runtime import (
    current_precision,
    derive_seed,
    precision_autocast_context,
    normalize_model_mode,
    seeded_worker_init,
    truthy,
    validate_sparse_precision,
)

# ---------------------------------------------------------------------------
# DT3-A — trainable-only update vector and frozen layout contract.
# ---------------------------------------------------------------------------
# The FL update vector is the model's **trainable** tensors ONLY (D1: the frozen
# ImageNet camera backbone — 94 % of params, zero update — is excluded and
# reconstructed byte-identically per node from the pinned ImageNet cache; see
# ``det-pretrained-backbone``). For the headline BEVFusion detector this is
# EXACTLY the 62 non-backbone param tensors, in this module order with these
# per-module counts. Frozen here because aggregation and checkpoint loading rely
# on the same ordered trainable vector (asserted by ``assert_trainable_layout``).
#
# Buffers are NOT in the update vector. VERIFIED (T3 introspection): all frozen
# *params* live under ``camera_backbone``; the only non-backbone buffers are 4
# node-invariant CONSTANTS (``preprocess._mean/_std``, ``view_transform.frustum/
# depth_values``) — byte-identical across nodes AND unchanged through training —
# so excluding them is lossless (each node rebuilds them from config). The 6
# trainable modules use GroupNorm (D6) and carry no running-stat buffers.
TRAINABLE_MODULE_SLICE_MAP: "OrderedDict[str, int]" = OrderedDict(
    [
        ("camera_neck", 15),
        ("view_transform", 5),
        ("lidar_encoder", 3),
        ("fusion", 6),
        ("bev_neck", 18),
        ("head", 15),
    ]
)
TRAINABLE_TENSOR_COUNT: int = sum(TRAINABLE_MODULE_SLICE_MAP.values())  # 62

# MCR Phase-3 (D17): the bb02d FL layout. D17 amends D1 — the camera backbone is now TRAINED (not frozen),
# and bb02d enables the 4-stage LidarBackbone2D. So the FL update vector is the FULL trainable model, NOT the
# frozen-62 subset: camera_backbone (169, trained) + lidar_backbone (39, 4-stage) ADD to the 6 non-backbone
# modules (still 62), giving exactly 270 tensors / 33.17M params, ZERO frozen. The 6 non-backbone module
# counts are byte-identical to the frozen contract (15/5/3/6/18/15) — bb02d is a strict superset. Empirically
# enumerated from configs/p1_bb02d.json (det-freeze-backbone=false, det-lidar-backbone=true, stages=4). This
# is a TEST/bookkeeping contract only: production FedAvg/FedAdam aggregate the requires_grad vector
# (trainable_state_dict), which has no hardcoded count — see assert_trainable_layout's note.
BB02D_TRAINABLE_MODULE_SLICE_MAP: "OrderedDict[str, int]" = OrderedDict(
    [
        ("camera_backbone", 169),
        ("camera_neck", 15),
        ("view_transform", 5),
        ("lidar_encoder", 3),
        ("lidar_backbone", 39),
        ("fusion", 6),
        ("bev_neck", 18),
        ("head", 15),
    ]
)
BB02D_TRAINABLE_TENSOR_COUNT: int = sum(BB02D_TRAINABLE_MODULE_SLICE_MAP.values())  # 270


def trainable_param_names(model: nn.Module) -> set:
    """Names of the parameters that get FL-aggregated (``requires_grad=True``)."""
    return {n for n, p in model.named_parameters() if p.requires_grad}


def trainable_state_dict(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    """The DT3-A update vector: the trainable tensors only, in ``state_dict`` order.

    A ``requires_grad``-based filter (NOT a ``camera_backbone`` prefix-drop) over
    ``model.state_dict()`` — so it degenerates to the FULL state_dict on a model with
    no frozen params (e.g. ``TinyMLP``: task-agnostic preserved). Buffers are dropped
    (they are not in ``named_parameters``); see the contract note above.
    """
    keep = trainable_param_names(model)
    sd = model.state_dict()
    return OrderedDict((k, v) for k, v in sd.items() if k in keep)


def load_trainable_state_dict(model: nn.Module, state) -> None:
    """Load a trainable-only state_dict into the FULL model with ``strict=False``.

    A partial (62-key) dict cannot ``load_state_dict(strict=True)`` — it RAISES on the
    missing frozen-backbone/buffer keys. We load non-strict and then assert the load was
    clean: NO unexpected keys, and NO **trainable** key left unfilled (missing keys may
    only be the frozen/buffer keys the model reconstructs itself). This catches a silent
    wrong-weights load (e.g. a key-order or schema drift) loudly.
    """
    sd = state.to_torch_state_dict() if hasattr(state, "to_torch_state_dict") else state
    incompat = model.load_state_dict(sd, strict=False)
    unexpected = list(incompat.unexpected_keys)
    if unexpected:
        raise RuntimeError(
            f"trainable-only load got {len(unexpected)} UNEXPECTED keys "
            f"(not in the model): e.g. {unexpected[:5]}"
        )
    trainable = trainable_param_names(model)
    missing_trainable = trainable.intersection(incompat.missing_keys)
    if missing_trainable:
        raise RuntimeError(
            f"trainable-only load left {len(missing_trainable)} TRAINABLE keys unfilled "
            f"(update vector incomplete): e.g. {sorted(missing_trainable)[:5]}"
        )


def trainable_module_layout(model: nn.Module) -> "OrderedDict[str, int]":
    """The model's ACTUAL trainable per-top-module tensor counts, in state_dict order."""
    keep = trainable_param_names(model)
    ordered = [k for k in model.state_dict().keys() if k in keep]
    counts: "OrderedDict[str, int]" = OrderedDict()
    for k in ordered:
        top = k.split(".")[0]
        counts[top] = counts.get(top, 0) + 1
    return counts


def assert_trainable_layout(
    model: nn.Module, expected: "Optional[OrderedDict[str, int]]" = None
) -> None:
    """Assert the model's trainable tensors match a frozen layout contract.

    Config-conditional (MCR Phase-3 / D17): ``expected`` defaults to the **frozen-backbone
    62-tensor** contract (:data:`TRAINABLE_MODULE_SLICE_MAP`) for back-compat; pass
    :data:`BB02D_TRAINABLE_MODULE_SLICE_MAP` (270) for the trained-backbone bb02d FL model.
    This is a TEST/bookkeeping guard against a silent param-structure drift — production
    FedAvg/FedAdam aggregate the ``requires_grad`` vector (:func:`trainable_state_dict`),
    which carries no hardcoded count, so this never gates a training/aggregation run.
    """
    if expected is None:
        expected = TRAINABLE_MODULE_SLICE_MAP
    counts = trainable_module_layout(model)
    total = sum(counts.values())
    expected_total = sum(expected.values())
    if total != expected_total:
        raise AssertionError(
            f"expected {expected_total} trainable tensors, got {total}"
        )
    if list(counts.keys()) != list(expected.keys()):
        raise AssertionError(
            f"module order {list(counts.keys())} != contract {list(expected.keys())}"
        )
    if counts != expected:
        raise AssertionError(
            f"per-module counts {dict(counts)} != contract {dict(expected)}"
        )

# A criterion maps (model_output, target) -> scalar loss tensor. NOT assumed to
# be classification — MSE, L1, a detection loss, etc. all satisfy this. ``target``
# is widened (T2) from ``torch.Tensor`` to ``Any`` so the AD detection criterion can
# take the multimodal batch dict as its target (the loop passes it through). Mirrors
# the widened alias in ``training/loop.py``.
Criterion = Callable[[Any, Any], torch.Tensor]

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
        whatever the task measures (e.g. ``eval_loss`` or official mAP/NDS)."""


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


# ---------------------------------------------------------------------------
# nuScenes multimodal detection task (T2) — the AD perception task.
# ---------------------------------------------------------------------------
def _det_config_from_run(run_config: dict):
    """Build a DetectorConfig from the flat run_config (config-injected knobs)."""
    from fl_v3.models.fusion.bev_grid import BEVConfig
    from fl_v3.models.fusion.detector import DetectorConfig

    production = bool(run_config.get("s06-production-runtime", False))
    if production:
        from fl_v3.config import ConfigError, validate_precision_partition

        required_arch = {
            "det-camera-arch", "det-camera-pretrained", "det-lidar-arch",
            "det-fusion-arch", "det-head-arch", "precision",
            "det-sparse-conv-precision", "det-camera-activation-checkpoint",
        }
        missing = sorted(required_arch - set(run_config))
        if missing:
            raise ValueError(f"resolved architecture mapping fields missing: {missing}")
        camera_arch = str(run_config["det-camera-arch"])
        lidar_arch = str(run_config["det-lidar-arch"])
        fusion_arch = str(run_config["det-fusion-arch"])
        head_arch = str(run_config["det-head-arch"])
        if camera_arch not in {"swin_t_stride8", "none"}:
            raise ValueError(f"unknown resolved camera architecture {camera_arch!r}")
        if lidar_arch not in {"second_075", "pillar_020", "none"}:
            raise ValueError(f"unknown resolved LiDAR architecture {lidar_arch!r}")
        if fusion_arch not in {"conv_fuser_256", "none"}:
            raise ValueError(f"unknown resolved fusion architecture {fusion_arch!r}")
        if head_arch != "centerhead_multitask":
            raise ValueError(f"unknown resolved head architecture {head_arch!r}")
        mode = normalize_model_mode(str(run_config["model-mode"]))
        architecture_matches_mode = {
            "camera_only": (
                camera_arch == "swin_t_stride8"
                and lidar_arch == "none"
                and fusion_arch == "none"
            ),
            "lidar_only": (
                camera_arch == "none"
                and lidar_arch in {"second_075", "pillar_020"}
                and fusion_arch == "none"
            ),
            "fusion": (
                camera_arch == "swin_t_stride8"
                and lidar_arch in {"second_075", "pillar_020"}
                and fusion_arch == "conv_fuser_256"
            ),
        }
        if not architecture_matches_mode[mode]:
            raise ValueError(
                "resolved architecture fields are inconsistent with "
                f"model-mode={mode!r}"
            )
        if lidar_arch == "second_075":
            bev = BEVConfig(
                point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
                bev_voxel=(0.6, 0.6),
                out_size_factor=1,
            )
            lidar_input_bev = BEVConfig(
                point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
                bev_voxel=(0.075, 0.075),
                out_size_factor=8,
            )
        elif lidar_arch == "pillar_020":
            bev = BEVConfig(
                point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                bev_voxel=(0.2, 0.2),
                out_size_factor=2,
            )
            lidar_input_bev = None
        else:
            bev = BEVConfig(
                point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
                bev_voxel=(0.6, 0.6),
                out_size_factor=1,
            )
            lidar_input_bev = None
        if "det-sparse-conv-fp16" in run_config:
            raise ValueError(
                "legacy det-sparse-conv-fp16 is forbidden in the strict S08 production runtime"
            )
        camera_activation_checkpoint = run_config["det-camera-activation-checkpoint"]
        if not isinstance(camera_activation_checkpoint, bool):
            raise ValueError("det-camera-activation-checkpoint must be boolean")
        precision = run_config["precision"]
        sparse_partition = run_config["det-sparse-conv-precision"]
        try:
            validate_precision_partition(precision, lidar_arch, sparse_partition)
        except ConfigError as exc:
            raise ValueError(f"invalid resolved precision partition: {exc}") from exc
        sparse_conv_fp16 = sparse_partition == "fp16"
        return DetectorConfig(
            model_mode=mode,
            required_spconv_version=("2.3.8" if lidar_arch == "second_075" else None),
            camera_backbone="swin_t",
            freeze_camera_backbone=False,
            pretrained_backbone=bool(run_config["det-camera-pretrained"]),
            activation_checkpoint=camera_activation_checkpoint,
            image_hw=(256, 704),
            feat_stride=8,
            neck_channels=128,
            context_channels=80,
            depth_bins=(1.0, 60.0, 0.5),
            reference_camera=camera_arch == "swin_t_stride8",
            camera_bev_output_dtype="input",
            lidar_channels=(256 if lidar_arch == "second_075" else 64),
            max_points_per_pillar=(10 if lidar_arch == "second_075" else 32),
            max_pillars=(120000 if lidar_arch == "pillar_020" else 30000),
            lidar_sweeps=int(run_config["det-lidar-sweeps"]),
            lidar_encoder=("voxel" if lidar_arch == "second_075" else "pillar"),
            lidar_z_voxel=(0.2 if lidar_arch == "second_075" else None),
            lidar_sparse_z_size=(41 if lidar_arch == "second_075" else None),
            sparse_conv_fp16=sparse_conv_fp16,
            lidar_input_bev=lidar_input_bev,
            max_voxels_train=120000,
            max_voxels_eval=160000,
            lidar_backbone=lidar_arch == "pillar_020",
            lidar_backbone_out=(128 if lidar_arch == "pillar_020" else 256),
            lidar_backbone_checkpoint=lidar_arch == "pillar_020",
            lidar_backbone_stages=4,
            fusion_channels=256,
            bev_neck_channels=256,
            head_channels=64,
            head_conv_layers=2,
            n_classes=10,
            score_threshold=0.1,
            bev=bev,
        )

    pcr = run_config.get("det-pc-range", None)
    voxel = float(run_config.get("det-bev-voxel", 0.4))
    osf = int(run_config.get("det-out-size-factor", 2))
    bev_kwargs = {"bev_voxel": (voxel, voxel), "out_size_factor": osf}
    if pcr:
        bev_kwargs["point_cloud_range"] = tuple(float(v) for v in pcr)
    bev = BEVConfig(**bev_kwargs)
    lidar_encoder = str(run_config.get("det-lidar-encoder", "pillar"))
    model_mode = normalize_model_mode(str(run_config.get("model-mode", "fusion")))
    precision = str(run_config.get("precision", "fp32")).strip().lower()
    validate_sparse_precision(precision, lidar_encoder)
    sparse_conv_fp16 = truthy(run_config.get("det-sparse-conv-fp16", False))
    if sparse_conv_fp16 and lidar_encoder != "voxel":
        raise ValueError("det-sparse-conv-fp16=true is only valid with det-lidar-encoder='voxel'")
    if sparse_conv_fp16 and precision != "fp16":
        raise ValueError("det-sparse-conv-fp16=true requires precision='fp16' (fp32 reference must stay fp32)")
    return DetectorConfig(
        model_mode=model_mode,
        required_spconv_version=run_config.get("dependency-spconv"),
        camera_backbone=str(run_config.get("det-camera-backbone", "swin_t")),
        freeze_camera_backbone=truthy(run_config.get("det-freeze-backbone", True)),
        pretrained_backbone=truthy(run_config.get("det-pretrained-backbone", True)),
        activation_checkpoint=truthy(run_config.get("det-activation-checkpoint", False)),
        swin_sdpa=truthy(run_config.get("det-swin-sdpa", False)),
        image_hw=(int(run_config.get("det-image-h", 256)), int(run_config.get("det-image-w", 704))),
        feat_stride=int(run_config.get("det-feat-stride", 16)),
        neck_channels=int(run_config.get("det-neck-channels", 128)),
        context_channels=int(run_config.get("det-context-channels", 80)),
        depth_bins=(
            float(run_config.get("det-depth-min", 1.0)),
            float(run_config.get("det-depth-max", 60.0)),
            float(run_config.get("det-depth-step", 1.0)),
        ),
        lidar_channels=int(run_config.get("det-lidar-channels", 64)),
        max_points_per_pillar=int(run_config.get("det-max-points-per-pillar", 32)),
        max_pillars=int(run_config.get("det-max-pillars", 30000)),
        lidar_sweeps=int(run_config.get("det-lidar-sweeps", 1)),
        lidar_encoder=lidar_encoder,
        lidar_z_voxel=(
            None
            if run_config.get("det-lidar-z-voxel", None) in (None, "")
            else float(run_config.get("det-lidar-z-voxel"))
        ),
        lidar_sparse_z_size=(
            None
            if run_config.get("det-lidar-sparse-z-size", None) in (None, "")
            else int(run_config.get("det-lidar-sparse-z-size"))
        ),
        sparse_conv_fp16=sparse_conv_fp16,
        lidar_backbone=truthy(run_config.get("det-lidar-backbone", False)),
        lidar_backbone_out=int(run_config.get("det-lidar-backbone-out", 128)),
        lidar_backbone_checkpoint=truthy(run_config.get("det-lidar-backbone-checkpoint", False)),
        lidar_backbone_stages=int(run_config.get("det-lidar-backbone-stages", 3)),
        fusion_channels=int(run_config.get("det-fusion-channels", 128)),
        bev_neck_channels=int(run_config.get("det-bev-neck-channels", 256)),
        head_channels=int(run_config.get("det-head-channels", 64)),
        head_conv_layers=int(run_config.get("det-head-conv-layers", 2)),
        n_classes=int(run_config.get("det-n-classes", 10)),
        max_objects=int(run_config.get("det-max-objects", 200)),
        score_threshold=float(run_config.get("det-score-threshold", 0.1)),
        bev=bev,
    )


def _apply_production_sampling(dataset, run_config: dict, *, shuffle: bool):
    """Apply the one resolved sampling recipe before DataLoader construction."""
    if not (
        bool(run_config.get("s06-production-runtime", False))
        and shuffle
        and bool(run_config.get("det-cbgs", False))
    ):
        return dataset
    if run_config.get("det-class-weights") is not None or run_config.get(
        "det-reg-class-weights"
    ) is not None:
        raise ValueError("F-CBGS replaces rather than stacks with class/regression weights")
    from fl_v3.data.nuscenes.cbgs import (
        CBGSWrapper,
        build_cbgs_indices,
        cbgs_index_identity,
        dataset_inrange_classes,
    )

    indices, stats = build_cbgs_indices(
        dataset_inrange_classes(dataset),
        n_classes=10,
        thresh=float(run_config["det-cbgs-thresh"]),
        seed=int(run_config["seed"]),
        max_repeat=float(run_config["det-cbgs-max-repeat"]),
    )
    sample_tokens = getattr(dataset, "sample_tokens", None)
    if sample_tokens is None:
        raise ValueError("CBGS production dataset must expose its role-restricted sample tokens")
    stats.update(cbgs_index_identity(sample_tokens, indices))
    wrapped = CBGSWrapper(dataset, indices)
    wrapped.stats = stats
    return wrapped


def _production_sampler(dataset, run_config: dict, *, shuffle: bool):
    """Return the epoch-addressable sampler required by the production trainer."""
    if not (bool(run_config.get("s06-production-runtime", False)) and shuffle):
        return None
    from fl_v3.training.runtime_state import EpochPermutationSampler

    return EpochPermutationSampler(dataset, seed=int(run_config["seed"]))


def _aug_from_run(run_config: dict):
    """BEV/3D aug params dict from the flat run_config, or None if disabled (the overfitting fix).

    TRAIN-ONLY — callers must pass this only to the training dataset; eval stays clean (None)."""
    if not truthy(run_config.get("det-aug-bev", False)):
        return None
    return {
        "rot": float(run_config.get("det-aug-rot", 0.785398)),
        "scale": (float(run_config.get("det-aug-scale-min", 0.9)),
                  float(run_config.get("det-aug-scale-max", 1.1))),
        "translate": float(run_config.get("det-aug-translate", 0.5)),
        "flip": truthy(run_config.get("det-aug-flip", True)),
        "img_flip": float(run_config.get("det-aug-img-flip", 0.0)),   # image-appearance flip prob (0=off)
    }


def _gtpaste_from_run(run_config: dict):
    """GT-paste (rare-class object copy-paste) params from the flat run_config, or None if disabled.

    TRAIN-ONLY — callers pass this ONLY to the training dataset; eval stays clean (None). ``det-gt-paste-counts``
    is a ``{class_name: K}`` map (default targets the lagging large/rare vehicles)."""
    if not truthy(run_config.get("det-gt-paste", False)):
        return None
    counts = run_config.get("det-gt-paste-counts") or {
        "trailer": 3, "construction_vehicle": 3, "bus": 2, "truck": 2, "bicycle": 3, "motorcycle": 3,
    }
    return {
        "db_path": str(run_config.get("det-gt-paste-db", "./fl_outputs/nuscenes/gt_database_msweep10")),
        "counts": {str(k): int(v) for k, v in dict(counts).items()},
        "min_points": int(run_config.get("det-gt-paste-min-points", 5)),
        "iou_thresh": float(run_config.get("det-gt-paste-iou-thresh", 0.0)),
        "yaw_jitter": float(run_config.get("det-gt-paste-yaw-jitter", 0.785398)),  # ±rad per-object diversity
    }


def _normalize_weights(cw):
    """Per-class loss weights → None when absent / empty / UNIFORM (all values equal).

    Accepts a **list** (centralized JSON reads it directly: ``[1,1.3,…]``) OR a **comma-string** (the FL
    path: ``flwr run`` only allows scalar/str config values — NOT arrays — so the FL config encodes the
    weights as ``"1.0,1.3,…"`` with a ``""`` default). A uniform/empty value maps back to ``None`` so the
    pre-MCR no-weight loss path stays byte-identical; a non-uniform value applies the capability weighting."""
    if cw is None:
        return None
    if isinstance(cw, str):
        s = cw.strip()
        if not s:
            return None
        cw = [float(x) for x in s.split(",") if x.strip() != ""]
    vals = [float(v) for v in cw]
    if not vals:
        return None
    if all(v == vals[0] for v in vals):
        return None
    return vals


def center_distance_proxy(
    decoded: List[dict],
    batch: dict,
    target_class: int = 0,
    max_dist: float = 2.0,
) -> Dict[str, float]:
    """Provisional center-distance proxy (not official DetectionEval).

    For the D8 target class, match decoded boxes to GT by **metric BEV-center L2 in the
    canonical frame** (greedy nearest, each GT matched at most once). Reports
    ``recall@max_dist`` (distinct GT matched / total GT), mean best-match distance, and
    decoded/GT counts — the falsifiable overfit signal + anti-collapse evidence."""
    import numpy as np

    gt_boxes_list = batch["gt_boxes"]
    gt_labels_list = batch["gt_labels"]
    total_gt = 0
    matched_gt = 0
    best_dists: List[float] = []
    total_decoded = 0
    for b, res in enumerate(decoded):
        gtb = gt_boxes_list[b].detach().cpu().numpy()
        gtl = gt_labels_list[b].detach().cpu().numpy()
        gt_xy = gtb[gtl == target_class][:, :2]
        det_mask = (res["labels"].detach().cpu().numpy() == target_class)
        det_xy = res["boxes"].detach().cpu().numpy()[det_mask][:, :2]
        total_gt += int(gt_xy.shape[0])
        total_decoded += int(det_xy.shape[0])
        used = np.zeros(det_xy.shape[0], dtype=bool)
        for g in range(gt_xy.shape[0]):
            if det_xy.shape[0] == 0:
                best_dists.append(float("inf"))
                continue
            d = np.linalg.norm(det_xy - gt_xy[g][None, :], axis=1)
            d_masked = np.where(used, np.inf, d)
            j = int(np.argmin(d_masked))
            best = float(d_masked[j])
            best_dists.append(best if np.isfinite(best) else float(np.min(d)))
            if best <= max_dist:
                matched_gt += 1
                used[j] = True
    finite = [d for d in best_dists if d != float("inf")]
    return {
        "recall_at": (matched_gt / total_gt) if total_gt else 0.0,
        "mean_best_center_dist": (float(sum(finite) / len(finite)) if finite else float("inf")),
        "n_gt": float(total_gt),
        "n_decoded": float(total_decoded),
        "matched_gt": float(matched_gt),
    }


class NuScenesDetectionTask(Task):
    """Federated multimodal AD detection task (BEVFusion-class model + CenterPoint loss).

    Consumes the **frozen T1 schema** (via the pre-built info-cache) + the log-group /
    IID partitioner; the loop trains it through the additive batch protocol; eval reports
    the training loss plus the center-distance proxy; official mAP/NDS is post-hoc."""

    name = "nuscenes_detection"

    def __init__(self):
        self._partition_cache: Dict[tuple, dict] = {}
        self._info_cache: Dict[tuple, tuple] = {}  # MCR P3: memoize the (heavy) info-list per (version,split)

    # --- model / criterion ---
    def build_model(self, run_config: dict) -> nn.Module:
        from fl_v3.models.fusion.detector import BEVFusionDetector

        return BEVFusionDetector(_det_config_from_run(run_config))

    def build_criterion(self, run_config: dict) -> Criterion:
        from fl_v3.models.fusion.losses import MultiTaskCenterPointLoss

        c = _det_config_from_run(run_config)
        return MultiTaskCenterPointLoss(
            cfg=c.bev,
            reg_weight=float(run_config.get("det-reg-weight", 0.25)),
            class_weights=_normalize_weights(run_config.get("det-class-weights")),
            reg_class_weights=_normalize_weights(run_config.get("det-reg-class-weights")),
        )

    # --- data ---
    def _load_info(self, run_config: dict, split: str):
        from fl_v3.data.nuscenes import info_cache as IC

        cache_dir = str(run_config["nuscenes-cache-dir"])
        version = str(run_config["nuscenes-version"])
        # MCR P3: memoize the unpickle. In FL, ``client_data`` is called per (client, round); without this
        # the ~580 MB trainval info-cache is re-unpickled on EVERY client invocation (~5 s × 25 × R of pure
        # I/O). The cache lives on the Task instance, which persists per Ray-actor process ⇒ each actor reads
        # the pickle once. (Memory-only; the result is read-only and identical across calls.)
        production = bool(run_config.get("s06-production-runtime", False))
        if production:
            required = ("det-lidar-sweeps", "nuscenes-cache-identities",
                        "nuscenes-zip-manifest", "nuscenes-zip-manifest-logical-sha256",
                        "nuscenes-zip-manifest-file-sha256")
            missing = [k for k in required if not run_config.get(k)]
            if missing:
                raise ValueError(f"S06 production cache/manifest identity fields missing: {missing}")
        n_sweeps = int(run_config["det-lidar-sweeps"]) if production else None
        cache_identity = None
        if production:
            if split == str(run_config["nuscenes-train-split"]):
                cache_identity = run_config["nuscenes-cache-identities"]["train"]
            elif split == str(run_config["nuscenes-val-split"]):
                cache_identity = run_config["nuscenes-cache-identities"]["val"]
            else:
                raise ValueError(f"split {split!r} is not the resolved train or val split")
        expected_hash = cache_identity["logical_sha256"] if cache_identity else None
        ck = (cache_dir, version, split, n_sweeps, expected_hash,
              run_config.get("resolved-config-sha256") if production else None)
        cached = self._info_cache.get(ck)
        if cached is not None:
            return cached
        try:
            if production:
                actual_pickle, actual_sidecar = IC.cache_paths(cache_dir, version, split, n_sweeps)
                declared = (os.path.abspath(str(cache_identity["path"])),
                            os.path.abspath(str(cache_identity["sidecar_path"])))
                actual = (os.path.abspath(actual_pickle), os.path.abspath(actual_sidecar))
                if actual != declared:
                    raise ValueError(f"canonical/physical cache path drift: declared={declared}, actual={actual}")
                checks = (
                    (actual_pickle, cache_identity["pickle_sha256"], "cache pickle"),
                    (actual_sidecar, cache_identity["sidecar_sha256"], "cache sidecar"),
                    (run_config["nuscenes-zip-manifest"],
                     run_config["nuscenes-zip-manifest-file-sha256"], "ZIP manifest"),
                )
                for path, expected, label in checks:
                    digest = hashlib.sha256()
                    with open(path, "rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    if digest.hexdigest() != expected:
                        raise ValueError(f"{label} physical identity drift")
                from fl_v3.data.nuscenes.zip_backend import manifest_summary
                manifest = manifest_summary(str(run_config["nuscenes-zip-manifest"]))
                if manifest["manifest_hash"] != run_config["nuscenes-zip-manifest-logical-sha256"]:
                    raise ValueError("ZIP manifest logical identity drift")
            info_list, meta = IC.load_cache(
                cache_dir, version, split, n_sweeps=n_sweeps,
                expected_cache_hash=expected_hash,
            )
            self._info_cache[ck] = (info_list, meta)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"nuScenes info-cache for ({version}, {split}) not found under "
                f"{cache_dir!r}. Build it on the LOGIN node first: "
                f"`source fl_v3/scripts/arrhenius_env.sh && arrhenius_activate_env && "
                f"python fl_v3/scripts/build_nuscenes_cache.py ...`. "
                f"client_data must NOT build the devkit during a training run."
            ) from e
        return info_list, meta

    def _partition(self, run_config: dict) -> dict:
        """Build (and memoize) the client partition for the train split."""
        from fl_v3.data.nuscenes.partition import (
            build_log_table, build_log_group_partition, coerce_partition_seed,
        )

        version = str(run_config["nuscenes-version"])
        split = str(run_config["nuscenes-train-split"])
        mode = str(run_config.get("nuscenes-partition-mode", "log_group"))
        floor = int(run_config.get("min-keyframes-per-client", 400))
        requested = int(run_config.get("nuscenes-num-clients", 50))
        run_seed = int(run_config.get("seed", 42))
        pseed = coerce_partition_seed(run_config.get("partition-seed", ""), run_seed)
        key = (version, split, mode, floor, requested, pseed)
        if key in self._partition_cache:
            return self._partition_cache[key]
        info_list, _ = self._load_info(run_config, split)
        if mode == "log_group":
            log_table = build_log_table(info_list)
            part = build_log_group_partition(log_table, floor=floor,
                                             requested_num_clients=requested, seed=pseed)
            client_tokens = {c["client_id"]: c["sample_tokens"] for c in part["clients"]}
            result = {"num_clients": part["num_clients"], "client_tokens": client_tokens,
                      "mode": mode, "raw": part}
        elif mode == "iid":
            from fl_v3.data.nuscenes.partition import iid_sample_partition

            cmap = iid_sample_partition(info_list, requested, seed=pseed)
            result = {"num_clients": requested, "client_tokens": cmap, "mode": mode, "raw": None}
        else:
            raise ValueError(f"unknown nuscenes-partition-mode {mode!r} (log_group|iid)")
        self._partition_cache[key] = result
        return result

    def num_clients(self, run_config: dict) -> int:
        # The DERIVED N (log_group may have fallen back to N∈{20,25}), NOT the request.
        return int(self._partition(run_config)["num_clients"])

    def _make_loader(self, run_config: dict, info_list, tokens, shuffle: bool,
                     augment: Optional[dict] = None):
        from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset, make_loader
        from fl_v3.data.nuscenes import paths as P
        from fl_v3.models.fusion.collate import detection_collate_fn

        dataroot = P.get_dataroot(run_config)
        production = bool(run_config.get("s06-production-runtime", False))
        model_mode = normalize_model_mode(str(run_config.get("model-mode", "fusion")))
        ds = NuScenesMultimodalDataset(info_list, dataroot, sample_tokens=tokens,
                                       n_sweeps=int(run_config.get("det-lidar-sweeps", 1)),
                                       augment=augment,
                                       gtpaste=_gtpaste_from_run(run_config) if shuffle else None,
                                       zip_manifest=(str(run_config["nuscenes-zip-manifest"])
                                                     if production else None),
                                       model_mode=model_mode)
        ds = _apply_production_sampling(ds, run_config, shuffle=shuffle)
        sampler = _production_sampler(ds, run_config, shuffle=shuffle)
        return make_loader(
            ds,
            batch_size=int(run_config.get("batch-size", 1)),
            shuffle=shuffle,
            num_workers=int(run_config.get("num-workers", 0)),
            seed=int(run_config.get("seed", 42)),
            collate_fn=detection_collate_fn,
            sampler=sampler,
        )

    def load_production_train_info(self, run_config: dict):
        """Expose the identity-checked train cache for a frozen S10 role consumer."""
        if not bool(run_config.get("s06-production-runtime", False)):
            raise ValueError("fixed S10 subsets require the strict production runtime")
        split = str(run_config["nuscenes-train-split"])
        return self._load_info(run_config, split)

    def fixed_train_subset_loader(self, run_config: dict, sample_tokens) -> object:
        """Build one ordered, unaugmented, unshuffled production subset loader.

        The caller must first bind the tokens to an accepted S10 manifest role.
        This method deliberately knows nothing about role names and cannot solve,
        sample, repeat, or rebalance a split.
        """
        if not bool(run_config.get("s06-production-runtime", False)):
            raise ValueError("fixed S10 subsets require the strict production runtime")
        if str(run_config.get("sampling", "uniform")) != "uniform":
            raise ValueError("STOP-B fixed subsets require uniform sampling")
        if truthy(run_config.get("det-cbgs", False)):
            raise ValueError("STOP-B fixed subsets forbid CBGS")
        if _aug_from_run(run_config) is not None or _gtpaste_from_run(run_config) is not None:
            raise ValueError("STOP-B fixed subsets forbid augmentation and GT-paste")
        tokens = [str(token) for token in sample_tokens]
        if not tokens or len(tokens) != len(set(tokens)):
            raise ValueError("fixed subset tokens must be non-empty and unique")
        info_list, _ = self.load_production_train_info(run_config)
        return self._make_loader(
            run_config,
            info_list,
            tokens,
            shuffle=False,
            augment=None,
        )

    def client_data(self, client_id: int, run_config: dict) -> ClientData:
        split = str(run_config["nuscenes-train-split"])
        info_list, _ = self._load_info(run_config, split)
        part = self._partition(run_config)
        tokens = part["client_tokens"][client_id]
        trainloader = self._make_loader(
            run_config,
            info_list,
            tokens,
            shuffle=True,
            augment=_aug_from_run(run_config),
        )
        return ClientData(trainloader=trainloader, valloader=None,
                          num_train=len(tokens), num_val=0)

    def eval_loader(self, run_config: dict) -> DataLoader:
        split = str(run_config["nuscenes-val-split"])
        info_list, _ = self._load_info(run_config, split)
        tokens = None
        limit = int(run_config.get("det-eval-limit", 0))
        if limit > 0:
            tokens = sorted(i["sample_token"] for i in info_list)[:limit]
        return self._make_loader(run_config, info_list, tokens, shuffle=False)

    def evaluate(self, model, eval_loader, criterion, device, run_config) -> dict:
        from fl_v3.training.loop import _float_tensors, _move_to_device

        target_class = int(run_config.get("det-target-class", 0))  # D8 car
        thr = float(run_config.get("det-score-threshold", 0.1))
        precision = current_precision()
        model.eval()
        total_loss, total_n = 0.0, 0
        agg = {"recall_at": 0.0, "mean_best_center_dist": 0.0, "n_gt": 0.0,
               "n_decoded": 0.0, "matched_gt": 0.0}
        n_batches = 0
        import torch as _torch

        with _torch.no_grad():
            for batch in eval_loader:
                batch = _move_to_device(batch, device)
                with precision_autocast_context(precision, device):
                    out = model(batch)
                if precision == "fp16" and device.type == "cuda":
                    out = _float_tensors(out)
                loss = criterion(out, batch)
                bs = len(batch["gt_boxes"])
                total_loss += float(loss.item()) * bs
                total_n += bs
                decoded = model.decode(out, score_threshold=thr)
                p = center_distance_proxy(decoded, batch, target_class=target_class)
                agg["n_gt"] += p["n_gt"]; agg["n_decoded"] += p["n_decoded"]
                agg["matched_gt"] += p["matched_gt"]
                n_batches += 1
        recall = (agg["matched_gt"] / agg["n_gt"]) if agg["n_gt"] else 0.0
        return {
            "eval_loss": total_loss / total_n if total_n else 0.0,
            "num-eval-examples": float(total_n),
            "proxy_recall_at_2m": recall,
            "proxy_n_decoded": agg["n_decoded"],
            "proxy_n_gt": agg["n_gt"],
            "proxy_matched_gt": agg["matched_gt"],
        }


register_task(NuScenesDetectionTask())
