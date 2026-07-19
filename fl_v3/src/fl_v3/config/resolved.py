"""Canonical, fail-closed S09/S10 production configuration.

No scientific field is inferred from the environment.  Callers resolve this
schema before constructing data, a model, or an optimizer and pass the resulting
object (and hash) through train, checkpoint/resume, evaluation, and later FL.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from fl_v3.source_identity import validate_source_state


class ConfigError(ValueError):
    """A production configuration is incomplete, unknown, or inconsistent."""


_HEX = frozenset("0123456789abcdef")
_MODES = frozenset({"camera_only", "lidar_only", "fusion"})
_PRECISIONS = frozenset({"fp32", "fp16"})
_SPARSE_CONV_PRECISIONS = frozenset({"fp32", "fp16", "not_applicable"})
_OPTIMIZERS = frozenset({"adam", "adamw"})
_SAMPLING = frozenset({"uniform", "cbgs"})
_CAMERAS = frozenset({"swin_t_stride8", "none"})
_LIDARS = frozenset({"second_075", "pillar_020", "none"})
_FUSIONS = frozenset({"conv_fuser_256", "none"})
_HEADS = frozenset({"centerhead_multitask"})
_SECOND_NORMALIZATIONS = frozenset({"group_norm", "batch_norm_1d", "not_applicable"})
_EXECUTION_MODES = frozenset({"train_eval", "readiness"})

_ROOT = frozenset({
    "schema_version", "model", "precision", "sparse_conv_precision", "optimizer",
    "training", "data", "dependencies", "evaluation", "execution",
})
_MODEL_V1 = frozenset({
    "mode", "camera_arch", "camera_pretrained", "lidar_arch", "fusion_arch", "head_arch",
})
_MODEL_V2 = _MODEL_V1 | frozenset({"camera_activation_checkpoint"})
_MODEL_V3 = _MODEL_V2 | frozenset({"second_normalization"})
_OPT = frozenset({"name", "learning_rate", "weight_decay"})
_TRAIN = frozenset({
    "max_optimizer_steps", "micro_batch_size", "world_size", "accumulation_steps",
    "effective_global_batch", "seed", "max_epochs", "num_workers", "ema_decay", "sampling",
})
_TRAIN_S10 = _TRAIN | frozenset({"grad_scaler_init_scale"})
_DATA = frozenset({
    "dataroot", "version", "train_split", "val_split", "n_sweeps", "caches", "zip_manifest",
})
_CACHES = frozenset({"train", "val"})
_CACHE = frozenset({
    "format", "path", "sidecar_path", "logical_sha256", "pickle_sha256", "sidecar_sha256",
})
_MANIFEST = frozenset({"path", "logical_sha256", "file_sha256"})
_DEPS = frozenset({
    "torch", "torch_build_sha256", "torch_source_sha",
    "spconv", "spconv_build_sha256", "spconv_source_sha", "spconv_source_state",
    "cumm", "cumm_build_sha256", "cumm_source_sha", "cumm_source_state",
})
_EVAL = frozenset({"timing", "checkpoint_weights"})
_CHECKPOINT_WEIGHTS = frozenset({"raw", "ema"})
_EXECUTION_V1 = frozenset({
    "mode", "max_attempted_windows", "timing_warmup_successful_windows",
    "loader_profile",
})
_EXECUTION_V2 = _EXECUTION_V1 | frozenset({"operator_profile"})
_LOADER_PROFILE = frozenset({
    "workers", "repeats", "determinism_batches", "warmup_batches", "measured_batches",
})
_OPERATOR_PROFILE = frozenset({
    "wait_attempted_windows", "warmup_attempted_windows", "active_attempted_windows",
    "record_shapes", "profile_memory", "row_limit",
})


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be an object")
    if not all(isinstance(k, str) for k in value):
        raise ConfigError(f"{where} keys must be strings")
    return value


def _keys(value: dict[str, Any], expected: frozenset[str], where: str) -> None:
    got = frozenset(value)
    missing, unknown = sorted(expected - got), sorted(got - expected)
    if missing or unknown:
        raise ConfigError(f"{where} keys invalid: missing={missing}, unknown={unknown}")


def _enum(value: Any, allowed: frozenset[str], where: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ConfigError(f"{where}={value!r}; expected one of {sorted(allowed)}")
    return value


def _integer(value: Any, where: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be numeric")
    out = float(value)
    if not (out > 0.0 if positive else out >= 0.0):
        raise ConfigError(f"{where} must be {'> 0' if positive else '>= 0'}")
    return out


def _sha(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a lowercase SHA-256 string")
    if len(value) != 64 or any(c not in _HEX for c in value):
        raise ConfigError(f"{where} must be exactly 64 lowercase hexadecimal characters")
    return value


def _path(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ConfigError(f"{where} must be a non-empty explicit path")
    return value


def _source_state(value: Any, where: str) -> dict[str, object]:
    try:
        return validate_source_state(value)
    except ValueError as exc:
        raise ConfigError(f"{where} is invalid: {exc}") from exc


def canonical_json(config: Mapping[str, Any]) -> bytes:
    """Locale/order-stable canonical UTF-8 representation used for hashing."""
    return json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    """Return JSON-serializable plain containers from the frozen config graph."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ResolvedConfig:
    """Validated canonical config and its content identity."""

    data: Mapping[str, Any]
    canonical_bytes: bytes
    sha256: str

    @property
    def model_mode(self) -> str:
        return str(self.data["model"]["mode"])

    @property
    def precision(self) -> str:
        return str(self.data["precision"])

    @property
    def sparse_conv_precision(self) -> str:
        return str(self.data["sparse_conv_precision"])

    @property
    def execution_mode(self) -> str:
        return str(self.data["execution"]["mode"])

    @property
    def data_identities(self) -> dict[str, Any]:
        d = self.data["data"]
        out = {
            "n_sweeps": d["n_sweeps"],
            "zip_manifest_logical_sha256": d["zip_manifest"]["logical_sha256"],
            "zip_manifest_file_sha256": d["zip_manifest"]["file_sha256"],
        }
        for role in ("train", "val"):
            cache = d["caches"][role]
            out[f"{role}_cache_format"] = cache["format"]
            out[f"{role}_cache_logical_sha256"] = cache["logical_sha256"]
            out[f"{role}_cache_pickle_sha256"] = cache["pickle_sha256"]
            out[f"{role}_cache_sidecar_sha256"] = cache["sidecar_sha256"]
        return out

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes.decode("utf-8"))

    def to_run_config(self) -> dict[str, Any]:
        """Bridge the validated S09 configuration to current task interfaces."""
        d, m, t, o = self.data["data"], self.data["model"], self.data["training"], self.data["optimizer"]
        schema_version = str(self.data["schema_version"])
        camera_activation_checkpoint = (
            True
            if schema_version == "s09.v1"
            else bool(m["camera_activation_checkpoint"])
        )
        operator_profile = (
            None
            if schema_version == "s09.v1"
            else _thaw(self.data["execution"]["operator_profile"])
        )
        second_normalization = (
            str(m["second_normalization"])
            if schema_version == "s10.v1"
            else ("group_norm" if m["lidar_arch"] == "second_075" else "not_applicable")
        )
        out = {
            "s06-production-runtime": True,
            "resolved-config-sha256": self.sha256,
            "resolved-schema-version": schema_version,
            "model-mode": m["mode"],
            "det-camera-arch": m["camera_arch"],
            "det-camera-pretrained": m["camera_pretrained"],
            "det-camera-activation-checkpoint": camera_activation_checkpoint,
            "det-lidar-arch": m["lidar_arch"],
            "det-second-normalization": second_normalization,
            "det-fusion-arch": m["fusion_arch"],
            "det-head-arch": m["head_arch"],
            "precision": self.data["precision"],
            "det-sparse-conv-precision": self.data["sparse_conv_precision"],
            "dependency-torch": self.data["dependencies"]["torch"],
            "dependency-torch-build-sha256": self.data["dependencies"]["torch_build_sha256"],
            "dependency-torch-source-sha": self.data["dependencies"]["torch_source_sha"],
            "dependency-spconv": self.data["dependencies"]["spconv"],
            "dependency-spconv-build-sha256": self.data["dependencies"]["spconv_build_sha256"],
            "dependency-spconv-source-sha": self.data["dependencies"]["spconv_source_sha"],
            "dependency-spconv-source-state": _thaw(
                self.data["dependencies"]["spconv_source_state"]
            ),
            "dependency-cumm": self.data["dependencies"]["cumm"],
            "dependency-cumm-build-sha256": self.data["dependencies"]["cumm_build_sha256"],
            "dependency-cumm-source-sha": self.data["dependencies"]["cumm_source_sha"],
            "dependency-cumm-source-state": _thaw(
                self.data["dependencies"]["cumm_source_state"]
            ),
            "seed": t["seed"],
            "batch-size": t["micro_batch_size"],
            "accumulation-steps": t["accumulation_steps"],
            "effective-global-batch": t["effective_global_batch"],
            "max-optimizer-steps": t["max_optimizer_steps"],
            "max-epochs": t["max_epochs"],
            "grad-scaler-init-scale": (
                float(t["grad_scaler_init_scale"])
                if schema_version == "s10.v1" else 512.0
            ),
            "num-workers": t["num_workers"],
            "det-ema-decay": t["ema_decay"],
            "det-cbgs": t["sampling"] == "cbgs",
            "det-cbgs-thresh": 0.5,
            "det-cbgs-max-repeat": 4.0,
            "det-class-weights": None,
            "det-reg-class-weights": None,
            "learning-rate": o["learning_rate"],
            "weight-decay": o["weight_decay"],
            "det-optimizer": o["name"],
            "nuscenes-dataroot": d["dataroot"],
            "nuscenes-version": d["version"],
            "nuscenes-train-split": d["train_split"],
            "nuscenes-val-split": d["val_split"],
            "nuscenes-cache-dir": str(Path(d["caches"]["train"]["path"]).parent),
            "nuscenes-cache-identities": json.loads(
                canonical_json(_thaw(d["caches"])).decode("utf-8")
            ),
            "det-lidar-sweeps": d["n_sweeps"],
            "nuscenes-zip-manifest": d["zip_manifest"]["path"],
            "nuscenes-zip-manifest-logical-sha256": d["zip_manifest"]["logical_sha256"],
            "nuscenes-zip-manifest-file-sha256": d["zip_manifest"]["file_sha256"],
            "evaluation-timing": self.data["evaluation"]["timing"],
            "evaluation-checkpoint-weights": self.data["evaluation"]["checkpoint_weights"],
            "execution-mode": self.data["execution"]["mode"],
            "readiness-max-attempted-windows": self.data["execution"][
                "max_attempted_windows"
            ],
            "readiness-timing-warmup-successful-windows": self.data["execution"][
                "timing_warmup_successful_windows"
            ],
            "readiness-loader-profile": _thaw(
                self.data["execution"]["loader_profile"]
            ),
            "readiness-operator-profile": operator_profile,
        }
        for role in ("train", "val"):
            cache = d["caches"][role]
            out[f"nuscenes-{role}-cache-logical-sha256"] = cache["logical_sha256"]
            out[f"nuscenes-{role}-cache-pickle-sha256"] = cache["pickle_sha256"]
            out[f"nuscenes-{role}-cache-sidecar-sha256"] = cache["sidecar_sha256"]
        return out


def validate_precision_partition(
    global_precision: Any,
    lidar_arch: Any,
    sparse_conv_precision: Any,
) -> str:
    """Validate the explicit S08 global/sparse precision partition.

    This helper is deliberately pure Python so config resolution and production
    construction share one fail-closed matrix without importing torch.
    """
    precision = _enum(global_precision, _PRECISIONS, "precision")
    lidar = _enum(lidar_arch, _LIDARS, "model.lidar_arch")
    partition = _enum(
        sparse_conv_precision,
        _SPARSE_CONV_PRECISIONS,
        "sparse_conv_precision",
    )
    if lidar != "second_075":
        if partition != "not_applicable":
            raise ConfigError(
                "sparse_conv_precision must be 'not_applicable' when "
                "model.lidar_arch is not 'second_075'"
            )
        return partition
    if partition == "not_applicable":
        raise ConfigError(
            "sparse_conv_precision must be explicit 'fp32' or 'fp16' for second_075"
        )
    if precision == "fp32" and partition != "fp32":
        raise ConfigError(
            "precision='fp32' requires sparse_conv_precision='fp32' for second_075"
        )
    return partition


def resolve_config(raw: Mapping[str, Any]) -> ResolvedConfig:
    """Validate and canonicalize one complete production config; never consult env vars."""
    root = _mapping(dict(raw), "config")
    _keys(root, _ROOT, "config")
    schema_version = root["schema_version"]
    if schema_version not in {"s09.v1", "s09.v2", "s10.v1"}:
        raise ConfigError(
            "schema_version must be exactly 's09.v1', 's09.v2', or 's10.v1'; "
            "legacy/partial configs are refused"
        )

    model_keys = {
        "s09.v1": _MODEL_V1,
        "s09.v2": _MODEL_V2,
        "s10.v1": _MODEL_V3,
    }[schema_version]
    model = _mapping(root["model"], "model"); _keys(model, model_keys, "model")
    mode = _enum(model["mode"], _MODES, "model.mode")
    camera = _enum(model["camera_arch"], _CAMERAS, "model.camera_arch")
    camera_pretrained = model["camera_pretrained"]
    if camera == "none":
        if camera_pretrained is not None:
            raise ConfigError("model.camera_pretrained must be null when camera_arch='none'")
    elif not isinstance(camera_pretrained, bool):
        raise ConfigError("model.camera_pretrained must explicitly be true or false")
    if schema_version != "s09.v1":
        camera_checkpoint = model["camera_activation_checkpoint"]
        if not isinstance(camera_checkpoint, bool):
            raise ConfigError("model.camera_activation_checkpoint must be boolean")
        if camera == "none" and camera_checkpoint:
            raise ConfigError(
                "model.camera_activation_checkpoint must be false when camera_arch='none'"
            )
    lidar = _enum(model["lidar_arch"], _LIDARS, "model.lidar_arch")
    if schema_version == "s10.v1":
        second_normalization = _enum(
            model["second_normalization"],
            _SECOND_NORMALIZATIONS,
            "model.second_normalization",
        )
        if lidar == "second_075" and second_normalization == "not_applicable":
            raise ConfigError(
                "model.second_normalization must be explicit for lidar_arch='second_075'"
            )
        if lidar != "second_075" and second_normalization != "not_applicable":
            raise ConfigError(
                "model.second_normalization must be 'not_applicable' when "
                "lidar_arch is not 'second_075'"
            )
    fusion = _enum(model["fusion_arch"], _FUSIONS, "model.fusion_arch")
    _enum(model["head_arch"], _HEADS, "model.head_arch")
    required = {
        "camera_only": (camera != "none" and lidar == "none" and fusion == "none"),
        "lidar_only": (camera == "none" and lidar != "none" and fusion == "none"),
        "fusion": (camera != "none" and lidar != "none" and fusion != "none"),
    }
    if not required[mode]:
        raise ConfigError(f"model architecture fields are inconsistent with mode={mode!r}")

    precision = _enum(root["precision"], _PRECISIONS, "precision")
    validate_precision_partition(precision, lidar, root["sparse_conv_precision"])
    opt = _mapping(root["optimizer"], "optimizer"); _keys(opt, _OPT, "optimizer")
    _enum(opt["name"], _OPTIMIZERS, "optimizer.name")
    _number(opt["learning_rate"], "optimizer.learning_rate", positive=True)
    _number(opt["weight_decay"], "optimizer.weight_decay")

    train = _mapping(root["training"], "training")
    _keys(train, _TRAIN_S10 if schema_version == "s10.v1" else _TRAIN, "training")
    sampling = _enum(train["sampling"], _SAMPLING, "training.sampling")
    if sampling == "cbgs" and mode != "fusion":
        raise ConfigError("training.sampling='cbgs' is frozen only for the F-CBGS fusion candidate")
    for key in ("max_optimizer_steps", "micro_batch_size", "world_size", "accumulation_steps",
                "max_epochs"):
        _integer(train[key], f"training.{key}")
    _integer(train["num_workers"], "training.num_workers", minimum=0)
    if train["ema_decay"] is not None:
        decay = _number(train["ema_decay"], "training.ema_decay")
        if not 0.0 < decay < 1.0:
            raise ConfigError("training.ema_decay must be null or strictly between 0 and 1")
    _integer(train["seed"], "training.seed", minimum=0)
    if schema_version == "s10.v1":
        scaler_init = _number(
            train["grad_scaler_init_scale"],
            "training.grad_scaler_init_scale",
            positive=True,
        )
        if not scaler_init.is_integer() or int(scaler_init) & (int(scaler_init) - 1):
            raise ConfigError("training.grad_scaler_init_scale must be a positive power of two")
    expected_batch = train["micro_batch_size"] * train["world_size"] * train["accumulation_steps"]
    if train["effective_global_batch"] != expected_batch:
        raise ConfigError(
            f"training.effective_global_batch={train['effective_global_batch']} != "
            f"micro_batch_size*world_size*accumulation_steps={expected_batch}"
        )

    data = _mapping(root["data"], "data"); _keys(data, _DATA, "data")
    for key in ("dataroot", "version", "train_split", "val_split"):
        _path(data[key], f"data.{key}")
    _integer(data["n_sweeps"], "data.n_sweeps")
    caches = _mapping(data["caches"], "data.caches"); _keys(caches, _CACHES, "data.caches")
    cache_parents = set()
    for role in ("train", "val"):
        cache = _mapping(caches[role], f"data.caches.{role}")
        _keys(cache, _CACHE, f"data.caches.{role}")
        if cache["format"] != "t1.v2":
            raise ConfigError(f"data.caches.{role}.format must be exactly 't1.v2'; t1.v1 is forbidden")
        _path(cache["path"], f"data.caches.{role}.path")
        _path(cache["sidecar_path"], f"data.caches.{role}.sidecar_path")
        cache_parents.add(str(Path(cache["path"]).parent))
        for key in ("logical_sha256", "pickle_sha256", "sidecar_sha256"):
            _sha(cache[key], f"data.caches.{role}.{key}")
    if len(cache_parents) != 1:
        raise ConfigError("train/val caches must share one explicit cache directory")
    manifest = _mapping(data["zip_manifest"], "data.zip_manifest")
    _keys(manifest, _MANIFEST, "data.zip_manifest")
    _path(manifest["path"], "data.zip_manifest.path")
    for key in ("logical_sha256", "file_sha256"):
        _sha(manifest[key], f"data.zip_manifest.{key}")

    deps = _mapping(root["dependencies"], "dependencies"); _keys(deps, _DEPS, "dependencies")
    if not isinstance(deps["torch"], str) or not deps["torch"]:
        raise ConfigError("dependencies.torch must be an explicit version")
    _sha(deps["torch_build_sha256"], "dependencies.torch_build_sha256")
    torch_source = deps["torch_source_sha"]
    if not isinstance(torch_source, str) or len(torch_source) != 40 or any(c not in _HEX for c in torch_source):
        raise ConfigError("dependencies.torch_source_sha must be an exact lowercase 40-character Git SHA")
    if lidar == "second_075":
        if deps["spconv"] != "2.3.8":
            raise ConfigError("lidar/fusion requires dependencies.spconv exactly '2.3.8'")
        if deps["cumm"] != "0.7.13":
            raise ConfigError("lidar/fusion requires dependencies.cumm exactly '0.7.13'")
        for key in ("spconv_build_sha256", "cumm_build_sha256"):
            _sha(deps[key], f"dependencies.{key}")
        for key in ("spconv_source_sha", "cumm_source_sha"):
            value = deps[key]
            if not isinstance(value, str) or len(value) != 40 or any(c not in _HEX for c in value):
                raise ConfigError(f"dependencies.{key} must be an exact lowercase 40-character Git SHA")
        for key in ("spconv_source_state", "cumm_source_state"):
            _source_state(deps[key], f"dependencies.{key}")
    elif any(deps[k] is not None for k in (
        "spconv", "spconv_build_sha256", "spconv_source_sha", "spconv_source_state",
        "cumm", "cumm_build_sha256", "cumm_source_sha", "cumm_source_state",
    )):
        raise ConfigError("non-SECOND modes must set spconv/cumm dependency fields to null")

    evaluation = _mapping(root["evaluation"], "evaluation"); _keys(evaluation, _EVAL, "evaluation")
    if not isinstance(evaluation["timing"], bool):
        raise ConfigError("evaluation.timing must be boolean")
    weights = _enum(
        evaluation["checkpoint_weights"], _CHECKPOINT_WEIGHTS,
        "evaluation.checkpoint_weights",
    )
    if weights == "ema" and train["ema_decay"] is None:
        raise ConfigError("evaluation.checkpoint_weights='ema' requires training.ema_decay")

    execution = _mapping(root["execution"], "execution")
    execution_keys = _EXECUTION_V1 if schema_version == "s09.v1" else _EXECUTION_V2
    _keys(execution, execution_keys, "execution")
    execution_mode = _enum(execution["mode"], _EXECUTION_MODES, "execution.mode")
    max_attempted = _integer(
        execution["max_attempted_windows"],
        "execution.max_attempted_windows",
        minimum=0,
    )
    timing_warmup = _integer(
        execution["timing_warmup_successful_windows"],
        "execution.timing_warmup_successful_windows",
        minimum=0,
    )
    loader_profile = execution["loader_profile"]
    operator_profile = None if schema_version == "s09.v1" else execution["operator_profile"]
    if execution_mode == "train_eval":
        if (
            max_attempted != 0
            or timing_warmup != 0
            or loader_profile is not None
            or operator_profile is not None
        ):
            raise ConfigError(
                "execution.mode='train_eval' requires max_attempted_windows=0, "
                "timing_warmup_successful_windows=0, loader_profile=null, "
                "and operator_profile=null"
            )
    else:
        if train["world_size"] != 1 or train["accumulation_steps"] != 1:
            raise ConfigError(
                "execution.mode='readiness' requires world_size=1 and accumulation_steps=1"
            )
        if max_attempted < train["max_optimizer_steps"]:
            raise ConfigError(
                "execution.max_attempted_windows must be >= training.max_optimizer_steps"
            )
        if timing_warmup >= train["max_optimizer_steps"]:
            raise ConfigError(
                "execution.timing_warmup_successful_windows must be below "
                "training.max_optimizer_steps"
            )
        if evaluation["timing"]:
            raise ConfigError(
                "execution.mode='readiness' requires evaluation.timing=false because "
                "official evaluation is not executed"
            )
        if loader_profile is not None:
            profile = _mapping(loader_profile, "execution.loader_profile")
            _keys(profile, _LOADER_PROFILE, "execution.loader_profile")
            workers = profile["workers"]
            if not isinstance(workers, list) or not workers:
                raise ConfigError(
                    "execution.loader_profile.workers must be a non-empty list"
                )
            normalized_workers = [
                _integer(value, f"execution.loader_profile.workers[{index}]", minimum=0)
                for index, value in enumerate(workers)
            ]
            if len(set(normalized_workers)) != len(normalized_workers):
                raise ConfigError("execution.loader_profile.workers must be unique")
            if train["num_workers"] not in normalized_workers:
                raise ConfigError(
                    "training.num_workers must be one declared loader-profile worker cell"
                )
            for key in ("repeats", "determinism_batches", "measured_batches"):
                _integer(profile[key], f"execution.loader_profile.{key}")
            _integer(
                profile["warmup_batches"],
                "execution.loader_profile.warmup_batches",
                minimum=0,
            )
        if operator_profile is not None:
            profile = _mapping(operator_profile, "execution.operator_profile")
            _keys(profile, _OPERATOR_PROFILE, "execution.operator_profile")
            wait = _integer(
                profile["wait_attempted_windows"],
                "execution.operator_profile.wait_attempted_windows",
                minimum=0,
            )
            warmup = _integer(
                profile["warmup_attempted_windows"],
                "execution.operator_profile.warmup_attempted_windows",
                minimum=0,
            )
            active = _integer(
                profile["active_attempted_windows"],
                "execution.operator_profile.active_attempted_windows",
            )
            _integer(profile["row_limit"], "execution.operator_profile.row_limit")
            for key in ("record_shapes", "profile_memory"):
                if not isinstance(profile[key], bool):
                    raise ConfigError(f"execution.operator_profile.{key} must be boolean")
            if wait + warmup + active > train["max_optimizer_steps"]:
                raise ConfigError(
                    "execution.operator_profile schedule must finish within "
                    "training.max_optimizer_steps even when no scaler window is skipped"
                )
            if timing_warmup < wait + warmup + active:
                raise ConfigError(
                    "execution.timing_warmup_successful_windows must be >= the "
                    "complete operator-profile schedule so diagnostic windows are "
                    "excluded from throughput evidence"
                )

    normalized = json.loads(canonical_json(root).decode("utf-8"))
    encoded = canonical_json(normalized)
    return ResolvedConfig(_freeze(normalized), encoded, hashlib.sha256(encoded).hexdigest())


def load_resolved_config(path: str | Path) -> ResolvedConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return resolve_config(raw)


def verify_physical_data_identities(config: ResolvedConfig) -> None:
    """Hash the exact cache pickle/sidecar/manifest files before construction."""
    data = config.data["data"]
    checks = []
    for role in ("train", "val"):
        cache = data["caches"][role]
        checks.extend(((cache["path"], cache["pickle_sha256"], f"{role} cache pickle"),
                       (cache["sidecar_path"], cache["sidecar_sha256"], f"{role} cache sidecar")))
    checks.append((data["zip_manifest"]["path"], data["zip_manifest"]["file_sha256"], "ZIP manifest"))
    for path, expected, label in checks:
        digest = hashlib.sha256()
        try:
            with Path(path).open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
        except FileNotFoundError as exc:
            raise ConfigError(f"{label} missing at explicit path {path!r}") from exc
        actual = digest.hexdigest()
        if actual != expected:
            raise ConfigError(f"{label} physical identity drift: expected={expected}, actual={actual}")
