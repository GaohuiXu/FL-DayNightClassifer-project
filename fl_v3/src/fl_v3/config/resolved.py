"""Canonical, fail-closed S06 production configuration.

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


class ConfigError(ValueError):
    """A production configuration is incomplete, unknown, or inconsistent."""


_HEX = frozenset("0123456789abcdef")
_MODES = frozenset({"camera_only", "lidar_only", "fusion"})
_PRECISIONS = frozenset({"fp32", "fp16"})
_OPTIMIZERS = frozenset({"adam", "adamw"})
_SAMPLING = frozenset({"uniform", "cbgs"})
_CAMERAS = frozenset({"swin_t_stride8", "none"})
_LIDARS = frozenset({"second_075", "pillar_020", "none"})
_FUSIONS = frozenset({"conv_fuser_256", "none"})
_HEADS = frozenset({"centerhead_multitask"})

_ROOT = frozenset({
    "schema_version", "model", "precision", "optimizer", "training", "data",
    "dependencies", "evaluation",
})
_MODEL = frozenset({
    "mode", "camera_arch", "camera_pretrained", "lidar_arch", "fusion_arch", "head_arch",
})
_OPT = frozenset({"name", "learning_rate", "weight_decay"})
_TRAIN = frozenset({
    "max_optimizer_steps", "micro_batch_size", "world_size", "accumulation_steps",
    "effective_global_batch", "seed", "max_epochs", "num_workers", "ema_decay", "sampling",
})
_DATA = frozenset({
    "dataroot", "version", "train_split", "val_split", "n_sweeps", "caches", "zip_manifest",
})
_CACHES = frozenset({"train", "val"})
_CACHE = frozenset({
    "format", "path", "sidecar_path", "logical_sha256", "pickle_sha256", "sidecar_sha256",
})
_MANIFEST = frozenset({"path", "logical_sha256", "file_sha256"})
_DEPS = frozenset({"torch", "spconv", "spconv_source_sha", "cumm", "cumm_source_sha"})
_EVAL = frozenset({"timing"})


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
        """Bridge to current task interfaces; S07-B owns final module enum wiring."""
        d, m, t, o = self.data["data"], self.data["model"], self.data["training"], self.data["optimizer"]
        out = {
            "s06-production-runtime": True,
            "resolved-config-sha256": self.sha256,
            "model-mode": m["mode"],
            "det-camera-arch": m["camera_arch"],
            "det-camera-pretrained": m["camera_pretrained"],
            "det-lidar-arch": m["lidar_arch"],
            "det-fusion-arch": m["fusion_arch"],
            "det-head-arch": m["head_arch"],
            "precision": self.data["precision"],
            "dependency-torch": self.data["dependencies"]["torch"],
            "dependency-spconv": self.data["dependencies"]["spconv"],
            "dependency-spconv-source-sha": self.data["dependencies"]["spconv_source_sha"],
            "dependency-cumm": self.data["dependencies"]["cumm"],
            "dependency-cumm-source-sha": self.data["dependencies"]["cumm_source_sha"],
            "seed": t["seed"],
            "batch-size": t["micro_batch_size"],
            "accumulation-steps": t["accumulation_steps"],
            "effective-global-batch": t["effective_global_batch"],
            "max-optimizer-steps": t["max_optimizer_steps"],
            "max-epochs": t["max_epochs"],
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
        }
        for role in ("train", "val"):
            cache = d["caches"][role]
            out[f"nuscenes-{role}-cache-logical-sha256"] = cache["logical_sha256"]
            out[f"nuscenes-{role}-cache-pickle-sha256"] = cache["pickle_sha256"]
            out[f"nuscenes-{role}-cache-sidecar-sha256"] = cache["sidecar_sha256"]
        return out


def resolve_config(raw: Mapping[str, Any]) -> ResolvedConfig:
    """Validate and canonicalize one complete S06 config; never consult env vars."""
    root = _mapping(dict(raw), "config")
    _keys(root, _ROOT, "config")
    if root["schema_version"] != "s06.v1":
        raise ConfigError("schema_version must be exactly 's06.v1'; legacy/partial configs are refused")

    model = _mapping(root["model"], "model"); _keys(model, _MODEL, "model")
    mode = _enum(model["mode"], _MODES, "model.mode")
    camera = _enum(model["camera_arch"], _CAMERAS, "model.camera_arch")
    camera_pretrained = model["camera_pretrained"]
    if camera == "none":
        if camera_pretrained is not None:
            raise ConfigError("model.camera_pretrained must be null when camera_arch='none'")
    elif not isinstance(camera_pretrained, bool):
        raise ConfigError("model.camera_pretrained must explicitly be true or false")
    lidar = _enum(model["lidar_arch"], _LIDARS, "model.lidar_arch")
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
    opt = _mapping(root["optimizer"], "optimizer"); _keys(opt, _OPT, "optimizer")
    _enum(opt["name"], _OPTIMIZERS, "optimizer.name")
    _number(opt["learning_rate"], "optimizer.learning_rate", positive=True)
    _number(opt["weight_decay"], "optimizer.weight_decay")

    train = _mapping(root["training"], "training"); _keys(train, _TRAIN, "training")
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
    if lidar == "second_075":
        if deps["spconv"] != "2.3.8":
            raise ConfigError("lidar/fusion requires dependencies.spconv exactly '2.3.8'")
        if deps["cumm"] != "0.7.13":
            raise ConfigError("lidar/fusion requires dependencies.cumm exactly '0.7.13'")
        for key in ("spconv_source_sha", "cumm_source_sha"):
            value = deps[key]
            if not isinstance(value, str) or len(value) != 40 or any(c not in _HEX for c in value):
                raise ConfigError(f"dependencies.{key} must be an exact lowercase 40-character Git SHA")
    elif any(deps[k] is not None for k in ("spconv", "spconv_source_sha", "cumm", "cumm_source_sha")):
        raise ConfigError("non-SECOND modes must set spconv/cumm dependency fields to null")

    evaluation = _mapping(root["evaluation"], "evaluation"); _keys(evaluation, _EVAL, "evaluation")
    if not isinstance(evaluation["timing"], bool):
        raise ConfigError("evaluation.timing must be boolean")

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
