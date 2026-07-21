"""Fail-closed, non-scientific configuration for S10 Phase I-P profiling.

The profiler specification is deliberately separate from the resolved Phase-I
recipe.  It may select measurement mechanics and default-off engineering
candidates, but it cannot silently create a new scientific config identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from fl_v3.config import ResolvedConfig


PHASE1_PROFILE_SCHEMA = "s10.phase1p.profile.v1"
PHASE1_PROFILE_SCHEMA_V2 = "s10.phase1p.profile.v2"
PHASE1_PROFILE_SCHEMA_V3 = "s10.phase1p.profile.v3"
PHASE1_PROFILE_SCHEMA_V4 = "s10.phase1p.profile.v4"

_HEX = frozenset("0123456789abcdef")
_ROOT_KEYS = frozenset({
    "schema",
    "phase",
    "envelope",
    "candidate_id",
    "branch_bindings",
    "measurement",
    "parity",
    "candidates",
    "boundaries",
})
_BRANCH_BINDING_KEYS = frozenset({
    "config_path",
    "config_file_sha256",
    "resolved_config_sha256",
})
_MEASUREMENT_KEYS = frozenset({
    "warmup_accepted_windows",
    "sustained_accepted_windows",
    "baseline_process_repeats",
    "repeat_instability_fraction",
    "trace_accepted_windows",
    "checkpoint_continuation_windows",
    "max_reserved_fraction",
    "system_sample_interval_seconds",
})
_MEASUREMENT_KEYS_V2 = _MEASUREMENT_KEYS | frozenset({"capacity_accepted_windows"})
_PARITY_KEYS = frozenset({"fp32", "fp16"})
_TOLERANCE_KEYS = frozenset({"rtol", "atol"})
_CANDIDATE_KEYS = frozenset({
    "camera_augmentation_transfer_cleanup",
    "training_field_whitelist",
    "camera_static_grid_cache",
    "camera_batched_affine_grid",
    "camera_batched_preprocess",
    "finite_loss_window_aggregation",
    "lidar_host_batch_offsets",
    "hungarian_batched_d2h",
    "checkpoint_snapshot_reuse",
    "checkpoint_async_write",
    "camera_sdpa",
    "lidar_sdpa",
    "torch_compile",
    "fused_adamw",
    "activation_checkpoint",
    "physical_batch_size",
    "checkpoint_cadence_epochs",
})
_CANDIDATE_KEYS_V4 = _CANDIDATE_KEYS | frozenset({
    "camera_vectorized_geometry",
    "camera_bulk_input_conversion",
})
_BOUNDARY_KEYS = frozenset({
    "allowed_data_role",
    "forbidden_roles",
    "capability_metrics",
    "output_root_prefix",
})

BASELINE_CANDIDATES: dict[str, Any] = {
    "camera_augmentation_transfer_cleanup": False,
    "training_field_whitelist": False,
    "camera_static_grid_cache": False,
    "camera_batched_affine_grid": False,
    "camera_batched_preprocess": False,
    "finite_loss_window_aggregation": False,
    "lidar_host_batch_offsets": False,
    "hungarian_batched_d2h": False,
    "checkpoint_snapshot_reuse": False,
    "checkpoint_async_write": False,
    "camera_sdpa": False,
    "lidar_sdpa": False,
    "torch_compile": False,
    "fused_adamw": False,
    "activation_checkpoint": False,
    "physical_batch_size": 4,
    "checkpoint_cadence_epochs": 1,
}
IP_E4_BASELINE_CANDIDATES: dict[str, Any] = {
    **BASELINE_CANDIDATES,
    "camera_vectorized_geometry": False,
    "camera_bulk_input_conversion": False,
}


def _candidate_options(**overrides: Any) -> dict[str, Any]:
    options = dict(BASELINE_CANDIDATES)
    unknown = set(overrides) - set(options)
    if unknown:
        raise RuntimeError(f"unknown Phase I-P candidate option(s): {sorted(unknown)}")
    options.update(overrides)
    return options


def _ip_e4_candidate_options(**overrides: Any) -> dict[str, Any]:
    options = dict(IP_E4_BASELINE_CANDIDATES)
    unknown = set(overrides) - set(options)
    if unknown:
        raise RuntimeError(f"unknown Phase I-P IP-E4 option(s): {sorted(unknown)}")
    options.update(overrides)
    return options


IP_E1_RUNNABLE_CANDIDATES: dict[str, dict[str, Any]] = {
    "reference_b4_accum8": {
        "branches": frozenset({"camera", "lidar"}),
        "options": _candidate_options(),
    },
    "camera_aug_transfer_cleanup_b4_accum8": {
        "branches": frozenset({"camera"}),
        "options": _candidate_options(camera_augmentation_transfer_cleanup=True),
    },
    "camera_static_grid_cache_b4_accum8": {
        "branches": frozenset({"camera"}),
        "options": _candidate_options(camera_static_grid_cache=True),
    },
    "camera_batched_affine_grid_b4_accum8": {
        "branches": frozenset({"camera"}),
        "options": _candidate_options(camera_batched_affine_grid=True),
    },
}

IP_E2_RUNNABLE_CANDIDATES: dict[str, dict[str, Any]] = {
    "camera_reference_b4_accum8": {
        "branches": frozenset({"camera"}),
        "options": _candidate_options(),
    },
    "camera_sdpa_b4_accum8": {
        "branches": frozenset({"camera"}),
        "options": _candidate_options(camera_sdpa=True),
    },
    "camera_compile_b4_accum8": {
        "branches": frozenset({"camera"}),
        "options": _candidate_options(torch_compile=True),
    },
    "camera_sdpa_compile_b4_accum8": {
        "branches": frozenset({"camera"}),
        "options": _candidate_options(camera_sdpa=True, torch_compile=True),
    },
    "camera_sdpa_compile_b8_accum4": {
        "branches": frozenset({"camera"}),
        "options": _candidate_options(
            camera_sdpa=True, torch_compile=True, physical_batch_size=8
        ),
    },
    "camera_sdpa_compile_fused_b8_accum4": {
        "branches": frozenset({"camera"}),
        "options": _candidate_options(
            camera_sdpa=True,
            torch_compile=True,
            fused_adamw=True,
            physical_batch_size=8,
        ),
    },
    "camera_sdpa_compile_b16_accum2": {
        "branches": frozenset({"camera"}),
        "options": _candidate_options(
            camera_sdpa=True, torch_compile=True, physical_batch_size=16
        ),
    },
    "camera_sdpa_compile_fused_b16_accum2": {
        "branches": frozenset({"camera"}),
        "options": _candidate_options(
            camera_sdpa=True,
            torch_compile=True,
            fused_adamw=True,
            physical_batch_size=16,
        ),
    },
}

IP_E3_RUNNABLE_CANDIDATES: dict[str, dict[str, Any]] = {
    "camera_sdpa_compile_fused_b16_accum2_followup_reference": {
        "branches": frozenset({"camera"}),
        "options": _candidate_options(
            camera_sdpa=True,
            torch_compile=True,
            fused_adamw=True,
            physical_batch_size=16,
        ),
    },
    "camera_sdpa_compile_fused_b16_accum2_followup_batched_affine_grid": {
        "branches": frozenset({"camera"}),
        "options": _candidate_options(
            camera_batched_affine_grid=True,
            camera_sdpa=True,
            torch_compile=True,
            fused_adamw=True,
            physical_batch_size=16,
        ),
    },
    "camera_sdpa_compile_fused_b16_accum2_followup_batched_rotation_grid_sample": {
        "branches": frozenset({"camera"}),
        "options": _candidate_options(
            camera_static_grid_cache=True,
            camera_batched_affine_grid=True,
            camera_batched_preprocess=True,
            camera_sdpa=True,
            torch_compile=True,
            fused_adamw=True,
            physical_batch_size=16,
        ),
    },
}

IP_E4_RUNNABLE_CANDIDATES: dict[str, dict[str, Any]] = {
    "camera_b16_batched_affine_reference": {
        "branches": frozenset({"camera"}),
        "options": _ip_e4_candidate_options(
            camera_batched_affine_grid=True,
            camera_sdpa=True,
            torch_compile=True,
            fused_adamw=True,
            physical_batch_size=16,
        ),
    },
    "camera_b16_batched_affine_vectorized_geometry": {
        "branches": frozenset({"camera"}),
        "options": _ip_e4_candidate_options(
            camera_batched_affine_grid=True,
            camera_vectorized_geometry=True,
            camera_sdpa=True,
            torch_compile=True,
            fused_adamw=True,
            physical_batch_size=16,
        ),
    },
}


def _runnable_candidates(envelope: str) -> dict[str, dict[str, Any]]:
    if envelope == "IP-E1":
        return IP_E1_RUNNABLE_CANDIDATES
    if envelope == "IP-E2":
        return IP_E2_RUNNABLE_CANDIDATES
    if envelope == "IP-E3":
        return IP_E3_RUNNABLE_CANDIDATES
    if envelope == "IP-E4":
        return IP_E4_RUNNABLE_CANDIDATES
    raise Phase1ProfileError(f"unknown Phase I-P envelope {envelope!r}")


class Phase1ProfileError(ValueError):
    """The Phase I-P measurement contract is missing or has drifted."""


def _keys(value: Any, expected: frozenset[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise Phase1ProfileError(f"{where} must be an object with string keys")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise Phase1ProfileError(
            f"{where} keys invalid: missing={missing}, unknown={unknown}"
        )
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Phase1ProfileError(f"{where} must be an integer >= {minimum}")
    return value


def _fraction(value: Any, where: str, *, lower_open: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase1ProfileError(f"{where} must be numeric")
    number = float(value)
    lower_ok = number > 0.0 if lower_open else number >= 0.0
    if not lower_ok or number >= 1.0:
        relation = "0 < value < 1" if lower_open else "0 <= value < 1"
        raise Phase1ProfileError(f"{where} must satisfy {relation}")
    return number


def _positive_number(value: Any, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) <= 0.0
    ):
        raise Phase1ProfileError(f"{where} must be numeric and > 0")
    return float(value)


def _sha256(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise Phase1ProfileError(f"{where} must be a lowercase SHA-256")
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class Phase1ProfileSpec:
    data: Mapping[str, Any]
    canonical_bytes: bytes
    sha256: str

    @property
    def measurement(self) -> Mapping[str, Any]:
        return self.data["measurement"]

    @property
    def candidates(self) -> Mapping[str, Any]:
        return self.data["candidates"]

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes.decode("utf-8"))

    def assert_baseline(self) -> None:
        if self.data["candidate_id"] != "reference_b4_accum8":
            raise Phase1ProfileError("IP-E1 baseline candidate identity drift")
        actual = dict(self.candidates)
        if actual != BASELINE_CANDIDATES:
            drift = {
                key: {"actual": actual.get(key), "expected": expected}
                for key, expected in BASELINE_CANDIDATES.items()
                if actual.get(key) != expected
            }
            raise Phase1ProfileError(
                f"IP-E1 baseline requires every candidate default-off: {drift}"
            )

    def assert_runnable(self, branch: str) -> None:
        candidate_id = str(self.data["candidate_id"])
        envelope = str(self.data["envelope"])
        specification = _runnable_candidates(envelope).get(candidate_id)
        if specification is None:
            raise Phase1ProfileError(
                f"{envelope} candidate {candidate_id!r} has no frozen runnable mapping"
            )
        if branch not in specification["branches"]:
            raise Phase1ProfileError(
                f"{envelope} candidate {candidate_id!r} is not runnable for {branch!r}"
            )
        actual = dict(self.candidates)
        expected = specification["options"]
        if actual != expected:
            drift = {
                key: {"actual": actual.get(key), "expected": value}
                for key, value in expected.items()
                if actual.get(key) != value
            }
            raise Phase1ProfileError(
                f"{envelope} candidate {candidate_id!r} option drift: {drift}"
            )

    def assert_branch_binding(
        self,
        branch: str,
        config_path: str | Path,
        config: ResolvedConfig,
    ) -> None:
        if branch not in {"camera", "lidar"}:
            raise Phase1ProfileError(f"unknown Phase I-P branch {branch!r}")
        binding = self.data["branch_bindings"][branch]
        path = Path(config_path)
        physical = hashlib.sha256(path.read_bytes()).hexdigest()
        if physical != binding["config_file_sha256"]:
            raise Phase1ProfileError(
                f"{branch} source config file identity drift: {physical}"
            )
        if config.sha256 != binding["resolved_config_sha256"]:
            raise Phase1ProfileError(
                f"{branch} resolved config identity drift: {config.sha256}"
            )
        normalized = path.as_posix()
        declared = str(binding["config_path"])
        if not normalized.endswith(declared):
            raise Phase1ProfileError(
                f"{branch} source config path drift: {normalized!r}"
            )
        if config.as_dict()["contract"]["branch"] != branch:
            raise Phase1ProfileError("profile branch and resolved config branch differ")


def validate_phase1_profile_spec(raw: Mapping[str, Any]) -> dict[str, Any]:
    root = _keys(dict(raw), _ROOT_KEYS, "profile")
    schema_to_envelope = {
        PHASE1_PROFILE_SCHEMA: "IP-E1",
        PHASE1_PROFILE_SCHEMA_V2: "IP-E2",
        PHASE1_PROFILE_SCHEMA_V3: "IP-E3",
        PHASE1_PROFILE_SCHEMA_V4: "IP-E4",
    }
    if root["schema"] not in schema_to_envelope:
        raise Phase1ProfileError(f"unsupported profile schema {root['schema']!r}")
    expected_envelope = schema_to_envelope[root["schema"]]
    if root["phase"] != "S10 Phase I-P" or root["envelope"] != expected_envelope:
        raise Phase1ProfileError("profile phase/envelope identity drift")
    runnable = _runnable_candidates(expected_envelope)
    if (
        not isinstance(root["candidate_id"], str)
        or root["candidate_id"] not in runnable
    ):
        raise Phase1ProfileError(f"{expected_envelope} candidate identity drift")

    bindings = _keys(
        root["branch_bindings"], frozenset({"camera", "lidar"}), "branch_bindings"
    )
    for branch, value in bindings.items():
        binding = _keys(value, _BRANCH_BINDING_KEYS, f"branch_bindings.{branch}")
        expected_path = f"fl_v3/configs/s10_phase1_{branch}.json"
        if binding["config_path"] != expected_path:
            raise Phase1ProfileError(f"{branch} config path drift")
        _sha256(binding["config_file_sha256"], f"{branch}.config_file_sha256")
        _sha256(binding["resolved_config_sha256"], f"{branch}.resolved_config_sha256")

    measurement_keys = (
        _MEASUREMENT_KEYS if expected_envelope == "IP-E1" else _MEASUREMENT_KEYS_V2
    )
    measurement = _keys(root["measurement"], measurement_keys, "measurement")
    expected_integers = {
        "warmup_accepted_windows": 16,
        "sustained_accepted_windows": 256,
        "baseline_process_repeats": 2,
        "trace_accepted_windows": 3,
        "checkpoint_continuation_windows": 8,
    }
    for key, expected in expected_integers.items():
        if _integer(measurement[key], f"measurement.{key}", minimum=1) != expected:
            raise Phase1ProfileError(f"measurement.{key} must remain {expected}")
    if expected_envelope in {"IP-E2", "IP-E3", "IP-E4"} and _integer(
        measurement["capacity_accepted_windows"],
        "measurement.capacity_accepted_windows",
        minimum=1,
    ) != 8:
        raise Phase1ProfileError("measurement.capacity_accepted_windows must remain 8")
    if _fraction(
        measurement["repeat_instability_fraction"],
        "measurement.repeat_instability_fraction",
        lower_open=True,
    ) != 0.03:
        raise Phase1ProfileError("repeat instability fraction must remain 0.03")
    if _fraction(
        measurement["max_reserved_fraction"],
        "measurement.max_reserved_fraction",
        lower_open=True,
    ) != 0.85:
        raise Phase1ProfileError("max reserved fraction must remain 0.85")
    if _positive_number(
        measurement["system_sample_interval_seconds"],
        "measurement.system_sample_interval_seconds",
    ) != 1.0:
        raise Phase1ProfileError("system sample interval must remain 1 second")

    parity = _keys(root["parity"], _PARITY_KEYS, "parity")
    expected_tolerances = {
        "fp32": {"rtol": 1e-4, "atol": 1e-6},
        "fp16": {"rtol": 2e-3, "atol": 2e-4},
    }
    for precision, expected in expected_tolerances.items():
        tolerance = _keys(
            parity[precision], _TOLERANCE_KEYS, f"parity.{precision}"
        )
        if tolerance != expected:
            raise Phase1ProfileError(f"parity.{precision} tolerance drift")

    candidate_keys = (
        _CANDIDATE_KEYS_V4 if expected_envelope == "IP-E4" else _CANDIDATE_KEYS
    )
    candidates = _keys(root["candidates"], candidate_keys, "candidates")
    for key in candidate_keys - {"physical_batch_size", "checkpoint_cadence_epochs"}:
        if not isinstance(candidates[key], bool):
            raise Phase1ProfileError(f"candidates.{key} must be boolean")
    _integer(candidates["physical_batch_size"], "candidates.physical_batch_size", minimum=1)
    _integer(
        candidates["checkpoint_cadence_epochs"],
        "candidates.checkpoint_cadence_epochs",
        minimum=1,
    )
    if expected_envelope in {"IP-E2", "IP-E3", "IP-E4"} and candidates[
        "physical_batch_size"
    ] not in {
        4, 8, 16
    }:
        raise Phase1ProfileError(
            f"{expected_envelope} physical batch must be one of 4, 8, or 16"
        )

    boundaries = _keys(root["boundaries"], _BOUNDARY_KEYS, "boundaries")
    if boundaries["allowed_data_role"] != "D_fit":
        raise Phase1ProfileError("Phase I-P may consume only D_fit")
    if boundaries["forbidden_roles"] != [
        "D_select", "D_audit", "official_validation"
    ]:
        raise Phase1ProfileError("Phase I-P forbidden-role boundary drift")
    if boundaries["capability_metrics"] is not False:
        raise Phase1ProfileError("Phase I-P capability metrics must remain disabled")
    prefix = boundaries["output_root_prefix"]
    expected_prefix = (
        "/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/"
        f"arrhenius_fl_v3/outputs/s10_phase1p_{expected_envelope.lower().replace('-', '_')}_"
    )
    if prefix != expected_prefix:
        raise Phase1ProfileError("Phase I-P output root prefix drift")

    # Detach custom containers and reject NaN/Infinity before hashing.
    return json.loads(json.dumps(root, allow_nan=False))


@dataclass(frozen=True)
class Phase1ProfileRuntimeConfig:
    """Profile-only effective runtime view over one validated Phase-I recipe.

    Production config bytes stay untouched.  Only physical batch/accumulation and
    the AdamW fused backend may differ.  A non-scientific ``phase1p_runtime``
    identity binds every profiler candidate (including SDPA/compile flags) into
    checkpoint and runtime provenance without making it a production recipe.
    """

    source: ResolvedConfig
    profile_sha256: str
    data: Mapping[str, Any]
    canonical_bytes: bytes
    sha256: str

    @property
    def schema_version(self) -> str:
        return str(self.data["schema_version"])

    @property
    def is_phase1(self) -> bool:
        return True

    @property
    def model_mode(self) -> str:
        return "camera_only" if self.data["contract"]["branch"] == "camera" else "lidar_only"

    @property
    def precision(self) -> str:
        return str(self.data["precision"]["global_autocast"])

    @property
    def data_identities(self) -> dict[str, Any]:
        return self.source.data_identities

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes.decode("utf-8"))

    def to_run_config(self) -> dict[str, Any]:
        run = self.source.to_run_config()
        raw = self.as_dict()
        run["resolved-config-sha256"] = self.sha256
        run["phase1"] = raw
        run["batch-size"] = int(raw["training"]["micro_batch_size"])
        run["accumulation-steps"] = int(raw["training"]["accumulation_steps"])
        run["effective-global-batch"] = int(raw["training"]["effective_global_batch"])
        return run


def derive_profile_runtime_config(
    source: ResolvedConfig,
    profile: Phase1ProfileSpec,
) -> Phase1ProfileRuntimeConfig:
    """Derive the explicit non-production runtime config bound by ``profile``."""
    if not source.is_phase1:
        raise Phase1ProfileError("Phase I-P runtime view requires a Phase-I source config")
    raw = source.as_dict()
    batch = int(profile.candidates["physical_batch_size"])
    if batch not in {4, 8, 16} or 32 % batch:
        raise Phase1ProfileError("physical batch must divide effective B32 exactly")
    accumulation = 32 // batch
    training = raw["training"]
    training["micro_batch_size"] = batch
    training["accumulation_steps"] = accumulation
    training["effective_global_batch"] = 32
    training["loss_accumulation"] = {
        8: "mean_over_eight_microbatches",
        4: "mean_over_four_microbatches",
        2: "mean_over_two_microbatches",
    }[accumulation]
    raw["optimizer"]["fused"] = bool(profile.candidates["fused_adamw"])
    raw["phase1p_runtime"] = {
        "schema": "s10.phase1p.runtime-config.v1",
        "source_resolved_config_sha256": source.sha256,
        "profile_sha256": profile.sha256,
        "envelope": str(profile.data["envelope"]),
        "candidate_id": str(profile.data["candidate_id"]),
        "candidate_options": dict(profile.candidates),
    }
    encoded = _canonical_bytes(raw)
    return Phase1ProfileRuntimeConfig(
        source=source,
        profile_sha256=profile.sha256,
        data=_freeze(raw),
        canonical_bytes=encoded,
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def load_phase1_profile_spec(path: str | Path) -> Phase1ProfileSpec:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    validated = validate_phase1_profile_spec(raw)
    encoded = _canonical_bytes(validated)
    return Phase1ProfileSpec(
        data=_freeze(validated),
        canonical_bytes=encoded,
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


__all__ = [
    "BASELINE_CANDIDATES",
    "IP_E1_RUNNABLE_CANDIDATES",
    "IP_E2_RUNNABLE_CANDIDATES",
    "IP_E3_RUNNABLE_CANDIDATES",
    "IP_E4_BASELINE_CANDIDATES",
    "IP_E4_RUNNABLE_CANDIDATES",
    "PHASE1_PROFILE_SCHEMA",
    "PHASE1_PROFILE_SCHEMA_V2",
    "PHASE1_PROFILE_SCHEMA_V3",
    "PHASE1_PROFILE_SCHEMA_V4",
    "Phase1ProfileError",
    "Phase1ProfileRuntimeConfig",
    "Phase1ProfileSpec",
    "derive_profile_runtime_config",
    "load_phase1_profile_spec",
    "validate_phase1_profile_spec",
]
