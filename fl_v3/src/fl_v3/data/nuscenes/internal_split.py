"""Deterministic S10 train-only split and ownership audit.

The split unit is a nuScenes ``log_token``.  A no-seed MILP chooses the
``D_fit/D_select/D_audit`` ownership and the nested ``D_low/D_mid`` rungs.  The
solver is followed by a separate reconstruction checker which consumes emitted
features/ownership rather than trusting solver summaries.

Two support definitions stay explicit throughout:

* ``training_support`` uses the accepted t1.v2 cache's ``gt_in_range`` targets.
  Its positive-frame prevalence drives the frozen class-support constraints.
* ``evaluation_support`` uses devkit ``load_gt -> add_center_dist ->
  filter_eval_boxes`` and drives eligible-box constraints and internal metrics.

This module neither trains a model nor reads sensor payload bytes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SCHEMA_PROTOCOL = "fl_v3.s10.split_protocol.v1"
SCHEMA_MANIFEST = "fl_v3.s10.split_manifest.v1"
SCHEMA_OWNERSHIP = "fl_v3.s10.sample_ownership.v1"
DETECTION_NAMES = (
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
)
LOCATIONS = (
    "boston-seaport",
    "singapore-onenorth",
    "singapore-queenstown",
    "singapore-hollandvillage",
)
BASE_ROLES = ("D_fit", "D_select", "D_audit")
BASE_LOG_QUOTAS = {
    "D_fit": (16, 11, 5, 2),
    "D_select": (3, 3, 1, 1),
    "D_audit": (3, 3, 1, 1),
}
BASE_TARGETS = {"D_fit": (7, 10), "D_select": (3, 20), "D_audit": (3, 20)}
BASE_SAMPLE_BOUNDS = {
    "D_fit": ((67, 100), (73, 100)),
    "D_select": ((12, 100), (18, 100)),
    "D_audit": ((12, 100), (18, 100)),
}
NESTED_ROLES = ("D_low", "D_mid")
NESTED_LOG_QUOTAS = {"D_low": (5, 3, 1, 1), "D_mid": (9, 7, 3, 1)}
NESTED_TARGETS = {"D_low": (3, 10), "D_mid": (3, 5)}
NESTED_SAMPLE_BOUNDS = {
    "D_low": ((27, 100), (33, 100)),
    "D_mid": ((57, 100), (63, 100)),
}
ASSIGNMENT_LEX_BLOCK_SIZE = 10


class SplitContractError(ValueError):
    """Fail-closed split, input-identity or ownership violation."""


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_sha256(value: str, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise SplitContractError(f"{label} must be a 64-character SHA-256 digest")
    return digest


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_jsonl(path: str, rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json_bytes(dict(row)).decode("utf-8"))
            stream.write("\n")


def read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise SplitContractError(f"blank JSONL record at {path}:{line_no}")
            out.append(json.loads(line))
    return out


def _scene_condition(description: str) -> str:
    text = str(description or "").lower()
    tod = "night" if "night" in text else "day"
    weather = "rain" if "rain" in text else "dry"
    return f"{tod}_{weather}"


def _new_class_counter() -> Dict[str, int]:
    return {name: 0 for name in DETECTION_NAMES}


def _validate_cache_tokens(info_list: Sequence[dict], expected_tokens: Sequence[str], split: str) -> None:
    actual = [str(info["sample_token"]) for info in info_list]
    if len(actual) != len(set(actual)):
        raise SplitContractError(f"{split} cache contains duplicate sample tokens")
    if sorted(actual) != sorted(str(token) for token in expected_tokens):
        expected = set(expected_tokens)
        got = set(actual)
        raise SplitContractError(
            f"{split} cache/devkit token identity drift: "
            f"missing={sorted(expected - got)[:8]}, extra={sorted(got - expected)[:8]}"
        )


def build_log_features(nusc, train_info: Sequence[dict]):
    """Build exact per-log MILP features and return filtered official train GT."""
    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_gt
    from nuscenes.eval.detection.data_classes import DetectionBox

    devkit_names = tuple(
        __import__("nuscenes.eval.detection.constants", fromlist=["DETECTION_NAMES"])
        .DETECTION_NAMES
    )
    if devkit_names != DETECTION_NAMES:
        raise SplitContractError(
            f"installed devkit class order drift: expected={DETECTION_NAMES}, actual={devkit_names}"
        )

    official_gt = load_gt(nusc, "train", DetectionBox, verbose=False)
    official_gt = add_center_dist(nusc, official_gt)
    official_gt = filter_eval_boxes(nusc, official_gt, _class_range(), verbose=False)
    official_tokens = set(official_gt.sample_tokens)

    logs: Dict[str, dict] = {}
    train_by_sample = {str(info["sample_token"]): info for info in train_info}
    if set(train_by_sample) != official_tokens:
        raise SplitContractError("accepted train cache differs from devkit official train split")

    for sample_token in sorted(train_by_sample):
        info = train_by_sample[sample_token]
        log_token = str(info["log_token"])
        scene_token = str(info["scene_token"])
        location = str(info["location"])
        if location not in LOCATIONS:
            raise SplitContractError(f"unexpected location {location!r}")
        entry = logs.setdefault(
            log_token,
            {
                "log_token": log_token,
                "location": location,
                "sample_tokens": [],
                "scene_tokens": set(),
                "conditions_by_scene": {},
                "training_frames": {name: set() for name in DETECTION_NAMES},
                "training_scenes": {name: set() for name in DETECTION_NAMES},
                "training_boxes": Counter(),
                "evaluation_frames": {name: set() for name in DETECTION_NAMES},
                "evaluation_scenes": {name: set() for name in DETECTION_NAMES},
                "evaluation_boxes": Counter(),
            },
        )
        if entry["location"] != location:
            raise SplitContractError(f"log {log_token} maps to multiple locations")
        entry["sample_tokens"].append(sample_token)
        entry["scene_tokens"].add(scene_token)
        scene = nusc.get("scene", scene_token)
        entry["conditions_by_scene"][scene_token] = _scene_condition(scene.get("description", ""))

        names = list(info["gt_names"])
        in_range = list(info["gt_in_range"])
        if len(names) != len(in_range):
            raise SplitContractError(f"cache target length mismatch for sample {sample_token}")
        present = set()
        for name, eligible in zip(names, in_range):
            if name not in DETECTION_NAMES:
                raise SplitContractError(f"unsupported cache detection class {name!r}")
            if bool(eligible):
                entry["training_boxes"][name] += 1
                present.add(name)
        for name in present:
            entry["training_frames"][name].add(sample_token)
            entry["training_scenes"][name].add(scene_token)

        eval_present = set()
        for box in official_gt[sample_token]:
            name = str(box.detection_name)
            if name not in DETECTION_NAMES:
                raise SplitContractError(f"unsupported official detection class {name!r}")
            entry["evaluation_boxes"][name] += 1
            eval_present.add(name)
        for name in eval_present:
            entry["evaluation_frames"][name].add(sample_token)
            entry["evaluation_scenes"][name].add(scene_token)

    records = []
    for log_token in sorted(logs):
        entry = logs[log_token]
        scenes = sorted(entry["scene_tokens"])
        conditions = Counter(entry["conditions_by_scene"].values())
        records.append(
            {
                "log_token": log_token,
                "location": entry["location"],
                "sample_tokens": sorted(entry["sample_tokens"]),
                "scene_tokens": scenes,
                "n_samples": len(entry["sample_tokens"]),
                "n_scenes": len(scenes),
                "condition_scenes": dict(sorted(conditions.items())),
                "training_support": {
                    "definition": "cache_gt_in_range",
                    "positive_frames": {
                        name: len(entry["training_frames"][name]) for name in DETECTION_NAMES
                    },
                    "boxes": {
                        name: int(entry["training_boxes"][name]) for name in DETECTION_NAMES
                    },
                    "positive_scenes": {
                        name: len(entry["training_scenes"][name]) for name in DETECTION_NAMES
                    },
                },
                "evaluation_support": {
                    "definition": "official_load_gt_add_center_dist_filter_eval_boxes",
                    "positive_frames": {
                        name: len(entry["evaluation_frames"][name]) for name in DETECTION_NAMES
                    },
                    "eligible_boxes": {
                        name: int(entry["evaluation_boxes"][name]) for name in DETECTION_NAMES
                    },
                    "positive_scenes": {
                        name: len(entry["evaluation_scenes"][name]) for name in DETECTION_NAMES
                    },
                },
            }
        )
    _assert_initial_topology(records)
    return records, official_gt


def _class_range() -> Dict[str, float]:
    from nuscenes.eval.common.config import config_factory

    cfg = config_factory("detection_cvpr_2019")
    return {name: float(cfg.class_range[name]) for name in DETECTION_NAMES}


def _assert_initial_topology(features: Sequence[dict]) -> None:
    if len(features) != 50:
        raise SplitContractError(f"expected exactly 50 train logs, found {len(features)}")
    counts = Counter(record["location"] for record in features)
    expected = dict(zip(LOCATIONS, (22, 17, 7, 4)))
    if dict(counts) != expected:
        raise SplitContractError(f"train log/location re-attestation failed: {counts} != {expected}")
    samples = sum(int(record["n_samples"]) for record in features)
    if samples != 28130:
        raise SplitContractError(f"expected 28130 accepted train samples, found {samples}")


class _MilpProblem:
    def __init__(self):
        self.names: List[str] = []
        self.lower: List[float] = []
        self.upper: List[float] = []
        self.integrality: List[int] = []
        self.constraints: List[Tuple[Dict[int, float], float, float, str]] = []

    def add_var(self, name: str, *, lower: float = 0, upper: float = 1, integer: bool = True) -> int:
        index = len(self.names)
        self.names.append(name)
        self.lower.append(lower)
        self.upper.append(upper)
        self.integrality.append(1 if integer else 0)
        return index

    def add_constraint(
        self,
        coefficients: Mapping[int, float],
        *,
        lower: float = -math.inf,
        upper: float = math.inf,
        name: str,
    ) -> None:
        merged = {int(k): float(v) for k, v in coefficients.items() if float(v) != 0.0}
        self.constraints.append((merged, float(lower), float(upper), name))

    def solve(self, objective: Mapping[int, float]):
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csc_matrix

        rows, cols, values = [], [], []
        for row, (coefs, _lo, _hi, _name) in enumerate(self.constraints):
            for col, value in coefs.items():
                rows.append(row)
                cols.append(col)
                values.append(value)
        # scipy.optimize.milp's bundled HiGHS Cython wrapper requires 32-bit
        # sparse index buffers on aarch64.  ``coo_array`` inherits platform
        # ``long`` here and fails before solving, so bind both index arrays
        # explicitly without changing any coefficient or constraint.
        matrix = csc_matrix(
            (
                np.asarray(values, dtype=np.float64),
                (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
            ),
            shape=(len(self.constraints), len(self.names)),
        )
        if matrix.indices.dtype != np.int32 or matrix.indptr.dtype != np.int32:
            raise SplitContractError(
                "MILP sparse matrix indices must remain int32 for the validated HiGHS wrapper"
            )
        c = np.zeros(len(self.names), dtype=float)
        for index, value in objective.items():
            c[index] = float(value)
        result = milp(
            c,
            integrality=np.asarray(self.integrality, dtype=np.uint8),
            bounds=Bounds(np.asarray(self.lower), np.asarray(self.upper)),
            constraints=LinearConstraint(
                matrix,
                np.asarray([row[1] for row in self.constraints]),
                np.asarray([row[2] for row in self.constraints]),
            ),
            options={"presolve": True, "mip_rel_gap": 0.0},
        )
        if result.status != 0 or not result.success or result.x is None:
            raise SplitContractError(
                f"MILP did not reach OPTIMAL: status={result.status}, message={result.message}"
            )
        return result


def _sum_expr(indices: Iterable[int], weights: Iterable[float] | None = None) -> Dict[int, float]:
    out: Dict[int, float] = defaultdict(float)
    if weights is None:
        for index in indices:
            out[int(index)] += 1.0
    else:
        for index, weight in zip(indices, weights):
            out[int(index)] += float(weight)
    return dict(out)


def _plus(*expressions: Mapping[int, float]) -> Dict[int, float]:
    out: Dict[int, float] = defaultdict(float)
    for expression in expressions:
        for index, value in expression.items():
            out[int(index)] += float(value)
    return {index: value for index, value in out.items() if value != 0.0}


def _scaled(expression: Mapping[int, float], factor: float) -> Dict[int, float]:
    return {index: float(value) * factor for index, value in expression.items()}


def _add_ppm_deviation(
    problem: _MilpProblem,
    expression: Mapping[int, float],
    total: int,
    target: Tuple[int, int],
    name: str,
) -> int:
    if total <= 0:
        raise SplitContractError(f"cannot define ppm objective {name} with total={total}")
    numerator, denominator = target
    target_ppm = 1_000_000.0 * numerator / denominator
    scale = 1_000_000.0 / total
    dev = problem.add_var(name, lower=0, upper=1_000_000, integer=True)
    problem.add_constraint(
        _plus(_scaled(expression, scale), {dev: -1}),
        upper=target_ppm,
        name=f"{name}:positive",
    )
    problem.add_constraint(
        _plus(_scaled(expression, -scale), {dev: -1}),
        upper=-target_ppm,
        name=f"{name}:negative",
    )
    return dev


def _integer_interval(total: int, bounds: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[int, int]:
    (lo_n, lo_d), (hi_n, hi_d) = bounds
    return (total * lo_n + lo_d - 1) // lo_d, (total * hi_n) // hi_d


def _feature_expr(
    features: Sequence[dict],
    variables: Mapping[Tuple[int, str], int],
    role: str,
    getter,
) -> Dict[int, float]:
    return {
        variables[(index, role)]: float(getter(record))
        for index, record in enumerate(features)
        if float(getter(record)) != 0.0
    }


def _fix_lex_stage(problem: _MilpProblem, objective: Mapping[int, float], label: str):
    result = problem.solve(objective)
    value = float(result.fun)
    rounded = round(value)
    if abs(value - rounded) > 1e-5:
        raise SplitContractError(f"non-integral lex objective {label}: {value}")
    problem.add_constraint(objective, lower=rounded, upper=rounded, name=f"fix:{label}")
    return result, int(rounded)


def _radix3_weights(length: int) -> Tuple[int, ...]:
    """Return safe exact weights for one contiguous assignment-vector block."""
    if not 1 <= length <= ASSIGNMENT_LEX_BLOCK_SIZE:
        raise SplitContractError(
            f"radix-3 block length must be in [1, {ASSIGNMENT_LEX_BLOCK_SIZE}], "
            f"found {length}"
        )
    weights = tuple(3 ** exponent for exponent in range(length - 1, -1, -1))
    if weights[0] > 19_683 or 2 * sum(weights) > 59_048:
        raise SplitContractError("radix-3 assignment objective exceeded its exact bound")
    return weights


def _ternary_digit(value: float, label: str) -> int:
    rounded = round(float(value))
    if abs(float(value) - rounded) > 1e-5 or rounded not in (0, 1, 2):
        raise SplitContractError(f"non-ternary assignment digit {label}: {value}")
    return int(rounded)


def _build_base_problem(features: Sequence[dict]):
    problem = _MilpProblem()
    variables = {
        (index, role): problem.add_var(f"x:{record['log_token']}:{role}")
        for index, record in enumerate(features)
        for role in BASE_ROLES
    }
    for index, record in enumerate(features):
        problem.add_constraint(
            {variables[(index, role)]: 1 for role in BASE_ROLES},
            lower=1,
            upper=1,
            name=f"one_role:{record['log_token']}",
        )
    for role in BASE_ROLES:
        for location_index, location in enumerate(LOCATIONS):
            expr = {
                variables[(index, role)]: 1
                for index, record in enumerate(features)
                if record["location"] == location
            }
            quota = BASE_LOG_QUOTAS[role][location_index]
            problem.add_constraint(expr, lower=quota, upper=quota, name=f"quota:{role}:{location}")

    total_samples = sum(record["n_samples"] for record in features)
    sample_exprs = {}
    for role in BASE_ROLES:
        expr = _feature_expr(features, variables, role, lambda r: r["n_samples"])
        sample_exprs[role] = expr
        lower, upper = _integer_interval(total_samples, BASE_SAMPLE_BOUNDS[role])
        problem.add_constraint(expr, lower=lower, upper=upper, name=f"sample_range:{role}")

    for role in ("D_select", "D_audit"):
        for class_name in DETECTION_NAMES:
            pos_expr = _feature_expr(
                features,
                variables,
                role,
                lambda r, c=class_name: r["training_support"]["positive_frames"][c],
            )
            box_expr = _feature_expr(
                features,
                variables,
                role,
                lambda r, c=class_name: r["evaluation_support"]["eligible_boxes"][c],
            )
            scene_expr = _feature_expr(
                features,
                variables,
                role,
                lambda r, c=class_name: r["training_support"]["positive_scenes"][c],
            )
            log_expr = _feature_expr(
                features,
                variables,
                role,
                lambda r, c=class_name: int(r["training_support"]["positive_frames"][c] > 0),
            )
            full_pos = sum(
                r["training_support"]["positive_frames"][class_name] for r in features
            )
            full_boxes = sum(
                r["evaluation_support"]["eligible_boxes"][class_name] for r in features
            )
            pos_lo, pos_hi = max(100, (2 * full_pos + 24) // 25), (11 * full_pos) // 50
            box_lo, box_hi = max(200, (2 * full_boxes + 24) // 25), (11 * full_boxes) // 50
            problem.add_constraint(pos_expr, lower=pos_lo, upper=pos_hi, name=f"pos:{role}:{class_name}")
            problem.add_constraint(box_expr, lower=box_lo, upper=box_hi, name=f"boxes:{role}:{class_name}")
            problem.add_constraint(scene_expr, lower=15, name=f"scenes:{role}:{class_name}")
            problem.add_constraint(log_expr, lower=3, name=f"logs:{role}:{class_name}")
            for index, record in enumerate(features):
                own_boxes = record["evaluation_support"]["eligible_boxes"][class_name]
                if own_boxes:
                    problem.add_constraint(
                        _plus(
                            {variables[(index, role)]: 2 * own_boxes},
                            _scaled(box_expr, -1),
                        ),
                        upper=0,
                        name=f"dominance:{role}:{class_name}:{record['log_token']}",
                    )

    stages: List[Tuple[str, Dict[int, float]]] = []
    sample_devs = [
        _add_ppm_deviation(
            problem, sample_exprs[role], total_samples, BASE_TARGETS[role], f"dev_sample:{role}"
        )
        for role in BASE_ROLES
    ]
    max_sample = problem.add_var("dev_sample:max", lower=0, upper=1_000_000, integer=True)
    for dev in sample_devs:
        problem.add_constraint({dev: 1, max_sample: -1}, upper=0, name=f"max_sample:{dev}")
    stages.append(("max_sample_deviation_ppm", {max_sample: 1}))
    stages.append(("total_sample_deviation_ppm", {dev: 1 for dev in sample_devs}))

    pos_devs, box_devs = [], []
    for role in BASE_ROLES:
        for class_name in DETECTION_NAMES:
            pos_total = sum(r["training_support"]["positive_frames"][class_name] for r in features)
            box_total = sum(r["evaluation_support"]["eligible_boxes"][class_name] for r in features)
            pos_devs.append(
                _add_ppm_deviation(
                    problem,
                    _feature_expr(
                        features, variables, role,
                        lambda r, c=class_name: r["training_support"]["positive_frames"][c],
                    ),
                    pos_total,
                    BASE_TARGETS[role],
                    f"dev_pos:{role}:{class_name}",
                )
            )
            box_devs.append(
                _add_ppm_deviation(
                    problem,
                    _feature_expr(
                        features, variables, role,
                        lambda r, c=class_name: r["evaluation_support"]["eligible_boxes"][c],
                    ),
                    box_total,
                    BASE_TARGETS[role],
                    f"dev_box:{role}:{class_name}",
                )
            )
    stages.append(("class_positive_frame_deviation_ppm", {dev: 1 for dev in pos_devs}))
    stages.append(("eligible_gt_deviation_ppm", {dev: 1 for dev in box_devs}))

    context_devs = []
    total_scenes = sum(record["n_scenes"] for record in features)
    condition_names = sorted({key for record in features for key in record["condition_scenes"]})
    for role in BASE_ROLES:
        context_devs.append(
            _add_ppm_deviation(
                problem,
                _feature_expr(features, variables, role, lambda r: r["n_scenes"]),
                total_scenes,
                BASE_TARGETS[role],
                f"dev_scene:{role}",
            )
        )
        for condition in condition_names:
            total = sum(record["condition_scenes"].get(condition, 0) for record in features)
            if total:
                context_devs.append(
                    _add_ppm_deviation(
                        problem,
                        _feature_expr(
                            features, variables, role,
                            lambda r, c=condition: r["condition_scenes"].get(c, 0),
                        ),
                        total,
                        BASE_TARGETS[role],
                        f"dev_condition:{role}:{condition}",
                    )
                )
        for location_index, location in enumerate(LOCATIONS):
            local_total = sum(
                record["n_samples"] for record in features if record["location"] == location
            )
            target = (BASE_LOG_QUOTAS[role][location_index], (22, 17, 7, 4)[location_index])
            context_devs.append(
                _add_ppm_deviation(
                    problem,
                    _feature_expr(
                        features,
                        variables,
                        role,
                        lambda r, loc=location: r["n_samples"] if r["location"] == loc else 0,
                    ),
                    local_total,
                    target,
                    f"dev_location_sample:{role}:{location}",
                )
            )
    stages.append(("scene_location_condition_deviation_ppm", {dev: 1 for dev in context_devs}))
    return problem, variables, stages


def _solve_base(features: Sequence[dict]):
    problem, variables, stages = _build_base_problem(features)
    objectives = []
    result = None
    for label, objective in stages:
        result, optimum = _fix_lex_stage(problem, objective, label)
        objectives.append({"name": label, "optimum": optimum})
    assignment_codes: List[int] = []
    for start in range(0, len(features), ASSIGNMENT_LEX_BLOCK_SIZE):
        stop = min(start + ASSIGNMENT_LEX_BLOCK_SIZE, len(features))
        weights = _radix3_weights(stop - start)
        objective: Dict[int, float] = defaultdict(float)
        for index, weight in zip(range(start, stop), weights):
            objective[variables[(index, "D_select")]] += weight
            objective[variables[(index, "D_audit")]] += 2 * weight
        result, optimum = _fix_lex_stage(
            problem,
            dict(objective),
            f"assignment_block:{start:02d}-{stop - 1:02d}",
        )
        block_codes = []
        for index in range(start, stop):
            code = _ternary_digit(
                result.x[variables[(index, "D_select")]]
                + 2 * result.x[variables[(index, "D_audit")]],
                features[index]["log_token"],
            )
            block_codes.append(code)
            select_value, audit_value = ((0, 0), (1, 0), (0, 1))[code]
            problem.add_constraint(
                {variables[(index, "D_select")]: 1},
                lower=select_value,
                upper=select_value,
                name=f"fix_assignment_select:{features[index]['log_token']}",
            )
            problem.add_constraint(
                {variables[(index, "D_audit")]: 1},
                lower=audit_value,
                upper=audit_value,
                name=f"fix_assignment_audit:{features[index]['log_token']}",
            )
        if sum(code * weight for code, weight in zip(block_codes, weights)) != optimum:
            raise SplitContractError("base radix-3 block failed exact objective reconstruction")
        assignment_codes.extend(block_codes)
    assert result is not None
    assignment = {}
    for index, (record, code) in enumerate(zip(features, assignment_codes)):
        chosen = [role for role in BASE_ROLES if result.x[variables[(index, role)]] > 0.5]
        if len(chosen) != 1:
            raise SplitContractError(f"non-integral base assignment for {record['log_token']}")
        if chosen[0] != BASE_ROLES[code]:
            raise SplitContractError(f"base assignment decode drift for {record['log_token']}")
        assignment[record["log_token"]] = chosen[0]
        objectives.append({"name": f"assignment:{record['log_token']}", "optimum": code})
    return assignment, {
        "status": "OPTIMAL",
        "status_code": int(result.status),
        "message": str(result.message),
        "objectives": objectives,
        "assignment_vector": [assignment[r["log_token"]] for r in features],
    }


def _build_nested_problem(features: Sequence[dict], base_assignment: Mapping[str, str]):
    fit = [record for record in features if base_assignment[record["log_token"]] == "D_fit"]
    if len(fit) != 34:
        raise SplitContractError(f"base assignment has {len(fit)} D_fit logs, expected 34")
    problem = _MilpProblem()
    variables = {
        (index, role): problem.add_var(f"nested:{record['log_token']}:{role}")
        for index, record in enumerate(fit)
        for role in NESTED_ROLES
    }
    for index, record in enumerate(fit):
        problem.add_constraint(
            {
                variables[(index, "D_low")]: 1,
                variables[(index, "D_mid")]: -1,
            },
            upper=0,
            name=f"nested:{record['log_token']}",
        )
    for role in NESTED_ROLES:
        quota_total = sum(NESTED_LOG_QUOTAS[role])
        problem.add_constraint(
            {variables[(index, role)]: 1 for index in range(len(fit))},
            lower=quota_total,
            upper=quota_total,
            name=f"nested_count:{role}",
        )
        for location_index, location in enumerate(LOCATIONS):
            expression = {
                variables[(index, role)]: 1
                for index, record in enumerate(fit)
                if record["location"] == location
            }
            quota = NESTED_LOG_QUOTAS[role][location_index]
            problem.add_constraint(
                expression, lower=quota, upper=quota, name=f"nested_quota:{role}:{location}"
            )

    fit_samples = sum(record["n_samples"] for record in fit)
    sample_exprs = {}
    for role in NESTED_ROLES:
        expression = _feature_expr(fit, variables, role, lambda r: r["n_samples"])
        sample_exprs[role] = expression
        lower, upper = _integer_interval(fit_samples, NESTED_SAMPLE_BOUNDS[role])
        problem.add_constraint(
            expression, lower=lower, upper=upper, name=f"nested_sample_range:{role}"
        )

    for role in NESTED_ROLES:
        for class_name in DETECTION_NAMES:
            pos_expr = _feature_expr(
                fit, variables, role,
                lambda r, c=class_name: r["training_support"]["positive_frames"][c],
            )
            scene_expr = _feature_expr(
                fit, variables, role,
                lambda r, c=class_name: r["training_support"]["positive_scenes"][c],
            )
            log_expr = _feature_expr(
                fit, variables, role,
                lambda r, c=class_name: int(r["training_support"]["positive_frames"][c] > 0),
            )
            minimum_frames = 50 if role == "D_low" else 100
            minimum_scenes = 5 if role == "D_low" else 10
            minimum_logs = 2 if role == "D_low" else 3
            problem.add_constraint(pos_expr, lower=minimum_frames, name=f"nested_pos:{role}:{class_name}")
            problem.add_constraint(scene_expr, lower=minimum_scenes, name=f"nested_scene:{role}:{class_name}")
            problem.add_constraint(log_expr, lower=minimum_logs, name=f"nested_log:{role}:{class_name}")

            fit_pos = sum(r["training_support"]["positive_frames"][class_name] for r in fit)
            if role == "D_low":
                # 0.5 <= (role_pos/role_samples)/(fit_pos/fit_samples) <= 1.5.
                problem.add_constraint(
                    _plus(_scaled(pos_expr, 2 * fit_samples), _scaled(sample_exprs[role], -fit_pos)),
                    lower=0,
                    name=f"nested_prevalence_low:{class_name}",
                )
                problem.add_constraint(
                    _plus(_scaled(pos_expr, 2 * fit_samples), _scaled(sample_exprs[role], -3 * fit_pos)),
                    upper=0,
                    name=f"nested_prevalence_high:{class_name}",
                )
            else:
                # 0.65 == 13/20 and 1.35 == 27/20.
                problem.add_constraint(
                    _plus(_scaled(pos_expr, 20 * fit_samples), _scaled(sample_exprs[role], -13 * fit_pos)),
                    lower=0,
                    name=f"nested_prevalence_low:{class_name}",
                )
                problem.add_constraint(
                    _plus(_scaled(pos_expr, 20 * fit_samples), _scaled(sample_exprs[role], -27 * fit_pos)),
                    upper=0,
                    name=f"nested_prevalence_high:{class_name}",
                )

    stages = []
    sample_devs = [
        _add_ppm_deviation(
            problem, sample_exprs[role], fit_samples, NESTED_TARGETS[role], f"nested_dev_sample:{role}"
        )
        for role in NESTED_ROLES
    ]
    max_sample = problem.add_var("nested_dev_sample:max", lower=0, upper=1_000_000, integer=True)
    for dev in sample_devs:
        problem.add_constraint({dev: 1, max_sample: -1}, upper=0, name=f"nested_max:{dev}")
    stages.append(("nested_max_sample_deviation_ppm", {max_sample: 1}))
    stages.append(("nested_total_sample_deviation_ppm", {dev: 1 for dev in sample_devs}))

    pos_devs, box_devs, context_devs = [], [], []
    for role in NESTED_ROLES:
        for class_name in DETECTION_NAMES:
            fit_pos = sum(r["training_support"]["positive_frames"][class_name] for r in fit)
            fit_boxes = sum(r["evaluation_support"]["eligible_boxes"][class_name] for r in fit)
            pos_devs.append(
                _add_ppm_deviation(
                    problem,
                    _feature_expr(
                        fit, variables, role,
                        lambda r, c=class_name: r["training_support"]["positive_frames"][c],
                    ),
                    fit_pos,
                    NESTED_TARGETS[role],
                    f"nested_dev_pos:{role}:{class_name}",
                )
            )
            box_devs.append(
                _add_ppm_deviation(
                    problem,
                    _feature_expr(
                        fit, variables, role,
                        lambda r, c=class_name: r["evaluation_support"]["eligible_boxes"][c],
                    ),
                    fit_boxes,
                    NESTED_TARGETS[role],
                    f"nested_dev_box:{role}:{class_name}",
                )
            )
        fit_scenes = sum(r["n_scenes"] for r in fit)
        context_devs.append(
            _add_ppm_deviation(
                problem,
                _feature_expr(fit, variables, role, lambda r: r["n_scenes"]),
                fit_scenes,
                NESTED_TARGETS[role],
                f"nested_dev_scene:{role}",
            )
        )
        condition_names = sorted({key for record in fit for key in record["condition_scenes"]})
        for condition in condition_names:
            total = sum(record["condition_scenes"].get(condition, 0) for record in fit)
            if total:
                context_devs.append(
                    _add_ppm_deviation(
                        problem,
                        _feature_expr(
                            fit, variables, role,
                            lambda r, c=condition: r["condition_scenes"].get(c, 0),
                        ),
                        total,
                        NESTED_TARGETS[role],
                        f"nested_dev_condition:{role}:{condition}",
                    )
                )
        for location_index, location in enumerate(LOCATIONS):
            local_total = sum(
                record["n_samples"] for record in fit if record["location"] == location
            )
            context_devs.append(
                _add_ppm_deviation(
                    problem,
                    _feature_expr(
                        fit,
                        variables,
                        role,
                        lambda r, loc=location: r["n_samples"] if r["location"] == loc else 0,
                    ),
                    local_total,
                    (
                        NESTED_LOG_QUOTAS[role][location_index],
                        BASE_LOG_QUOTAS["D_fit"][location_index],
                    ),
                    f"nested_dev_location_sample:{role}:{location}",
                )
            )
    stages.append(("nested_class_positive_frame_deviation_ppm", {dev: 1 for dev in pos_devs}))
    stages.append(("nested_eligible_gt_deviation_ppm", {dev: 1 for dev in box_devs}))
    stages.append(("nested_scene_location_condition_deviation_ppm", {dev: 1 for dev in context_devs}))
    return problem, variables, stages, fit


def _solve_nested(features: Sequence[dict], base_assignment: Mapping[str, str]):
    problem, variables, stages, fit = _build_nested_problem(features, base_assignment)
    objectives = []
    result = None
    for label, objective in stages:
        result, optimum = _fix_lex_stage(problem, objective, label)
        objectives.append({"name": label, "optimum": optimum})
    assignment_codes: List[int] = []
    for start in range(0, len(fit), ASSIGNMENT_LEX_BLOCK_SIZE):
        stop = min(start + ASSIGNMENT_LEX_BLOCK_SIZE, len(fit))
        weights = _radix3_weights(stop - start)
        objective: Dict[int, float] = defaultdict(float)
        for index, weight in zip(range(start, stop), weights):
            # Code is D_low=0, D_mid-only=1, D_fit-only=2.  The constant
            # ``2 * weight`` is omitted from each digit without changing argmin.
            objective[variables[(index, "D_low")]] -= weight
            objective[variables[(index, "D_mid")]] -= weight
        result, optimum = _fix_lex_stage(
            problem,
            dict(objective),
            f"nested_assignment_block:{start:02d}-{stop - 1:02d}",
        )
        block_codes = []
        for index in range(start, stop):
            code = _ternary_digit(
                2
                - result.x[variables[(index, "D_low")]]
                - result.x[variables[(index, "D_mid")]],
                fit[index]["log_token"],
            )
            block_codes.append(code)
            low_value, mid_value = ((1, 1), (0, 1), (0, 0))[code]
            problem.add_constraint(
                {variables[(index, "D_low")]: 1},
                lower=low_value,
                upper=low_value,
                name=f"fix_nested_low:{fit[index]['log_token']}",
            )
            problem.add_constraint(
                {variables[(index, "D_mid")]: 1},
                lower=mid_value,
                upper=mid_value,
                name=f"fix_nested_mid:{fit[index]['log_token']}",
            )
        encoded = sum(code * weight for code, weight in zip(block_codes, weights))
        if encoded - 2 * sum(weights) != optimum:
            raise SplitContractError("nested radix-3 block failed exact objective reconstruction")
        assignment_codes.extend(block_codes)
    assert result is not None
    low, mid = set(), set()
    for index, (record, code) in enumerate(zip(fit, assignment_codes)):
        if result.x[variables[(index, "D_mid")]] > 0.5:
            mid.add(record["log_token"])
        if result.x[variables[(index, "D_low")]] > 0.5:
            low.add(record["log_token"])
        observed_code = _ternary_digit(
            2
            - result.x[variables[(index, "D_low")]]
            - result.x[variables[(index, "D_mid")]],
            record["log_token"],
        )
        if observed_code != code:
            raise SplitContractError(f"nested assignment decode drift for {record['log_token']}")
        objectives.append({"name": f"nested_assignment:{record['log_token']}", "optimum": code})
    if not low <= mid:
        raise SplitContractError("D_low is not nested inside D_mid")
    return low, mid, {
        "status": "OPTIMAL",
        "status_code": int(result.status),
        "message": str(result.message),
        "objectives": objectives,
        "assignment_vector": [
            "D_low" if r["log_token"] in low else "D_mid" if r["log_token"] in mid else "D_fit"
            for r in fit
        ],
    }


def solve_split(features: Sequence[dict]):
    """Run both lexicographic no-seed MILPs and reconstruct their hard gates."""
    features = sorted(features, key=lambda record: record["log_token"])
    _assert_initial_topology(features)
    base, base_report = _solve_base(features)
    low, mid, nested_report = _solve_nested(features, base)
    checker = check_constraints(features, base, low, mid)
    return base, low, mid, {
        "solver": "scipy.optimize.milp/HiGHS",
        "seed": None,
        "integer_ppm_objectives": True,
        "base": base_report,
        "nested": nested_report,
        "checker_summary": checker,
    }


def _selected(features: Sequence[dict], logs: set[str]) -> List[dict]:
    return [record for record in features if record["log_token"] in logs]


def _role_summary(records: Sequence[dict]) -> dict:
    return {
        "logs": len(records),
        "samples": sum(r["n_samples"] for r in records),
        "scenes": sum(r["n_scenes"] for r in records),
        "locations": dict(Counter(r["location"] for r in records)),
        "training_positive_frames": {
            c: sum(r["training_support"]["positive_frames"][c] for r in records)
            for c in DETECTION_NAMES
        },
        "training_positive_scenes": {
            c: sum(r["training_support"]["positive_scenes"][c] for r in records)
            for c in DETECTION_NAMES
        },
        "training_positive_logs": {
            c: sum(r["training_support"]["positive_frames"][c] > 0 for r in records)
            for c in DETECTION_NAMES
        },
        "evaluation_eligible_boxes": {
            c: sum(r["evaluation_support"]["eligible_boxes"][c] for r in records)
            for c in DETECTION_NAMES
        },
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitContractError(message)


def check_constraints(
    features: Sequence[dict],
    base_assignment: Mapping[str, str],
    low_logs: set[str],
    mid_logs: set[str],
) -> dict:
    """Independent exact reconstruction of every frozen split hard constraint."""
    features = sorted(features, key=lambda r: r["log_token"])
    all_logs = {r["log_token"] for r in features}
    _require(set(base_assignment) == all_logs, "base assignment does not cover exact train logs")
    _require(set(base_assignment.values()) == set(BASE_ROLES), "base assignment misses a role")
    base_sets = {role: {log for log, value in base_assignment.items() if value == role} for role in BASE_ROLES}
    _require(low_logs <= mid_logs <= base_sets["D_fit"], "nested ownership relation failed")
    total_samples = sum(r["n_samples"] for r in features)
    summaries = {}
    for role in BASE_ROLES:
        records = _selected(features, base_sets[role])
        summary = _role_summary(records)
        summaries[role] = summary
        expected_locations = dict(zip(LOCATIONS, BASE_LOG_QUOTAS[role]))
        _require(summary["locations"] == expected_locations, f"{role} location quota failed")
        lower, upper = _integer_interval(total_samples, BASE_SAMPLE_BOUNDS[role])
        _require(lower <= summary["samples"] <= upper, f"{role} sample-size gate failed")
        if role in {"D_select", "D_audit"}:
            for class_name in DETECTION_NAMES:
                full_pos = sum(r["training_support"]["positive_frames"][class_name] for r in features)
                full_boxes = sum(r["evaluation_support"]["eligible_boxes"][class_name] for r in features)
                pos = summary["training_positive_frames"][class_name]
                boxes = summary["evaluation_eligible_boxes"][class_name]
                _require(max(100, (2 * full_pos + 24) // 25) <= pos <= (11 * full_pos) // 50,
                         f"{role}/{class_name} positive-frame gate failed")
                _require(max(200, (2 * full_boxes + 24) // 25) <= boxes <= (11 * full_boxes) // 50,
                         f"{role}/{class_name} eligible-box gate failed")
                _require(summary["training_positive_scenes"][class_name] >= 15,
                         f"{role}/{class_name} scene-support gate failed")
                _require(summary["training_positive_logs"][class_name] >= 3,
                         f"{role}/{class_name} log-support gate failed")
                for record in records:
                    own = record["evaluation_support"]["eligible_boxes"][class_name]
                    _require(2 * own <= boxes, f"{role}/{class_name} >50% log dominance")

    fit_records = _selected(features, base_sets["D_fit"])
    fit_summary = summaries["D_fit"]
    fit_samples = fit_summary["samples"]
    for role, logs in (("D_low", low_logs), ("D_mid", mid_logs)):
        records = _selected(features, logs)
        summary = _role_summary(records)
        summaries[role] = summary
        _require(summary["logs"] == sum(NESTED_LOG_QUOTAS[role]), f"{role} log count failed")
        _require(summary["locations"] == dict(zip(LOCATIONS, NESTED_LOG_QUOTAS[role])),
                 f"{role} location quota failed")
        lower, upper = _integer_interval(fit_samples, NESTED_SAMPLE_BOUNDS[role])
        _require(lower <= summary["samples"] <= upper, f"{role} sample-size gate failed")
        for class_name in DETECTION_NAMES:
            pos = summary["training_positive_frames"][class_name]
            scenes = summary["training_positive_scenes"][class_name]
            logs_with_class = summary["training_positive_logs"][class_name]
            if role == "D_low":
                _require(pos >= 50 and scenes >= 5 and logs_with_class >= 2,
                         f"{role}/{class_name} support gate failed")
                lo_n, lo_d, hi_n, hi_d = 1, 2, 3, 2
            else:
                _require(pos >= 100 and scenes >= 10 and logs_with_class >= 3,
                         f"{role}/{class_name} support gate failed")
                lo_n, lo_d, hi_n, hi_d = 13, 20, 27, 20
            fit_pos = fit_summary["training_positive_frames"][class_name]
            role_samples = summary["samples"]
            _require(lo_d * pos * fit_samples >= lo_n * fit_pos * role_samples,
                     f"{role}/{class_name} prevalence lower bound failed")
            _require(hi_d * pos * fit_samples <= hi_n * fit_pos * role_samples,
                     f"{role}/{class_name} prevalence upper bound failed")
    return {"status": "PASS", "roles": summaries}


def build_manifest(
    features: Sequence[dict],
    base_assignment: Mapping[str, str],
    low_logs: set[str],
    mid_logs: set[str],
    source_identities: Mapping[str, Any],
) -> dict:
    by_log = {record["log_token"]: record for record in features}

    def role_record(logs: Iterable[str]) -> dict:
        ordered = sorted(logs)
        return {
            "log_tokens": ordered,
            "scene_tokens": sorted(
                scene for log in ordered for scene in by_log[log]["scene_tokens"]
            ),
            "sample_tokens": sorted(
                sample for log in ordered for sample in by_log[log]["sample_tokens"]
            ),
        }

    roles = {
        role: role_record(log for log, value in base_assignment.items() if value == role)
        for role in BASE_ROLES
    }
    roles["D_low"] = role_record(low_logs)
    roles["D_mid"] = role_record(mid_logs)
    return {
        "schema": SCHEMA_MANIFEST,
        "parent_version": "v1.0-trainval",
        "parent_split": "train",
        "source_identities": dict(source_identities),
        "roles": roles,
        "audit_policy": {
            "D_select": "repeatable_train_only_selection",
            "D_audit": "locked_until_candidate_freeze_then_open_exactly_once",
            "official_val": "sealed_until_STOP_F",
        },
    }


def build_sample_ownership(
    nusc,
    train_info: Sequence[dict],
    val_info: Sequence[dict],
    base_assignment: Mapping[str, str],
    low_logs: set[str],
    mid_logs: set[str],
) -> List[dict]:
    """Materialize every inherited token/path owner for train and official val."""
    records = []
    for parent_split, info_list in (("train", train_info), ("val", val_info)):
        for info in sorted(info_list, key=lambda row: row["sample_token"]):
            sample_token = str(info["sample_token"])
            log_token = str(info["log_token"])
            sample = nusc.get("sample", sample_token)
            if parent_split == "train":
                if log_token not in base_assignment:
                    raise SplitContractError(f"train sample {sample_token} has unassigned log")
                owner = base_assignment[log_token]
            else:
                owner = "official_val"
            annotations, instances = [], []
            for ann_token in sample["anns"]:
                annotation = nusc.get("sample_annotation", ann_token)
                annotations.append(str(ann_token))
                instances.append(str(annotation["instance_token"]))
            camera_paths = [str(path) for path in info["cam_rel_paths"]]
            if len(camera_paths) != 6 or len(set(camera_paths)) != 6:
                raise SplitContractError(f"sample {sample_token} does not bind six unique cameras")
            sweep_paths = [str(sweep["rel_path"]) for sweep in info.get("lidar_sweeps", [])]
            records.append(
                {
                    "schema": SCHEMA_OWNERSHIP,
                    "parent_split": parent_split,
                    "owner": owner,
                    "in_D_low": bool(parent_split == "train" and log_token in low_logs),
                    "in_D_mid": bool(parent_split == "train" and log_token in mid_logs),
                    "log_token": log_token,
                    "scene_token": str(info["scene_token"]),
                    "sample_token": sample_token,
                    "annotation_tokens": sorted(annotations),
                    "instance_tokens": sorted(set(instances)),
                    "camera_paths": sorted(camera_paths),
                    "key_lidar_path": str(info["lidar_rel_path"]),
                    "lidar_sweep_paths": sorted(sweep_paths),
                }
            )
    return records


def check_ownership(
    ownership: Sequence[dict],
    manifest: Mapping[str, Any],
) -> dict:
    """Prove role disjointness for every inherited token and raw dependency."""
    sample_seen = set()
    domains = {
        "log": defaultdict(set),
        "scene": defaultdict(set),
        "sample": defaultdict(set),
        "annotation": defaultdict(set),
        "instance": defaultdict(set),
        "raw_path": defaultdict(set),
    }
    train_by_owner: Dict[str, set[str]] = defaultdict(set)
    val_count = 0
    for row in ownership:
        if row.get("schema") != SCHEMA_OWNERSHIP:
            raise SplitContractError("sample ownership schema drift")
        owner = str(row["owner"])
        sample = str(row["sample_token"])
        _require(sample not in sample_seen, f"duplicate ownership record for sample {sample}")
        sample_seen.add(sample)
        if row["parent_split"] == "train":
            _require(owner in BASE_ROLES, f"train sample has invalid owner {owner}")
            train_by_owner[owner].add(sample)
            _require((not row["in_D_low"]) or row["in_D_mid"], "D_low sample outside D_mid")
            _require((not row["in_D_mid"]) or owner == "D_fit", "D_mid sample outside D_fit")
        else:
            _require(row["parent_split"] == "val" and owner == "official_val",
                     "non-train ownership must be sealed official_val")
            _require(not row["in_D_low"] and not row["in_D_mid"], "official val enters a train rung")
            val_count += 1
        domains["log"][row["log_token"]].add(owner)
        domains["scene"][row["scene_token"]].add(owner)
        domains["sample"][sample].add(owner)
        for token in row["annotation_tokens"]:
            domains["annotation"][token].add(owner)
        for token in row["instance_tokens"]:
            domains["instance"][token].add(owner)
        paths = [*row["camera_paths"], row["key_lidar_path"], *row["lidar_sweep_paths"]]
        for path in paths:
            domains["raw_path"][path].add(owner)
    overlaps = {
        domain: {key: sorted(owners) for key, owners in mapping.items() if len(owners) > 1}
        for domain, mapping in domains.items()
    }
    bad = {domain: values for domain, values in overlaps.items() if values}
    if bad:
        preview = {domain: list(values.items())[:3] for domain, values in bad.items()}
        raise SplitContractError(f"cross-owner leakage detected: {preview}")
    for role in BASE_ROLES:
        expected = set(manifest["roles"][role]["sample_tokens"])
        _require(train_by_owner[role] == expected, f"ownership/manifest mismatch for {role}")
    _require(val_count == 6019, f"official val ownership count {val_count} != 6019")
    return {
        "status": "PASS",
        "overlap_counts": {domain: 0 for domain in domains},
        "unique_counts": {domain: len(values) for domain, values in domains.items()},
        "train_samples": sum(len(values) for values in train_by_owner.values()),
        "official_val_samples": val_count,
    }


def verify_artifact_directory(output_dir: str) -> dict:
    """Reload emitted artifacts and independently reconstruct split + leakage gates."""
    required = {
        "split_protocol.json",
        "log_features.jsonl",
        "sample_ownership.jsonl",
        "split_manifest.json",
    }
    missing = sorted(name for name in required if not os.path.isfile(os.path.join(output_dir, name)))
    if missing:
        raise SplitContractError(f"split artifact directory is missing {missing}")
    if os.path.exists(os.path.join(output_dir, "candidate_freeze.json")):
        raise SplitContractError("candidate_freeze.json must be absent before terminal STOP-D")
    protocol = json.load(open(os.path.join(output_dir, "split_protocol.json"), encoding="utf-8"))
    manifest = json.load(open(os.path.join(output_dir, "split_manifest.json"), encoding="utf-8"))
    features = read_jsonl(os.path.join(output_dir, "log_features.jsonl"))
    ownership = read_jsonl(os.path.join(output_dir, "sample_ownership.jsonl"))
    _require(protocol.get("schema") == SCHEMA_PROTOCOL, "split protocol schema mismatch")
    _require(manifest.get("schema") == SCHEMA_MANIFEST, "split manifest schema mismatch")
    base = {}
    for role in BASE_ROLES:
        for log in manifest["roles"][role]["log_tokens"]:
            _require(log not in base, f"log {log} appears in multiple base roles")
            base[log] = role
    low = set(manifest["roles"]["D_low"]["log_tokens"])
    mid = set(manifest["roles"]["D_mid"]["log_tokens"])
    constraint_report = check_constraints(features, base, low, mid)
    ownership_report = check_ownership(ownership, manifest)
    return {
        "schema": "fl_v3.s10.leakage_report.v1",
        "status": "PASS",
        "constraint_report": constraint_report,
        "ownership_report": ownership_report,
        "candidate_freeze_state": "ABSENT_LOCKED_UNTIL_STOP_D",
    }


def materialize_split_artifacts(
    nusc,
    train_info: Sequence[dict],
    val_info: Sequence[dict],
    output_dir: str,
    source_identities: Mapping[str, Any],
) -> dict:
    """Build, solve, emit, reload-check and checksum the frozen STOP-A split."""
    os.makedirs(output_dir, exist_ok=True)
    candidate_freeze = os.path.join(output_dir, "candidate_freeze.json")
    if os.path.exists(candidate_freeze):
        raise SplitContractError("refusing to overwrite an output containing candidate_freeze.json")
    from fl_v3.data.nuscenes.info_cache import split_sample_tokens

    train_tokens = split_sample_tokens(nusc, "train")
    val_tokens = split_sample_tokens(nusc, "val")
    _validate_cache_tokens(train_info, train_tokens, "train")
    _validate_cache_tokens(val_info, val_tokens, "val")
    if len(train_info) != 28130 or len(val_info) != 6019:
        raise SplitContractError("accepted train/val sample count identity drift")

    features, _official_gt = build_log_features(nusc, train_info)
    base, low, mid, solve_report = solve_split(features)
    manifest = build_manifest(features, base, low, mid, source_identities)
    ownership = build_sample_ownership(nusc, train_info, val_info, base, low, mid)
    protocol = {
        "schema": SCHEMA_PROTOCOL,
        "parent_version": "v1.0-trainval",
        "parent_split": "train",
        "source_identities": dict(source_identities),
        "support_semantics": {
            "positive_frames_and_prevalence": "training_support.cache_gt_in_range",
            "eligible_boxes": "evaluation_support.official_devkit_filtered",
            "official_filter_order": "load_gt->restrict_manifest->add_center_dist->filter_eval_boxes",
        },
        "quotas": {
            "locations": list(LOCATIONS),
            "base": {role: list(values) for role, values in BASE_LOG_QUOTAS.items()},
            "nested": {role: list(values) for role, values in NESTED_LOG_QUOTAS.items()},
        },
        "solver_report": solve_report,
        "candidate_freeze_policy": "file_absent_and_locked_until_terminal_STOP_D",
    }
    write_json(os.path.join(output_dir, "split_protocol.json"), protocol)
    write_jsonl(os.path.join(output_dir, "log_features.jsonl"), features)
    write_jsonl(os.path.join(output_dir, "sample_ownership.jsonl"), ownership)
    write_json(os.path.join(output_dir, "split_manifest.json"), manifest)
    leakage = verify_artifact_directory(output_dir)
    write_json(os.path.join(output_dir, "leakage_report.json"), leakage)

    names = (
        "split_protocol.json",
        "log_features.jsonl",
        "sample_ownership.jsonl",
        "split_manifest.json",
        "leakage_report.json",
    )
    with open(os.path.join(output_dir, "sha256sums.txt"), "w", encoding="utf-8") as stream:
        for name in sorted(names):
            stream.write(f"{sha256_file(os.path.join(output_dir, name))}  {name}\n")
    return {
        "status": "PASS",
        "output_dir": os.path.abspath(output_dir),
        "split_manifest_sha256": sha256_file(os.path.join(output_dir, "split_manifest.json")),
        "sha256sums_sha256": sha256_file(os.path.join(output_dir, "sha256sums.txt")),
        "roles": leakage["constraint_report"]["roles"],
    }
