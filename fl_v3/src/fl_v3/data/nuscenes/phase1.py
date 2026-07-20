"""Single-source S10 Phase-I D_fit dataset and sampler construction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from fl_v3.config import ResolvedConfig, verify_physical_data_identities
from fl_v3.data.nuscenes import info_cache
from fl_v3.data.nuscenes.cbgs import (
    OfficialCBGSWrapper,
    Phase1EpochPermutationSampler,
)
from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset, make_loader
from fl_v3.data.nuscenes.s10_binding import load_frozen_split_role
from fl_v3.models.fusion.collate import detection_collate_fn
from fl_v3.phase1_sampling import load_official_cbgs_artifact


@dataclass
class Phase1DataBundle:
    """The role-restricted base data, sealed CBGS view, sampler, and loader."""

    base_dataset: NuScenesMultimodalDataset
    dataset: OfficialCBGSWrapper
    sampler: Phase1EpochPermutationSampler
    loader: DataLoader
    role_identity: dict[str, Any]
    cache_meta: dict[str, Any]
    seed: int

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)
        self.sampler.set_epoch(epoch)
        # Phase I deliberately respawns workers at each epoch.  Addressing the
        # worker seed by epoch makes an epoch-boundary recovery checkpoint replay
        # the same augmentation/GT-paste streams instead of inheriting opaque
        # persistent-worker RNG state.
        self.loader.generator.manual_seed(self.seed + int(epoch))

    def close(self) -> None:
        self.base_dataset.close()


def phase1_augmentation_parameters(config: ResolvedConfig) -> dict[str, Any]:
    """Translate the complete resolved bundle into the canonical loader hook."""
    if not config.is_phase1:
        raise ValueError("Phase-I augmentation requires a Phase-I config")
    raw = config.as_dict()
    scene = raw["augmentation"]["scene_3d"]
    point_range = raw["model"]["head"]["train"]["point_cloud_range"]
    image = raw["augmentation"]["image"]
    reference_image = None
    if raw["contract"]["branch"] == "camera":
        if image is None:
            raise ValueError("Phase-I Camera recipe is missing image augmentation")
        reference_image = {
            "output_size": tuple(int(value) for value in image["output_size"]),
            "resize_limits": tuple(float(value) for value in image["train_resize"]),
            "bottom_crop_limits": tuple(
                float(value) for value in image["bottom_crop_percent"]
            ),
            "rotation_limits_degrees": tuple(
                float(value) for value in image["train_rotation_degrees"]
            ),
            "horizontal_flip_probability": float(
                image["train_horizontal_flip_probability"]
            ),
        }
    elif image is not None:
        raise ValueError("Phase-I LiDAR recipe must not contain image augmentation")
    return {
        "rot": float(scene["train_yaw_radians"][1]),
        "scale": tuple(float(value) for value in scene["train_scale"]),
        "translate": float(scene["train_translation_limit"]),
        "horizontal_flip_probability": float(
            scene["train_horizontal_flip_probability"]
        ),
        "vertical_flip_probability": float(
            scene["train_vertical_flip_probability"]
        ),
        "img_flip": 0.0,
        "allow_camera_only": True,
        "point_cloud_range": tuple(float(value) for value in point_range),
        "object_classes": tuple(raw["taxonomy"]["reference_object_classes"]),
        "point_shuffle": bool(raw["augmentation"]["point_shuffle"]),
        "reference_image_augmentation": reference_image,
    }


def phase1_gt_paste_parameters(
    config: ResolvedConfig,
    *,
    materialized_manifest_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Construct the role-bound paste contract, including immutable GTDB proof.

    ``materialized_manifest_sha256`` is allowed only for Envelope-A engineering
    calibration in the same job that creates a pending database.  Qualified
    recipes must carry the digest directly in the ResolvedConfig.
    """
    raw = config.as_dict()
    paste = raw["gt_paste"]
    if not paste["enabled"]:
        if materialized_manifest_sha256 is not None:
            raise ValueError("Camera recipe cannot accept a GTDB override")
        return None
    manifest_sha = paste["manifest_sha256"]
    if manifest_sha is None:
        if raw["contract"]["lifecycle"] != "envelope_a_implementation":
            raise ValueError("only Envelope-A implementation may bind a fresh GTDB digest")
        manifest_sha = materialized_manifest_sha256
    elif materialized_manifest_sha256 not in {None, manifest_sha}:
        raise ValueError("GTDB digest override differs from the accepted config")
    if manifest_sha is None:
        raise ValueError("Phase-I LiDAR loader requires a materialized GTDB digest")
    minimums = {int(value) for value in paste["min_points"].values()}
    if minimums != {5}:
        raise ValueError("Phase-I GTDB per-class minimum-point policy drift")
    data = raw["data"]
    return {
        "algorithm": "mit_bevfusion_reference",
        "database_root": paste["database_root"],
        "manifest_path": paste["manifest_path"],
        "manifest_sha256": manifest_sha,
        "class_names": tuple(paste["all_classes"]),
        "sample_groups": dict(paste["sample_groups"]),
        "min_points": 5,
        "filter_by_difficulty": tuple(paste["filter_by_difficulty"]),
        "rate": float(paste["rate"]),
        "stop_epoch": len(paste["active_zero_based_epochs"]),
        "yaw_jitter_radians": float(paste["yaw_jitter_radians"]),
        "expected_contract": {
            "plan_sha": raw["contract"]["plan_sha"],
            "request_commit": raw["contract"]["request_commit"],
            "candidate_id": raw["contract"]["candidate_id"],
        },
        "expected_source": {
            "role": "D_fit",
            "sample_count": int(data["roles"]["fit"]["samples"]),
            "sample_tokens_sha256": data["roles"]["fit"]["sample_tokens_sha256"],
            "split_manifest_path": str(Path(data["split_manifest"]["path"]).resolve()),
            "split_manifest_sha256": data["split_manifest"]["sha256"],
            "cache_capacity_sweeps": int(data["cache_capacity_sweeps"]),
            "consumed_point_sweeps": int(data["train_point_sweeps"]),
        },
        "expected_semantics": {
            "class_order": list(paste["all_classes"]),
            "min_points": dict(paste["min_points"]),
            "filter_by_difficulty": list(paste["filter_by_difficulty"]),
            "build_truncation": None,
            "source_point_dims": list(paste["source_point_dims"]),
            "point_coordinate_frame": "center_relative_lidar",
            "box_fields": ["x", "y", "z", "dx", "dy", "dz", "yaw", "vx", "vy"],
            "reference": "mit-han-lab/bevfusion:create_groundtruth_database",
        },
    }


def build_phase1_train_data(
    config: ResolvedConfig,
    *,
    materialized_gtdb_manifest_sha256: str | None = None,
) -> Phase1DataBundle:
    """Build the exact D_fit -> official-CBGS -> B4 production loader."""
    if not config.is_phase1:
        raise ValueError("build_phase1_train_data requires a Phase-I config")
    verify_physical_data_identities(config)
    raw = config.as_dict()
    data = raw["data"]
    cache = data["cache"]
    cache_path = Path(cache["path"]).resolve()
    expected_pickle, expected_sidecar = info_cache.cache_paths(
        str(cache_path.parent),
        data["version"],
        "train",
        n_sweeps=int(data["cache_capacity_sweeps"]),
    )
    if Path(expected_pickle).resolve() != cache_path or Path(expected_sidecar).resolve() != Path(
        cache["sidecar_path"]
    ).resolve():
        raise ValueError("Phase-I cache paths do not match the declared capacity")
    infos, meta = info_cache.load_cache(
        str(cache_path.parent),
        data["version"],
        "train",
        n_sweeps=int(data["cache_capacity_sweeps"]),
        expected_cache_hash=cache["logical_sha256"],
    )
    role = load_frozen_split_role(
        data["split_manifest"]["path"],
        expected_manifest_sha256=data["split_manifest"]["sha256"],
        role="D_fit",
        expected_source_identities={
            "train_cache_logical_sha256": cache["logical_sha256"],
            "train_cache_pickle_sha256": cache["pickle_sha256"],
            "train_cache_sidecar_sha256": cache["sidecar_sha256"],
            "zip_manifest_logical_sha256": data["zip_manifest"]["logical_sha256"],
            "zip_manifest_file_sha256": data["zip_manifest"]["file_sha256"],
        },
    )
    expected_role = data["roles"]["fit"]
    if (
        len(role.log_tokens) != int(expected_role["logs"])
        or len(role.scene_tokens) != int(expected_role["scenes"])
        or len(role.sample_tokens) != int(expected_role["samples"])
        or role.log_tokens_sha256 != expected_role["log_tokens_sha256"]
        or role.scene_tokens_sha256 != expected_role["scene_tokens_sha256"]
        or role.sample_tokens_sha256 != expected_role["sample_tokens_sha256"]
    ):
        raise ValueError("Phase-I D_fit role identity drift")

    branch = raw["contract"]["branch"]
    base = NuScenesMultimodalDataset(
        infos,
        data["dataroot"],
        sample_tokens=list(role.sample_tokens),
        n_sweeps=int(data["train_point_sweeps"]),
        cache_capacity_sweeps=int(data["cache_capacity_sweeps"]),
        target_class_names=raw["taxonomy"]["reference_object_classes"],
        augment=phase1_augmentation_parameters(config),
        gtpaste=phase1_gt_paste_parameters(
            config,
            materialized_manifest_sha256=materialized_gtdb_manifest_sha256,
        ),
        zip_manifest=data["zip_manifest"]["path"],
        model_mode="camera_only" if branch == "camera" else "lidar_only",
    )
    sampling = raw["sampling"]
    artifact = load_official_cbgs_artifact(
        sampling["artifact"]["path"],
        expected_sha256=sampling["artifact"]["sha256"],
        expected_sampling=sampling,
    )
    dataset = OfficialCBGSWrapper(base, artifact)
    training = raw["training"]
    sampler = Phase1EpochPermutationSampler(
        dataset,
        seed=int(training["seed"]),
        consumed_samples=int(training["consumed_samples_per_epoch"]),
        expected_twenty_epoch_order_sha256=sampling["twenty_epoch_order_sha256"],
        expected_twenty_epoch_remainder_sha256=sampling[
            "twenty_epoch_remainder_sha256"
        ],
        epochs=int(training["epochs"]),
    )
    loader = make_loader(
        dataset,
        batch_size=int(training["micro_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        seed=int(training["seed"]),
        collate_fn=detection_collate_fn,
        sampler=sampler,
        drop_last=True,
        persistent_workers=False,
    )
    return Phase1DataBundle(
        base_dataset=base,
        dataset=dataset,
        sampler=sampler,
        loader=loader,
        role_identity=role.identity(),
        cache_meta=dict(meta),
        seed=int(training["seed"]),
    )
