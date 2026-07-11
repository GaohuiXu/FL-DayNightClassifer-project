"""CPU/static golden fixtures for the S04 SECOND resolution contract."""
from __future__ import annotations

import inspect

import pytest

from fl_v3.models.fusion.second_sparse_backbone import (
    SECONDShapeContract,
    SECONDSparseBackbone,
    validate_reference_stage_channels,
)
from fl_v3.models.fusion.sparse_voxel_encoder import SparseVoxelEncoder


def test_reference_shape_stride_channel_and_receptive_field_golden():
    contract = SECONDShapeContract.reference_075()
    assert contract.stage_shapes_zyx == (
        (41, 1440, 1440),
        (21, 720, 720),
        (11, 360, 360),
        (5, 180, 180),
        (5, 180, 180),
        (2, 180, 180),
    )
    assert contract.bev_hw == (180, 180)
    assert contract.output_stride_xy == 8
    assert contract.collapsed_channels == 256
    assert contract.output_cell_xy == pytest.approx((0.6, 0.6))
    assert contract.receptive_field_voxels_zyx == (153, 137, 137)
    assert contract.receptive_field_metres_zyx == pytest.approx((30.6, 10.275, 10.275))


def test_metric_coordinate_and_camera_fusion_alignment_golden():
    contract = SECONDShapeContract.reference_075()
    assert contract.output_cell_center_xy(0, 0) == pytest.approx((-53.7, -53.7))
    assert contract.output_cell_center_xy(90, 90) == pytest.approx((0.3, 0.3))
    assert contract.output_cell_center_xy(179, 179) == pytest.approx((53.7, 53.7))
    # The camera contract consumed by S07 must use these exact origin/cell/shape values.
    assert contract.range_xyzxyz[:2] == (-54.0, -54.0)
    assert contract.bev_hw == (180, 180)


def test_densification_is_bounded_to_reduced_resolution():
    contract = SECONDShapeContract.reference_075()
    assert contract.final_dense_numel(batch_size=4) == 33_177_600
    assert contract.forbidden_fine_dense_numel(batch_size=4) == 43_529_011_200
    assert contract.forbidden_fine_dense_numel(4) == 1312 * contract.final_dense_numel(4)
    # fp16 final sparse-dense conversion is 63.28125 MiB; fp32 is 126.5625 MiB.
    assert contract.final_dense_numel(4) * 2 / 2**20 == pytest.approx(63.28125)
    assert contract.final_dense_numel(4) * 4 / 2**20 == pytest.approx(126.5625)
    assert ".dense(" not in inspect.getsource(SECONDSparseBackbone)
    assert inspect.getsource(SparseVoxelEncoder.forward).count(".dense()") == 1


def test_official_reference_stage_channel_mapping_is_fail_closed():
    validate_reference_stage_channels(
        ((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128))
    )
    with pytest.raises(ValueError, match="stage channels"):
        validate_reference_stage_channels(((16, 32), (64, 128)))
