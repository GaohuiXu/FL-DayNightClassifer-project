"""nuScenes multimodal data module (fl_v3 T1).

Bit-deterministic loader returning the canonical sample schema (synchronized 6-cam
+ LIDAR_TOP + full calibration + canonical-frame 3D-box GT), the geographic
log-group partitioner, and the host-portable info-cache. No mmdet3d/mmcv/spconv;
the nuscenes-devkit is the geometry oracle. See ``conventions.md``.
"""
from fl_v3.data.nuscenes import (  # noqa: F401
    class_map,
    dataset,
    info_cache,
    partition,
    paths,
    transforms,
)
