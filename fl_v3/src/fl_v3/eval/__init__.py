"""Official clean nuScenes detection evaluation.

``box_to_global`` converts canonical ``LIDAR_TOP`` boxes to global
``DetectionBox`` records, and ``detection_eval`` runs the official mAP/NDS,
per-class AP, and true-positive error evaluation.
"""
