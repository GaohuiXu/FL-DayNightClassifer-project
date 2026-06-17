"""fl_v3 T4 — official metric (mAP/NDS) + ASR-eligibility eval harness.

Turns the platform's decoded detections (``models.fusion.detector.decode``) into
scientifically valid utility + attack-success metrics:

  * :mod:`box_to_global` — canonical ``LIDAR_TOP`` box7 → nuScenes **global**
    ``DetectionBox`` (the submission conversion, anchored to the devkit oracle).
  * :mod:`detection_eval` — the official ``nuscenes.eval.detection`` ``DetectionEval``
    (mAP/NDS + per-class AP + the 5 TP errors) on the converted boxes.
  * :mod:`frustum_visibility` — ASR eligibility criterion (2): GT visible in ≥1 cam.
  * :mod:`asr` — the strict 6-criterion ASR-eligibility harness, the frozen held-out
    eligible-clean-detected subset, the disappearance metric + false-disappearance
    baseline + the denominator-N (built + validated on CLEAN data; T5 plugs the trigger).
  * :mod:`report` — the frozen 6-tuple reporting schema every attack×defense cell fills.

There is **no attack** in T4 (the trigger is T5). T4 builds + validates the machinery
on clean data and emits the benchmark-readiness verdict.
"""
