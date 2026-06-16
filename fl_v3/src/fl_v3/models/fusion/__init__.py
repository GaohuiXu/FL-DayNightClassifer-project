"""Deterministic BEVFusion-class detector (fl_v3 T2).

A package of cleanly-named sub-networks (so per-module param counts + gradient
slicing for T5/T6 work without surgery), all obeying the single shared BEV
convention in :mod:`fl_v3.models.fusion.bev_grid`. Imports are kept lazy/local in
the submodules to avoid pulling torchvision at package import time.
"""
