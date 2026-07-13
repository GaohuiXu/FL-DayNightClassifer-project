"""Deterministic BEVFusion-class detector.

A package of cleanly named sub-networks with stable per-module parameter
accounting, all obeying the single shared BEV convention in
:mod:`fl_v3.models.fusion.bev_grid`. Imports are kept lazy/local in the submodules
to avoid pulling torchvision at package import time.
"""
