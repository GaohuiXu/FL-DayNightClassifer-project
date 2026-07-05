"""LSS camera→BEV view transform (fl_v3 T2) — the deterministic ``cumsum_trick`` splat.

Per the Architecture table: per-pixel **depth softmax** over bins → lift context
features into the camera frustum (outer product — determinism-clean) → transform the
frustum to the canonical LIDAR_TOP frame via ``inv(lidar2img)`` → **splat into BEV via
the LSS ``cumsum_trick``** (sort by BEV rank → float cumsum → segment-boundary diff).
**NO ``scatter_add`` / ``index_add`` / atomic scatter; NO ``grid_sample``** (the lift is
grid_sample-free by construction — the ban is belt-and-suspenders).

## The two determinism musts (SPEC §0 + view_transform bullet)

1. **The LSS reference sorts with a non-stable ``ranks.argsort()``.** Float-add is
   non-associative, so a tie permutation drifts the pooled cell value at ULP level and
   ``strict`` mode does NOT catch it. We sort by a **canonical composite key**
   ``rank·G + geom_id`` where ``geom_id`` is the frustum cell's canonical index
   ``((n·D + d)·fH + i)·fW + j`` — making ``(rank, geom_id)`` **globally unique**, so the
   sort order (and thus the per-cell summation order) is **independent of the order the
   frustum points are presented in**. This is what makes the splat pass the
   permutation-invariance gate AND be architecture-portable (a fixed summation order on
   every GPU / the future ARM H200). ``argsort(stable=True)`` is used on top (a no-op for
   unique keys, but it honors the contract and is safe if a key ever ties).

2. **The cumsum is float-on-CUDA**, which the torch-2.7 docstring still lists as
   *raising* under deterministic mode but **empirically does NOT** on this build (§0).
   It is pinned by the GPU ``cumsum`` guard test (forward+backward ``torch.equal``) and a
   note that the **ARM/H200 rebuild must re-run it**; if a future torch raises, the
   fallback is an int/segment-reduce reformulation, **NOT ``scatter_add``**.

The per-cell sums are written into the BEV canvas with ``index_copy_`` on the FLAT
canvas at the **unique** cell ranks (assignment, not accumulation — #76176-safe), then
reshaped to ``[B, Cc, ny, nx]`` per the shared :mod:`bev_grid` convention (``W→x, H→y``).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fl_v3.models.fusion.bev_grid import BEVConfig, metric_to_grid, in_grid_mask, flat_index


class QuickCumsum(torch.autograd.Function):
    """LSS segment-sum via cumsum + boundary diff (deterministic, atomic-free).

    ``x`` is sorted so equal ``ranks`` are contiguous. Forward: ``cumsum`` then take the
    last element of each rank-segment and difference adjacent segments → per-segment
    sums. Backward: ``cumsum`` of the kept-mask gives, for each input, the output index
    it contributed to — a **gather** (deterministic), never a scatter."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, ranks: torch.Tensor):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]
        x = x[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))
        ctx.save_for_backward(kept)
        return x

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1  # 0-based index of the output each input fed
        return grad_x[back], None


def bev_splat(
    x: torch.Tensor,       # [P, Cc] lifted features (P surviving frustum points)
    ranks: torch.Tensor,   # [P] int64 global flat rank = b*(nx*ny) + row*nx + col
    geom_id: torch.Tensor,  # [P] int64 canonical frustum index (per-point, order-invariant)
    B: int,
    nx: int,
    ny: int,
) -> torch.Tensor:
    """Splat ``x`` into a ``[B, Cc, ny, nx]`` BEV canvas via the canonical cumsum_trick."""
    Cc = x.shape[1]
    if x.numel() == 0:
        return x.new_zeros((B, Cc, ny, nx))
    # canonical, permutation-invariant total order: (rank primary, geom_id tiebreak).
    G = int(geom_id.max().item()) + 1
    key = ranks * G + geom_id
    order = torch.argsort(key, stable=True)
    x, ranks_s = x[order], ranks[order]
    summed = QuickCumsum.apply(x, ranks_s)            # [U, Cc] per-cell sums
    # the kept ranks (one per unique cell), in the same order as `summed`
    kept = torch.ones_like(ranks_s, dtype=torch.bool)
    kept[:-1] = ranks_s[1:] != ranks_s[:-1]
    unique_ranks = ranks_s[kept]                      # [U]
    canvas = x.new_zeros((B * ny * nx, Cc))
    canvas.index_copy_(0, unique_ranks, summed)       # unique → assignment (#76176-safe)
    bev = canvas.view(B, ny, nx, Cc).permute(0, 3, 1, 2).contiguous()  # [B,Cc,ny,nx]
    return bev


class DepthLSSTransform(nn.Module):
    """Depth-distribution LSS lift-splat (camera features → BEV).

    ``depthnet`` (Conv-GN-ReLU stack → ``Conv(D + Cc)``) produces per-pixel depth logits
    (softmax over ``D`` bins) and a context feature; the outer product lifts context into
    the frustum, which is splatted to BEV. All trainable (the AD camera→BEV projection is
    FL-learned, D1)."""

    def __init__(
        self,
        in_channels: int = 256,
        context_channels: int = 80,
        depth_bins: Tuple[float, float, float] = (1.0, 60.0, 1.0),  # (min, max, step)
        image_hw: Tuple[int, int] = (256, 704),
        feat_stride: int = 16,
        cfg: BEVConfig = BEVConfig(),
    ):
        super().__init__()
        self.cfg = cfg
        self.context_channels = int(context_channels)
        dmin, dmax, dstep = depth_bins
        ds = torch.arange(dmin, dmax, dstep, dtype=torch.float32)
        self.D = int(ds.numel())
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.feat_stride = int(feat_stride)
        fH = self.image_hw[0] // self.feat_stride
        fW = self.image_hw[1] // self.feat_stride
        self.fH, self.fW = fH, fW

        # depthnet: a small Conv-GN-ReLU then the (D + Cc) projection.
        mid = in_channels
        groups = min(32, mid)
        while mid % groups != 0:
            groups -= 1
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.GroupNorm(groups, mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, self.D + self.context_channels, 1),
        )

        # --- frustum (D, fH, fW, 3) in (u, v, d) input-pixel coords (LSS create_frustum) ---
        # feature-grid pixel centers sampled across the input image (align_corners=True
        # linspace — the LSS convention; sub-pixel offset is dominated by depth-bin error).
        H_in, W_in = self.image_hw
        xs = torch.linspace(0.0, W_in - 1.0, fW).view(1, 1, fW).expand(self.D, fH, fW)
        ys = torch.linspace(0.0, H_in - 1.0, fH).view(1, fH, 1).expand(self.D, fH, fW)
        dd = ds.view(self.D, 1, 1).expand(self.D, fH, fW)
        frustum = torch.stack((xs, ys, dd), dim=-1)  # (D, fH, fW, 3) = (u, v, d)
        self.register_buffer("frustum", frustum, persistent=False)
        # canonical per-cell geom index (n,d,i,j) added at forward (depends on N).
        self.register_buffer("depth_values", ds, persistent=False)
        self.record_debug = False
        self.last_projection_meta = {}

    def _img2lidar(self, lidar2img: torch.Tensor) -> torch.Tensor:
        """Inverse of (rescaled) ``lidar2img`` in float64 (calibration, not hot-path).

        Computed in float64 then cast to feature dtype: a fixed function of fixed
        calibration → identical every run; ARM portability re-verified at migration."""
        return torch.inverse(lidar2img.to(torch.float64)).to(lidar2img.dtype)

    def forward(self, feat: torch.Tensor, lidar2img: torch.Tensor, B: int, N: int) -> dict:
        """``feat`` ``[B*N, C, fH, fW]``; ``lidar2img`` ``[B, N, 4, 4]`` (rescaled).

        Returns ``{"bev": [B,Cc,ny,nx], "depth_prob": [B*N,D,fH,fW], "context":
        [B*N,Cc,fH,fW]}`` (the extra maps drive V2)."""
        cfg = self.cfg
        x = self.depthnet(feat)                                # [BN, D+Cc, fH, fW]
        depth_logits = x[:, : self.D]
        context = x[:, self.D :]
        depth_prob = depth_logits.softmax(dim=1)               # [BN, D, fH, fW]
        Cc = self.context_channels
        fH, fW, D = self.fH, self.fW, self.D
        P = D * fH * fW
        # D15: relaxed level (set by enforce_determinism) → fast atomic splat; strict → the
        # deterministic cumsum_trick (byte-identical to the reference).
        relaxed = not torch.backends.cudnn.deterministic

        # --- frustum → lidar (x,y,z) for every (b,n,d,i,j) [shared by both paths] ---
        u, v, d = self.frustum.unbind(-1)                      # each (D,fH,fW)
        ones = torch.ones_like(d)
        pts_img = torch.stack((u * d, v * d, d, ones), dim=-1).view(-1, 4)  # (P,4)
        img2lidar = self._img2lidar(lidar2img)                 # [B,N,4,4]
        pts_lidar = torch.einsum("bnij,pj->bnpi", img2lidar, pts_img)  # [B,N,P,4]
        xy = pts_lidar[..., :2]
        z = pts_lidar[..., 2]
        col, row = metric_to_grid(xy[..., 0], xy[..., 1], cfg.x_min, cfg.y_min, cfg.vx, cfg.vy)
        valid = in_grid_mask(col, row, cfg.nx, cfg.ny) & (z >= cfg.z_min) & (z < cfg.z_max)  # [B,N,P]
        b_idx = torch.arange(B, device=feat.device).view(B, 1, 1).expand(B, N, P)
        flat = flat_index(col, row, cfg.nx)
        ranks = b_idx * (cfg.nx * cfg.ny) + flat               # [B,N,P]
        m = valid.reshape(-1)
        ranks_pt = ranks.reshape(-1)[m]
        projection_meta = {}
        if self.record_debug:
            valid_counts = valid.detach().sum(dim=2)            # [B,N]
            per_camera = valid_counts.sum(dim=0)
            valid_total = int(valid_counts.sum().detach().cpu())
            total = int(B * N * P)
            denom_per_camera = max(int(B * P), 1)
            projection_meta = {
                "B": int(B),
                "N": int(N),
                "D": int(D),
                "fH": int(fH),
                "fW": int(fW),
                "frustum_points_per_camera": int(P),
                "frustum_points_total": total,
                "valid_points_total": valid_total,
                "valid_ratio": (float(valid_total) / float(total)) if total else 0.0,
                "valid_points_per_camera": [int(v) for v in per_camera.detach().cpu().tolist()],
                "valid_ratio_per_camera": [
                    float(v) / float(denom_per_camera) for v in per_camera.detach().cpu().tolist()
                ],
                "bev_grid_hw": [int(cfg.ny), int(cfg.nx)],
                "point_cloud_range": [
                    float(cfg.x_min),
                    float(cfg.y_min),
                    float(cfg.z_min),
                    float(cfg.x_max),
                    float(cfg.y_max),
                    float(cfg.z_max),
                ],
            }
            self.last_projection_meta = projection_meta

        if relaxed:
            # L3a+L3b (D15 relaxed): mask-then-lift (lift ONLY at valid points — the atomicAdd backward
            # of the shared-context gather is now allowed) + a single native scatter_add_ splat (no
            # argsort, no cumsum, no geom_id, no 1.28 GB materialization). Cell sums are the intended
            # values; per-cell float-add order is now non-deterministic (acceptable under D15).
            idx = m.nonzero(as_tuple=False).squeeze(1)
            bn = torch.div(idx, P, rounding_mode="floor")
            p = idx % P
            ij = p % (fH * fW)
            depth_vals = depth_prob.reshape(B * N, P)[bn, p]
            ctx_vals = context.reshape(B * N, Cc, fH * fW)[bn, :, ij]
            # AMP STABILITY: accumulate the BEV splat in fp32 (thousands of points sum into one cell).
            # The forward stays autocast; only this reduction is fp32 (cheap). Output is fp32;
            # downstream fusion re-casts under autocast.
            xf = (depth_vals.unsqueeze(1) * ctx_vals).float()  # [P_valid, Cc] fp32
            canvas = xf.new_zeros((B * cfg.ny * cfg.nx, Cc))
            canvas.scatter_add_(0, ranks_pt.unsqueeze(1).expand(-1, Cc), xf)  # D15-relaxed-ok
            bev = canvas.view(B, cfg.ny, cfg.nx, Cc).permute(0, 3, 1, 2).contiguous()
        else:
            # STRICT: the deterministic cumsum_trick splat — byte-identical to the reference.
            lifted = depth_prob.unsqueeze(1) * context.unsqueeze(2)  # [BN,Cc,D,fH,fW]
            n_idx = torch.arange(N, device=feat.device).view(1, N, 1)
            cell_id = torch.arange(P, device=feat.device).view(1, 1, -1)
            geom_id = (n_idx * P + cell_id).expand(B, N, P)
            feats_pt = lifted.view(B, N, Cc, P).permute(0, 1, 3, 2).contiguous()  # [B,N,P,Cc]
            x_pt = feats_pt.reshape(-1, Cc)[m]
            geom_pt = geom_id.reshape(-1)[m]
            bev = bev_splat(x_pt, ranks_pt, geom_pt, B, cfg.nx, cfg.ny)
        out = {"bev": bev, "depth_prob": depth_prob, "context": context}
        if projection_meta:
            out["projection_meta"] = projection_meta
        return out


def P_(D: int, fH: int, fW: int) -> int:
    return D * fH * fW
