"""In-tree SDPA rewrite of torchvision Swin-T ``ShiftedWindowAttention`` (MCR Phase-2, D16 envelope).

torchvision's ``shifted_window_attention`` does the attention core as **manual fp32-ish math**
(``(q*scale) @ kᵀ + rel_pos_bias [+ shift_mask] → softmax → @ v``), which leaves ``aten::bmm`` +
softmax as separate kernels (verified in the A100 kernel profile — ~188 ms ``bmm`` even after
``torch.compile``). This module swaps ONLY that core for ``F.scaled_dot_product_attention`` while keeping
torchvision's exact padding / window-partition / cyclic-shift / unpad — so the output is numerically
equivalent (validated within tolerance) but the attention runs as one fused, tensor-core kernel.

**Mapping (exact):** torchvision computes ``softmax((q·scale)@kᵀ + M) @ v`` with
``scale = head_dim**-0.5`` and ``M = rel_pos_bias [+ shift_mask]``. SDPA computes
``softmax((q@kᵀ)·scale + attn_mask) @ v`` — identical with ``scale=head_dim**-0.5`` and
``attn_mask = M`` (so q is NOT pre-scaled here; SDPA applies the scale).

**Backend:** Swin's additive float ``attn_mask`` (rel-pos-bias) is rejected by the FLASH backend (which
takes only none/causal/padding) → SDPA routes to EFFICIENT (≈1.3–2×, the D16-addendum expectation). Under
the fp32 dev-regression regime (``cudnn.deterministic``) the call is pinned to the deterministic **MATH**
backend so the strict tool stays byte-reproducible. Swin-T uses ``attention_dropout=dropout=0`` → no RNG.
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def _sdpa_swa_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for ``ShiftedWindowAttention.forward`` using SDPA for the attention core.

    Faithful port of ``torchvision.models.swin_transformer.shifted_window_attention`` (logit_scale=None
    / Swin-V1 branch); only the ``q@kᵀ + bias + softmax + @v`` core is replaced by SDPA."""
    relative_position_bias = self.get_relative_position_bias()        # [1, nH, N, N] (torchvision adds dim0)
    qkv_weight, qkv_bias = self.qkv.weight, self.qkv.bias
    proj_weight, proj_bias = self.proj.weight, self.proj.bias
    window_size = self.window_size
    num_heads = self.num_heads
    shift_size = list(self.shift_size)
    attention_dropout = self.attention_dropout
    dropout = self.dropout
    training = self.training

    B, H, W, C = x.shape
    # pad feature maps to multiples of window size (identical to torchvision)
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    xw = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)
    N = xw.size(1)
    head_dim = C // num_heads

    qkv = F.linear(xw, qkv_weight, qkv_bias)
    qkv = qkv.reshape(xw.size(0), N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]                                  # each [B*nW, nH, N, head_dim]

    # additive attention mask = rel_pos_bias [+ shift mask] (the EXACT thing torchvision adds pre-softmax)
    attn_mask = relative_position_bias                              # [1, nH, N, N] → broadcasts over B*nW
    if sum(shift_size) > 0:
        sm = xw.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                sm[h[0]:h[1], w[0]:w[1]] = count
                count += 1
        sm = sm.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        sm = sm.permute(0, 2, 1, 3).reshape(num_windows, N)
        sm = sm.unsqueeze(1) - sm.unsqueeze(2)                       # [nW, N, N]
        sm = sm.masked_fill(sm != 0, float(-100.0)).masked_fill(sm == 0, float(0.0))
        sm = sm.unsqueeze(0).expand(B, num_windows, N, N).reshape(B * num_windows, 1, N, N)
        attn_mask = attn_mask + sm                                  # [B*nW, nH, N, N]
    attn_mask = attn_mask.to(q.dtype)

    p = attention_dropout if training else 0.0
    if torch.backends.cudnn.deterministic:                          # fp32 dev-regression → deterministic MATH
        with sdpa_kernel(SDPBackend.MATH):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p,
                                                 scale=head_dim ** -0.5)
    else:                                                           # fp16 relaxed path -> EFFICIENT (float-mask)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p,
                                             scale=head_dim ** -0.5)

    out = out.transpose(1, 2).reshape(xw.size(0), N, C)
    out = F.linear(out, proj_weight, proj_bias)
    out = F.dropout(out, p=dropout, training=training)

    # reverse windows
    out = out.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)
    if sum(shift_size) > 0:
        out = torch.roll(out, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
    out = out[:, :H, :W, :].contiguous()
    return out


def apply_sdpa_to_swin(module: torch.nn.Module) -> int:
    """Patch every torchvision ``ShiftedWindowAttention`` under ``module`` to use the SDPA core.

    Returns the number of attention modules patched. Idempotent-ish (re-patching just re-binds the same
    method). Matches by class name so it works whether or not torchvision is importable by symbol here."""
    n = 0
    for m in module.modules():
        if type(m).__name__ in ("ShiftedWindowAttention", "ShiftedWindowAttentionV2"):
            if type(m).__name__ == "ShiftedWindowAttentionV2":
                # V2 uses cosine attention (logit_scale) — not Swin-T; refuse rather than mis-handle it.
                raise NotImplementedError("SDPA rewrite supports Swin-V1 ShiftedWindowAttention only")
            m.forward = types.MethodType(_sdpa_swa_forward, m)
            n += 1
    return n
