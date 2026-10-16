"""
WPO (Wave Propagation Operator) modules for weakly supervised video anomaly detection.
Based on WaveFormer (AAAI 2026) and the research plan in CLIP WPO视觉分支研究方案.md.

Modules:
- SpatialWPO: 2D FFT-based wave propagation on spatial feature maps
- TemporalWPO: 1D FFT-based wave propagation along time dimension
- MambaBlock: SSM block for temporal modeling
- SpatialAggregation: CLS token or average pooling for spatial aggregation
- ClassificationHead: Frame-level anomaly score head

Correctness notes (2026-09 audit fixes):
1. ANGULAR FREQUENCY. torch.fft.*freq returns *cycles per sample* (0..0.5), while the
   wave equation needs the angular wavenumber k = 2*pi*f. The frequency grids are now
   multiplied by TWO_PI. Without this the reachable omega_d was capped at c*0.5 = 1.5 rad,
   which is below pi/2, so H(w) could never go negative and the operator degenerated into
   a near-constant gain (~1.96 with the residual). See WPO方法合理性审查报告.md P0-1.
2. OVERDAMPED REGIME. When 4c^2*w^2 < alpha^2/4 the system is overdamped and the closed
   form uses cosh/sinh, not cos/sin. Clamping omega_d to a tiny constant flattened 38% of
   the low-frequency band and killed dH/dc there (~200x smaller gradient). Now handled by
   `wpo_transfer()` with a torch.where branch. See report P0-3.
3. MAMBA BLOCK. MambAttentionBlock now applies pre-norm + residual + dropout around the
   SSM (the official mamba-ssm block owns no norm/residual), and the pure-PyTorch fallback
   is a real diagonal SSM with a parallel scan instead of a rank-d_state bottleneck gate.
   See report P0-2.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi

try:
    from mamba_ssm import Mamba as MambaSSM
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False


class LayerNorm(nn.LayerNorm):
    """LayerNorm that preserves dtype (same as model.py)."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def wpo_transfer(omega, alpha, c):
    """
    Frequency-domain transfer function of the damped wave equation
        u_tt + alpha * u_t = c^2 * Laplacian(u)
    evaluated at t = 1, for angular wavenumber `omega`.

        omega_d = sqrt(c^2 * omega^2 - alpha^2 / 4)

    Underdamped (c^2 w^2 > alpha^2/4):
        H = exp(-alpha/2) * [cos(omega_d) + alpha/(2 omega_d) sin(omega_d)]
    Overdamped (c^2 w^2 < alpha^2/4), with gamma = sqrt(alpha^2/4 - c^2 w^2):
        H = exp(-alpha/2) * [cosh(gamma) + alpha/(2 gamma) sinh(gamma)]
    Both branches agree at the critical point (limit = exp(-alpha/2) * (1 + alpha/2)).

    Args:
        omega: angular wavenumber tensor, any shape
        alpha: scalar tensor, damping coefficient (already clamped by the caller)
        c: scalar tensor, wave speed (already clamped by the caller)
    Returns:
        H with the same shape as `omega`
    """
    wd_sq = c ** 2 * omega ** 2 - alpha ** 2 / 4.0
    under = wd_sq > 0

    wd = torch.sqrt(torch.clamp(wd_sq, min=1e-12))
    gamma = torch.sqrt(torch.clamp(-wd_sq, min=1e-12))

    decay = torch.exp(-alpha / 2.0)
    h_under = decay * (torch.cos(wd) + alpha / (2.0 * wd) * torch.sin(wd))
    h_over = decay * (torch.cosh(gamma) + alpha / (2.0 * gamma) * torch.sinh(gamma))

    return torch.where(under, h_under, h_over)


class SpatialWPO(nn.Module):
    """
    Spatial WPO: 2D FFT -> frequency-domain WPO -> 2D iFFT.
    Applies the damped wave equation closed-form solution in the frequency domain.

    Input:  (B, T, H, W, C) patch tokens, WPO applied per-frame on (H, W) spatial dims.
    Frame-level features (B, T, C) are NOT supported: they carry no spatial structure,
    so reshaping them into a 2D map would impose a meaningless spatial layout.
    Use TemporalWPO for frame-level features.

    Flow: LayerNorm -> 2D FFT -> WPO -> 2D iFFT -> residual -> LayerNorm
    """

    def __init__(self, dim, alpha_init=0.5, c_init=1.0, spatial_h=16, spatial_w=32,
                 c_max=20.0):
        super().__init__()
        self.dim = dim
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.c_max = c_max
        # Learnable physical parameters (damped wave equation)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(c_init, dtype=torch.float32))
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(
                f"SpatialWPO requires 5D patch tokens (B, T, H, W, C), got {x.dim()}D. "
                f"Frame-level features (B, T, C) have no spatial structure; use TemporalWPO."
            )
        B, T, H, W, C = x.shape
        x_norm = self.norm1(x)
        x_flat = x_norm.reshape(B * T, H, W, C)
        x_out = self._apply_wpo_2d(x_flat, H, W, C)
        x_out = x_out.reshape(B, T, H, W, C)
        x = x + x_out
        return self.norm2(x)

    def _apply_wpo_2d(self, x, H, W, C):
        """Apply 2D WPO on (B, H, W, C) input, return (B, H, W, C)."""
        # 2D FFT: (B, H, W, C) -> (B, H, W//2+1, C) complex
        x_fft = torch.fft.rfft2(x, dim=(-3, -2))

        # Angular frequency grid (cycles/sample -> rad/sample)
        freq_h = TWO_PI * torch.fft.fftfreq(H, device=x.device)[:, None]   # (H, 1)
        freq_w = TWO_PI * torch.fft.rfftfreq(W, device=x.device)[None, :]  # (1, W//2+1)
        omega = torch.sqrt(freq_h ** 2 + freq_w ** 2)                      # (H, W//2+1)

        # WPO transfer function
        alpha = torch.clamp(self.alpha, min=0.01, max=5.0)
        c = torch.clamp(self.c, min=0.1, max=self.c_max)

        H_wpo = wpo_transfer(omega, alpha, c)          # (H, W//2+1)

        # Frequency-domain element-wise multiplication
        H_wpo = H_wpo[None, :, :, None]                # (1, H, W//2+1, 1)
        x_fft = x_fft * H_wpo

        # 2D iFFT
        x_out = torch.fft.irfft2(x_fft, s=(H, W), dim=(-3, -2))
        return x_out


class TemporalWPO(nn.Module):
    """
    Temporal WPO: 1D FFT along time dimension -> WPO -> 1D iFFT.
    Enhances inter-frame motion signals by selective frequency amplification.

    Input:  (B, T, C)
    Output: (B, T, C)

    Flow: LayerNorm -> replicate-tail -> 1D FFT -> WPO -> 1D iFFT -> residual -> LayerNorm

    `padding_mask` (B, T), True = padding. The FFT is a *global* (non-causal) operator, so
    a zero-padding tail injects a step discontinuity whose spectrum spreads over the whole
    valid region. Padded frames are therefore replaced by the last valid frame (replicate)
    before the transform.
    """

    def __init__(self, dim, alpha_init=0.3, c_init=0.8, c_max=20.0):
        super().__init__()
        self.dim = dim
        self.c_max = c_max
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(c_init, dtype=torch.float32))
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

    @staticmethod
    def _replicate_tail(x, padding_mask):
        """Replace padded frames with the last valid frame. x: (B,T,C), mask: (B,T) bool."""
        B, T, C = x.shape
        lengths = (~padding_mask).sum(dim=1).clamp(min=1)        # (B,)
        idx = torch.arange(T, device=x.device)[None, :].expand(B, T)
        last = (lengths - 1)[:, None]
        safe = torch.where(idx < lengths[:, None], idx, last)
        return x.gather(1, safe.unsqueeze(-1).expand(-1, -1, C))

    def forward(self, x, padding_mask=None):
        # x: (B, T, C)
        B, T, C = x.shape

        x_norm = self.norm1(x)

        if padding_mask is not None:
            padding_mask = padding_mask.to(dtype=torch.bool, device=x.device)
            x_norm = self._replicate_tail(x_norm, padding_mask)

        # 1D FFT along time: (B, T, C) -> (B, T//2+1, C) complex
        x_fft = torch.fft.rfft(x_norm, dim=1)

        # Angular temporal frequency (cycles/frame -> rad/frame)
        omega = TWO_PI * torch.fft.rfftfreq(T, device=x.device)   # (T//2+1,)

        # WPO transfer function
        alpha = torch.clamp(self.alpha, min=0.01, max=2.0)
        c = torch.clamp(self.c, min=0.1, max=self.c_max)

        H_wpo = wpo_transfer(omega, alpha, c)          # (T//2+1,)

        # Frequency-domain multiplication
        H_wpo = H_wpo[None, :, None]                   # (1, T//2+1, 1)
        x_fft = x_fft * H_wpo

        # 1D iFFT
        x_out = torch.fft.irfft(x_fft, n=T, dim=1)     # (B, T, C)

        x = x + x_out
        return self.norm2(x)


class _SimpleMamba(nn.Module):
    """
    Pure-PyTorch fallback SSM block, used only when the official mamba-ssm library is
    unavailable. It is a *diagonal* (per-channel) linear state-space model

        h_t = a_t * h_{t-1} + u_t,      a_t = exp(-softplus(W_a x_t))  in (0, 1]
        y_t = norm(h_t) * silu(gate_t)

    evaluated with a numerically stable parallel scan (log-cumsum-exp), so it is depth
    O(log T) instead of a Python loop. This is an S4D/LRU-style diagonal SSM with
    input-dependent decay -- it has real cross-timestep recurrence, but it is NOT
    selective Mamba (no d_state expansion, no hardware-aware scan). Install `mamba-ssm`
    for the real thing; `HAS_MAMBA_SSM` reports which one is active.

    Input:  (B, T, C) -> (B, T, C). No internal norm/residual: the enclosing
    MambAttentionBlock owns those.
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_inner = dim * expand

        self.in_proj = nn.Linear(dim, self.d_inner * 2)
        self.out_proj = nn.Linear(self.d_inner, dim)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.act = nn.SiLU()

        # Input-dependent diagonal state transition and input projection
        self.decay_proj = nn.Linear(self.d_inner, self.d_inner)
        self.input_proj = nn.Linear(self.d_inner, self.d_inner)
        self.h_norm = nn.LayerNorm(self.d_inner)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape

        x1, x2 = self.in_proj(x).chunk(2, dim=-1)          # each (B, T, d_inner)

        x1 = self.conv1d(x1.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x1 = self.act(x1)
        u = self.input_proj(x1)                            # (B, T, d_inner)

        # d >= 0 is the per-step "decay distance"; D is its cumulative sum, non-decreasing.
        # Clamped so the cumulative sum cannot overflow on long sequences.
        d = torch.clamp(F.softplus(self.decay_proj(x1)), min=0.0, max=20.0)
        D = torch.cumsum(d, dim=1)                         # (B, T, d_inner), >= 0

        # Parallel scan of h_t = a_t h_{t-1} + u_t with a_t = exp(-d_t):
        #   h_t = sum_{s<=t} exp(-(D_t - D_s)) * u_s
        # Done in log space with sign separation so nothing overflows:
        #   h_t = exp(L+_t - D_t) - exp(L-_t - D_t)
        m = torch.log(u.abs() + 1e-12) + D
        neg_inf = torch.finfo(m.dtype).min
        m_pos = torch.where(u > 0, m, torch.full_like(m, neg_inf))
        m_neg = torch.where(u < 0, m, torch.full_like(m, neg_inf))

        lp = torch.logcumsumexp(m_pos, dim=1) - D
        ln = torch.logcumsumexp(m_neg, dim=1) - D
        h = torch.exp(lp) - torch.exp(ln)                  # (B, T, d_inner)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.h_norm(h)

        y = h * self.act(x2)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """
    Mamba SSM block for temporal modeling.
    Uses the official mamba-ssm library when available, with a pure-PyTorch diagonal-SSM
    fallback for environments without the compiled extension.

    Input:  (B, T, C)
    Output: (B, T, C)
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.using_official = HAS_MAMBA_SSM
        if HAS_MAMBA_SSM:
            self.mamba = MambaSSM(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba = _SimpleMamba(dim, d_state, d_conv, expand)

    def forward(self, x):
        return self.mamba(x)


class MambAttentionBlock(nn.Module):
    """
    MHA -> Mamba -> FFN block.

    The SSM sub-layer is wrapped with pre-norm + residual + dropout, matching the research
    plan and the contract of both `mamba_ssm.Mamba` and `_SimpleMamba` (neither owns a
    norm/residual of its own). Without the residual the SSM output replaced the whole
    stream, which collapsed the per-frame representations onto a single direction.

    `attn_mask` is the additive (T, T) window mask from the original VadCLIP; pass None
    for full attention. `padding_mask` (B, T) marks padded frames (True = padding).

    Input:  (B, T, C)
    Output: (B, T, C)
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_heads=8, dropout=0.1,
                 attn_mask=None):
        super().__init__()

        self.norm1 = LayerNorm(dim)

        # Lightweight multi-head attention (generalization plugin)
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

        # Mamba core
        self.norm2 = LayerNorm(dim)
        self.mamba_block = MambaBlock(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # FFN
        self.norm3 = LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        # Stored as a BOOL mask (True = masked out) so it matches the dtype of
        # key_padding_mask; mixing a float attn_mask with a bool key_padding_mask is
        # deprecated in nn.MultiheadAttention.
        if attn_mask is not None:
            bool_mask = (attn_mask == float('-inf')) if attn_mask.is_floating_point() \
                else attn_mask.to(torch.bool)
            self.register_buffer('attn_mask', bool_mask.clone().detach())
        else:
            self.register_buffer('attn_mask', None)

    def forward(self, x, padding_mask=None, attn_mask=None):
        # x: (B, T, D). `attn_mask` overrides the registered window mask when given
        # (the caller may pass an already-merged (B*heads, T, T) mask).
        if padding_mask is not None:
            padding_mask = padding_mask.to(dtype=torch.bool, device=x.device)

        if attn_mask is None:
            attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(device=x.device)

        # MHA
        h = self.norm1(x)
        h, _ = self.mha(h, h, h, key_padding_mask=padding_mask, attn_mask=attn_mask)
        x = x + self.dropout(h)

        # Mamba (pre-norm + residual). The SSM is causal, so a tail of padded frames
        # cannot leak backwards; the pad positions of the delta are still zeroed.
        h = self.mamba_block(self.norm2(x))
        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = x + h

        # FFN
        h = self.ffn(self.norm3(x))
        x = x + h

        return x


class SpatialAggregation(nn.Module):
    """
    Spatial aggregation: CLS token cross-attention or average pooling.
    Aggregates (B, T, H, W, C) patch features into (B, T, C) frame vectors.

    mode: 'cls' (cross-attention with learnable CLS token) or 'avg' (mean pooling)
    """

    def __init__(self, dim, num_patches=576, mode='cls', num_heads=8):
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        elif mode == 'avg':
            pass
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")

    def forward(self, x):
        # x: (B, T, H, W, C) -> (B, T, C)
        B, T, H, W, C = x.shape
        x = x.reshape(B * T, H * W, C)  # (B*T, H*W, C)

        if self.mode == 'cls':
            cls = self.cls_token.expand(B * T, -1, -1)  # (B*T, 1, C)
            cls, _ = self.cross_attn(cls, x, x)  # (B*T, 1, C)
            return cls.reshape(B, T, C)
        else:  # 'avg'
            return x.mean(dim=1).reshape(B, T, C)


class ClassificationHead(nn.Module):
    """
    Frame-level anomaly score head.
    Input:  (B, T, C)
    Output: (B, T)
    """

    def __init__(self, dim, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (B, T, C)
        logits = self.mlp(x)  # (B, T, 1)
        return logits.squeeze(-1)  # (B, T)
