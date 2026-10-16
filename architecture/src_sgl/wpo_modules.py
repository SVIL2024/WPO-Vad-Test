"""
Identity-anchored TemporalWPO for the single-branch model (src_sgl).

Lesson from the src/ audit + the 2026-09-12 training logs: the learnable wave
parameters (alpha, c) receive ~zero net task gradient — their entire observed
movement over 4 epochs equaled the weight-decay pull — so a plain learnable WPO
behaves as a FIXED, randomly-initialized filter injected into the feature stream.
That cost ~2 AP once the corrected (non-degenerate, strongly oscillatory)
operator replaced the old near-constant-gain one.

This version fixes the failure mode instead of repeating it:
- identity anchor: out = x + g * (WPO_core(norm(x)) - norm(x)), with the gate
  g learnable and initialized to 0. Training starts EXACTLY at the baseline and
  admits the wave filter only where the loss wants it. d(out)/d(alpha, c)
  scales with g, so the wave parameters unlock progressively instead of being
  noise-dominated from step one.
- alpha stays clamped to [0.01, 2.0], c to [0.1, c_max] (same bounds as src/).
- replicate-tail padding: padded frames are replaced by the last valid frame
  before the global FFT (zero-padding leaks a step discontinuity spectrum).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi


class LayerNorm(nn.LayerNorm):
    """LayerNorm that preserves dtype (same convention as the rest of the repo)."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def wpo_transfer(omega, alpha, c):
    """
    Frequency-domain transfer function of the damped wave equation
        u_tt + alpha * u_t = c^2 * Laplacian(u)
    evaluated at t = 1, for angular wavenumber `omega` (identical to src/).

        omega_d = sqrt(c^2 * omega^2 - alpha^2 / 4)

    Underdamped (c^2 w^2 > alpha^2/4):
        H = exp(-alpha/2) * [cos(omega_d) + alpha/(2 omega_d) sin(omega_d)]
    Overdamped, with gamma = sqrt(alpha^2/4 - c^2 w^2):
        H = exp(-alpha/2) * [cosh(gamma) + alpha/(2 gamma) sinh(gamma)]
    """
    wd_sq = c ** 2 * omega ** 2 - alpha ** 2 / 4.0
    under = wd_sq > 0

    wd = torch.sqrt(torch.clamp(wd_sq, min=1e-12))
    gamma = torch.sqrt(torch.clamp(-wd_sq, min=1e-12))

    decay = torch.exp(-alpha / 2.0)
    h_under = decay * (torch.cos(wd) + alpha / (2.0 * wd) * torch.sin(wd))
    h_over = decay * (torch.cosh(gamma) + alpha / (2.0 * gamma) * torch.sinh(gamma))

    return torch.where(under, h_under, h_over)


class FixedBandFilter(nn.Module):
    """
    Parameter-free band view generator — the control arm for the wave operator.

    Produces the SAME identity-anchored admixture shape as TemporalWPO,
    out = (1 - g0) * x + g0 * band(x), but with a fixed canonical filter:
    kind='low' is a moving average (kernel K), kind='high' its complement
    (x − moving average). The admixture g0 is a fixed buffer (default 0.3,
    the converged gate level observed in the 2026-09-12 Plan B runs), so a
    fixed-mode band differs from a learnable band ONLY in "learnable wave
    transfer function + learned gate" vs "fixed canonical filter + fixed gate".
    Zero learnable parameters: nothing in this module trains.
    """

    def __init__(self, dim, kernel, kind='low', anchor=0.3):
        super().__init__()
        assert kind in ('low', 'high')
        assert kernel >= 2
        self.dim = dim
        self.kernel = int(kernel)
        self.kind = kind
        self.register_buffer('anchor', torch.tensor(float(anchor)))

    def forward(self, x, padding_mask=None):
        xt = x
        if padding_mask is not None:
            xt = TemporalWPO._replicate_tail(
                x, padding_mask.to(dtype=torch.bool, device=x.device))
        xp = xt.permute(0, 2, 1)
        pad_l = self.kernel // 2
        ma = F.avg_pool1d(F.pad(xp, (pad_l, self.kernel - 1 - pad_l), mode='replicate'),
                          kernel_size=self.kernel, stride=1)
        band = ma if self.kind == 'low' else xt - ma
        band = band.to(x.dtype)
        return x + self.anchor * (band - xt)


class TemporalWPO(nn.Module):
    """
    Identity-anchored temporal WPO: (B, T, C) -> (B, T, C).

    With the gate g at 0 the module is the exact identity, so a block that wraps
    `wpo(ln_1(x))` ahead of Mamba reduces bit-for-bit to the baseline block. As g
    grows, the damped-wave filter is admixed and alpha/c start to learn.

    `internal_norm=True` applies LayerNorm inside (standalone use); when the
    module sits right after a block's own LayerNorm use internal_norm=False.

    `padding_mask` (B, T), True = padding — replicated to the last valid frame
    before the FFT.
    """

    def __init__(self, dim, alpha_init=0.3, c_init=0.8, c_max=20.0, internal_norm=False,
                 adaptive_params=False, motion_cond=False):
        super().__init__()
        self.dim = dim
        self.c_max = c_max
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.c = nn.Parameter(torch.tensor(float(c_init)))
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm = LayerNorm(dim) if internal_norm else nn.Identity()
        # Plan ①: video-level conditioned wave parameters (FFTNet-style adaptive
        # spectral filtering at minimal dimensionality). A zero-initialised
        # hypernetwork predicts per-video (d_alpha, d_c) offsets from the pooled
        # video context, added on top of the global learnable (alpha_0, c_0).
        # Zero init => alpha/c exactly the global values at start, so the module
        # is bit-exact identical to the non-adaptive version until the offsets move.
        self.adaptive_params = adaptive_params
        # Plan ②: motion-aware conditioning — the context doubles up with a
        # temporal-difference descriptor (|x_t − x_{t−1}| pooled), the natural
        # motion measure in feature space, so the wave speed/damping condition on
        # HOW MUCH a video moves instead of only what it looks like.
        self.motion_cond = motion_cond
        if adaptive_params:
            cond_in = dim * 2 if motion_cond else dim
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_in, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 2),
            )
            nn.init.zeros_(self.cond_mlp[-1].weight)
            nn.init.zeros_(self.cond_mlp[-1].bias)

    def _video_context(self, x, padding_mask):
        """Per-video conditioning: mean appearance (and pooled |Δx| motion when
        motion_cond is on). Honours the padding mask."""
        if padding_mask is None:
            app = x.mean(dim=1)
            if not (self.adaptive_params and self.motion_cond):
                return app
            dx = (x[:, 1:, :] - x[:, :-1, :]).abs().mean(dim=1)
            return torch.cat([app, dx], dim=-1)
        m = (~padding_mask).float().unsqueeze(-1)
        app = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        if not (self.adaptive_params and self.motion_cond):
            return app
        d = (x[:, 1:, :] - x[:, :-1, :]).abs()
        mm = (m[:, 1:, :] * m[:, :-1, :])
        dx = (d * mm).sum(dim=1) / mm.sum(dim=1).clamp(min=1)
        return torch.cat([app, dx], dim=-1)

    @staticmethod
    def _replicate_tail(x, padding_mask):
        """Replace padded frames with the last valid frame. x: (B,T,C), mask: (B,T) bool."""
        B, T, C = x.shape
        lengths = (~padding_mask).sum(dim=1).clamp(min=1)
        idx = torch.arange(T, device=x.device)[None, :].expand(B, T)
        last = (lengths - 1)[:, None]
        safe = torch.where(idx < lengths[:, None], idx, last)
        return x.gather(1, safe.unsqueeze(-1).expand(-1, -1, C))

    def forward(self, x, padding_mask=None):
        B, T, C = x.shape
        xn = self.norm(x)
        if padding_mask is not None:
            padding_mask = padding_mask.to(dtype=torch.bool, device=x.device)
            xn = self._replicate_tail(xn, padding_mask)

        omega = TWO_PI * torch.fft.rfftfreq(T, device=x.device)
        if self.adaptive_params:
            # per-video (alpha_i, c_i): global base + zero-init predicted offsets
            ctx = self._video_context(x, padding_mask)              # (B, C)
            da, dc = self.cond_mlp(ctx).split(1, dim=-1)            # (B, 1) each
            alpha = torch.clamp(self.alpha + da.squeeze(-1), min=0.01, max=2.0)   # (B,)
            c = torch.clamp(self.c + dc.squeeze(-1), min=0.1, max=self.c_max)     # (B,)
            H = wpo_transfer(omega[None, :], alpha[:, None], c[:, None])          # (B, F)
            y = torch.fft.irfft(
                torch.fft.rfft(xn, dim=1) * H.unsqueeze(-1), n=T, dim=1)
        else:
            alpha = torch.clamp(self.alpha, min=0.01, max=2.0)
            c = torch.clamp(self.c, min=0.1, max=self.c_max)
            H = wpo_transfer(omega, alpha, c)
            y = torch.fft.irfft(torch.fft.rfft(xn, dim=1) * H[None, :, None], n=T, dim=1)
        return x + self.gate * (y - xn)
