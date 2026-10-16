# -*- coding: utf-8 -*-
"""Damped-wave operator on the CLIP FEATURE FIELD (the "direction 1+2" module).

Why this module exists
----------------------
The previous wave prior acted on the scalar per-frame anomaly score `logits1`.
It measured exactly zero - both as a training-time penalty and as a test-time
projection - and the reason is now measured, not guessed:

    on the scalar score   |A s| is only 0.33% of |s|   (447 XD videos)
    on the feature field  |A u| is 68.97% of |u|      (60 XD videos, D=512)

The scalar trajectory is already (numerically) a free solution
s = A + B*exp(-alpha t), so a wave constraint on it is satisfied before
training starts and can never produce a gradient. The feature field is not:
it is full of structure, and the channel axis varies even more than time
(mean |u[d+1]-u[d]| = 0.404 vs mean |u[t+1]-u[t]| = 0.128), which is exactly
what the missing c^2 * Laplacian term needs.

Equation
--------
    u_tt + alpha * u_t - c^2 * Delta_c u = f(t)

`Delta_c u` is a Laplacian across a LEARNED set of K "wave modes" (channels
have no natural order, so the spatial axis itself is learned). f is the
driving source; sparsity of f is the prior.

Two failure modes are explicitly designed against:

1. LTI collapse (proven earlier: a non-adaptive linear operator plus a linear
   fusion collapses to a single linear map, adding zero expressivity).
   -> `alpha` and `c` are PREDICTED FROM THE INPUT, so the operator is not
      linear-time-invariant.
2. The fused evidence being linear (same collapse).
   -> what the model consumes is the per-frame energy ||f[t]||_1, which is a
      NORM and therefore nonlinear no matter what.

Usage
-----
    wf = WaveField(dim=512, n_modes=16)
    out = wf(u)                      # u: (B, T, D)
    out['energy']                    # (B, T)  per-frame wave-source energy
    loss = out['energy'].mean()      # the prior: the field should satisfy the PDE
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveField(nn.Module):
    """Damped wave operator with input-predicted coefficients.

    Args:
        dim: feature dimension D (512 for CLIP ViT-B/16).
        n_modes: number of learned wave modes K. This is the "space" the
            Laplacian acts on. Small on purpose (8-32): it is a physical
            coordinate, not another feature map.
        hidden: width of the coefficient predictor.
        alpha_range, c_range: (lo, hi) ranges for the predicted coefficients,
            so they stay in a physically sane regime no matter what the MLP
            does early in training.
    """

    def __init__(self, dim, n_modes=16, hidden=64,
                 alpha_range=(0.05, 2.0), c_range=(0.05, 2.0)):
        super().__init__()
        self.dim = dim
        self.n_modes = n_modes
        self.alpha_range = alpha_range
        self.c_range = c_range

        # Projection to wave modes: this defines the "spatial" axis the
        # Laplacian lives on. Learned, so the axis is task-adapted.
        self.mix = nn.Linear(dim, n_modes, bias=False)
        nn.init.orthogonal_(self.mix.weight)

        # Coefficients predicted from the clip's global descriptor -> the
        # operator is input-dependent (breaks LTI).
        #
        # NOTE (bug found by _smoke_wave_field.py): zero-initialising the last
        # layer makes the head output a CONSTANT, so alpha/c would not depend
        # on the input at all and the LTI-collapse guard would be decorative.
        # The residual form below starts exactly at mid-range while keeping a
        # live, input-dependent path from step 0.
        self.gnorm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        # tanh(0) = 0 -> raw starts at 0.5 (mid-range); the 0.5 std is what
        # makes different clips actually get different coefficients.
        nn.init.normal_(self.head[-1].weight, std=0.5)
        nn.init.zeros_(self.head[-1].bias)

    def _coeffs(self, u):
        """(B, T, D) -> alpha, c of shape (B, 1, 1)."""
        g = self.gnorm(u.mean(dim=1))           # (B, D) global descriptor
        raw = 0.5 + 0.5 * torch.tanh(self.head(g))   # (B, 2) in (0, 1)
        lo, hi = self.alpha_range
        alpha = lo + (hi - lo) * raw[:, 0]
        lo, hi = self.c_range
        c = lo + (hi - lo) * raw[:, 1]
        return alpha.view(-1, 1, 1), c.view(-1, 1, 1)

    @staticmethod
    def _temporal(z):
        """Second and first temporal differences.

        MUST match wave_loss.py:wave_source_loss exactly, otherwise the two
        "wave" terms in the paper would be solving different equations:

            d2[i] = z[i+2] - 2 z[i+1] + z[i]
            d1[i] = z[i+1] - z[i]          <-- forward difference, NOT centred

        Returns tensors of length T-2 whose index i is anchored at frame i.
        """
        d2 = z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2]
        d1 = z[:, 1:-1] - z[:, :-2]
        return d2, d1

    @staticmethod
    def _laplacian_modes(z):
        """Laplacian along the learned mode axis, zero-flux at the ends."""
        # pad mode axis by 1 on each side, replicating the edge -> the ends
        # see a one-sided difference instead of a phantom zero.
        zp = F.pad(z, (1, 1), mode="replicate")
        return zp[:, :, :-2] - 2 * zp[:, :, 1:-1] + zp[:, :, 2:]

    @staticmethod
    def _unit_rms(u, lengths=None):
        """Rescale each video to unit RMS over its VALID frames.

        Padding frames must not participate, otherwise a clip that is 90%
        padding gets a different constant than an identical full-length clip.
        """
        B, T, D = u.shape
        if lengths is None:
            m = torch.ones((B, T), dtype=u.dtype, device=u.device)
        else:
            m = (torch.arange(T, device=u.device).unsqueeze(0)
                 < lengths.to(u.device).clamp(max=T).unsqueeze(1)).to(u.dtype)
        denom = (m.sum(1, keepdim=True) * D).clamp(min=1.0)      # (B, 1)
        # float32 accumulator: u is fp32 here, but u**2 can still be large
        # enough that the sum is worth doing in the wider type.
        sq = (u.float() ** 2 * m.unsqueeze(-1)).sum(dim=(1, 2), keepdim=True)
        rms = (sq / denom.unsqueeze(-1)).clamp(min=1e-12).sqrt()  # (B, 1, 1)
        return (u.float() / rms).to(u.dtype)

    def forward(self, u, lengths=None):
        """u: (B, T, D). lengths: optional (B,) valid frame counts.

        Returns dict with:
            source : (B, T-2, K) the driving source f
            energy : (B, T)     per-frame ||f[t]||_1  (nonlinear -> fusion-safe)
            alpha, c : (B,)     the predicted coefficients (for logging)
        """
        if u.dim() != 3:
            raise ValueError(f"WaveField expects (B, T, D), got {tuple(u.shape)}")
        B, T, D = u.shape
        if D != self.dim:
            raise ValueError(f"feature dim {D} != configured dim {self.dim}")

        # ---- dtype guard -------------------------------------------------
        # The trunk usually runs under autocast: activations are fp16/bf16 while
        # the parameters here stay fp32. F.linear rejects mismatched dtypes
        # ("expected mat1 and mat2 to have the same dtype, but got: c10::Half !=
        # float"), which killed the first E1 run a minute into training.
        # Do the wave math in the PARAMETER dtype, then hand back the caller's
        # dtype so the downstream fusion (logits1 + gain*energy) still type-checks.
        out_dtype = u.dtype
        calc_dtype = self.mix.weight.dtype
        if out_dtype != calc_dtype:
            u = u.to(calc_dtype)

        # ---- per-video RMS normalisation ----------------------------------
        # Numerically REQUIRED, and mathematically free. The trunk is protected
        # from arbitrarily-scaled inputs by the LayerNorm at its mouth; this
        # branch is not, so a clip whose features are stored at a large scale
        # (the XD .npy files are a mix of float16 and float32 with very
        # different levels) sends the second difference d2 straight past
        # fp32's range. That is what produced
        #     [WaveField] WARNING: non-finite wave energy (24576/24576)
        # i.e. a WHOLE (96, 256) training batch non-finite, on E1's first step.
        #
        # Why it cannot change any number the model sees: f is linear in u, so
        # scaling u by a per-video constant s scales `energy` by s; the caller
        # then z-scores `energy` per video, which is invariant to s. And
        # _coeffs() feeds LayerNorm, which is scale-invariant too. So this is a
        # pure conditioning fix - see smoke group 9.
        u = self._unit_rms(u, lengths)

        z = self.mix(u)                       # (B, T, K)
        alpha, c = self._coeffs(u)
        d2, d1 = self._temporal(z)
        lap = self._laplacian_modes(z)[:, 1:-1] if T > 2 else z[:, :0]
        f = d2 + alpha * d1 - (c ** 2) * lap  # (B, T-2, K)

        energy = f.abs().mean(dim=-1)         # (B, T-2) - a norm => nonlinear
        # Re-align to T: the source is undefined at the two boundary frames.
        # (F.pad's 'replicate' needs >=3-D input, and energy is 2-D, so repeat
        # the edge frames by hand.)
        if T > 2:
            energy = torch.cat([energy[:, :1], energy, energy[:, -1:]], dim=1)
        else:
            energy = u.new_zeros(B, T)

        if lengths is not None:
            L = T
            m = torch.arange(L, device=u.device).unsqueeze(0) < lengths.to(u.device).clamp(max=L).unsqueeze(1)
            energy = energy * m.to(energy.dtype)

        # NOTE (2026-09-16): the outputs used to be cast BACK to the caller's
        # dtype here. That silently broke training: the pre-extracted features
        # are float16, `energy` legitimately reaches ~5e3 once the (unbounded)
        # mode basis trains up, and fp16 tops out at 65504 - so the whole batch
        # overflowed to Inf, the per-video mean became Inf, and every z-score
        # turned into NaN. It only ever happened on fp16 batches, which is why
        # it looked data-dependent.
        # Everything now stays in the compute dtype. The caller
        # (CLIPVAD._wave_evidence) works in fp32 and pins the fusion to
        # logits1's dtype, so nothing downstream needs the downcast.
        return {
            "source": f,
            "energy": energy,
            "alpha": alpha.reshape(-1),
            "c": c.reshape(-1),
        }


def wave_field_loss(u, lengths=None, n_modes=16, module=None):
    """Convenience: mean wave-source energy for a feature batch.

    Pass `module=` to reuse a WaveField registered in the model (so the mode
    basis is shared and actually trained). Without it a throwaway module is
    used, which is only useful for probing.
    """
    wf = module if module is not None else WaveField(u.shape[-1], n_modes=n_modes).to(u.device)
    out = wf(u, lengths=lengths)
    e = out["energy"]
    if lengths is not None:
        w = (torch.arange(e.shape[1], device=u.device).unsqueeze(0)
             < lengths.to(u.device).clamp(max=e.shape[1]).unsqueeze(1)).to(e.dtype)
        return (e * w).sum() / w.sum().clamp(min=1)
    return e.mean()
