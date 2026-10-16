"""
Driven damped-wave resonator bank for the src_sgl visual branch.

WHY THIS FILE EXISTS
--------------------
`wpo_modules.TemporalWPO` evaluates the *initial-value* (free-decay) solution of

    u_tt + alpha * u_t = c^2 * u_ss

at evolution time t = 1. That solution carries the initial condition u(0) = x,
u_t(0) = 0, and one can show analytically that it forces

    H_free(0) == 1     for every (alpha, c)          <-- provable, see report

so the operator provably CANNOT attenuate the DC component of a clip. Whatever
(alpha, c) the optimiser picks, the static / slowly-varying background of a
normal surveillance clip passes through unchanged, and what is left is a
zero-phase symmetric smoothing kernel -- functionally the same thing that
bidirectional Mamba and multi-scale average pooling already do. That is the
mathematical reason WPO measured ~0 AP on the 0.8271 strong baseline.

THE FIX: USE THE FORCED (STEADY-STATE) RESPONSE
----------------------------------------------
Drive the same medium with the observed sequence at evolution frequency Omega:

    u_tt + alpha * u_t - c^2 * u_ss = f(s) e^{i Omega t}

Looking for u = U(s) e^{i Omega t} and transforming over the frame axis s
(wavenumber w) gives

    (-Omega^2 + i alpha Omega + c^2 w^2) U(w) = X(w)
    H(w) = 1 / (c^2 w^2 - Omega^2 + i alpha Omega)

Two free physical knobs remain after normalising the peak gain to 1:
  * w* = resonant wavenumber  -> period P = 2*pi/w* (frames)
  * lambda = damping          -> quality factor Q = w*^2 / lambda

    Hn(w) = lambda / (w^2 - w*^2 + i lambda),     lambda = w*^2 / Q

    |Hn(w*)| = 1                      unit peak gain
    |Hn(0)|  = 1 / sqrt(1 + Q^2)      <-- DC suppression, learnable, was 1.0
    |Hn(w)|  <= 1 everywhere          cannot amplify noise
    |Hn(w)|  ~ lambda / w^2  (w >> w*)  high-frequency roll-off

So the forced solution is a genuine **band-pass resonator bank** whose DC
rejection is a closed-form function of a single learnable scalar. It delivers
exactly the "selectively enhance specific temporal-frequency motion patterns"
that the original proposal claimed and the free-decay code could not express.

CONTROL ARMS (matched parameterisation, for the ablation the paper needs)
-------------------------------------------------------------------------
  kind='resonator' : hyperbolic + forced      (ours)          |Hn(0)| = 1/sqrt(1+Q^2)
  kind='diffusion' : parabolic  + forced      w*^2/(w*^2+w^2) |Hn(0)| = 1 (provably)
  kind='free'      : hyperbolic + initial value (old WPO)     |Hn(0)| = 1 (provably)

The parabolic arm has no Q because a first-order-in-t operator cannot resonate;
that absence IS the claim being tested.
"""

import math

import torch
import torch.nn as nn

TWO_PI = 2.0 * math.pi

KINDS = ('resonator', 'diffusion', 'free')

# Physically sensible clamps. P is a period in frames; Q is a quality factor.
P_MIN, P_MAX = 2.0, 512.0
Q_MIN, Q_MAX = 0.25, 64.0


def _resonator(omega2, w_star, q):
    """Forced hyperbolic response, peak-normalised. |H(w*)| = 1, |H(0)| = 1/sqrt(1+Q^2)."""
    lam = w_star ** 2 / q
    re = omega2 - w_star ** 2
    lam = lam.expand_as(re)
    return lam / torch.complex(re, lam)


def _diffusion(omega2, w_star, q):
    """Forced parabolic response. Real, zero-phase, unit DC gain, no resonance."""
    ws2 = w_star ** 2
    return ws2 / (ws2 + omega2) + 0j


def _free(omega2, w_star, q):
    """Initial-value hyperbolic response (the old TemporalWPO semantics).

    Matched knobs: wave speed c = P frames per unit evolution time, damping
    alpha = 1/Q. Reproduces the provable |H(0)| == 1 degeneracy.
    """
    c = TWO_PI / w_star
    alpha = 1.0 / q
    w = torch.sqrt(omega2)
    wd_sq = c ** 2 * w ** 2 - alpha ** 2 / 4.0
    wd = torch.sqrt(torch.clamp(wd_sq, min=1e-12))
    gamma = torch.sqrt(torch.clamp(-wd_sq, min=1e-12))
    decay = torch.exp(-alpha / 2.0)
    h_u = decay * (torch.cos(wd) + alpha / (2.0 * wd) * torch.sin(wd))
    h_o = decay * (torch.cosh(gamma) + alpha / (2.0 * gamma) * torch.sinh(gamma))
    return torch.where(wd_sq > 0, h_u, h_o)


_BUILDER = {'resonator': _resonator, 'diffusion': _diffusion, 'free': _free}


class WaveResonatorBank(nn.Module):
    """Learnable bank of physically-parameterised temporal filters.

    Two learnable scalars per (band, channel group): the resonant period P
    (frames) and the quality factor Q. Optionally a per-video hyper-network
    predicts (dlogP, dlogQ) offsets from the pooled clip context, which makes
    the bank *selective* (input-dependent) instead of LTI -- the property the
    old scalar-alpha/c WPO lacked and the reason it could not compete with a
    selective SSM.

    forward(x, padding_mask):
        residual=True  -> x + gain * R(x)        (in-block insertion, gain init 0.1)
        residual=False -> R(x)                   (single band view)
    band_views(x, mask) -> list of per-band R_b(x)   (fusion insertion)

    Padding: frames at/after the true length are replaced by the last valid
    frame before the FFT (a zero pad would inject a step-discontinuity
    spectrum), and the filtered output is zeroed there afterwards.
    """

    def __init__(self, dim, num_bands=1, periods=(64.0,), q=(4.0,), kind='resonator',
                 n_groups=1, adaptive=False, motion_cond=False, gain_init=0.1,
                 residual=True):
        super().__init__()
        assert kind in KINDS, f"kind must be one of {KINDS}"
        assert dim % n_groups == 0, "dim must be divisible by n_groups"
        assert len(periods) == num_bands and len(q) == num_bands

        self.dim = dim
        self.num_bands = num_bands
        self.n_groups = n_groups
        self.kind = kind
        self.residual = residual
        self.adaptive = adaptive
        self.motion_cond = motion_cond

        self.log_P = nn.Parameter(torch.log(torch.tensor([float(p) for p in periods]))
                                  .unsqueeze(-1).repeat(1, n_groups).clone())
        self.log_Q = nn.Parameter(torch.log(torch.tensor([float(v) for v in q]))
                                  .unsqueeze(-1).repeat(1, n_groups).clone())
        self.gain = nn.Parameter(torch.full((num_bands, n_groups), float(gain_init)))
        self.norm = nn.LayerNorm(dim)

        if adaptive:
            # Zero-init hyper-network: at step 0 the offsets are exactly 0, so the
            # adaptive bank is bit-identical to the global one.
            cond_in = dim * 2 if motion_cond else dim
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_in, max(16, dim // 8)),
                nn.GELU(),
                nn.Linear(max(16, dim // 8), num_bands * n_groups * 2),
            )
            nn.init.zeros_(self.cond_mlp[-1].weight)
            nn.init.zeros_(self.cond_mlp[-1].bias)

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _replicate_tail(x, padding_mask):
        B, T, C = x.shape
        lengths = (~padding_mask).sum(dim=1).clamp(min=1)
        idx = torch.arange(T, device=x.device)[None, :].expand(B, T)
        last = (lengths - 1)[:, None]
        safe = torch.where(idx < lengths[:, None], idx, last)
        return x.gather(1, safe.unsqueeze(-1).expand(-1, -1, C))

    def _context(self, x, padding_mask):
        """(B, cond_in) pooled appearance (+ pooled |dx| motion)."""
        if padding_mask is None:
            app = x.mean(dim=1)
            if not self.motion_cond:
                return app
            dx = (x[:, 1:, :] - x[:, :-1, :]).abs().mean(dim=1)
            return torch.cat([app, dx], dim=-1)
        m = (~padding_mask).float().unsqueeze(-1)
        app = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        if not self.motion_cond:
            return app
        d = (x[:, 1:, :] - x[:, :-1, :]).abs()
        mm = m[:, 1:, :] * m[:, :-1, :]
        dx = (d * mm).sum(dim=1) / mm.sum(dim=1).clamp(min=1)
        return torch.cat([app, dx], dim=-1)

    def _pq(self, x, padding_mask, T):
        """Returns P, Q with shape (B, bands, groups, 1)."""
        B = x.shape[0]
        P = torch.clamp(torch.exp(self.log_P), P_MIN, float(T)).view(1, *self.log_P.shape, 1)
        Q = torch.clamp(torch.exp(self.log_Q), Q_MIN, Q_MAX).view(1, *self.log_Q.shape, 1)
        if self.adaptive:
            d = self.cond_mlp(self._context(x, padding_mask))
            d = d.view(B, self.num_bands, self.n_groups, 2)
            P = torch.clamp(P * torch.exp(d[..., 0:1]), P_MIN, float(T))
            Q = torch.clamp(Q * torch.exp(d[..., 1:2]), Q_MIN, Q_MAX)
        else:
            P = P.expand(B, -1, -1, -1)
            Q = Q.expand(B, -1, -1, -1)
        return P, Q

    def _responses(self, x, padding_mask):
        """(x) -> list of num_bands responses, each (B, T, D) in x.dtype."""
        B, T, D = x.shape
        dtype = x.dtype
        xn = self.norm(x.float())
        if padding_mask is not None:
            xn = self._replicate_tail(xn, padding_mask.to(dtype=torch.bool, device=x.device))

        X = torch.fft.rfft(xn, dim=1)                       # (B, F, D) complex
        n_f = X.shape[1]
        omega = TWO_PI * torch.fft.rfftfreq(T, device=x.device, dtype=torch.float32)
        omega2 = (omega ** 2).view(1, 1, n_f)                  # (1, 1, F)

        P, Q = self._pq(x, padding_mask, T)
        w_star = TWO_PI / P                                 # (B, bands, groups, 1)

        G, Dg = self.n_groups, D // self.n_groups
        Xv = X.reshape(B, n_f, G, Dg)

        outs = []
        for b in range(self.num_bands):
            H = _BUILDER[self.kind](omega2, w_star[:, b], Q[:, b])   # (B, G, F)
            H = H.permute(0, 2, 1).unsqueeze(-1)                     # (B, F, G, 1)
            Y = Xv * H
            y = torch.fft.irfft(Y.reshape(B, n_f, D), n=T, dim=1)    # (B, T, D)
            y = y.view(B, T, G, Dg) * self.gain[b].view(1, 1, G, 1)
            y = y.reshape(B, T, D)
            if padding_mask is not None:
                y = y.masked_fill(padding_mask.to(dtype=torch.bool, device=x.device)
                                  .unsqueeze(-1), 0.0)
            outs.append(y.to(dtype))
        return outs

    # ---------------------------------------------------------------- forward
    def band_views(self, x, padding_mask=None):
        return self._responses(x, padding_mask)

    def forward(self, x, padding_mask=None):
        y = self._responses(x, padding_mask)
        if len(y) == 1:
            y = y[0]
        else:
            y = torch.stack(y, dim=0).sum(dim=0)
        return x + y if self.residual else y

    # ---------------------------------------------------------------- introspection
    @torch.no_grad()
    def describe(self):
        P = torch.clamp(torch.exp(self.log_P), P_MIN, P_MAX)
        Q = torch.clamp(torch.exp(self.log_Q), Q_MIN, Q_MAX)
        return (f"kind={self.kind} bands={self.num_bands} groups={self.n_groups} "
                f"P={[round(v, 1) for v in P[:, 0].tolist()]} "
                f"Q={[round(v, 2) for v in Q[:, 0].tolist()]} "
                f"gain={[round(v, 3) for v in self.gain[:, 0].tolist()]}")
