# -*- coding: utf-8 -*-
"""Time-varying wave operators + a wave-equation source loss.

Why this file exists
--------------------
The FFT bank in ``wave_resonator.py`` is a *global LTI* filter: one (P, Q) per
band for the whole clip, inserted **after** the Mamba encoder and concatenated
into a plain ``Linear`` fusion. With ``--adaptive-fusion`` off that whole branch
collapses to a single linear map on ``x`` (sum of linear operators is linear),
so it cannot add expressivity - and it inflates the fusion layer by 67%
(3*512 -> 5*512), which is a net negative on a model that already peaks at
epoch 1. Measured: 0.8247 / 0.8234 / 0.8165 vs the 0.8298 no-branch baseline.

Two things fix that, and both keep the *physics* (a damped wave equation) as the
actual mechanism instead of a decoration:

1. ``TimeVaryingWaveBank`` - the medium changes over time.
   Instead of one (P, Q) per clip we keep K fixed damped-oscillator kernels and
   predict a **per-frame** softmax gate over them from the local feature:
       y_t = gain * sum_k g_k(t) * (h_k * x)_t
   The gate depends on x, so the operator is no longer linear in x (no collapse)
   and it is **localised in time** (it can boost the band that matches the burst
   at frames 120-160 and stay quiet elsewhere). Kernels are short (33 taps)
   depthwise convs: no FFT, no periodic-boundary assumption, ~30k params.

2. ``wave_source_loss`` - the wave equation as a *prior on the score*.
   Treat the frame score s(t) as a displaced string. Away from any event the
   damped wave equation with no source gives  s_tt + alpha*s_t = 0, whose
   solutions are  A + B*exp(-alpha*t)  - piecewise flat with exponential
   transitions. Deviations from it are the *source term*, and events are sparse
   sources, so penalising |s_tt + alpha*s_t| with L1 kills isolated one-frame
   spikes (the main false-positive source under top-k MIL) while leaving a
   genuine multi-frame event untouched. This adds **no capacity at all**, which
   matters because every run peaks at epoch 1.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# 1. Kernel tiling: which (P, Q) pairs are physically meaningful               #
# --------------------------------------------------------------------------- #
def log_period_tiling(size, k=5, p_min=4.0, p_max=None):
    """Log-spaced periods that actually fit inside the window.

    A kernel of ``size`` taps can only resolve a period it *sees at least
    twice*, i.e. ``P <= size/2``. Anything longer is not an oscillator any
    more - after the Hann taper it degenerates into an edge/trend detector.
    The old default (P up to 64 with size=33) violated this for half the bank.
    """
    if p_max is None:
        p_max = size / 2.0
    p_max = max(p_max, p_min * 1.01)
    r = (p_max / p_min) ** (1.0 / max(int(k) - 1, 1))
    return [float(p_min * (r ** i)) for i in range(int(k))]


def quality_for_period(p, size, q_min=1.0, q_max=20.0):
    """Q such that the envelope decays over ~half the window.

    ``alpha = w0/(2Q) = pi/(P*Q)`` so the decay time is ``P*Q/pi``. Setting
    that to ``size/4`` (half of the Hann-tapered support) gives

        **Q = pi * size / (4 P)**      -> Q *decreases* as P grows.

    This is the opposite of the old hand-written ``P=8,16,32,64 / Q=2,3,4,6``
    pairing, which made the long-period kernels *more* selective even though
    they see fewer cycles - i.e. guaranteed trend detectors.
    """
    q = math.pi * float(size) / (4.0 * max(float(p), 1e-6))
    return float(min(max(q, q_min), q_max))


def auto_quality(periods, size, q_min=1.0, q_max=20.0):
    return [quality_for_period(p, size, q_min, q_max) for p in periods]


# --------------------------------------------------------------------------- #
# 2. Kernels: impulse response of the damped wave equation                     #
# --------------------------------------------------------------------------- #
def damped_oscillator_kernel(period, q, size, bandpass=True):
    """h[n] = exp(-alpha*n) * cos(wd*n), the impulse response of

        u_tt + 2*alpha*u_t + w0^2 * u = 0,   w0 = 2*pi/P,  alpha = w0 / (2Q)

    (the same damped wave used by ``wpo_transfer`` / ``wave_resonator``, just
    written in the time domain). ``bandpass`` removes the DC component so the
    view carries the *oscillation*, not another smoothed copy of x.
    """
    n = torch.arange(size, device=period.device, dtype=torch.float32)
    w0 = TWO_PI / torch.clamp(period, min=2.0)
    alpha = w0 / (2.0 * torch.clamp(q, min=0.5))
    wd_sq = w0 * w0 - alpha * alpha
    wd = torch.sqrt(torch.clamp(wd_sq, min=1e-8))

    h = torch.exp(-alpha * n) * torch.cos(wd * n)
    # Hann window: a hard truncation of the tail would put ripples in H(w).
    h = h * (0.5 - 0.5 * torch.cos(TWO_PI * n / (size - 1)))
    if bandpass:
        h = h - h.mean()
        s = h.abs().sum()
    else:
        s = h.sum()
    h = h / s.clamp(min=1e-6)
    return h


# --------------------------------------------------------------------------- #
# 2. Time-varying wave bank                                                    #
# --------------------------------------------------------------------------- #
class TimeVaryingWaveBank(nn.Module):
    """K damped-oscillator depthwise convs + a per-frame softmax gate.

    ``forward(x, padding_mask)`` -> same shape as x.
      residual=True  -> x + gain * y     (in-block / pre-encoder insertion,
                                          gain init 0.1 so it starts at identity)
      residual=False -> y                (extra fusion view)

    Padding: the tail is replicated before the conv (a zero pad would inject a
    step) and the output is zeroed there afterwards, same convention as
    ``WaveResonatorBank``.
    """

    def __init__(self, dim, periods=None, qs=None,
                 size=33, gate_hidden=64, gain_init=0.1, residual=True,
                 bandpass=True, learnable_pq=True, k_bands=5, p_min=4.0,
                 p_max_frac=0.5):
        super().__init__()
        assert dim % 1 == 0
        self.dim = dim
        self.size = size
        # Hard physical bound: a kernel of `size` taps Hann-tapered to ~size/2
        # cannot resolve a period longer than p_max_frac*size. Without this
        # clamp a learnable P happily drifts past it and the "oscillator"
        # silently degrades into a trend/edge detector (that is exactly what
        # killed the P=32/64 members of the old 8,16,32,64 @ size=33 bank).
        self.p_max_frac = float(p_max_frac)
        if periods is None:
            periods = log_period_tiling(size, k=k_bands, p_min=p_min)
        periods = [float(p) for p in periods]
        if qs is None:
            qs = auto_quality(periods, size)
        qs = [float(q) for q in qs]
        assert len(qs) == len(periods), "periods and qs must have the same length"
        self.K = len(periods)
        self.residual = residual
        self.bandpass = bandpass
        # Diagnostics / regularisation hooks, set from the training loop:
        #   last_gate : (K, T) batch-averaged gate time-course, detached
        #   _gate_bkt : (B, K, T) *differentiable* gate, only kept when
        #               ent_weight > 0 (otherwise it would pin the graph alive)
        #   ent_weight: weight of the gate-entropy term (see gate_entropy)
        self.last_gate = None
        self._gate_bkt = None
        self.ent_weight = 0.0

        self.log_P = nn.Parameter(torch.log(torch.tensor([float(p) for p in periods])),
                                  requires_grad=learnable_pq)
        self.log_Q = nn.Parameter(torch.log(torch.tensor([float(q) for q in qs])),
                                  requires_grad=learnable_pq)
        self.gain = nn.Parameter(torch.tensor(float(gain_init)))
        self.norm = nn.LayerNorm(dim)

        # Zero-init last layer -> softmax is uniform at step 0, i.e. the bank
        # starts as the *average* kernel and cannot disturb the init.
        self.gate = nn.Sequential(
            nn.Conv1d(dim, gate_hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(gate_hidden, self.K, kernel_size=5, padding=2),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    # ---------------------------------------------------------------- utils
    def kernels(self, dtype=torch.float32):
        P = torch.clamp(torch.exp(self.log_P), 2.0, self.size * self.p_max_frac)
        Q = torch.clamp(torch.exp(self.log_Q), 0.5, 1e3)
        ks = [damped_oscillator_kernel(P[k], Q[k], self.size, self.bandpass)
              for k in range(self.K)]
        return torch.stack(ks, 0).to(dtype)                       # (K, size)

    @staticmethod
    def _replicate_tail(x, padding_mask):
        B, T, C = x.shape
        lengths = (~padding_mask).sum(dim=1).clamp(min=1)
        idx = torch.arange(T, device=x.device)[None, :].expand(B, T)
        last = (lengths - 1)[:, None]
        safe = torch.where(idx < lengths[:, None], idx, last)
        return x.gather(1, safe.unsqueeze(-1).expand(-1, -1, C))

    # ---------------------------------------------------------------- forward
    def forward(self, x, padding_mask=None):
        B, T, D = x.shape
        dtype = x.dtype
        xn = self.norm(x.float())
        if padding_mask is not None:
            xn = self._replicate_tail(xn, padding_mask.to(torch.bool))

        h = self.kernels(dtype=xn.dtype)                          # (K, S)
        # Grouped conv: out channel o = d*K + k (d-major!), so that
        # u[:, d, k] is the depthwise conv of channel d with kernel k.
        w = h[None, :, :].expand(D, self.K, self.size).reshape(D * self.K, 1, self.size)
        u = F.conv1d(xn.transpose(1, 2), w,
                     padding=self.size // 2, groups=D)             # (B, D*K, T)
        if u.shape[-1] != T:                                       # even-size kernels
            u = u[..., :T]
        u = u.view(B, D, self.K, T)                                       # (B, D, K, T)

        g = torch.softmax(self.gate(xn.transpose(1, 2)).float(), dim=1)   # (B, K, T)
        # (K, T) average time-course, detached: used only by describe().
        self.last_gate = g.mean(dim=0).detach()
        # The differentiable copy is kept ONLY when the entropy term is on -
        # otherwise the reference would pin the gate's graph alive for free.
        self._gate_bkt = g if self.ent_weight > 0 else None
        y = (u * g[:, None, :, :]).sum(dim=2)                             # (B, D, T)
        y = y.transpose(1, 2) * self.gain                                 # (B, T, D)

        if padding_mask is not None:
            y = y.masked_fill(padding_mask.to(torch.bool).unsqueeze(-1), 0.0)
        y = y.to(dtype)
        return x + y if self.residual else y

    def gate_stats(self):
        """(H_perframe, H_marginal), both normalised to [0, 1].

        ``H_perframe``  = mean over (batch, time) of the entropy of g[:, :, t]:
                          how *decided* each individual frame is.
        ``H_marginal``  = entropy of the gate averaged over (batch, time):
                          how *spread* the band usage is overall.

        Differentiable when ``ent_weight > 0`` (that is when ``_gate_bkt`` is
        kept); otherwise falls back to the detached (K, T) average and the
        values are for display only.
        """
        g = self._gate_bkt if self._gate_bkt is not None else self.last_gate
        if g is None:
            return None
        if g.dim() == 2:                                      # (K, T) -> B=1
            g = g.unsqueeze(0)
        lnK = math.log(max(self.K, 2))
        Ht = -(g * (g + 1e-8).log()).sum(dim=1)                # (B, T)
        H_per = Ht.mean() / lnK
        gbar = g.mean(dim=(0, 2))                              # (K,)
        H_marg = -(gbar * (gbar + 1e-8).log()).sum() / lnK
        return H_per, H_marg

    def gate_entropy(self):
        """Mean normalised PER-FRAME entropy (see gate_stats)."""
        st = self.gate_stats()
        if st is None:
            return float("nan")
        return st[0]

    def gate_mi(self):
        """Mutual information between the selected band and (sample, time).

            MI = H_marginal - H_perframe  >= 0     (entropy is concave)

        This is the quantity that actually matters, and it is the one V1 showed
        to be ~0. BOTH degenerate gates are linear operators:

          * uniform and constant    -> gain * (mean kernel) * x
          * one-hot and CONSTANT    -> gain * (h_k * x)

        Neither is time-varying, so neither escapes the LTI collapse. Only a
        gate whose choice *carries information about where we are in the clip*
        does - and that is exactly MI. Maximising it is the right regulariser;
        merely sharpening the per-frame entropy (which is what a naive
        "negative entropy" term does) accelerates the collapse instead.
        """
        st = self.gate_stats()
        if st is None:
            return None
        H_per, H_marg = st
        return H_marg - H_per

    @torch.no_grad()
    def describe(self):
        P = [round(v, 1) for v in torch.exp(self.log_P).tolist()]
        Q = [round(v, 2) for v in torch.exp(self.log_Q).tolist()]
        st = self.gate_stats()
        H_per, H_marg = st if st is not None else (float("nan"), float("nan"))
        mi = self.gate_mi()
        mi = float(mi) if mi is not None else float("nan")
        out = (f"TVWO P={P} Q={Q} gain={self.gain.item():.3f} "
               f"Hper={H_per:.3f} Hmarg={H_marg:.3f} MI={mi:.3f}")
        g = self.last_gate
        if g is not None:
            # per-kernel mean gate: which periods the model actually listens to
            gbar = g.mean(dim=1)
            # cosine similarity between the K band time-courses (redundancy)
            gn = F.normalize(g, dim=1)
            cos = (gn @ gn.t()).fill_diagonal_(0.0)
            out += (f" gbar={[round(float(v), 3) for v in gbar]} "
                    f"cosmax={float(cos.abs().max()):.3f}")
        return out


# --------------------------------------------------------------------------- #
# 3. Wave equation as a loss: the score is a string, events are sources         #
# --------------------------------------------------------------------------- #
def wave_source_loss(logits, lengths=None, alpha=0.25, reduction="mean"):
    """L1 on the source term of   s_tt + alpha * s_t = f(t).

        s(t) = sigmoid(frame logit)      (B, T[, 1])
        f(t) = s_tt + alpha * s_t        -> 0 for the free (no-event) solution
                                            A + B*exp(-alpha t)

    Sparse f  <=>  piecewise-flat score with exponential onsets, i.e. exactly
    what a video-level anomaly should look like. Isolated single-frame spikes
    have a huge |f| and get smoothed; a 40-frame fighting segment barely moves.
    """
    s = torch.sigmoid(logits.float())
    if s.dim() == 3:
        s = s.squeeze(-1)                                          # (B, T)
    if s.shape[1] < 3:
        return s.new_zeros(())

    d1 = s[:, 1:-1] - s[:, :-2]
    d2 = s[:, 2:] - 2.0 * s[:, 1:-1] + s[:, :-2]
    f = d2 + alpha * d1                                            # (B, T-2)

    if lengths is not None:
        T = s.shape[1]
        idx = torch.arange(T - 2, device=s.device)[None, :].expand(s.shape[0], T - 2)
        m = (idx + 2) < lengths.clamp(max=T)[:, None].to(idx.device)
        f = f * m.to(f.dtype)
        denom = m.to(f.dtype).sum().clamp(min=1.0)
    else:
        denom = f.numel()

    if reduction == "sum":
        return f.abs().sum()
    return f.abs().sum() / denom
