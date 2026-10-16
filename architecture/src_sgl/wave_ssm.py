# -*- coding: utf-8 -*-
"""WaveSSM -- a SECOND-ORDER (damped-oscillator) selective state-space mixer.

WHY THIS EXISTS
---------------
Mamba's S6 layer is a *real diagonal* state-space model::

    h_t = a_t * h_{t-1} + b_t,      a_t = exp(dt_t * A),   A real and < 0

so `a_t` is a positive real number strictly in (0, 1). Such a system can only
DECAY. It cannot oscillate -- no parameter setting makes a positive real number
change sign.

A damped oscillator -- the wave equation  u_tt + 2*zeta*w*u_t + w^2*u = f,
written as a first-order system in (u, u_t) -- has eigenvalues::

    lambda = -zeta*w  +/-  i*w*sqrt(1 - zeta^2)

which are COMPLEX CONJUGATES whenever the system is underdamped. The
exponential of a complex number rotates as well as shrinks. That rotation is
precisely the capacity Mamba structurally lacks, and it is not a cosmetic
difference: fights, riots, explosions and shaking are physically oscillatory,
while ordinary surveillance footage is comparatively smooth.

THE KEY TRICK -- COMPLEXIFY, DO NOT BLOCK-DIAGONALISE
-----------------------------------------------------
Writing the second-order system in companion form gives a 2x2 real block per
mode. Block-diagonal matrices destroy Mamba's parallel scan: the whole point of
the diagonal form is that the recurrence factorises elementwise, so the scan can
be done in O(log T) depth instead of O(T).

But a 2x2 real block with complex-conjugate eigenvalues is *isomorphic to a
single complex number*::

    [[a, -b], [b, a]]   <->   a + i*b

So we keep the diagonal structure -- and therefore the parallel scan -- and
simply let the diagonal be COMPLEX::

    h_t = z_t * h_{t-1} + b_t
    z_t = exp(dt_t * (-lam + i*omega))
        = exp(-dt_t*lam) * [ cos(dt_t*omega) + i*sin(dt_t*omega) ]

|z_t| = exp(-dt_t * lam) < 1 for every dt_t > 0 and lam > 0, so the layer is
stable BY CONSTRUCTION: no eigenvalue clamping, no matrix exponential at run
time, no 2x2 blocks to multiply.

EXACT ABLATION
--------------
When `omega == 0` the layer degenerates to a real-diagonal (Mamba-style) SSM.
`omega_init=0` keeps `log_omega` as a *frozen parameter*, so the ablation has
the identical code path, the identical parameter count and the identical
initialisation -- only the imaginary part of A differs. Any measured difference
is therefore attributable to oscillation and nothing else.

IMPLEMENTATION NOTES
--------------------
* The scan carries native `complex64` tensors. A complex multiply is one fused
  kernel instead of four real ones -- measured 1.9x faster than carrying
  (real, imag) pairs, bit-identical results.
* The scan is chunked (`chunk`, default 32): a Mamba1-style scan materialises
  (B, d_inner, d_state, T) many times over, which at the project's shape is
  ~0.5 GB per tensor in fp32. Chunking bounds the working set; chunk
  aggregates are combined by a tiny scan over T/chunk steps. The maths is
  unchanged -- the same recurrence, just not parallelised across boundaries.
  On CUDA the doubling scan is the fast path; see CF_ON_CPU_ONLY.

See also: ops/_verify_wave_ssm.py (local gate) and
ops/_diag_second_order_ssm.py (the capacity-gap study that motivated this).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# NOTE: `import torch` does NOT make `torch.utils.checkpoint` available -- the
# AttributeError only shows up at the first checkpointed call, so
# --wave-checkpoint-scan silently did nothing until this line existed.
import torch.utils.checkpoint

# Floor on the cumulative product in the closed-form scan; guards
# `b / cumprod(a)` against overflow without touching ordinary cases.
PFLOOR = 1e-18
# Longest chunk for which the closed form is used; beyond this the doubling
# scan (no division, no underflow risk) takes over. See _scan_chunked_cf.
MAX_CF_CHUNK = 32
# The closed form wins on CPU (5.3x) but LOSES on CUDA, so it is gated on
# device. Measured on the 2080 Ti at (16, 1024, 8, 256) complex64 -- the
# project's exact shape -- one scan takes:
#     chunk=32   doubling  44.1 ms   closed-form  70.6 ms
#     chunk=64   doubling  48.4 ms   closed-form  48.3 ms
# i.e. at the default chunk the closed form is 1.6x SLOWER, and at chunk=64 it
# merely ties. The individual primitives are cheap (complex mul 1.48 ms, div
# 3.67 ms, abs 1.75 ms, cumsum 1.11 ms, cumprod 1.12 ms), so the CPU advantage
# was an artefact of CPU kernel-launch overhead, not of fewer bytes moved.
# Reverting this to "always closed-form" costs ~35 min per 4-arm run.
CF_ON_CPU_ONLY = True

__all__ = ["complex_scan", "complex_scan_naive", "WaveSSM"]


# --------------------------------------------------------------------------- scan
def complex_scan_naive(A, B):
    """Reference O(T) loop for `complex_scan`. A, B complex, shape (..., T)."""
    T = A.shape[-1]
    h = torch.zeros_like(A[..., 0])
    out = []
    for t in range(T):
        h = A[..., t] * h + B[..., t]
        out.append(h)
    return torch.stack(out, dim=-1)


def _scan_body(A, B):
    """Hillis-Steele inclusive scan, O(log T) depth. A, B complex, (..., T).

    Returns (A_agg, B_agg) where the pair encodes the affine map
    h_t = A_agg[t] * h_{-1} + B_agg[t]. With h_{-1} = 0 the state is B_agg.

    Element t carries the map over its block. Merging an earlier block (A1,B1)
    with a later one (A2,B2) gives (A2*A1, A2*B1 + B2). Positions with no
    earlier block are padded with the identity (1+0i, 0+0i), which is what
    makes the prefix truncation at t < s come out right automatically.

    NOTE ON MEMORY.  This allocates `Ap`, `Bp`, `An`, `Bn` -- four full-size
    tensors -- on top of `A` and `B`, so a single scan peaks at ~6x its input.
    An in-place variant was tried and is numerically identical, but autograd
    rejects it (the elementwise products need the pre-write values for their
    backward).  Memory is instead bounded by slicing along the STATE axis and
    checkpointing each slice -- see `complex_scan`.
    """
    T = A.shape[-1]
    s = 1
    while s < T:
        Ap = torch.cat([torch.ones_like(A[..., :s]), A[..., :-s]], dim=-1)
        Bp = torch.cat([torch.zeros_like(B[..., :s]), B[..., :-s]], dim=-1)
        An = A * Ap
        Bn = A * Bp + B
        A, B = An, Bn
        s *= 2
    return A, B


def _scan_chunked(A, B, chunk):
    """Chunked `_scan_body`: bounds memory to (..., chunk) instead of (..., T).

    Within-chunk scans run in parallel; the chunk aggregates are then combined
    by a scan over T/chunk steps (4 steps at chunk=64, T=256). Exact.
    """
    T = A.shape[-1]
    if chunk is None or chunk <= 0 or T <= chunk:
        return _scan_body(A, B)[1]

    pad = (-T) % chunk
    if pad:
        # Pad with the IDENTITY transition (a = 1+0i, b = 0+0i). Padding with 0
        # would zero the aggregate and wipe the state.
        A = torch.cat([A, torch.ones_like(A[..., :pad])], dim=-1)
        B = torch.cat([B, torch.zeros_like(B[..., :pad])], dim=-1)
    T2 = T + pad
    n = T2 // chunk
    lead = A.shape[:-1]

    # Reshape to (-1, chunk), NOT (-1, T2): each chunk must be scanned as its
    # OWN short sequence. Scanning the whole T2 row would run across chunk
    # boundaries and the recombination below would then be applied twice.
    Al, Bl = _scan_body(A.reshape(-1, chunk), B.reshape(-1, chunk))
    Al = Al.reshape(*lead, n, chunk)
    Bl = Bl.reshape(*lead, n, chunk)

    # Inclusive scan over the chunk aggregates, then shift right -> EXCLUSIVE
    # prefix. PB[k] is the state at the end of chunk k-1, i.e. the state
    # entering chunk k (PB[0] = 0).
    # The local scan says h_local[k,t] = Al[k,t]*h_enter_k + Bl[k,t], so the
    # true state is  h[k,t] = Al[k,t]*PB[k] + Bl[k,t].
    # (Note the order: the WITHIN-chunk A multiplies the ACROSS-chunk prefix.)
    _, GB = _scan_body(Al[..., -1], Bl[..., -1])
    PB = torch.cat([torch.zeros_like(GB[..., :1]), GB[..., :-1]], dim=-1)
    h = Al * PB[..., :, None] + Bl
    return h.reshape(*lead, T2)[..., :T]


def _scan_chunked_cf(A, B, chunk):
    """Chunked scan via the CLOSED FORM of the linear recurrence.

    h_t = a_t h_{t-1} + b_t has, inside a chunk,

        P_t = prod_{s<=t} a_s          (cumprod)
        Q_t = sum_{s<=t} b_s / P_s     (cumsum)
        h_t = P_t (h_chunk_start + Q_t)

    so a chunk needs TWO reductions instead of the log2(chunk) doubling passes
    of `_scan_chunked`. Measured on CPU at (8, 512, 8, 256) complex64:
    417 ms -> 78 ms (5.3x).

    THE CPU WIN DOES NOT TRANSFER TO GPU. On the 2080 Ti at the project's shape
    (16, 1024, 8, 256) complex64 the same comparison reads 44.1 vs 70.6 ms at
    chunk=32 (closed form SLOWER) and 48.4 vs 48.3 ms at chunk=64 (a tie). The
    primitives are individually cheap there, so the CPU result was dominated by
    launch overhead, not by bytes moved. Hence the device gate in `_scan_one`.

    The hope that it would remove the need for gradient checkpointing also did
    not materialise: with `ckpt=False` the run still OOMs at 10.52 GiB, because
    the peak is set by the retained activation tensors, not by the scan's
    internal intermediates. Keep checkpointing on.

    NUMERICAL GUARD.  `b/P` overflows if the cumulative product P underflows.
    |a| = exp(-dt*lam), so the worst case is a large dt*lam accumulated over a
    long chunk. Two guards, both required:
      * P is floored at PFLOOR (~41 nepers of decay) so the division cannot
        produce inf; see the comment at the clamp.
      * BEFORE computing anything, if the floor would actually fire for this
        input the call is handed to the doubling scan, so the function is exact
        whenever it returns. The floor alone is not enough -- it keeps the
        numbers finite but changes them.
    """
    # The closed form divides by cumprod(a), so a long chunk can underflow it.
    # The pre-check further down is what actually guarantees exactness; this cap
    # just avoids paying for the check on absurd chunk sizes.
    if chunk <= 0 or chunk > MAX_CF_CHUNK:
        return _scan_chunked(A, B, chunk)

    lead = A.shape[:-1]
    T = A.shape[-1]
    pad = (-T) % chunk
    if pad:
        A = torch.cat([A, torch.ones_like(A[..., :pad])], dim=-1)
        B = torch.cat([B, torch.zeros_like(B[..., :pad])], dim=-1)
    T2 = T + pad
    n = T2 // chunk

    Al = A.reshape(-1, n, chunk)
    Bl = B.reshape(-1, n, chunk)

    # Decide BEFORE doing the work whether the floor below will fire. If any
    # chunk's smallest |a| raised to the chunk length underflows PFLOOR, then
    # `b / cumprod(a)` overflows there and the floor would silently corrupt the
    # result -- so hand the whole call to the doubling scan, which has no
    # division and is exact unconditionally. Measured relative error vs the
    # naive loop, (2, 64, 3, 4) complex64, dt drawn uniformly:
    #     dt<=0.1 (the model's range, min|a|~0.45)  chunk=16 1.2e-07  32 1.3e-07
    #     dt<=0.5 (extreme,            min|a|~0.011) chunk=16 2.7e-01  32 8.0e-01
    # i.e. the failure is real and the chunk cap alone did not prevent it
    # (MAX_CF_CHUNK=32 still allowed the 8e-01 case). The test is a min over A
    # plus a scalar pow, so it costs one extra pass on a path that only runs on
    # CPU anyway; correctness beats the pass.
    with torch.no_grad():
        if float(Al.abs().amin()) ** chunk < PFLOOR:
            return _scan_chunked(A, B, chunk)

    P = torch.cumprod(Al, dim=-1)
    # Guard against `b / P` overflowing.  Clamp the CUMULATIVE product, not the
    # per-step a: a per-step floor is far too aggressive (at chunk=64 a 60-neper
    # budget lifts everything below 0.39, which corrupts normal cases and made
    # the exactness tests fail at chunk>=32).  Clamping P instead leaves every
    # ordinary case bit-identical and only acts once the state has decayed by
    # more than ~41 nepers inside one chunk, where its contribution is nil.
    floor = PFLOOR
    mag = P.abs()
    P = torch.where(mag < floor, P / mag.clamp_min(1e-38) * floor, P)
    Q = torch.cumsum(Bl / P, dim=-1)
    Aagg = P
    Bagg = P * Q

    # across-chunk prefix: n is tiny (256/64 = 4), so the doubling scan is fine
    _, GB = _scan_body(Aagg[..., -1].reshape(-1, n), Bagg[..., -1].reshape(-1, n))
    PB = torch.cat([torch.zeros_like(GB[..., :1]), GB[..., :-1]], dim=-1)
    h = Aagg * PB[..., :, None] + Bagg
    return h.reshape(*lead, T2)[..., :T]


def _scan_one(A, B, checkpoint, chunk):
    # Device-gated dispatch. See CF_ON_CPU_ONLY: the closed form is 5.3x faster
    # on CPU but 1.6x slower on CUDA at the default chunk, so pick per device
    # rather than per taste. `_scan_chunked_cf` still falls back to the doubling
    # scan for chunk > MAX_CF_CHUNK.
    use_cf = _scan_chunked_cf if (not CF_ON_CPU_ONLY or not A.is_cuda) else _scan_chunked
    if checkpoint and torch.is_grad_enabled() and (A.requires_grad or B.requires_grad):
        return torch.utils.checkpoint.checkpoint(
            use_cf, A, B, chunk, use_reentrant=False)
    return use_cf(A, B, chunk)


def complex_scan(A, B, checkpoint=False, chunk=32, state_chunk=1):
    """Parallel (O(log T) depth) complex selective scan.

    A, B: complex64, shape (..., S, T) -> state h, same shape (h_{-1} = 0).
    S is the SSM state axis; the scan is diagonal, so the S channels are
    independent and can be scanned in separate passes.

    `chunk`, `state_chunk` and `checkpoint` are memory controls only -- the
    returned value is identical for any setting.

    Why `state_chunk` matters.  `_scan_body` peaks at ~6x its input, and the
    input here is (B, d_inner, d_state, T) complex64: at the project's shape
    (64, 1024, 8, 256) that is 1.07 GiB, so ONE scan peaks near 6.4 GiB and,
    with 12 scans in the model (2 layers x 2 directions x 3 scales), an 11 GB
    card OOMs even with gradient checkpointing on.  Slicing the state axis down
    to `state_chunk` channels per pass shrinks that peak proportionally.
    """
    # A dim < 3 means there is no separate state axis (the unit tests call the
    # scan directly on (..., T)); the state slicing below would misread it.
    S = A.shape[-2] if A.dim() >= 3 else 1
    if A.dim() < 3 or state_chunk is None or state_chunk <= 0 or S <= state_chunk:
        return _scan_one(A, B, checkpoint, chunk)
    outs = [_scan_one(A[..., i:i + state_chunk, :],
                      B[..., i:i + state_chunk, :], checkpoint, chunk)
            for i in range(0, S, state_chunk)]
    return torch.cat(outs, dim=-2)


# -------------------------------------------------------------------------- module
def _softplus_inv(v, eps=1e-6):
    return torch.log(torch.expm1(torch.as_tensor(v, dtype=torch.float32).clamp(min=eps)))


class WaveSSM(nn.Module):
    """Drop-in replacement for `mamba_ssm.Mamba` with a complex diagonal A.

    Interface matches Mamba1: `WaveSSM(d_model=..., d_state=..., d_conv=...,
    expand=...)` and `y = layer(x)` with x of shape (B, T, D).

    Args:
        omega_init: 'spread' -> log-spread resonator bank over the d_state slots
            (default; removes the initialisation sensitivity of the frequency);
            a float -> every slot starts at that frequency;
            0 -> FREEZE omega at zero, making this an exact real-diagonal
            (Mamba-style) SSM. Use it for the ablation.
        lam_init: 'mamba' (1..d_state, as in Mamba) or a float.
        chunk: scan chunk length; smaller = less memory, slightly slower.
            On CUDA 32 is the measured optimum for the doubling scan.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 omega_init="spread", omega_min=0.2, omega_max=8.0,
                 lam_init="mamba", dt_min=1e-3, dt_max=1e-1,
                 checkpoint_scan=False, chunk=32, state_chunk=1, **ignored):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.checkpoint_scan = bool(checkpoint_scan)
        self.chunk = int(chunk)
        # Memory knob only: scan this many state channels per pass. See
        # `complex_scan` for why it is needed -- at the project's shape one
        # unsliced scan alone peaks near 6 GiB.
        self.state_chunk = int(state_chunk)

        # --- input / gate projection (Mamba1 layout) ---
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        # dt (per channel), B (per state), C = C_r + i*C_i (complex readout)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 3 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # --- the complex diagonal: A = -lam + i*omega -----------------------
        if isinstance(lam_init, str) and lam_init == "mamba":
            lam0 = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        else:
            lam0 = torch.full((self.d_state,), float(lam_init), dtype=torch.float32)
        self.log_lam = nn.Parameter(_softplus_inv(lam0).repeat(self.d_inner, 1))

        if isinstance(omega_init, str) and omega_init == "spread":
            w0 = torch.exp(torch.linspace(math.log(omega_min), math.log(omega_max),
                                          self.d_state))
        else:
            w0 = torch.full((self.d_state,), float(omega_init), dtype=torch.float32)
        self.omega_is_zero = bool(w0.abs().max().item() == 0.0)
        # A frozen Parameter, not a buffer: keeps the parameter count identical
        # to the omega>0 model so the ablation is strictly matched.
        self.log_omega = nn.Parameter(
            torch.full((self.d_inner, self.d_state), -20.0) if self.omega_is_zero
            else _softplus_inv(w0).repeat(self.d_inner, 1),
            requires_grad=not self.omega_is_zero)

        # --- init, Mamba1-style ---
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        with torch.no_grad():
            dt = torch.rand(self.d_inner) * (dt_max - dt_min) + dt_min
            self.dt_proj.bias.copy_(_softplus_inv(dt))
        nn.init.xavier_uniform_(self.dt_proj.weight)
        with torch.no_grad():                      # depthwise conv starts as identity
            self.conv1d.weight.zero_()
            self.conv1d.weight[..., -1] = 1.0
            self.conv1d.bias.zero_()

    @property
    def lam(self):
        return F.softplus(self.log_lam.float())          # > 0 : decay rate

    @property
    def omega(self):
        return F.softplus(self.log_omega.float())        # >= 0 : angular frequency

    def transition(self, dt, scan_layout=False, state_slice=None):
        """Complex diagonal z = exp(dt*(-lam + i*omega)).

        dt: (B, T, d_inner) or (B, T, d_inner, 1) -> complex (B, T, d_inner, d_state)

        `scan_layout=True` takes dt as (B, d_inner, T) and returns
        (B, d_inner, d_state, T) directly.  Building it in the layout the scan
        wants avoids a `permute(...).contiguous()` of the whole tensor, which
        at the project's shape is an extra ~1 GiB copy per scan.

        `state_slice=(lo, hi)` builds only state channels [lo, hi).  The full
        tensor is (B, d_inner, d_state, T) complex64 = 1.07 GiB at the project's
        shape, and every checkpointed scan retains its inputs, so materialising
        it once per scan costs 12 GiB across the model. Building it a slice at a
        time keeps only ~1/d_state of that alive.
        """
        d = dt.float()
        if scan_layout:
            d = d.unsqueeze(2)                           # (B,d_inner,1,T)
            lo, hi = (0, self.d_state) if state_slice is None else state_slice
            lam = self.lam[:, lo:hi].view(1, self.d_inner, hi - lo, 1)
            om = self.omega[:, lo:hi].view(1, self.d_inner, hi - lo, 1)
        else:
            d = d.unsqueeze(-1) if d.dim() == 3 else d
            lam = self.lam
            om = self.omega
        r = torch.exp(-d * lam)                          # |z| in (0, 1)
        th = d * om
        return torch.polar(r, th)                        # r*(cos th + i sin th)

    def forward(self, x):
        """x: (B, T, D) -> (B, T, D)."""
        B, T, _ = x.shape
        xz = self.in_proj(x)
        xb, gate = xz.chunk(2, dim=-1)                       # (B,T,d_inner)

        xb = xb.transpose(1, 2)                              # (B,d_inner,T)
        xb = self.conv1d(xb)[..., :T]
        xb = F.silu(xb).transpose(1, 2)                      # (B,T,d_inner)

        p = self.x_proj(xb)
        dt_r, Bv, Cr, Ci = torch.split(
            p, [self.dt_rank, self.d_state, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_r))                  # (B,T,d_inner) > 0

        # Everything below is built directly in SCAN LAYOUT (B, d_inner, d_state, T):
        # permuting the small (B, d_inner, T) factors is cheap, whereas permuting
        # the full (B, T, d_inner, d_state) result would double the memory of the
        # biggest tensors in the model.
        dtT = dt.transpose(1, 2)                             # (B,d_inner,T)
        # b = dt * B_s * u_c  (Mamba1 selective-scan input); real drive, the
        # state still turns complex through repeated multiplication by z.
        # (dt*xb) is shared by every state channel, so it is computed once and
        # stays (B,d_inner,1,T).
        dtx = (dtT * xb.transpose(1, 2)).unsqueeze(2)        # (B,d_inner,1,T)
        BvT = Bv.transpose(1, 2)                             # (B,d_state,T)

        # Scan the state channels in slices of `state_chunk`.  A and Bc are
        # never materialised for all channels at once: each is ~1 GiB at the
        # project's shape and a checkpointed scan retains its inputs.
        g = max(1, int(self.state_chunk))
        hs = []
        for i in range(0, self.d_state, g):
            A_i = self.transition(dtT, scan_layout=True, state_slice=(i, i + g))
            b_i = dtx * BvT[:, i:i + g, :].unsqueeze(1)     # (B,d_inner,g,T)
            Bc_i = torch.complex(b_i, torch.zeros_like(b_i))
            hs.append(complex_scan(A_i, Bc_i,
                                   checkpoint=self.checkpoint_scan,
                                   chunk=self.chunk, state_chunk=0))
        h = hs[0] if len(hs) == 1 else torch.cat(hs, dim=-2)

        h = h.permute(0, 3, 1, 2)                            # (B,T,d_inner,d_state)
        y = (h.real * Cr.unsqueeze(2) - h.imag * Ci.unsqueeze(2)).sum(-1)
        y = y * F.silu(gate)
        return self.out_proj(y)

    def damping_report(self):
        """Per-state damping ratio zeta = lam / sqrt(lam^2 + omega^2).

        zeta < 1 means the mode oscillates; zeta -> 1 is pure decay.
        """
        with torch.no_grad():
            lam, om = self.lam, self.omega
            return lam / torch.sqrt(lam ** 2 + om ** 2 + 1e-12)
