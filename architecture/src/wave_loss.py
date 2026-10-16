# -*- coding: utf-8 -*-
"""Wave-source sparsity loss for the ORIGINAL VadCLIP branch (src/).

Byte-for-byte the same operator as `src_sgl/wave_tv.py:wave_source_loss`, copied
here so `src/` stays self-contained (the remote job runs from `src/`, which has
no `wave_tv.py`). Keeping the two implementations identical is what makes the
src_sgl result transferable to VadCLIP - if you change one, change both.

The score is a string; an anomaly is a source.

    s(t) = sigmoid(frame logit)          (B, T) or (B, T, 1)
    f(t) = s_tt + alpha * s_t

`f ~ 0` is the free (source-free) damped solution A + B*exp(-alpha t), i.e. a
piecewise-flat score with exponential onsets - exactly what a video-level
anomaly label should look like. Isolated single-frame spikes have a huge |f| and
are penalised; a 40-frame event barely moves. This is a TEMPORAL prior only: it
never sees the video label, so it cannot leak supervision.
"""
import torch

__all__ = ["wave_source_loss"]


def wave_source_loss(logits, lengths=None, alpha=0.25, reduction="mean"):
    """L1 on the source term of  s_tt + alpha * s_t = f(t).

    logits : (B, T) or (B, T, 1) raw frame logits (sigmoid applied internally).
    lengths: (B,) valid frame counts; padding is masked out of the mean.
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
