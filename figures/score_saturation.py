# -*- coding: utf-8 -*-
"""Is the base model's score saturated? A check on how to read the AP numbers.

`fig_qualitative.py` picked videos by position in the separation distribution and
the lower three looked flat: the score sat near 0.97 across the whole video while
the ground-truth window came and went. If that is common rather than a quirk of
three videos, then frame-level AP is not being earned by within-video localisation
on those clips, and the paper should say so instead of leaving the reader to
assume otherwise.

Measures, per video, on the segment scores:
  saturated   fraction of segments with score > 0.9
  dynamic     max - min
  separability  mean(score | anomalous segments) - mean(score | normal segments)

usage: python score_saturation.py
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np

DUMPS = None


def dumps():
    global DUMPS
    if DUMPS is None:
        import glob
        pat = r'D:\program\waveClipVad\logs_fetch\scores_xd_*.npz'
        DUMPS = sorted(glob.glob(pat))
    return DUMPS


def measure(path):
    z = np.load(path, allow_pickle=True)
    probs, gt, rep = z['probs'], z['gt'], int(z['repeat'])
    seg = np.array([len(p) for p in probs])
    starts = np.concatenate([[0], np.cumsum(seg)[:-1]]) * rep
    sat, dyn, sep = [], [], []
    for i, p in enumerate(probs):
        p = np.asarray(p, dtype=np.float64)
        if len(p) < 20:
            continue
        g = gt[int(starts[i]):int(starts[i]) + len(p) * rep].reshape(len(p), rep)
        anom = (g > 0.5).any(axis=1)
        if anom.sum() < 5 or (~anom).sum() < 5:
            continue
        sat.append(float((p > 0.9).mean()))
        dyn.append(float(p.max() - p.min()))
        sep.append(float(p[anom].mean() - p[~anom].mean()))
    return (np.asarray(sat), np.asarray(dyn), np.asarray(sep))


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    fs = dumps()
    if not fs:
        print('no dumps under D:\\program\\waveClipVad\\logs_fetch')
        return 1
    print('score dumps: %d arms' % len(fs))
    print('')
    print('%-22s %6s %9s %9s %9s %9s'
          % ('arm', 'n_vid', 'sat>0.9', 'dyn range',
             'separab.', 'sep<0.05'))
    print('-' * 70)
    agg = {'sat': [], 'dyn': [], 'sep': []}
    for f in fs:
        sat, dyn, sep = measure(f)
        if not len(sat):
            continue
        name = os.path.basename(f)[len('scores_xd_'):-len('_s234.npz')]
        agg['sat'].append(sat)
        agg['dyn'].append(dyn)
        agg['sep'].append(sep)
        print('%-22s %6d %9.3f %9.3f %9.3f %8.0f%%'
              % (name, len(sat), float(np.median(sat)),
                 float(np.median(dyn)), float(np.median(sep)),
                 100.0 * float((np.abs(sep) < 0.05).mean())))
    if not agg['sep']:
        return 1
    sat = np.concatenate(agg['sat'])
    dyn = np.concatenate(agg['dyn'])
    sep = np.concatenate(agg['sep'])
    n = len(sep)
    print('-' * 70)
    print('%-22s %6d %9.3f %9.3f %9.3f %8.0f%%'
          % ('ALL ARMS pooled', n, float(np.median(sat)),
             float(np.median(dyn)), float(np.median(sep)),
             100.0 * float((np.abs(sep) < 0.05).mean())))
    print('')
    print('videos more than 50%% saturated : %4d of %d (%.0f%%)'
          % (int((sat > 0.5).sum()), n, 100.0 * (sat > 0.5).sum() / n))
    print('videos more than 99%% saturated : %4d of %d (%.0f%%)'
          % (int((sat > 0.99).sum()), n, 100.0 * (sat > 0.99).sum() / n))
    print('videos with dynamic range < 0.1 : %4d of %d (%.0f%%)'
          % (int((dyn < 0.1).sum()), n, 100.0 * (dyn < 0.1).sum() / n))
    print('videos with separability < 0.05: %4d of %d (%.0f%%)'
          % (int((np.abs(sep) < 0.05).sum()), n,
             100.0 * (np.abs(sep) < 0.05).sum() / n))
    print('')
    print('CAVEAT: every dump here is seed 234. A single-seed statistic in a')
    print('paper whose thesis is that single-seed numbers are unreliable is the')
    print('weakest form this claim can take. Across arms it is consistent, but')
    print('arms share a seed, so this bounds the ARM effect and says nothing')
    print('about seed-to-seed spread.')
    return 0


if __name__ == '__main__':
    sys.exit(main())