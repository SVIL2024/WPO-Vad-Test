# -*- coding: utf-8 -*-
"""Figure: what the paper measures, applied to an actual clip.

R3 #5 objected that the manuscript "mainly presents statistical tables, spectral
measurements, and variance analyses, without providing representative anomaly
detection results or temporal anomaly-score visualizations. It remains unclear how
the reported spectral differences and evaluation inconsistencies translate into
actual anomaly localization performance."

All eight existing figures are aggregate measurements. This one is not: it shows
per-frame anomaly score against ground truth for named XD-Violence videos, so a
reader can see the thing the paper argues about.

The data is already computed. `scores_xd_base_s234.npz` holds

    probs    object array, one float16/32 array per video, SEGMENT scores
    gt       2331296 float32, frame-level ground truth
    repeat   16, the frames-per-segment factor

so probs has one score per segment and gt has one label per FRAME. Segment k's
score therefore covers frames [16k, 16k+16), and the overlay has to be built by
repeating the segment axis, not by indexing the two arrays the same way. Getting
that wrong plots scores against the wrong frames and looks plausible.

usage: python fig_qualitative.py [--n 4]
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP_CANDIDATES = [
    r'D:\program\waveClipVad\logs_fetch\scores_xd_base_s234.npz',
    os.path.join(HERE, 'scores_xd_base_s234.npz'),
]
OUT = os.path.join(HERE, 'F21_qualitative_localization.png')


def load():
    for p in DUMP_CANDIDATES:
        if os.path.exists(p):
            print('reading %s' % p)
            return np.load(p, allow_pickle=True), p
    raise SystemExit('score dump not found; looked in:\n  ' +
                     '\n  '.join(DUMP_CANDIDATES))


def video_name(p):
    b = os.path.basename(str(p))
    return b[:-4] if b.endswith('.npy') else b


def smooth(x, k):
    """Centred moving average with edge handling, so the overlay reads as a
    curve rather than as 16-frame steps."""
    if k <= 1 or len(x) < k:
        return x
    pad = k // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    ker = np.ones(k) / k
    return np.convolve(xp, ker, mode='same')[pad:pad + len(x)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4, help='videos to show')
    ap.add_argument('--smooth', type=int, default=9, help='smoothing frames')
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    z, path = load()
    probs, gt, rep = z['probs'], z['gt'], int(z['repeat'])
    print('videos %d, frame-level gt %d, repeat %d' % (len(probs), len(gt), rep))
    total_seg = sum(len(p) for p in probs)
    print('segment scores %d -> frames %d (gt has %d, match=%s)'
          % (total_seg, total_seg * rep, len(gt),
             total_seg * rep == len(gt)))

    # A video is worth showing if it has a clear anomaly window: the standard
    # deviation of its own score should exceed the median video's by a margin.
    stats = []
    # gt is the concatenation of every video's frame labels in probs order, so the
    # start of video i is the CUMULATIVE segment count, not i times this video's
    # own length. Using i*len(p) silently reads past the end from the second video
    # onward, which is what the first version did.
    seg_counts = np.array([len(p) for p in probs])
    starts = np.concatenate([[0], np.cumsum(seg_counts)[:-1]]) * rep
    for i, p in enumerate(probs):
        p = np.asarray(p, dtype=np.float64)
        if len(p) < 20:
            continue
        s0 = int(starts[i])
        g = gt[s0:s0 + len(p) * rep].reshape(len(p), rep)
        # Reduce the FRAME-level ground truth to a per-segment anomaly fraction
        # before comparing it with the SEGMENT-level score. Indexing one with the
        # other directly is a shape error, and the first version of this script
        # did exactly that.
        frac = (g > 0.5).mean(axis=1)
        pos = frac > 0.5
        if pos.sum() < 5 or (~pos).sum() < 5:
            continue
        an, no = p[pos].mean(), p[~pos].mean()
        stats.append((an - no, i, len(p), frac.mean()))
    stats.sort(reverse=True)
    print('videos with a usable anomaly window: %d' % len(stats))
    print('best separation (anomalous minus normal score): %s'
          % ', '.join('%+.3f' % s[0] for s in stats[:8]))

    # Pick by POSITION in the separation distribution, not by taking the top N.
    # Showing the best few is what R3 #5's "representative" is objecting to: the
    # first version of this figure picked the four highest-scoring videos, which
    # makes the method look cleaner than it is. One video from each part of the
    # distribution is honest and is also more informative, because it shows the
    # false positives the aggregate numbers hide.
    stats.sort(key=lambda s: s[0])
    if not stats:
        print('no usable video: cannot draw the figure')
        return 1
    picks = []
    n = len(stats)
    for frac in (0.95, 0.75, 0.50, 0.25, 0.05)[:a.n]:
        picks.append(stats[min(n - 1, int(n * frac))])
    picks = sorted(picks, key=lambda s: -s[0])
    picked = picks
    print('separation range over %d usable videos: %+.3f .. %+.3f'
          % (n, stats[0][0], stats[-1][0]))
    QUARTILE = {0: 'worst decile', 1: 'lower quartile', 2: 'median',
                3: 'upper quartile', 4: 'best decile'}

    def band_of(sep_, n_):
        q = (sum(1 for s in stats if s[0] < sep_) + 0.5) / n_
        if q >= 0.9:
            return QUARTILE[4]
        if q >= 0.75:
            return QUARTILE[3]
        if q >= 0.25:
            return QUARTILE[2]
        if q >= 0.10:
            return QUARTILE[1]
        return QUARTILE[0]

    # Real video names, so a reader can find the clip.
    # The test CSV has one row per video in the same order as probs, so the
    # i-th row names the i-th video. Strip the __N.npy chunk suffix.
    STEMS = []
    try:
        import csv as _csv
        for root in (r'D:\program\waveClipVad\list',
                     os.path.join(HERE, 'list'),
                     r'C:\Users\limax\AppData\Local\Temp\opencode\wpo_final'
                     r'\architecture\src_sgl\list'):
            f = os.path.join(root, 'xd_CLIP_rgbtest.csv')
            if os.path.exists(f):
                with io.open(f, encoding='utf-8') as fh:
                    for row in _csv.DictReader(fh):
                        b = os.path.basename(str(row['path']))[:-5]
                        STEMS.append((b.rsplit('__', 1)[0], str(row['label'])))
                break
    except Exception as e:
        print('  (video names unavailable: %s)' % e)

    def name_of(i):
        if i < len(STEMS):
            return '%s  [%s]' % (STEMS[i][0][:58], STEMS[i][1])
        return 'video %d of %d' % (i, len(probs))

    fig, axes = plt.subplots(len(picked), 1, figsize=(9.0, 2.05 * len(picked)),
                             sharex=False)
    if len(picked) == 1:
        axes = [axes]
    for ax, (sep, i, nseg, frac) in zip(axes, picked):
        p = np.asarray(probs[i], dtype=np.float64)
        s0 = int(starts[i])
        g = gt[s0:s0 + nseg * rep].reshape(nseg, rep)
        seg_anom = (g > 0.5).any(axis=1)          # segment is anomalous if any frame is
        # segment score -> per frame. This is the step that makes the plot honest:
        # probs holds one score per 16-frame segment.
        frame_score = np.repeat(p, rep)
        frame_anom = np.repeat(seg_anom, rep)
        x = np.arange(len(frame_score))
        ax.fill_between(x, 0, 1, where=frame_anom, color='0.85', step='mid',
                        label='ground truth anomaly')
        ax.plot(x, smooth(frame_score, a.smooth), color='C0', lw=1.1,
                label='anomaly score')
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel('score', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title('%s%s   |  %d segments   |  anomalous frames %.0f%%   '
                     '|  score gap %+.3f'
                     % (name_of(i), '  (%s)' % band_of(sep, n) if band_of(sep, n)
                        else '', nseg, 100 * frac, sep), fontsize=8)
        for s in ax.spines.values():
            s.set_linewidth(0.6)
    axes[0].legend(fontsize=7, loc='upper right', frameon=False)
    axes[-1].set_xlabel('frame', fontsize=8)
    fig.suptitle('Temporal anomaly score against frame-level ground truth, '
                 'XD-Violence (seed 234, base arm)', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=170)
    fig.savefig(OUT.replace('.png', '.pdf'))
    print('\nwrote %s and %s'
          % (OUT, OUT.replace('.png', '.pdf')))
    print('selection is by position in the separation distribution, so this is a')
    print('representative panel rather than a best-case one. The lower three')
    print('panels look flat: see score_saturation.py for how common that is.')
    print('pick table (video index, segments, anomalous fraction, score gap):')
    for sep, i, nseg, frac in picked:
        print('  %4d  %6d  %.3f  %+.4f' % (i, nseg, frac, sep))
    return 0


def _path_of(z, i):
    return ''


if __name__ == '__main__':
    sys.exit(main())