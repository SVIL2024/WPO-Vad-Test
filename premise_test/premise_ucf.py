# -*- coding: utf-8 -*-
"""UCF-Crime premise test -- the band defined in seconds, not in frames.

Why this exists
---------------
R1 #4 and R3 both objected that UCF-Crime was missing from the premise tests.
Adding it naively would repeat R3 #3, which is sharper: "the choice of this
frequency band is not sufficiently justified across datasets with different frame
rates."

And the frame rate genuinely differs. XD features sit at the native 15 fps -- a
window stamped 00-25-20..00-29-20 (240 s) holds 3600 rows, and the loader comments
say "native 15 fps". UCF features, by contrast, are ten equal-length copies of one
sampled clip per video, and the row counts land on 5-8 s per clip only at ~30 fps,
which is UCF-Crime's native rate. The paper's mid band is bins 3-5 of a 64-point
transform, i.e. periods 10-21 frames: 0.67-1.40 s on XD, 0.33-0.70 s on UCF. Same
bin edges, twice the frequency range. So both readings are reported:

  period-matched   the XD period re-expressed in UCF's bins -- the claim that
                   transfers is about a temporal period, so this answers R3 #3
  frame-matched    XD's bins applied to UCF as if its rows were XD frames -- so the
                   reader can see how much the frame-rate assumption matters

An earlier version of this script picked a *band index* instead of shifting bin
edges. That is wrong and is the same mistake in disguise: the five bands are cut
from a 64-point transform at extraction time, so at another frame rate band 1
still covers a different period. Matching a period requires different bins.

Two limits to state up front:

  * UCF features are pooled 512-d vectors only, with no patch tokens, so this is
    the GLOBAL view. In the paper's tab:cluster the Ped2 global view is the one
    view whose 95% CI covers zero, so a null here would be consistent with that
    and would not refute the premise.
  * UCF's exact extraction rate is recorded nowhere on this machine. It is assumed
    to be 30 fps and the z is reported at 25 fps too, so whether the conclusion
    depends on the assumption is visible rather than buried.

usage:
  python premise_ucf.py [--limit N] [--seed 234]
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT_ROOT = '/root/autodl-tmp/dataset'
LIST_TEST = '/root/autodl-tmp/program/waveClipVad/list/ucf_CLIP_rgbtest.csv'
LIST_TRAIN = '/root/autodl-tmp/program/waveClipVad/list/ucf_CLIP_rgb.csv'

XD_FPS = 15.0
XD_BAND_FRAMES = (10, 21)          # the paper's mid band, in XD frames
WIN, STRIDE = 64, 16
N_PERM = 20000
PERIOD_LO_S = XD_BAND_FRAMES[1] / XD_FPS   # 1.40 s
PERIOD_HI_S = XD_BAND_FRAMES[0] / XD_FPS   # 0.667 s


# ------------------------------------------------------------------ statistic
def power(win_arr, hann):
    """eq:power in one line: Hann-windowed |rfft|^2, averaged over dimensions,
    DC dropped. Returns (len(win_arr)//2,) bins."""
    w = win_arr * hann[:, None]
    pw = np.abs(np.fft.rfft(w, axis=0)) ** 2
    pw = pw.mean(1)
    pw[0] = 0.0
    return pw


def share(pw, bin_lo, bin_hi):
    a, b = max(1, int(bin_lo)), min(len(pw), int(bin_hi) + 1)
    tot = pw.sum()
    return pw[a:b].sum() / tot if tot > 0 else 0.0


def detrender(x):
    t = np.arange(x.shape[0], dtype=np.float64)
    tc = t - t.mean()
    y = x - x.mean(0, keepdims=True)
    slope = (tc[:, None] * y).sum(0) / (tc ** 2).sum()
    return y - slope[None, :] * tc[:, None]


def bin_range_for_period(win, fps, lo_s, hi_s):
    """Bins of a `win`-point transform whose periods fall in [lo_s, hi_s] seconds
    at the given rate. Period of bin m is win/m frames = win/(m*fps) seconds.

    The transform length has to be passed in. Reading the module-level WIN here
    while the caller had scaled it to win_for(fps) is what made the first run
    report "period-matched" while actually summing bins 1-4 -- the longest
    periods in the spectrum, not the mid band. The z that came back was real and
    it was measuring the wrong thing.
    """
    lo_f = lo_s * fps
    hi_f = hi_s * fps
    m_lo = max(1, int(np.floor(win / hi_f)))     # longest period -> lowest bin
    m_hi = int(np.ceil(win / lo_f))
    return m_lo, m_hi


def win_for(fps):
    """Window length that keeps the window DURATION equal to XD's.

    XD analyses 64 frames at 15 fps = 4.27 s. Holding WIN at 64 on UCF would give a
    2.13 s window at 30 fps, which barely spans the 1.40 s period of interest and
    leaves the band resting on two FFT bins. Scaling WIN with the rate makes bin m
    cover WIN/(m*fps) seconds, i.e. the same seconds as XD's bin m, so the band
    edges carry across datasets by period rather than by bin index -- which is the
    assumption the manuscript states and the one R3 #3 asked to be justified.
    """
    return int(round(WIN * fps / XD_FPS))


def load(path):
    a = np.load(path, allow_pickle=True)
    if hasattr(a, 'files'):
        a = a['global_feat']
    return np.asarray(a, dtype=np.float32)


def video_shares(path, win, hann, bin_lo, bin_hi):
    v = detrender(load(path))
    out = []
    for st in range(0, len(v) - win + 1, STRIDE):
        out.append(share(power(v[st:st + win], hann), bin_lo, bin_hi))
    return out


def usable(path, win):
    return np.load(path, allow_pickle=True).shape[0] >= win


def one_video_per_file(paths):
    """UCF stores ten equal-length copies of one sampled clip per video, so keep
    one file per stem. Using all ten would repeat the same clip and inflate the
    window count without adding information."""
    seen, out = set(), []
    for p in paths:
        s = os.path.basename(p)[:-5].rsplit('__', 1)[0]
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def read(path):
    import csv
    rows = []
    with io.open(path, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            rows.append((str(r['path']).replace('../../../dataset/',
                                               FEAT_ROOT + '/'),
                         str(r['label'])))
    return rows


def z_of(norm, anom, rng):
    """Difference in mean band share, anom minus norm, with the VIDEO as the
    resampling unit (per-video means, then a two-sided permutation test)."""
    if len(norm) < 3 or len(anom) < 3:
        return float('nan'), len(norm), len(anom)
    obs = np.mean(anom) - np.mean(norm)
    pool = np.concatenate([norm, anom])
    al = len(anom)
    hits = 0
    for _ in range(N_PERM):
        idx = rng.permutation(pool.shape[0])
        d = pool[idx[al:]].mean() - pool[idx[:al]].mean()
        if abs(d) >= abs(obs) - 1e-15:
            hits += 1
    # normal-approximation z for reporting, p from the permutation
    se = pool.std(ddof=1) * np.sqrt(1.0 / al + 1.0 / (len(pool) - al))
    return (obs / se if se > 0 else float('nan'), (hits + 1.0) / (N_PERM + 1.0),
            len(norm), len(anom))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=234)
    ap.add_argument('--out', default=os.path.join(HERE, 'results_ucf_premise.txt'))
    a = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    te = read(LIST_TEST)
    if a.limit:
        # limit PER CLASS, not the row list: truncating rows silently dropped all
        # the normal videos, because Normal sits after the anomaly classes in the
        # CSV and the run then reported 0 normal videos without saying why.
        per = {}
        for p, lab in te:
            per.setdefault(lab, []).append((p, lab))
        te = [x for lab in sorted(per) for x in per[lab][:a.limit]]
    tr = read(LIST_TRAIN)
    te_by = {}
    for p, lab in te:
        te_by.setdefault(lab, []).append(p)
    tr_by = {}
    for p, lab in tr:
        tr_by.setdefault(lab, []).append(p)

    te_norm = one_video_per_file(te_by.get('Normal', []))
    te_anom = one_video_per_file([p for lab, ps in te_by.items()
                                  if 'Normal' not in lab for p in ps])
    tr_norm = one_video_per_file(tr_by.get('Normal', []))
    print('test: %d normal videos, %d anomalous videos (from %d classes)'
          % (len(te_norm), len(te_anom), len([k for k in te_by if 'Normal' not in k])))
    print('train normal videos: %d' % len(tr_norm))

    hann = np.hanning(WIN + 1)[:-1]
    lo_s, hi_s = PERIOD_HI_S, PERIOD_LO_S
    print('\nXD mid band : bins 3-5 of a %d-point transform' % WIN)
    print('             = %d-%d frames at %g fps' % (XD_BAND_FRAMES[0],
                                                    XD_BAND_FRAMES[1], XD_FPS))
    print('             = %.2f-%.2f s' % (lo_s, hi_s))

    rows = []
    for fps, tag in ((30.0, 'period-matched @30fps'),
                     (25.0, 'period-matched @25fps'),
                     (None, 'frame-matched (10-21 UCF frames)')):
        win = win_for(fps) if fps else WIN
        if fps is None:
            # same bin edges the paper uses on XD, applied to UCF rows as if they
            # were XD frames: bins WIN/21 .. WIN/10
            m_lo = max(1, int(np.floor(WIN / XD_BAND_FRAMES[1])))
            m_hi = int(np.ceil(WIN / XD_BAND_FRAMES[0]))
        else:
            m_lo, m_hi = bin_range_for_period(win, fps, lo_s, hi_s)
        hann = np.hanning(win + 1)[:-1]
        nm = [video_shares(p, win, hann, m_lo, m_hi) for p in te_norm]
        am = [video_shares(p, win, hann, m_lo, m_hi) for p in te_anom]
        nm = [x for x in nm if x]
        am = [x for x in am if x]
        rng = np.random.default_rng(a.seed)
        if not nm or not am:
            rows.append((tag, win, m_lo, m_hi, float('nan'), float('nan'), 0, 0))
            continue
        z, p, nn, na = z_of([np.mean(x) for x in nm], [np.mean(x) for x in am], rng)
        # report the period the chosen bins actually cover, in seconds
        per_lo = win / (m_hi * (fps or XD_FPS))
        per_hi = win / (m_lo * (fps or XD_FPS))
        rows.append((tag, win, m_lo, m_hi, z, p, nn, na, per_lo, per_hi))
        print('  %-32s WIN=%3d bins %2d-%2d (%.2f-%.2f s)  z=%7.3f  p=%.4f  '
              '(%d normal, %d anomalous)'
              % (tag, win, m_lo, m_hi, per_lo, per_hi, z, p, nn, na))

    o = []
    o.append('UCF-Crime premise test -- the band defined in seconds, not frames')
    o.append('=' * 78)
    o.append('')
    o.append('why: XD features are at the native 15 fps (a 240 s window holds 3600')
    o.append('rows); UCF features are one sampled clip per video at ~30 fps. The')
    o.append("paper's mid band is bins 3-5 of a 64-point transform = 10-21 frames =")
    o.append('%.2f-%.2f s. On UCF those same bins cover 0.33-0.70 s, so the two' % (lo_s, hi_s))
    o.append('datasets are compared at different periods unless the bins move.')
    o.append('')
    o.append('scope: UCF features are pooled 512-d vectors only, with no patch')
    o.append('tokens, so this is the GLOBAL view. The paper\'s tab:cluster reports')
    o.append('the Ped2 global view as the one whose 95% CI covers zero, so a null')
    o.append('here is consistent with that and would not refute the premise.')
    o.append('')
    o.append('%-32s %5s %10s %10s %8s %8s %7s'
             % ('reading', 'WIN', 'bins', 'period(s)', 'z', 'perm p',
                'n_vids'))
    for row in rows:
        tag, win, m_lo, m_hi, z, p, nn, na, plo, phi = row
        o.append('%-32s %5d %4d-%-5d %4.2f-%4.2f %8.3f %8.4f %7d'
                 % (tag, win, m_lo, m_hi, plo, phi, z, p, nn + na))
    # ---- per class, because UCF-Crime's 13 anomaly types are not alike ----
    # The pooled null above could be an average over classes that differ in the
    # opposite directions, which is exactly what the paper's own per-class table
    # found on XD: the deficit holds in every class there, but the classes sit at
    # different periods. So repeat the period-matched test one class at a time.
    o.append('')
    o.append('per anomaly class, period-matched @30fps. Each class is compared')
    o.append('against the same normal videos, with the video as the resampling')
    o.append('unit.')
    o.append('')
    # How many videos survive the window is reported first, because at the scaled
    # window many UCF clips are shorter than the window itself: Abuse002 holds 54
    # frames against a 128-frame window. Dropping those silently compares the long
    # videos of one group against the long videos of another, which is a selection
    # effect on both sides.
    o.append('%-8s %-26s %12s %12s' % ('WIN', 'band (s)', 'n_norm kept',
                                        'n_anom kept'))
    win = win_for(30.0)
    hann = np.hanning(win + 1)[:-1]
    m_lo, m_hi = bin_range_for_period(win, 30.0, lo_s, hi_s)
    lo_p, hi_p = win / (m_hi * 30.0), win / (m_lo * 30.0)
    keep_n = [p for p in te_norm if usable(p, win)]
    keep_a = [p for lab in te_by if 'Normal' not in lab
              for p in one_video_per_file(te_by[lab]) if usable(p, win)]
    o.append('%-8d %-26s %12d %12d   (of %d / %d)'
             % (win, '%.2f-%.2f' % (lo_p, hi_p), len(keep_n), len(keep_a),
                len(te_norm),
                sum(len(one_video_per_file(te_by[l]))
                    for l in te_by if 'Normal' not in l)))
    o.append('')
    o.append('%-22s %9s %9s %8s' % ('class', 'n_vids', 'z', 'perm p'))
    def _shares(paths):
        out = []
        for p in paths:
            s = video_shares(p, win, hann, m_lo, m_hi)
            if len(s) >= 2:
                out.append(float(np.mean(s)))
        return out
    nm = _shares(keep_n)
    per_class = []
    for lab in sorted(k for k in te_by if 'Normal' not in k):
        ps = [p for p in one_video_per_file(te_by[lab]) if usable(p, win)]
        am = _shares(ps)
        if len(am) < 3 or len(nm) < 3:
            continue
        rng = np.random.default_rng(a.seed)
        z, p, _, _ = z_of(nm, am, rng)
        per_class.append((lab, len(am), z, p))
        o.append('%-22s %9d %9.3f %8.4f' % (lab[:22], len(am), z, p))
    if per_class:
        sig = [c for c in per_class if c[3] < 0.05]
        o.append('')
        o.append('classes with perm p < 0.05: %d of %d  (%s)'
                 % (len(sig), len(per_class),
                    ', '.join(c[0] for c in sig) or 'none'))

    text = '\n'.join(o) + '\n'
    io.open(a.out, 'w', encoding='utf-8').write(text)
    print('\nwrote %s' % a.out)
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
