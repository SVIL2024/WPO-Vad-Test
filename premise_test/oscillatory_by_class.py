# -*- coding: utf-8 -*-
"""Robustness of "anomalies are NOT oscillatory" -- per class, and across scales.

`_diag_is_anomaly_oscillatory.py` pooled every anomaly class into one group and
found a broadband tilt toward high frequencies rather than a single-band bump.
The obvious objection: fighting and explosion might genuinely ring, and simply
get diluted by abuse / car accident. A reviewer will ask this first, so answer
it before writing anything up.

This script repeats the measurement separately for each XD-Violence class
(fighting B1, shooting B2, riot B4, abuse B5, car accident B6, explosion G)
against the normal class A, and additionally repeats it at three window sizes
(32 / 64 / 128 frames) so the conclusion does not depend on one arbitrary scale.
The bands are the ones the paper declares for the statistic (see PERIOD_EDGES
below): at a 64-frame window the band edges are the FFT bins {1,3,6,12,24,33}.

For each (class, window) it reports:
  * the per-band difference vs normal, with a z score
  * tilt R^2 of that difference -- high = monotone slide to high frequency
    (roughness), low = a localised bump (rhythmic)
  * whether the largest band is the top-frequency one (tilt) or interior (bump)

Run from repo root:
    python premise_test/oscillatory_by_class.py
"""
import io
import os
import re
import sys
import zipfile

import numpy as np

ZIP = os.environ.get("XD_FEATURES_ZIP", r"D:\dataset\XDTrainClipFeatures.zip")
N_PER_CLASS = 80
WINDOWS = (32, 64, 128)
# Bands are the ones the paper declares for the statistic: five log-spaced bands
# with FFT-bin edges {1,3,6,12,24,33} at the reference window W=64, i.e. bins
# 1--2, 3--5, 6--11, 12--23, 24--32 (_diag_is_anomaly_oscillatory.py (not in this release) and the
# released premise_test/spectral.py both use exactly this set).
# This script also runs at W=32 and W=128, so the bands are carried across
# window sizes by their PERIOD boundaries instead of by bin index, and at
# win == REF_WIN that reproduces the declared bins exactly.  (An earlier version
# carried a separate period grid [64,21,10,5,2.5,2.0], which at W=64 gave bins
# [1,3,6,13,26,32]: bands 2--4 were then built on different bins than the paper
# declares.  See the note in `shape_vector`.)
REF_WIN = 64
BAND_EDGES_REF = [1, 3, 6, 12, 24, 33]
PERIOD_EDGES = [REF_WIN / float(e) for e in BAND_EDGES_REF]   # 64 .. 1.94 fr
CLASSES = [("A", "normal"), ("B1", "fighting"), ("B2", "shooting"),
           ("B4", "riot"), ("B5", "abuse"), ("B6", "car accident"),
           ("G", "explosion")]
RNG = np.random.default_rng(0)
PCA = None


def band_label(i, win):
    """The band name the paper uses: the nominal period at the band's edges.

    Truncating the period boundaries to integers is what the paper's labels
    are (21--64, 10--21, 5--10, 2--5, 1--2 frames), so the header printed here
    and the table header in the manuscript name the same bands.
    """
    lo, hi = PERIOD_EDGES[i], PERIOD_EDGES[i + 1]
    return "period %2d-%2d fr" % (int(hi), int(lo))


def load_index(z):
    vids = {}
    for n in z.namelist():
        if not n.endswith(".npy"):
            continue
        m = re.match(r"^(.*)__(\d+)$", os.path.basename(n)[:-4])
        if m:
            vids.setdefault(m.group(1), []).append((int(m.group(2)), n))
    for v in vids.values():
        v.sort()
    return vids


def class_of(vid):
    m = re.search(r"_label_([A-G]\d?)", vid)
    return m.group(1) if m else "?"


def load_video(z, entries):
    out = []
    for _, n in entries:
        with z.open(n) as fh:          # must close: see the note in the other script
            out.append(np.load(io.BytesIO(fh.read())).astype(np.float32))
    return np.concatenate(out, axis=0)


def fit_pca(z, vids, sel, n_comp=16, max_frames=200000):
    tot, X = 0, []
    for v in sel:
        try:
            x = load_video(z, vids[v])
        except Exception:
            continue
        X.append(x[::3])
        tot += x[::3].shape[0]
        if tot > max_frames:
            break
    X = np.concatenate(X, axis=0).astype(np.float64)
    X -= X.mean(0, keepdims=True)
    w, V = np.linalg.eigh(X.T @ X / len(X))
    return V[:, ::-1][:, :n_comp].astype(np.float32)


def shape_vector(x, win):
    """Normalised spectral shape for one video at window size `win`."""
    s = x @ PCA
    T = s.shape[0]
    if T < win * 2:
        return None
    stride = win // 2
    # bin index for a given period: f = 1/period  ->  bin = win * f
    # Clip to the legal slice bounds.  rfft(win) has win//2 + 1 entries, so
    # win//2 + 1 is a legal EXCLUSIVE end -- the Nyquist bin is bin win//2.
    # The previous filter used a strict `<` against that end, which at
    # win == REF_WIN silently deleted the last edge (33) and with it the whole
    # 24--32 band; the paper's table has five bands, so that was a bug.
    top = win // 2 + 1
    edges = [min(top, max(1, int(np.floor(win / p + 0.5)))) for p in PERIOD_EDGES]
    edges = sorted(set(edges))
    if len(edges) < 3:
        return None
    hann = np.hanning(win).astype(np.float32)[:, None]
    acc, nw = np.zeros(len(edges) - 1), 0
    for st in range(0, T - win + 1, stride):
        w = s[st:st + win]
        t = np.arange(win, dtype=np.float32)
        A = np.stack([np.ones_like(t), t], axis=1)
        w = w - A @ np.linalg.lstsq(A, w, rcond=None)[0]
        P = (np.abs(np.fft.rfft(w * hann, axis=0)) ** 2).mean(1)
        tot = P[1:].sum()
        if tot <= 1e-12:
            continue
        P = P / tot
        for i in range(len(edges) - 1):
            acc[i] += P[edges[i]:edges[i + 1]].sum()
        nw += 1
    return acc / nw if nw else None


def tilt_r2(v):
    i = np.arange(len(v), dtype=np.float64)
    A = np.stack([np.ones(len(v)), i], 1)
    p = A @ np.linalg.lstsq(A, v, rcond=None)[0]
    return 1 - float(((v - p) ** 2).sum()) / (float(((v - v.mean()) ** 2).sum()) + 1e-18)


def main():
    global PCA
    if not os.path.exists(ZIP):
        print("feature zip not found:", ZIP)
        return 1
    z = zipfile.ZipFile(ZIP)
    vids = load_index(z)
    by = {}
    for v in vids:
        by.setdefault(class_of(v), []).append(v)
    print("videos per class: " +
          ", ".join("%s=%d" % (c, len(by.get(c, []))) for c, _ in CLASSES))

    allsel = []
    for c, _ in CLASSES:
        s = sorted(by.get(c, []))
        RNG.shuffle(s)
        allsel += s[:N_PER_CLASS]
    print("fitting PCA on a mixed sample ...")
    PCA = fit_pca(z, vids, allsel[:120])
    print("  %d temporal directions" % PCA.shape[1])

    # load each class once, reuse across window sizes
    feats = {}
    for c, name in CLASSES:
        s = sorted(by.get(c, []))
        RNG.shuffle(s)
        cache = []
        for v in s[:N_PER_CLASS]:
            try:
                x = load_video(z, vids[v])
            except Exception:
                continue
            if x.shape[0] >= 256:
                cache.append(x)
        feats[c] = (name, cache)
        print("  %-14s %d videos" % (name, len(cache)))

    for win in WINDOWS:
        print("\n" + "=" * 74)
        print("WINDOW = %d frames" % win)
        print("=" * 74)
        S = {}
        for c, (name, cache) in feats.items():
            S[c] = np.array([r for r in (shape_vector(x, win) for x in cache)
                             if r is not None])
        if S["A"] is None or len(S["A"]) < 5:
            print("  not enough normal videos at this window")
            continue
        nb = S["A"].shape[1]
        print("  %-14s %-4s %s" % ("class", "n", "".join("%12s" % band_label(i, win)
                                                          for i in range(nb))))
        print("  %-14s %-4d %s" % ("normal", len(S["A"]),
                                   "".join("%12.4f" % v for v in S["A"].mean(0))))
        for c, name in CLASSES:
            if c == "A" or S.get(c) is None or len(S[c]) < 5:
                continue
            diff = S[c].mean(0) - S["A"].mean(0)
            zs = diff / np.sqrt(S[c].var(0) / len(S[c]) + S["A"].var(0) / len(S["A"]) + 1e-18)
            r2 = tilt_r2(diff)
            top = int(np.argmax(diff))
            kind = "TILT->fastest" if top == nb - 1 else "peak at band %d" % top
            print("  %-14s %-4d %s   tiltR2=%.2f %s"
                  % (name, len(S[c]), "".join("%+12.4f" % v for v in diff), r2, kind))
            print("  %-14s %-4s %s"
                  % ("", "", "".join("%+12.1f" % v for v in zs)) + "   (z)")
        print("  reading: a rhythmic class shows ONE positive band with the rest")
        print("  negative and a LOW tilt R^2. A tilt is: slow bands negative,")
        print("  fastest band the most positive, tilt R^2 high.")
    print("\n" + "=" * 74)
    print("If every class shows a tilt at every window size, the 'anomalies are")
    print("oscillatory' premise is dead for all of XD-Violence, not just pooled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
