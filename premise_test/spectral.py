# -*- coding: utf-8 -*-
"""Does the XD "anomalies are not oscillatory" finding survive SPATIAL resolution?

SETUP
    The XD result (_diag_is_anomaly_oscillatory.py (not in this release)) was measured on GLOBAL
    CLIP features: one 512-d vector per frame, with the 14x14 patch tokens
    thrown away at extraction. So the honest challenge is: maybe the mid-band
    (periods 10-21 frames) DEFICIT we reported is an artefact of averaging 196
    patch tokens into one vector. If a fight is a localised rhythm in one corner
    of the frame, global pooling would smear it into broadband noise -- and we
    would wrongly conclude "there is no rhythm".

    UCSD Ped2 lets us test exactly that, because the raw frames AND per-pixel
    ground truth are local. We re-extracted CLIP ViT-B/16 with patch tokens kept
    (_ped2_extract.py (not in this release)), so on the SAME frames we can compare:
      GLOBAL view  : the 512-d pooled vector (identical in kind to XD)
      PATCH view    : each of the 196 patch tokens, separately, with the GT mask
                      telling us which patches actually contain the anomaly

THE DECISIVE TEST IS THE CALIBRATION, NOT THE MEASUREMENT
    If we inject a period-16 ringing into the patch tokens and the 10-21 frame
    band lights up, then the patch-view statistic CAN see a rhythm. If real
    anomalous patches still show no mid-band excess under that same statistic,
    the XD finding is NOT a pooling artefact -- it replicates in a spatially
    resolved feature space. Conversely, if real anomalous patches DO show a
    mid-band bump that the global vector misses, the paper's premise section
    must be revised.

STATISTIC (identical to the XD script, do not "improve" it)
    WIN=64, STRIDE=16, remove mean + linear trend, Hann, rfft, average power
    over the 16 PCA directions, drop bin 0, normalise to sum 1, aggregate into
    BAND_EDGES = [1,3,6,12,24,33] -> periods 21-64 / 10-21 / 5-10 / 2-5 / 1-2 fr.
    z = (mean_A - mean_N) / sqrt(var_N/n_N + var_A/n_A)
"""
import glob
import os

import numpy as np
from scipy.ndimage import uniform_filter1d

FEAT = os.environ.get("PED2_FEAT", r"D:\program\waveClipVad\ped2_feat")
WIN = 64
STRIDE = 16
BAND_EDGES = [1, 3, 6, 12, 24, 33]
N_PCA = 16
RNG = np.random.default_rng(0)
GRID = 14


def band_label(i):
    lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
    return "%2d-%2d fr" % (WIN // hi, WIN // lo)


def detrender():
    """Projection matrix that removes mean + linear trend from a WIN-length axis."""
    t = np.arange(WIN, dtype=np.float64)
    A = np.stack([np.ones_like(t), t], axis=1)
    return np.eye(WIN) - A @ np.linalg.pinv(A)


def spec_shape(series, P, hann):
    """series: (WIN, M, D). Returns (n_band, M) normalised spectral shape."""
    M, D = series.shape[1], series.shape[2]
    w = series.reshape(WIN, M * D)
    w = P @ w
    w = w.reshape(WIN, M, D) * hann[:, None, None]
    pw = (np.abs(np.fft.rfft(w, axis=0)) ** 2).mean(2)      # (33, M)
    tot = pw[1:].sum(0)
    ok = tot > 1e-12
    out = np.zeros((len(BAND_EDGES) - 1, M))
    if not ok.any():
        return out, ok
    pw[:, ok] /= tot[ok]
    for i in range(len(BAND_EDGES) - 1):
        out[i, ok] = pw[BAND_EDGES[i]:BAND_EDGES[i + 1], ok].sum(0)
    return out, ok


def zscore(a, b):
    """a = normal samples, b = anomalous samples, both (n,)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return (b.mean() - a.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b) + 1e-18)


def fit_pca(X, k=N_PCA, cap=200000):
    X = X[:: max(1, len(X) // cap)].astype(np.float64)
    X -= X.mean(0, keepdims=True)
    w, V = np.linalg.eigh(X.T @ X / len(X))
    return V[:, ::-1][:, :k].astype(np.float32)


def main():
    tests = sorted(glob.glob(os.path.join(FEAT, "Test_*.npz")))
    trains = sorted(glob.glob(os.path.join(FEAT, "Train_*.npz")))
    print("clips: %d test (with pixel GT), %d train (all normal)" % (len(tests), len(trains)))
    if not tests:
        print("no features yet - run _ped2_extract.py (not in this release) first")
        return

    clips = []
    for f in tests:
        d = np.load(f)
        clips.append(dict(name=os.path.basename(f)[5:-4], g=d["global_feat"].astype(np.float32),
                          p=d["patch"].astype(np.float32), gt=d["gt"].astype(np.float32),
                          ok=d["frame_ok"]))
    trains_x = [np.load(f)["global_feat"].astype(np.float32) for f in trains]
    trains_p = [np.load(f)["patch"].astype(np.float32) for f in trains]

    print("fitting PCA ...")
    PCA_G = fit_pca(np.concatenate([c["g"] for c in clips] + trains_x))
    # every 5th PATCH, matching controls/backbone.py.  The two scripts used
    # to disagree here (7 vs 5), which produced two different PATCH z values for
    # the same backbone (-6.08 vs -6.97) and is now disclosed in section 4.4.
    PP = np.concatenate([c["p"][:, ::5, :].reshape(-1, 768) for c in clips] +
                        [x[:, ::5, :].reshape(-1, 768) for x in trains_p])
    PCA_P = fit_pca(PP)
    print("  global 512 -> %d,  patch 768 -> %d" % (PCA_G.shape[1], PCA_P.shape[1]))

    P = detrender()
    hann = np.hanning(WIN).astype(np.float32)
    nb = len(BAND_EDGES) - 1

    # ---------------------------------------------------------------- GLOBAL
    gA, gN = [], []
    for c in clips:
        s = (c["g"] @ PCA_G)[:, None, :]
        fr = c["ok"].astype(np.float32)
        for st in range(0, len(s) - WIN + 1, STRIDE):
            sh, ok = spec_shape(s[st:st + WIN], P, hann)
            if not ok[0]:
                continue
            frac = fr[st:st + WIN].mean()
            (gA if frac >= 0.5 else gN if frac == 0.0 else []).append(sh[:, 0])
    gA = np.array(gA) if gA else np.zeros((0, nb))
    gN = np.array(gN) if gN else np.zeros((0, nb))

    # ----------------------------------------------------------------- PATCH
    pA, pN = [], []
    for c in clips:
        pp = (c["p"] @ PCA_P)                       # (T,196,16)
        gtw = c["gt"]                               # (T,14,14)
        for st in range(0, len(pp) - WIN + 1, STRIDE):
            sh, ok = spec_shape(pp[st:st + WIN], P, hann)
            if not ok.any():
                continue
            gfrac = gtw[st:st + WIN].mean(0).reshape(-1)   # (196,)
            a = (gfrac > 0.05) & ok
            n = (gfrac == 0.0) & ok
            if a.any():
                pA.append(sh[:, a].mean(1))
            if n.any():
                pN.append(sh[:, n].mean(1))
    pA = np.array(pA) if pA else np.zeros((0, nb))
    pN = np.array(pN) if pN else np.zeros((0, nb))

    def table(tag, A, N):
        print("\n=== %s ===" % tag)
        print("  n_windows: anomalous=%d  normal=%d" % (len(A), len(N)))
        print("  %-12s %9s %9s %8s %8s" % ("band", "normal", "anomal", "ratio", "z"))
        zs = []
        for i in range(nb):
            zz = zscore(N[:, i], A[:, i])
            zs.append(zz)
            print("  %-12s %9.4f %9.4f %8.3f %+8.2f"
                  % (band_label(i), N[:, i].mean(), A[:, i].mean(),
                     A[:, i].mean() / (N[:, i].mean() + 1e-12), zz))
        return zs

    zg = table("GLOBAL view (512-d pooled, same kind as XD)", gA, gN)
    zp = table("PATCH view (per-patch tokens, GT-selected)", pA, pN)

    # ------------------------------------------------------------ CALIBRATION
    # Inject a KNOWN ringing into the patch tokens of NORMAL windows. If band
    # "10-21 fr" lights up for a period-16 tone, the patch statistic can see a
    # rhythm -- which is what makes the real-data null above interpretable.
    print("\n=== CALIBRATION (patch view): inject a known tone into normal patches ===")
    # Use a TRAIN clip: Ped2 train is all normal, so the injected tone is the
    # only thing that can move the bands. (Using a test window would mix the
    # injection with the real anomaly.)
    cal = np.load(os.path.join(FEAT, "Train_Train001.npz"))["patch"].astype(np.float32)
    seg = cal[:WIN]                                    # (64,196,768)
    clean, ok = spec_shape((seg @ PCA_P), P, hann)
    base = clean[:, ok].mean(1)

    def inject(period, amp=1.0):
        x = seg.copy()
        n_fr = WIN
        # The direction MUST be built from the PCA basis (as the XD script does).
        # A random 768-d direction lands mostly OUTSIDE the 16-d subspace we
        # measure in, and the injected power then shrinks by ~50x -- which is
        # exactly what made the first run of this calibration read all zeros.
        u = (PCA_P @ RNG.normal(size=PCA_P.shape[1]).astype(np.float32)).astype(np.float32)
        u /= np.linalg.norm(u)
        proj = x @ u                                   # (64,196)
        # np.convolve is 1-D only; uniform_filter1d does the same moving
        # average along the time axis for every patch column at once.
        nat = np.std(proj - uniform_filter1d(proj, size=15, axis=0, mode="nearest"), axis=0)
        t = np.arange(n_fr, dtype=np.float32)
        tone = np.sin(2 * np.pi * t / period)
        tone /= np.sqrt(np.mean(tone ** 2)) + 1e-9
        s = x.shape
        delta = np.zeros_like(x)
        delta[:n_fr] = (amp * nat * tone[:, None])[:, :, None] * u[None, None, :]
        y = x + delta
        sh, o = spec_shape((y @ PCA_P), P, hann)
        return sh[:, o].mean(1) - base

    cal = {}
    print("  %-22s %s" % ("injection", "".join("%11s" % band_label(i) for i in range(nb))))
    print("  %-22s %s" % ("clean", "".join("%11.4f" % v for v in base)))
    print("  (natural fluctuation scale along the injected direction: %.3f)"
          % float(np.std(seg @ (PCA_P @ RNG.normal(size=PCA_P.shape[1]).astype(np.float32)))))
    for per in (16, 8):
        for amp in (0.5, 1.0, 2.0):
            d = inject(per, amp)
            cal.setdefault(per, {})[amp] = d
            print("  %-22s %s" % ("ring period %d %.1fx" % (per, amp),
                                  "".join("%+11.4f" % v for v in d)))

    print("\n=== VERDICT ===")
    mid = 1  # the 10-21 fr band
    print("  global mid-band (10-21 fr) z = %+.2f" % zg[mid])
    print("  patch  mid-band (10-21 fr) z = %+.2f" % zp[mid])
    print("  -> calibration must show a positive mid-band shift for period 16;")
    print("     if real data then shows z <= 0 in BOTH views, the XD finding is")
    print("     NOT a spatial-pooling artefact.")

    # Returned so figures/figs_ped2.py plots the SAME numbers instead of
    # recomputing them independently (two sources of truth drift apart).
    return dict(gA=gA, gN=gN, pA=pA, pN=pN, base=base, cal=cal,
                labels=[band_label(i) for i in range(nb)])


if __name__ == "__main__":
    main()
