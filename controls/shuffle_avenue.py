# -*- coding: utf-8 -*-
"""P5 + P7 replicated on CUHK Avenue (cross-dataset, second anomaly type).

WHY
    Everything in P5 / P5b / P6 / P7 was measured on UCSD Ped2, whose only
    anomaly is a PERSISTENT FOREIGN OBJECT (bike / cart / wheelchair).  Avenue's
    anomalies are motion-based instead -- running, thrown object, loitering,
    walking against the flow -- so it is a genuinely different regime.  Its
    frames are also JPEG (Ped2 is uncompressed TIFF), which makes it a
    real-data counterpart to the J3 JPEG round-trip proxy.

    Statistic is IDENTICAL to the Ped2 pipeline (imported from premise_test/spectral.py so
    there is exactly one implementation): WIN=64, STRIDE=16, detrend mean+linear
    trend, Hann, rfft, average power over 16 PCA directions, drop bin 0,
    normalise to sum 1, aggregate into the 5 bands.

OUTPUTS
    (a) PATCH view  : per-window, GT-selected anomalous vs normal patch tokens
    (b) GLOBAL view : same windows, but the pooled 512-d vector (the only view
                      that exists for XD, so this is the apples-to-apples one)
    (c) P7 shuffle  : permute the frame axis, recompute the between-class delta

    Written to results/avenue_spectral.txt and _avenue_cache.npz.

USAGE
    python controls/shuffle_avenue.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)


# The statistic exists in exactly one place (premise_test/spectral.py).  This
# script is run as `python <dir>/<name>.py`, so the directory holding that
# module is not on sys.path by default; add it rather than keep a second copy
# of the statistic here.
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _d in ("premise_test", "controls"):
    _p = _os.path.normpath(_os.path.join(_HERE, "..", _d))
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)

import spectral as S

FEAT = os.path.join(REPO, "feat_avenue_clip_vit_b16")
N_SHUFFLES = 5
RNG_SEED = 0


def load():
    def rd(pat, need_gt):
        out = []
        for f in sorted(glob.glob(pat)):
            d = np.load(f)
            if need_gt and "gt" not in d.files:
                continue
            p = d["patch"].astype(np.float32)
            g = d["global_feat"].astype(np.float32)
            if need_gt:
                out.append((p, g, d["gt"].astype(np.float32), d["frame_ok"]))
            else:
                out.append((p, g, None, None))
        return out
    return (rd(os.path.join(FEAT, "Test_*.npz"), True),
            rd(os.path.join(FEAT, "Train_*.npz"), False))


def class_shapes(X, gt, pca_or_none, P, hann, nb):
    """Per-WINDOW accumulation. X: (T, M, D) or (T, 1, D) for the global view.

    Returns (pN (n_win, nb), pA (n_win, nb)).
    """
    proj = X if pca_or_none is None else X @ pca_or_none
    pA, pN = [], []
    gtw = gt.reshape(len(proj), -1)
    for st in range(0, len(proj) - S.WIN + 1, S.STRIDE):
        sh, ok = S.spec_shape(proj[st:st + S.WIN], P, hann)
        if not ok.any():
            continue
        gf = gtw[st:st + S.WIN].mean(0)
        a = (gf > 0.05) & ok
        n = (gf == 0.0) & ok
        if a.any():
            pA.append(sh[:, a].mean(1))
        if n.any():
            pN.append(sh[:, n].mean(1))
    return (np.array(pN) if pN else np.zeros((0, nb)),
            np.array(pA) if pA else np.zeros((0, nb)))


def report(tag, pN, pA, fh):
    d = pA.mean(0) - pN.mean(0)
    print("\n=== %s ===" % tag, flush=True)
    print("  n_windows: normal=%d  anomalous=%d" % (len(pN), len(pA)), flush=True)
    print("  band            normal    anomal    ratio        z", flush=True)
    for i in range(len(d)):
        a, b = pN[:, i], pA[:, i]
        print("  %-14s  %.4f    %.4f    %.3f   %+7.2f"
              % (S.band_label(i), a.mean(), b.mean(),
                 b.mean() / (a.mean() + 1e-12), S.zscore(a, b)), flush=True)
        fh.write("  %-14s  %.4f    %.4f    %.3f   %+7.2f\n"
                 % (S.band_label(i), a.mean(), b.mean(),
                    b.mean() / (a.mean() + 1e-12), S.zscore(a, b)))
    print("  delta: %s" % "".join("%+9.4f" % v for v in d), flush=True)
    fh.write("  delta: %s\n" % "".join("%+9.4f" % v for v in d))
    return d


def main():
    tests, trains = load()
    print("[AV] %d test clips, %d train clips from %s"
          % (len(tests), len(trains), FEAT), flush=True)
    if not tests:
        print("[FATAL] no Avenue features -- run _avenue_extract_bb.py (not in this release) first",
              flush=True)
        sys.exit(2)

    P = S.detrender()
    hann = np.hanning(S.WIN).astype(np.float32)
    nb = len(S.BAND_EDGES) - 1

    # PCA on test[::5] + train[::5] patch tokens, exactly as on Ped2
    sub = np.concatenate([c[0][:, ::5, :].reshape(-1, c[0].shape[2]) for c in tests] +
                         [t[0][:, ::5, :].reshape(-1, t[0].shape[2]) for t in trains])
    pca = S.fit_pca(sub)
    print("[AV] PCA %s -> %s" % (sub.shape, pca.shape), flush=True)

    out_txt = os.path.join(REPO, "results", "avenue_spectral.txt")
    fh = open(out_txt, "w", encoding="utf-8")
    fh.write("P5 replicated on CUHK Avenue (CLIP ViT-B/16, %d test + %d train clips)\n"
             % (len(tests), len(trains)))

    # ---------------- (a) PATCH view ----------------
    pNs, pAs = [], []
    for Xp, Xg, gt, ok in tests:
        pN, pA = class_shapes(Xp, gt, pca, P, hann, nb)
        if len(pN):
            pNs.append(pN)
        if len(pA):
            pAs.append(pA)
    pN = np.concatenate(pNs); pA = np.concatenate(pAs)
    fh.write("\n=== PATCH view (per-patch tokens, GT-selected) ===\n")
    fh.write("n_windows: normal=%d anomalous=%d\n" % (len(pN), len(pA)))
    d_patch = report("PATCH view (16-d PCA of patch tokens)", pN, pA, fh)

    # ---------------- (b) GLOBAL view ----------------
    gNs, gAs = [], []
    for Xp, Xg, gt, ok in tests:
        g = Xg[:, None, :]                     # (T, 1, 512)
        gt1 = (gt.sum(axis=(1, 2)) > 0).astype(np.float32)[:, None]   # (T, 1)
        gN, gA = class_shapes(g, gt1, None, P, hann, nb)
        if len(gN):
            gNs.append(gN)
        if len(gA):
            gAs.append(gA)
    gN = np.concatenate(gNs); gA = np.concatenate(gAs)
    fh.write("\n=== GLOBAL view (512-d pooled, XD-comparable) ===\n")
    fh.write("n_windows: normal=%d anomalous=%d\n" % (len(gN), len(gA)))
    d_glob = report("GLOBAL view (512-d pooled vector)", gN, gA, fh)

    # ---------------- (c) P7 shuffle ----------------
    rng = np.random.default_rng(RNG_SEED)
    shuf_d = []
    for s in range(N_SHUFFLES):
        pNs2, pAs2 = [], []
        for Xp, Xg, gt, ok in tests:
            perm = rng.permutation(len(Xp))
            pN2, pA2 = class_shapes(Xp[perm], gt[perm], pca, P, hann, nb)
            if len(pN2):
                pNs2.append(pN2)
            if len(pA2):
                pAs2.append(pA2)
        pN2 = np.concatenate(pNs2); pA2 = np.concatenate(pAs2)
        shuf_d.append(pA2.mean(0) - pN2.mean(0))
        print("[AV] shuffle %d delta %s" % (s, "".join("%+9.4f" % v for v in shuf_d[-1])),
              flush=True)
    shuf_d = np.array(shuf_d)
    d_shuf = shuf_d.mean(0)
    cos = float(d_patch @ d_shuf /
                (np.linalg.norm(d_patch) * np.linalg.norm(d_shuf) + 1e-12))
    ratio = float(np.linalg.norm(d_shuf) / (np.linalg.norm(d_patch) + 1e-12))

    print("\n" + "=" * 74, flush=True)
    print("P7 on Avenue: does the spectral shape depend on temporal order?", flush=True)
    print("=" * 74, flush=True)
    print("  cos(d_orig, d_shuf)  = %.3f" % cos, flush=True)
    print("  |d_shuf| / |d_orig|  = %.3f" % ratio, flush=True)
    print("  Ped2 reference       : ratio = 0.026, cos = -0.16", flush=True)
    print("  XD  reference        : ratio = 0.015, cos = +0.59", flush=True)
    v = ("temporal structure -- 'temporal' framing OK" if ratio < 0.3
         else "largely marginal -- 'temporal' wording misleading" if ratio > 0.8
         else "inconclusive band")
    print("  verdict: %s" % v, flush=True)

    fh.write("\n=== P7 shuffle control (patch view) ===\n")
    fh.write("cos = %+0.3f\nratio |d_shuf|/|d_orig| = %0.3f\nverdict: %s\n"
             % (cos, ratio, v))
    fh.close()

    np.savez(os.path.join(REPO, "results", "_avenue_cache.npz"),
             d_patch=d_patch, d_glob=d_glob, d_shuf=d_shuf, shuf_d=shuf_d,
             cos=cos, ratio=ratio, n_norm=len(pN), n_anom=len(pA),
             n_norm_glob=len(gN), n_anom_glob=len(gA))
    print("\n[AV] wrote %s and results/_avenue_cache.npz" % out_txt, flush=True)


if __name__ == "__main__":
    main()
