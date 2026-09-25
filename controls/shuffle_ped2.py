# -*- coding: utf-8 -*-
"""Experiment S (corrected) -- does the spectral shape depend on temporal order?

Redo of _ped2_shuffle.py (not in this release), fixing two things:

  1. It re-encoded every clip from raw frames (16 min) even though the patch
     features are already on disk in ped2_feat/.  Shuffling can be done on the
     FEATURES (they are per-frame and deterministic), so no re-encoding is
     needed -- this version is ~instant.

  2. Its d_orig did not reproduce the published P5 delta (band 0 even flipped
     sign) because it fitted PCA on test data only and averaged per CLIP
     instead of per WINDOW.  This version reproduces the P5 pipeline exactly
     (PCA on test+train patches, per-window accumulation) and ASSERTIS that
     d_orig matches results/backbone_robustness.txt before trusting the
     shuffle comparison.

The question: if we destroy the temporal order of frames within each clip
(keeping the same marginal distribution of per-frame features), does the
anomaly-vs-normal spectral-shape difference survive?

PRE-REGISTERED READING
    |d_shuf| / |d_orig| < 0.3  ->  shuffling mostly preserves the difference
        -> the "temporal dynamics" framing is justified.
    |d_shuf| / |d_orig| > 0.8  ->  the difference is largely a MARGINAL
        distribution property; the paper's "temporal" wording is misleading
        and must be reframed.
    cos(d_shuf, d_orig) > 0.7  ->  shape preserved (robust)
    cos(d_shuf, d_orig) < 0    ->  shuffling reverses it (bad)
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

FEAT = os.environ.get("PED2_FEAT", r"D:\program\waveClipVad\ped2_feat")
N_SHUFFLES = 5
RNG_SEED = 0


def load():
    """Return (tests, trains).

    tests  : [(patch (T,196,768), gt (T,14,14), frame_ok (T,)), ...]
    trains : [(patch, None, None), ...]   -- Train clips are all-normal and
             have NO gt, but their patches still feed the PCA basis, exactly
             as controls/backbone.py's analyze does.
    """
    def rd(pat, need_gt):
        out = []
        for f in sorted(glob.glob(pat)):
            d = np.load(f)
            if need_gt and "gt" not in d.files:
                continue
            p = d["patch"].astype(np.float32)
            if need_gt:
                out.append((p, d["gt"].astype(np.float32), d["frame_ok"]))
            else:
                out.append((p, None, None))
        return out
    return (rd(os.path.join(FEAT, "Test_*.npz"), True),
            rd(os.path.join(FEAT, "Train_*.npz"), False))


def class_shapes(Xp, gt, pca, P, hann, nb):
    """Per-WINDOW accumulation, exactly as controls/backbone.py's analyze does.

    Returns (pN (n_win, nb), pA (n_win, nb)).
    """
    proj = Xp @ pca
    pA, pN = [], []
    gtw = gt.reshape(len(Xp), -1)
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


def main():
    tests, trains = load()
    print("[S2] %d test clips, %d train clips" % (len(tests), len(trains)))

    P = S.detrender()
    hann = np.hanning(S.WIN).astype(np.float32)
    nb = len(S.BAND_EDGES) - 1

    # ---- PCA exactly as the P5 / backbone script does: test[::5] + train[::5]
    sub = np.concatenate([c[0][:, ::5, :].reshape(-1, c[0].shape[2]) for c in tests] +
                         [t[0][:, ::5, :].reshape(-1, t[0].shape[2]) for t in trains])
    pca = S.fit_pca(sub)
    print("[S2] PCA %s -> %s" % (sub.shape, pca.shape))

    # ---- ORIGINAL (unshuffled) : must reproduce the published table --------
    pNs, pAs = [], []
    for Xp, gt, ok in tests:
        pN, pA = class_shapes(Xp, gt, pca, P, hann, nb)
        if len(pN):
            pNs.append(pN)
        if len(pA):
            pAs.append(pA)
    pN = np.concatenate(pNs); pA = np.concatenate(pAs)
    d_orig = pA.mean(0) - pN.mean(0)

    # published reference (results/backbone_robustness.txt, clip_vit_b16)
    ref = np.array([0.6267, 0.2037, 0.0957, 0.0527, 0.0212]) - \
          np.array([0.4999, 0.2290, 0.1356, 0.0930, 0.0425])
    print("\n[S2] d_orig          %s" % "".join("%+9.4f" % v for v in d_orig))
    print("[S2] published P5    %s" % "".join("%+9.4f" % v for v in ref))
    err = np.abs(d_orig - ref).max()
    print("[S2] max abs diff vs published = %.4f" % err)
    if err > 0.005:
        print("[FATAL] pipeline does not reproduce P5 -- aborting", flush=True)
        sys.exit(1)
    print("[S2] OK: pipeline reproduces the published delta\n")

    # ---- SHUFFLED : permute the FRAME axis (and GT) within each clip ------
    rng = np.random.default_rng(RNG_SEED)
    shuf_d = []
    for s in range(N_SHUFFLES):
        pNs2, pAs2 = [], []
        for Xp, gt, ok in tests:
            perm = rng.permutation(len(Xp))
            pN2, pA2 = class_shapes(Xp[perm], gt[perm], pca, P, hann, nb)
            if len(pN2):
                pNs2.append(pN2)
            if len(pA2):
                pAs2.append(pA2)
        pN2 = np.concatenate(pNs2); pA2 = np.concatenate(pAs2)
        shuf_d.append(pA2.mean(0) - pN2.mean(0))
        print("[S2] shuffle %d delta %s" % (s, "".join("%+9.4f" % v for v in shuf_d[-1])))
    shuf_d = np.array(shuf_d)
    d_shuf = shuf_d.mean(0)

    cos = float(d_orig @ d_shuf /
                (np.linalg.norm(d_orig) * np.linalg.norm(d_shuf) + 1e-12))
    ratio = float(np.linalg.norm(d_shuf) / (np.linalg.norm(d_orig) + 1e-12))
    per_shuffle_cos = [float(d_orig @ d / (np.linalg.norm(d_orig) * np.linalg.norm(d) + 1e-12))
                       for d in shuf_d]

    print("\n" + "=" * 74)
    print("S (corrected): does the spectral shape depend on temporal order?")
    print("=" * 74)
    print("  cos(d_orig, d_shuf)  = %.3f   (per-shuffle: %s)"
          % (cos, " ".join("%.2f" % c for c in per_shuffle_cos)))
    print("  |d_shuf| / |d_orig|  = %.3f" % ratio)
    if ratio < 0.3:
        v = "difference largely PRESERVED -> 'temporal' framing justified"
    elif ratio > 0.8:
        v = ("difference largely INDISTINGUISHABLE from a marginal-distribution "
             "property -> 'temporal' wording is misleading")
    else:
        v = "shuffling partially reduces the signal"
    print("  verdict: %s" % v)
    if cos < 0:
        print("  WARNING: shuffling REVERSES the direction.")

    np.savez(os.path.join(REPO, "results", "_shuffle2_cache.npz"),
             d_orig=d_orig, d_shuf=d_shuf, shuf_d=shuf_d,
             cos=cos, ratio=ratio)
    print("\n[S2] cached -> results/_shuffle2_cache.npz")


if __name__ == "__main__":
    main()
