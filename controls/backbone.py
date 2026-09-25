# -*- coding: utf-8 -*-
"""Backbone-robustness version of premise_test/spectral.py.

Same statistic (WIN=64, detrend, Hann, rfft, 16-d PCA, normalised to 1,
5 log-spaced bands) and the same injection calibration, run on several
backbones so the P1/P5 finding can be checked against:
    clip_vit_b16  CLIP objective, ViT arch, 14x14
    clip_rn50     CLIP objective, CNN arch,  7x7    (isolates architecture)
    inet_rn50     ImageNet objective, CNN,   7x7    (isolates pretraining)

The calibration direction is resampled with a FRESH seed for every backbone so
that no backbone gets an easier/harder injection direction by luck.
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
from scipy.ndimage import uniform_filter1d

BBS = ["clip_vit_b16", "clip_rn50", "inet_rn50"]


def analyze(featdir, tag):
    tests = sorted(glob.glob(os.path.join(featdir, "Test_*.npz")))
    trains = sorted(glob.glob(os.path.join(featdir, "Train_*.npz")))
    if not tests:
        return None
    d0 = np.load(tests[0])
    # the original CLIP ViT-B/16 dump predates the `grid` key
    grid = int(d0["grid"]) if "grid" in d0.files else 14

    clips = []
    for f in tests:
        d = np.load(f)
        clips.append(dict(g=d["global_feat"].astype(np.float32),
                          p=d["patch"].astype(np.float32),
                          gt=d["gt"].astype(np.float32), ok=d["frame_ok"]))
    tr_g = [np.load(f)["global_feat"].astype(np.float32) for f in trains]
    tr_p = [np.load(f)["patch"].astype(np.float32) for f in trains]

    PCA_G = S.fit_pca(np.concatenate([c["g"] for c in clips] + tr_g))
    sub = np.concatenate([c["p"][:, ::5, :].reshape(-1, c["p"].shape[2]) for c in clips] +
                         [x[:, ::5, :].reshape(-1, x.shape[2]) for x in tr_p])
    PCA_P = S.fit_pca(sub)
    # diagnostics for the limitation section: how much of the patch-token
    # variance the 16 measured directions actually carry
    sub_c = sub[::max(1, len(sub) // 200000)].astype(np.float64)
    sub_c -= sub_c.mean(0, keepdims=True)
    ev_all = np.linalg.eigvalsh(sub_c.T @ sub_c / len(sub_c))[::-1]
    evr16 = float(ev_all[:PCA_P.shape[1]].sum() / ev_all.sum())

    P = S.detrender()
    hann = np.hanning(S.WIN).astype(np.float32)
    nb = len(S.BAND_EDGES) - 1

    gA, gN = [], []
    for c in clips:
        s = (c["g"] @ PCA_G)[:, None, :]
        fr = c["ok"].astype(np.float32)
        for st in range(0, len(s) - S.WIN + 1, S.STRIDE):
            sh, ok = S.spec_shape(s[st:st + S.WIN], P, hann)
            if not ok[0]:
                continue
            frac = fr[st:st + S.WIN].mean()
            (gA if frac >= 0.5 else gN if frac == 0.0 else []).append(sh[:, 0])

    pA, pN = [], []
    for c in clips:
        pp = c["p"] @ PCA_P
        gtw = c["gt"].reshape(len(pp), -1)
        for st in range(0, len(pp) - S.WIN + 1, S.STRIDE):
            sh, ok = S.spec_shape(pp[st:st + S.WIN], P, hann)
            if not ok.any():
                continue
            gf = gtw[st:st + S.WIN].mean(0)
            a = (gf > 0.05) & ok
            n = (gf == 0.0) & ok
            if a.any():
                pA.append(sh[:, a].mean(1))
            if n.any():
                pN.append(sh[:, n].mean(1))

    gA = np.array(gA) if gA else np.zeros((0, nb))
    gN = np.array(gN) if gN else np.zeros((0, nb))
    pA = np.array(pA) if pA else np.zeros((0, nb))
    pN = np.array(pN) if pN else np.zeros((0, nb))

    # ---------------- calibration: inject a known tone into NORMAL patches ----
    rng = np.random.default_rng(0)          # fresh per backbone
    cal_src = np.load(trains[0])["patch"].astype(np.float32) if trains else clips[0]["p"]
    seg = cal_src[:S.WIN]
    clean, ok = S.spec_shape(seg @ PCA_P, P, hann)
    base = clean[:, ok].mean(1)

    nat_mean = 0.0

    def inject(period, amp):
        nonlocal nat_mean
        x = seg.copy()
        u = (PCA_P @ rng.normal(size=PCA_P.shape[1]).astype(np.float32)).astype(np.float32)
        u /= np.linalg.norm(u)
        proj = x @ u
        nat = np.std(proj - uniform_filter1d(proj, size=15, axis=0, mode="nearest"), axis=0)
        nat_mean = float(nat.mean())
        t = np.arange(S.WIN, dtype=np.float32)
        tone = np.sin(2 * np.pi * t / period)
        tone /= np.sqrt(np.mean(tone ** 2)) + 1e-9
        delta = np.zeros_like(x)
        delta[:] = (amp * nat * tone[:, None])[:, :, None] * u[None, None, :]
        sh, o = S.spec_shape((x + delta) @ PCA_P, P, hann)
        return sh[:, o].mean(1) - base

    cal = {per: {amp: inject(per, amp) for amp in (0.5, 1.0, 2.0)} for per in (16, 8)}

    # ---- POWER-MATCHED calibration -------------------------------------
    # The `nat`-scaled injection above is not comparable across backbones:
    # `nat` is the fluctuation along ONE direction, while the spectrum is
    # averaged over 16 dims, so any backbone whose window power is dominated
    # by slow drift dilutes the tone by a backbone-specific factor. Scaling
    # the tone to a fixed FRACTION of the window's own detrended power removes
    # that factor and makes the calibration comparable everywhere.
    w = seg @ PCA_P                                    # (64, M, 16)
    wd = np.einsum("ij,jkl->ikl", P, w)                # detrend along time
    pwr_dim = float((wd ** 2).mean())                  # power per (patch, dim)
    pwr_tot = pwr_dim * PCA_P.shape[1]                 # summed over the 16 dims

    def inject_pw(period, frac):
        x = seg.copy()
        u = (PCA_P @ rng.normal(size=PCA_P.shape[1]).astype(np.float32)).astype(np.float32)
        u /= np.linalg.norm(u)
        t = np.arange(S.WIN, dtype=np.float32)
        tone = np.sin(2 * np.pi * t / period)
        tone /= np.sqrt(np.mean(tone ** 2)) + 1e-9
        # injected power (summed over dims) = s^2 ; want s^2 = frac * pwr_tot
        s = float(np.sqrt(frac * pwr_tot))
        delta = np.zeros_like(x)
        delta[:] = (s * tone[:, None])[:, :, None] * u[None, None, :]
        sh, o = S.spec_shape((x + delta) @ PCA_P, P, hann)
        return sh[:, o].mean(1) - base

    cal_pw = {per: {f: inject_pw(per, f) for f in (0.10, 0.25, 0.50)} for per in (16, 8)}
    # how much of the window's detrended power sits in the slowest band:
    # a large value means slow drift dominates and dilutes any injected tone
    slow_share = float(base[0])

    return dict(tag=tag, grid=grid, gA=gA, gN=gN, pA=pA, pN=pN, base=base, cal=cal,
                cal_pw=cal_pw, pwr_tot=pwr_tot,
                evr16=evr16, nat=nat_mean, slow_share=slow_share,
                labels=[S.band_label(i) for i in range(nb)])


def report(R):
    nb = len(R["labels"])
    print("\n" + "=" * 74)
    print("%s   (grid %dx%d)" % (R["tag"], R["grid"], R["grid"]))
    print("=" * 74)
    print("  %-11s | %-24s | %-24s" % ("band", "GLOBAL  n=%d/%d" % (len(R["gA"]), len(R["gN"])),
                                       "PATCH   n=%d/%d" % (len(R["pA"]), len(R["pN"]))))
    print("  %-11s | %8s %7s %7s | %8s %7s %7s" %
          ("", "normal", "anom", "z", "normal", "anom", "z"))
    for i in range(nb):
        zz_g = S.zscore(R["gN"][:, i], R["gA"][:, i])
        zz_p = S.zscore(R["pN"][:, i], R["pA"][:, i])
        print("  %-11s | %8.4f %7.4f %+7.2f | %8.4f %7.4f %+7.2f"
              % (R["labels"][i], R["gN"][:, i].mean(), R["gA"][:, i].mean(), zz_g,
                 R["pN"][:, i].mean(), R["pA"][:, i].mean(), zz_p))
    print("  calibration, nat-scaled (amp 1.0x):")
    print("    %-11s %s" % ("", "".join("%11s" % l for l in R["labels"])))
    for per in (16, 8):
        print("    period %-3d   %s" % (per, "".join("%+11.4f" % v for v in R["cal"][per][1.0])))
    print("  calibration, POWER-MATCHED (tone carries `frac` of window detrended power):")
    print("    %-11s %s" % ("", "".join("%11s" % l for l in R["labels"])))
    for per in (16, 8):
        for f in (0.10, 0.25, 0.50):
            print("    p%-2d f=%.2f %s" % (per, f, "".join("%+11.4f" % v for v in R["cal_pw"][per][f])))
    print("  slow-band share of window power: %.3f   window detrended power: %.4g"
          % (R["slow_share"], R["pwr_tot"]))


def main():
    res = []
    for bb in BBS:
        d = os.path.join(REPO, "ped2_feat_%s" % bb)
        if not os.path.isdir(d) and bb == "clip_vit_b16":
            d = os.path.join(REPO, "ped2_feat")   # original dump name
        R = analyze(d, bb)
        if R is None:
            print("[skip] %s (no features)" % bb)
            continue
        report(R)
        res.append(R)

    print("\n" + "=" * 74)
    print("SUMMARY - the band a behavioural rhythm would occupy (10-21 fr)")
    print("=" * 74)
    print("  %-14s %8s %8s %11s %11s %10s %11s" %
          ("backbone", "GLOB z", "PATCH z", "cal nat", "real dMid", "te-nat", "te-POWER"))
    for R in res:
        i = R["labels"].index("10-21 fr")
        z_p = S.zscore(R["pN"][:, i], R["pA"][:, i])
        z_g = S.zscore(R["gN"][:, i], R["gA"][:, i])
        cal = R["cal"][16][1.0][i]
        calp = R["cal_pw"][16][0.25][i]
        dmid = R["pA"][:, i].mean() - R["pN"][:, i].mean()
        print("  %-14s %8.2f %8.2f %11.4f %11.4f %10.2f %11.2f"
              % (R["tag"], z_g, z_p, cal, dmid,
                 dmid / (cal + 1e-12), dmid / (calp + 1e-12)))
    print()
    print("  POWER-MATCHED calibration (25%% tone) per backbone, mid band:")
    for R in res:
        i = R["labels"].index("10-21 fr")
        print("    %-14s +%.4f" % (R["tag"], R["cal_pw"][16][0.25][i]))
    print()
    print("  te-POWER = real mid-band shift / response to a 25%-power pure tone.")
    print("  If this is clearly negative for EVERY backbone, the 'no mid-band")
    print("  rhythm' finding is not a backbone artefact -- including ImageNet RN50.")
    print()
    print("  tone-equiv = real mid-band shift / response to a 1x pure tone.")
    print("  Negative means the data moves AWAY from tone-like. |tone-equiv| >= 1")
    print("  means the deviation is at least as large as a full-amplitude tone.")
    print("  PCA16 = variance fraction carried by the 16 measured directions.")
    print("\n  Minimum detectable tone: the tone power (as a fraction of the")
    print("  window's own detrended power) that would move the mid band by as much")
    print("  as the REAL anomalous-vs-normal shift did. Smaller = more sensitive.")
    print("  %-14s %12s %12s %14s" % ("backbone", "real dMid", "cal@25%", "min tone pow"))
    for R in res:
        i = R["labels"].index("10-21 fr")
        d = R["pA"][:, i].mean() - R["pN"][:, i].mean()
        c25 = R["cal_pw"][16][0.25][i]
        print("  %-14s %12.4f %12.4f %13.2f%%" % (R["tag"], d, c25,
                                                  25.0 * abs(d) / (c25 + 1e-12)))
    print("\n  Read: calibration must be POSITIVE (statistic sees the tone).")
    print("  If PATCH z stays clearly NEGATIVE for every backbone, the finding is")
    print("  not an artefact of one backbone's architecture or pretraining.")


if __name__ == "__main__":
    main()
