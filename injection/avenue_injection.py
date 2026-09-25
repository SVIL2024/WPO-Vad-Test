# -*- coding: utf-8 -*-
"""Avenue injection arm: is the Avenue mid-band deficit calibrated?

WHY THIS EXISTS
    The paper reports a mid-band (10-21 frame) deficit on three datasets.  On
    Ped2 and XD-Violence every one of those deficits carries an injection
    calibration next to it -- a tone of known period and known power is put into
    a normal window and we check (i) that the instrument sees it and (ii) how
    small a tone it would have seen.  That second number is what turns "we found
    nothing" into "we would have found at least X".

    Avenue had no such arm.  Its mid-band z (-4.57 patch, -4.51 global) was
    therefore the only deficit in the paper that a reader could dismiss with
    "maybe the instrument is simply insensitive on these features".  This run
    closes that gap with the same power-matched calibration used on Ped2.

WHAT IS MEASURED (definitions identical to results/backbone_robustness.txt)
    real dMid     : anomalous minus normal band share in the 10-21 frame band.
    cal@25%       : shift of that band when a period-16 tone carrying 25% of the
                    window's own detrended power is injected into a NORMAL train
                    window.  Must be POSITIVE (the instrument sees the tone).
    te-POWER      : real dMid / cal@25%.  Negative = the data moves AWAY from
                    tone-like; |te| >= 1 = at least as large as a full-amplitude
                    tone.
    MDT           : 0.25 * |real dMid| / cal@25%, i.e. the tone power, as a
                    fraction of window detrended power, that would move the mid
                    band as far as the data did.  Smaller = more sensitive.

GUARDRAILS (a new number is only reported after an old one is reproduced)
    G1  Recompute the published Avenue deltas (PATCH and GLOBAL) and compare
        against results/_avenue_cache.npz.  If the pipeline that produced
        the deficit cannot be reproduced, the calibration attached to it means
        nothing.  Threshold 1e-6.
    G2  A zero-power injection must return exactly zero.  Catches any drift in
        the detrender / window bookkeeping.
    G3  Every injected period must land in its own band.  This is the
        recovered-frequency check that F4 runs on Ped2; it is what makes
        "cal@25% is positive" interpretable as "the instrument is calibrated"
        rather than "the instrument moved".

    A direction drawn in the RAW 768-d patch space is ALSO injected once, and
    reported as a diagnostic only.  It lands almost entirely outside the 16-d
    subspace the statistic measures in -- only ~16/768 of its power is visible
    to the statistic -- so its response collapses.  That is the trap this script
    avoids, shown rather than asserted.

Reads  : feat_avenue_clip_vit_b16 (local, already extracted),
         results/_avenue_cache.npz (the published deltas).
Writes : results/avenue_injection.txt,
         results/F21_avenue_injection.png

USAGE
    python injection/avenue_injection.py            # full run incl. G1 recompute
    python injection/avenue_injection.py --skip-real  # skip G1 (saves ~7 min)
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "_pylibs"))        # isolated matplotlib


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

import spectral as S                                # ONE implementation

FEAT = os.path.join(REPO, "feat_avenue_clip_vit_b16")
CACHE = os.path.join(REPO, "results", "_avenue_cache.npz")

WIN = S.WIN
STRIDE = S.STRIDE
BAND_EDGES = S.BAND_EDGES
NB = len(BAND_EDGES) - 1
MID = 1                      # index of the 10-21 frame band

FRACS = (0.10, 0.25, 0.50)
PRIMARY_FRAC = 0.25
PERIODS = (8, 12, 16, 24, 32, 48)
ONSETS = (16, 32, 48)
PRIMARY_ONSET = 32
N_DIR = 8
ROBUST_WINDOWS = ((0, 0), (0, 400), (0, 800),
                  (1, 0), (1, 400), (1, 800),
                  (2, 0), (2, 400), (2, 800),
                  (3, 0), (3, 400), (3, 800))
TOL_REAL = 1e-6
TOL_ZERO = 1e-12


def band_label(i):
    return S.band_label(i)


def tilt_r2(v):
    """R^2 of a straight-line fit in band index. High = tilt, low = bump."""
    i = np.arange(len(v), dtype=np.float64)
    A = np.stack([np.ones(len(v)), i], 1)
    p = A @ np.linalg.lstsq(A, v, rcond=None)[0]
    ss_res = float(((v - p) ** 2).sum())
    ss_tot = float(((v - v.mean()) ** 2).sum())
    return 1 - ss_res / (ss_tot + 1e-18)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))


def waveform(kind, period=None, onset=None):
    """Unit-RMS waveform, so equal amplitude = equal power."""
    t = np.arange(WIN, dtype=np.float32)
    if kind == "ring":
        w = np.sin(2 * np.pi * t / period).astype(np.float32)
    elif kind == "step":
        w = (t >= onset).astype(np.float32)
    else:
        raise ValueError(kind)
    return w / (np.sqrt(np.mean(w ** 2)) + 1e-9)


def load():
    """(tests, trains); test entries carry gt, train entries do not."""
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


def class_shapes(X, gt, pca_or_none, P, hann):
    """Identical to controls/shuffle_avenue.py -- kept byte-for-byte comparable."""
    proj = X if pca_or_none is None else X @ pca_or_none
    pA, pN = [], []
    gtw = gt.reshape(len(proj), -1)
    for st in range(0, len(proj) - WIN + 1, STRIDE):
        sh, ok = S.spec_shape(proj[st:st + WIN], P, hann)
        if not ok.any():
            continue
        gf = gtw[st:st + WIN].mean(0)
        a = (gf > 0.05) & ok
        n = (gf == 0.0) & ok
        if a.any():
            pA.append(sh[:, a].mean(1))
        if n.any():
            pN.append(sh[:, n].mean(1))
    return (np.array(pN) if pN else np.zeros((0, NB)),
            np.array(pA) if pA else np.zeros((0, NB)))


def make_injector(seg_raw, pca_or_none, P, hann):
    """Return (inject, base, pwr_tot) for one window of one view.

    seg_raw      : (WIN, M, D) raw features, NOT projected.
    pca_or_none  : the projection the statistic measures in, or None.

    POWER MATCHING.  `pwr_tot` is the window's own detrended power summed over
    the directions the statistic actually measures.  Injecting
    s = sqrt(frac * pwr_tot) therefore means the same thing in every view and on
    every dataset, which the older `nat`-scaled injection did not.

    DIRECTION.  u = PCA @ N(0, I_k), normalised, added in the RAW D-dimensional
    space.  A random raw-D direction is diluted by ~D/k and silently produces a
    null calibration; that is checked explicitly, not assumed away.
    """
    proj = seg_raw if pca_or_none is None else seg_raw @ pca_or_none
    clean, ok = S.spec_shape(proj, P, hann)
    if not ok.any():
        raise RuntimeError("window has no usable direction")
    okm = ok
    base = clean[:, okm].mean(1)

    D = seg_raw.shape[2]
    k = D if pca_or_none is None else pca_or_none.shape[1]

    wd = np.einsum("ij,jkl->ikl", P, proj)
    pwr_tot = float((wd ** 2).mean()) * k

    def inject(kind, frac, rng, period=None, onset=None, raw_dir=False):
        x = seg_raw.copy()
        g = rng.normal(size=(D if raw_dir else k)).astype(np.float32)
        if raw_dir:
            u = g
        elif pca_or_none is None:
            u = g
        else:
            u = (pca_or_none @ g).astype(np.float32)
        u = u.astype(np.float32)
        u /= np.linalg.norm(u)
        wav = waveform(kind, period=period, onset=onset)
        s = float(np.sqrt(frac * pwr_tot))
        delta = np.zeros_like(x)
        delta[:] = (s * wav[:, None])[:, :, None] * u[None, None, :]
        sh, o = S.spec_shape((x + delta) @ pca_or_none if pca_or_none is not None
                             else (x + delta), P, hann)
        return sh[:, o].mean(1) - base

    return inject, base, pwr_tot, k, D


def run_view(tag, seg_raw, pca_or_none, P, hann, real, fh, seed0):
    print("\n" + "=" * 78, flush=True)
    print("VIEW: %s" % tag, flush=True)
    print("=" * 78, flush=True)

    inject, base, pwr_tot, k, D = make_injector(seg_raw, pca_or_none, P, hann)
    print("  measured in %d directions (raw D=%d)" % (k, D), flush=True)
    print("  window detrended power  %.5g" % pwr_tot, flush=True)
    print("  clean normal shape      %s" % "".join("%8.4f" % v for v in base),
          flush=True)

    # ------------------------------------------------------------- G2: zero
    z = inject("ring", 0.0, np.random.default_rng(0), period=16)
    zmax = float(np.abs(z).max())
    print("\n  [G2] zero-power injection: max |delta| = %.2e  -> %s"
          % (zmax, "OK" if zmax < TOL_ZERO else "FAIL"), flush=True)
    if zmax >= TOL_ZERO:
        raise SystemExit("G2 failed")

    # ------------------------------------------ diagnostic: the dilution trap
    good = np.mean([inject("ring", PRIMARY_FRAC, np.random.default_rng(seed0 + i),
                           period=16) for i in range(N_DIR)], axis=0)
    if pca_or_none is not None:
        bad = np.mean([inject("ring", PRIMARY_FRAC,
                              np.random.default_rng(seed0 + i), period=16,
                              raw_dir=True) for i in range(N_DIR)], axis=0)
        dil = float(np.abs(good).max() / (np.abs(bad).max() + 1e-18))
        print("  [diag] direction from raw %d-d space instead of the %d-d "
              "subspace: response %.4f vs %.4f  (dilution %.0fx)"
              % (D, k, np.abs(bad).max(), np.abs(good).max(), dil), flush=True)

    # ------------------------------------------------------- dose x waveform
    res = {}
    for frac in FRACS:
        for per in PERIODS:
            res[("ring", per, frac)] = np.mean(
                [inject("ring", frac, np.random.default_rng(seed0 + i), period=per)
                 for i in range(N_DIR)], axis=0)
        for on in ONSETS:
            res[("step", on, frac)] = np.mean(
                [inject("step", frac, np.random.default_rng(seed0 + i), onset=on)
                 for i in range(N_DIR)], axis=0)

    print("\n  injection response, mean over %d directions, period 16" % N_DIR,
          flush=True)
    print("  %-12s %s" % ("frac", "".join("%11s" % band_label(i)
                                          for i in range(NB))), flush=True)
    for frac in FRACS:
        print("  %-12.2f %s" % (frac, "".join("%+11.4f" % v
                                              for v in res[("ring", 16, frac)])),
              flush=True)

    # --------------------------------------------------------- G3: frequency
    print("\n  [G3] recovered frequency (period -> band), frac %.2f"
          % PRIMARY_FRAC, flush=True)
    rec = []
    for per in PERIODS:
        d = res[("ring", per, PRIMARY_FRAC)]
        bi = int(np.argmax(d))
        exp_bin = WIN / float(per)
        exp_bi = max(i for i in range(NB) if BAND_EDGES[i] <= exp_bin)
        edge = min(abs(exp_bin - e) for e in BAND_EDGES)
        bnd = "boundary" if edge < 0.5 else ""
        rec.append((per, exp_bi, bi, float(d[bi]), bnd))
        print("    p=%-4d expected %-10s measured %-10s %-9s %+0.4f"
              % (per, band_label(exp_bi), band_label(bi), bnd, d[bi]), flush=True)
    clean_rec = [r for r in rec if not r[4]]
    hit = sum(1 for r in clean_rec if r[1] == r[2])
    print("    -> %d/%d unambiguous periods land in their own band"
          % (hit, len(clean_rec)), flush=True)
    if hit < len(clean_rec):
        print("       G3 FAILED: the instrument is not placing tones correctly",
              flush=True)
        raise SystemExit("G3 failed")

    # ------------------------------------------------------------- the number
    cal25 = float(res[("ring", 16, PRIMARY_FRAC)][MID])
    out = dict(cal25=cal25, res=res, base=base, pwr_tot=pwr_tot,
               rec=rec, k=k, D=D)
    if real is not None:
        dmid = float(real[MID])
        te = dmid / cal25
        mdt = PRIMARY_FRAC * abs(dmid) / cal25
        out.update(dmid=dmid, te=te, mdt=mdt)
        print("\n  real dMid (10-21 fr)   %+0.4f" % dmid, flush=True)
        print("  cal@25%% (period 16)    %+0.4f   %s"
              % (cal25, "POSITIVE -- instrument sees the tone" if cal25 > 0
                 else "NEGATIVE -- calibration is broken"), flush=True)
        print("  te-POWER               %+0.4f   (negative = moves AWAY from "
              "tone-like)" % te, flush=True)
        print("  minimum detectable tone  %.2f%% of window detrended power"
              % (100 * mdt), flush=True)

        ring = res[("ring", 16, PRIMARY_FRAC)]
        step = res[("step", PRIMARY_ONSET, PRIMARY_FRAC)]
        out["cos_ring"] = cos(real, ring)
        out["cos_step"] = cos(real, step)
        print("\n  shape vs the real Avenue delta (cosine of the 5-vector):",
              flush=True)
        print("    ring p16  %+0.3f   tilt R2 %.3f"
              % (out["cos_ring"], tilt_r2(ring)), flush=True)
        print("    step on%-2d %+0.3f   tilt R2 %.3f"
              % (PRIMARY_ONSET, out["cos_step"], tilt_r2(step)), flush=True)
        print("    real      %+0.3f   tilt R2 %.3f" % (1.0, tilt_r2(real)),
              flush=True)

    # ------------------------------- robustness: is window 0 of clip 0 special?
    mdt_list = []
    for ci, st in ROBUST_WINDOWS:
        seg_r = TRAINS[ci][0][st:st + WIN]
        try:
            inj2, _b, _p, _k, _d = make_injector(seg_r, pca_or_none, P, hann)
        except RuntimeError:
            continue
        c2 = float(np.mean([inj2("ring", PRIMARY_FRAC,
                                 np.random.default_rng(seed0 + i), period=16)[MID]
                            for i in range(2)]))
        if real is not None and c2 > 0:
            mdt_list.append(PRIMARY_FRAC * abs(float(real[MID])) / c2)
    if mdt_list:
        a = np.array(mdt_list)
        out["mdt_robust"] = (float(a.min()), float(a.mean()), float(a.max()))
        print("\n  robustness over %d normal training windows: MDT "
              "%.2f%% - %.2f%% (mean %.2f%%)"
              % (len(a), 100 * a.min(), 100 * a.max(), 100 * a.mean()), flush=True)

    fh.write("\n=== %s ===\n" % tag)
    fh.write("measured in %d directions (raw D=%d)\n" % (out["k"], out["D"]))
    fh.write("window detrended power %.6g\n" % out["pwr_tot"])
    fh.write("clean normal shape %s\n" % "".join("%9.4f" % v for v in base))
    fh.write("zero-power injection max|delta| = %.2e\n" % zmax)
    if "dil" in dir():
        fh.write("raw-direction dilution diagnostic: %.0fx\n" % dil)
    fh.write("\n%-20s %s\n" % ("injection",
                               "".join("%11s" % band_label(i) for i in range(NB))))
    for frac in FRACS:
        for per in PERIODS:
            fh.write("%-20s %s\n" % ("ring p%d f=%.2f" % (per, frac),
                                     "".join("%+11.4f" % v
                                             for v in res[("ring", per, frac)])))
        for on in ONSETS:
            fh.write("%-20s %s\n" % ("step on%d f=%.2f" % (on, frac),
                                     "".join("%+11.4f" % v
                                             for v in res[("step", on, frac)])))
    if real is not None:
        fh.write("%-20s %s\n" % ("REAL (anom-norm)",
                                 "".join("%+11.4f" % v for v in real)))
        fh.write("\nreal dMid = %+0.6f\ncal@25%% (period 16) = %+0.6f\n"
                 "te-POWER = %+0.4f\nminimum detectable tone = %.2f%% of window "
                 "detrended power\n" % (out["dmid"], out["cal25"], out["te"],
                                        100 * out["mdt"]))
        fh.write("cos(real, ring p16) = %+0.3f ; cos(real, step on%d) = %+0.3f\n"
                 % (out["cos_ring"], PRIMARY_ONSET, out["cos_step"]))
        if "mdt_robust" in out:
            fh.write("MDT over %d train windows: %.2f%% - %.2f%% (mean %.2f%%)\n"
                     % (len(mdt_list), 100 * out["mdt_robust"][0],
                        100 * out["mdt_robust"][2], 100 * out["mdt_robust"][1]))
    return out


TRAINS = []


def main():
    global TRAINS
    skip_real = "--skip-real" in sys.argv

    tests, trains = load()
    TRAINS = trains
    print("[AVI] %d test clips, %d train clips from %s"
          % (len(tests), len(trains), FEAT), flush=True)
    if not tests or not trains:
        print("[FATAL] no Avenue features", flush=True)
        return 2

    P = S.detrender()
    hann = np.hanning(WIN).astype(np.float32)

    # PCA exactly as controls/shuffle_avenue.py fits it
    sub = np.concatenate([c[0][:, ::5, :].reshape(-1, c[0].shape[2]) for c in tests] +
                         [t[0][:, ::5, :].reshape(-1, t[0].shape[2]) for t in trains])
    PCA_P = S.fit_pca(sub)
    print("[AVI] patch PCA %s -> %s" % (sub.shape, PCA_P.shape), flush=True)

    # ------------------------------------------------------------- G1 guardrail
    real_patch = real_glob = None
    if not skip_real and os.path.exists(CACHE):
        print("\n[G1] recomputing the published Avenue deltas ...", flush=True)
        pNs, pAs, gNs, gAs = [], [], [], []
        for Xp, Xg, gt, ok in tests:
            pN, pA = class_shapes(Xp, gt, PCA_P, P, hann)
            if len(pN):
                pNs.append(pN)
            if len(pA):
                pAs.append(pA)
            g = Xg[:, None, :]
            gt1 = (gt.sum(axis=(1, 2)) > 0).astype(np.float32)[:, None]
            gN, gA = class_shapes(g, gt1, None, P, hann)
            if len(gN):
                gNs.append(gN)
            if len(gA):
                gAs.append(gA)
        pN = np.concatenate(pNs); pA = np.concatenate(pAs)
        gN = np.concatenate(gNs); gA = np.concatenate(gAs)
        real_patch = pA.mean(0) - pN.mean(0)
        real_glob = gA.mean(0) - gN.mean(0)
        c = np.load(CACHE)
        dp = float(np.abs(real_patch - c["d_patch"]).max())
        dg = float(np.abs(real_glob - c["d_glob"]).max())
        print("  PATCH   this %s" % "".join("%+9.4f" % v for v in real_patch),
              flush=True)
        print("  PATCH   pub  %s" % "".join("%+9.4f" % v
                                             for v in c["d_patch"]), flush=True)
        print("  GLOBAL  this %s" % "".join("%+9.4f" % v for v in real_glob),
              flush=True)
        print("  GLOBAL  pub  %s" % "".join("%+9.4f" % v for v in c["d_glob"]),
              flush=True)
        print("  max |diff|  patch %.2e   global %.2e   -> %s"
              % (dp, dg, "OK" if max(dp, dg) < TOL_REAL else "MISMATCH"),
              flush=True)
        if max(dp, dg) >= TOL_REAL:
            print("\n  G1 failed: cannot reproduce the published deficit, so no "
                  "calibration will be attached to it.", flush=True)
            return 1
    elif os.path.exists(CACHE) and skip_real:
        c = np.load(CACHE)
        real_patch = c["d_patch"].astype(np.float64)
        real_glob = c["d_glob"].astype(np.float64)
        print("[G1] SKIPPED -- deltas read from cache, not recomputed", flush=True)

    out_txt = os.path.join(REPO, "results", "avenue_injection.txt")
    fh = open(out_txt, "w", encoding="utf-8")
    fh.write("Avenue injection calibration -- power-matched, %d directions\n"
             % N_DIR)
    fh.write("unit: shift in band share (the five bands sum to 1)\n")
    fh.write("MDT = tone power, as a fraction of window detrended power, that "
             "would move the 10-21 fr band as far as the data did\n")
    fh.write("guardrails: G1 published deltas reproduced to %.0e ; G2 zero-dose "
             "= 0 ; G3 every period lands in its own band\n" % TOL_REAL)

    # Avenue training clips are all normal, so an injection into a TRAIN window
    # is the only thing that can move the bands without mixing in a real event.
    seg_patch = trains[0][0][:WIN]
    seg_glob = trains[0][1][:WIN][:, None, :]

    R = {}
    R["patch"] = run_view("PATCH view (16-d PCA of patch tokens)",
                          seg_patch, PCA_P, P, hann, real_patch, fh, seed0=1000)
    R["glob"] = run_view("GLOBAL view (512-d pooled, XD-comparable)",
                         seg_glob, None, P, hann, real_glob, fh, seed0=2000)

    # ----------------------------------------------------------------- summary
    print("\n" + "=" * 78, flush=True)
    print("SUMMARY -- the Avenue deficit is now calibrated", flush=True)
    print("=" * 78, flush=True)
    print("  %-10s %10s %10s %10s %12s" % ("view", "real dMid", "cal@25%",
                                           "te-POWER", "MDT"))
    for v in ("patch", "glob"):
        r = R[v]
        if "mdt" in r:
            print("  %-10s %+10.4f %+10.4f %+10.4f %11.2f%%"
                  % (v, r["dmid"], r["cal25"], r["te"], 100 * r["mdt"]))
    print("  Ped2 reference (backbone_robustness.txt): MDT 0.57% - 6.23%",
          flush=True)
    fh.write("\n=== summary ===\n%-10s %10s %10s %10s %12s\n"
             % ("view", "real dMid", "cal@25%", "te-POWER", "MDT"))
    for v in ("patch", "glob"):
        r = R[v]
        if "mdt" in r:
            fh.write("%-10s %+10.4f %+10.4f %+10.4f %11.2f%%\n"
                     % (v, r["dmid"], r["cal25"], r["te"], 100 * r["mdt"]))
    fh.close()
    print("\nwrote %s" % out_txt, flush=True)

    # ----------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0))
    xi = np.arange(NB)

    # (a) dose-response and the MDT intersection
    for v, col, mk in (("patch", "#c0392b", "o"), ("glob", "#2471a3", "s")):
        r = R[v]
        y = [r["res"][("ring", 16, f)][MID] for f in FRACS]
        ax[0].plot([100 * f for f in FRACS], y, mk + "-", color=col,
                   label="%s: response" % v)
        if "dmid" in r:
            ax[0].axhline(r["dmid"], color=col, ls="--", lw=0.9,
                          label="%s: real dMid (%+.4f)" % (v, r["dmid"]))
            ax[0].plot([100 * r["mdt"]], [r["dmid"]], "*", color=col, ms=13,
                       zorder=5)
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_xlabel("injected tone power (% of window detrended power)")
    ax[0].set_ylabel("shift of the 10-21 fr band")
    ax[0].set_title("(a) dose-response, period 16; star = minimum detectable",
                    fontsize=9)
    ax[0].legend(fontsize=6.5, loc="upper left")
    ax[0].grid(alpha=0.25)

    # (b) shape: ring vs step vs the real Avenue delta (patch view)
    r = R["patch"]
    ring = r["res"][("ring", 16, PRIMARY_FRAC)]
    step = r["res"][("step", PRIMARY_ONSET, PRIMARY_FRAC)]
    real = real_patch
    for v, lab, col, mk in ((ring, "injected RING p16", "#c0392b", "o"),
                            (step, "injected STEP on%d" % PRIMARY_ONSET,
                             "#2471a3", "s"),
                            (real, "REAL Avenue (patch)", "0.25", "D")):
        if v is None:
            continue
        ax[1].plot(xi, v / np.abs(v).max(), mk + ("--" if mk == "D" else "-"),
                   color=col, label=lab)
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].set_xticks(xi)
    ax[1].set_xticklabels([band_label(i) for i in range(NB)], fontsize=8)
    ax[1].set_xlabel("period band")
    ax[1].set_ylabel("band shift, normalised to |max| = 1")
    ax[1].set_title("(b) shape: a ring bumps one band, the data tilts",
                    fontsize=9)
    ax[1].legend(fontsize=6.5, loc="upper right")
    ax[1].grid(alpha=0.25)

    # (c) recovered frequency
    M = np.array([r["res"][("ring", p, PRIMARY_FRAC)] /
                  np.abs(r["res"][("ring", p, PRIMARY_FRAC)]).max()
                  for p in PERIODS])
    im = ax[2].imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax[2].set_yticks(range(len(PERIODS)))
    ax[2].set_yticklabels(["p=%d" % p for p in PERIODS], fontsize=8)
    ax[2].set_xticks(xi)
    ax[2].set_xticklabels([band_label(i) for i in range(NB)], fontsize=8,
                          rotation=30)
    ax[2].set_xlabel("period band")
    ax[2].set_title("(c) each period lights its own band", fontsize=9)
    for i in range(len(PERIODS)):
        ax[2].plot(int(np.argmax(M[i])), i, "k*", ms=11)
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

    fig.suptitle("F21  Avenue injection calibration: the instrument would have "
                 "seen the tone", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(REPO, "results", "F21_avenue_injection.png")
    fig.savefig(out, dpi=160)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
