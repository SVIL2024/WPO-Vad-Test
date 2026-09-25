# -*- coding: utf-8 -*-
"""F13 and F14 for the paper: the backbone-robustness check (section 2.6).

F13  the same spectral-shape statistic on the same UCSD Ped2 frames, computed
     on three backbones that differ in BOTH architecture and pretraining
     objective.  If the mid-band deficit only appeared for CLIP ViT-B/16 the
     finding would be a backbone artefact; it appears for all three.

F14  the POWER-MATCHED injection calibration.  A pure tone is injected so that
     it carries a fixed fraction of the window's own detrended power, which
     makes the response comparable across backbones.  Every curve goes UP
     (the statistic sees a rhythm); every observed real shift is BELOW zero
     (the data moves away from rhythmic).  No tone amplitude explains it.

Everything is computed by controls/backbone.py so figures and tables can
never disagree.  The results are cached to results/_bb_cache.npz because
re-running the analysis reloads ~4 GB of patch tokens.

NOTE: matplotlib lives in _pylibs/ (the conda env's `packaging` dist-info is
broken, so `pip install matplotlib` into the env fails).  Keep it first on
sys.path.
"""
import contextlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # .../waveClipVad/ops
ROOT = os.path.dirname(HERE)                               # .../waveClipVad
sys.path.insert(0, os.path.join(ROOT, "_pylibs"))          # isolated matplotlib
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- vector output for the IEEE submission -------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def save(fig, path):
    """Save as vector PDF, keeping the figure's exact figsize so the aspect
    ratio -- and hence the typeset layout -- is unchanged."""
    p = os.path.splitext(path)[0] + ".pdf"
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)
    return p
import numpy as np


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
import backbone as B

OUT = os.environ.get("FIG_OUT", r"D:\program\waveClipVad\results_ped2")
CACHE = os.path.join(OUT, "_bb_cache.npz")
MID = 1          # 10-21 fr : the band a behavioural rhythm would occupy
FAST = 2         # 5-10 fr  : where a period-8 tone lands

NAME = {"clip_vit_b16": "CLIP ViT-B/16\n(CLIP obj, ViT, 14x14)",
        "clip_rn50": "CLIP RN50\n(CLIP obj, CNN, 7x7)",
        "inet_rn50": "ImageNet RN50\n(sup. obj, CNN, 7x7)"}
COL = {"clip_vit_b16": "#0072B2", "clip_rn50": "#D55E00", "inet_rn50": "#009E73"}


def pick(d, k):
    """cal/cal_pw keys are ints/floats in memory and strings after the
    JSON round-trip through the cache -- accept both."""
    if k in d:
        return d[k]
    for kk in (str(k), str(float(k)), int(k) if float(k).is_integer() else k):
        if kk in d:
            return d[kk]
    raise KeyError(k)


def _rekey(d):
    """JSON turns the numeric dict keys (period, amplitude, power fraction)
    into strings.  Restore them so the dicts look exactly like the ones
    B.analyze() returns."""
    out = {}
    for k, v in d.items():
        try:
            f = float(k)
            kk = int(f) if f.is_integer() else f
        except ValueError:
            kk = k
        out[kk] = _rekey(v) if isinstance(v, dict) else v
    return out


def load():
    """Run (or load) the backbone analysis; returns [dict, ...]."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        res = []
        for tag in z["tags"]:
            res.append(dict(tag=str(tag),
                            grid=int(z["%s_grid" % tag]),
                            gA=z["%s_gA" % tag], gN=z["%s_gN" % tag],
                            pA=z["%s_pA" % tag], pN=z["%s_pN" % tag],
                            base=z["%s_base" % tag],
                            cal=_rekey(json.loads(str(z["%s_cal" % tag]))),
                            cal_pw=_rekey(json.loads(str(z["%s_calpw" % tag]))),
                            pwr_tot=float(z["%s_pwr" % tag]),
                            evr16=float(z["%s_evr" % tag]),
                            slow_share=float(z["%s_slow" % tag]),
                            labels=[str(s) for s in z["labels"]]))
        print("[cache] loaded %d backbones from %s" % (len(res), CACHE))
        return res

    res = []
    buf = io.StringIO()
    for bb in B.BBS:
        d = os.path.join(ROOT, "ped2_feat_%s" % bb)
        if not os.path.isdir(d) and bb == "clip_vit_b16":
            d = os.path.join(ROOT, "ped2_feat")
        print("analysing %s ..." % bb, flush=True)
        with contextlib.redirect_stdout(buf):
            R = B.analyze(d, bb)
        res.append(R)
    os.makedirs(OUT, exist_ok=True)
    d = {"tags": [R["tag"] for R in res],
         "labels": res[0]["labels"]}
    for R in res:
        t = R["tag"]
        d["%s_grid" % t] = R["grid"]
        d["%s_gA" % t] = R["gA"]; d["%s_gN" % t] = R["gN"]
        d["%s_pA" % t] = R["pA"]; d["%s_pN" % t] = R["pN"]
        d["%s_base" % t] = R["base"]
        d["%s_cal" % t] = json.dumps({str(k): {str(a): list(map(float, v))
                                               for a, v in av.items()}
                                      for k, av in R["cal"].items()})
        d["%s_calpw" % t] = json.dumps({str(k): {str(a): list(map(float, v))
                                                 for a, v in av.items()}
                                        for k, av in R["cal_pw"].items()})
        d["%s_pwr" % t] = R["pwr_tot"]
        d["%s_evr" % t] = R["evr16"]
        d["%s_slow" % t] = R["slow_share"]
    np.savez(CACHE, **d)
    print("[cache] wrote %s" % CACHE)
    return res


def fig13(res):
    labels = res[0]["labels"]
    x = np.arange(len(labels))
    w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # ---- (a) PATCH view: anomalous - normal, per band, per backbone --------
    ax = axes[0]
    for k, R in enumerate(res):
        dmid = R["pA"].mean(0) - R["pN"].mean(0)
        ax.bar(x + (k - 1) * w, dmid, w, label=NAME[R["tag"]].replace("\n", " "),
               color=COL[R["tag"]], edgecolor="white", lw=0.5)
    ax.axhline(0, color="#333", lw=1.0)
    ax.axvspan(MID - 0.5, MID + 0.5, color="#ffd166", alpha=0.22, zorder=0)
    # panel (a) deliberately keeps the plotting area clean: panel (b) below
    # shows the z values directly, so no per-bar z labels here.
    ax.set_ylim(-0.115, 0.27)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("anomalous - normal band power (PATCH view)")
    ax.set_title("(a)  spatial-resolved shift: same sign on all three backbones",
                 fontsize=9)
    ax.legend(fontsize=6.6, frameon=False, loc="lower left")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- (b) PATCH z per band, per backbone --------------------------------
    ax = axes[1]
    for R in res:
        zs = [S.zscore(R["pN"][:, i], R["pA"][:, i]) for i in range(len(labels))]
        ax.plot(x, zs, marker="o", lw=1.8, color=COL[R["tag"]],
                label=NAME[R["tag"]].replace("\n", " "))
    ax.axhline(0, color="#333", lw=1.0)
    ax.axhline(1.96, color="#888", lw=0.8, ls=":")
    ax.axhline(-1.96, color="#888", lw=0.8, ls=":")
    ax.axvspan(MID - 0.5, MID + 0.5, color="#ffd166", alpha=0.22, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("z (anomalous vs normal windows)")
    ax.set_title("(b)  significance per band: no backbone shows a mid-band surplus",
                 fontsize=9)
    ax.legend(fontsize=6.6, frameon=False, loc="lower left")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # no in-figure title (F4); the caption carries it
    fig.tight_layout()
    p = os.path.join(OUT, "F13_backbone_robustness.png")
    return save(fig, p)


def fig14(res):
    fracs = [0.0, 0.10, 0.25, 0.50]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for ax, (per, band) in zip(axes, [(16, MID), (8, FAST)]):
        for R in res:
            cp = pick(R["cal_pw"], per)
            y = [0.0] + [float(pick(cp, f)[band]) for f in fracs[1:]]
            ax.plot(fracs, y, marker="o", lw=2.0, color=COL[R["tag"]],
                    label=NAME[R["tag"]].replace("\n", " "))
            real = float(R["pA"][:, band].mean() - R["pN"][:, band].mean())
            ax.axhline(real, color=COL[R["tag"]], lw=1.0, ls="--", alpha=0.85)
            ax.text(0.505, real, "  real %+.4f" % real, fontsize=6.4,
                    color=COL[R["tag"]], va="center")
        ax.axhline(0, color="#333", lw=1.0)
        ax.set_xlabel("injected tone power  /  window detrended power")
        ax.set_xticks(fracs)
        ax.set_xticklabels(["0", "10%", "25%", "50%"])
        ax.set_title("period-%d tone -> %s band" % (per, res[0]["labels"][band]),
                     fontsize=9)
        ax.legend(fontsize=6.6, frameon=False, loc="upper left")
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("change in normalised band power")
    # no in-figure title (F4)
    fig.tight_layout()
    p = os.path.join(OUT, "F14_power_matched_calibration.png")
    return save(fig, p)


def main():
    os.makedirs(OUT, exist_ok=True)
    res = load()
    # keep the printed table in sync with the figures
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for R in res:
            B.report(R)
    print(fig13(res))
    print(fig14(res))
    print("\nmin detectable tone power (period-16, mid band):")
    for R in res:
        d = float(R["pA"][:, MID].mean() - R["pN"][:, MID].mean())
        c = float(pick(pick(R["cal_pw"], 16), 0.25)[MID])
        print("  %-14s real %+.4f   cal@25%% %+.4f   -> %.2f%%"
              % (R["tag"], d, c, 25.0 * abs(d) / (c + 1e-12)))


if __name__ == "__main__":
    main()
