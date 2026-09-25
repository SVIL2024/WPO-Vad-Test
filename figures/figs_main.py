# -*- coding: utf-8 -*-
"""F5, F6, F7, F8, F9 -- the five body figures that had no rendered file yet.

Every number below is transcribed from this project's own evidence ledger
(the project's internal evidence ledger) or from `results/*.txt`.  Nothing is
recomputed: these plots exist only because the tables in the paper cite them.

  F5  per-class spectral difference on XD-Violence      (ledger / draft §1.5)
  F6  per-band AUC vs frequency + branch Nyquist lines  (ledger §2.1)
  F7  multi-scale arms: AP with seed spread             (ledger §2.2, §3.1)
  F8  seed swing at fixed configuration                 (ledger §3.1)
  F9  effect size vs the noise floor                    (ledger §3.2)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "_pylibs"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Vector output for the IEEE submission: line art must not be raster.  Serif +
# TrueType embedding keeps figure text consistent with the Times body font and
# removes the font-substitution warnings.
plt.rcParams.update({
    "font.family": "serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

OUTDIR = os.path.join(REPO, "results")

# Okabe-Ito palette: distinguishable under deuteranopia / protanopia.
OKABE = {
    "blue": "#0072B2", "orange": "#D55E00", "green": "#009E73",
    "purple": "#CC79A7", "yellow": "#E69F00", "sky": "#56B4E9",
    "grey": "#999999",
}


def save(fig, name, dpi=None):
    """Save as vector PDF.  The figure keeps its exact figsize (no
    bbox_inches='tight') so the aspect ratio -- and therefore the typeset
    layout -- is identical to the raster version it replaces.

    `dpi` only controls the resolution of raster content inside the PDF
    (imshow); pure line art is unaffected."""
    p = os.path.join(OUTDIR, name + ".pdf")
    if dpi:
        fig.savefig(p, dpi=dpi)
    else:
        fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


BANDS = ["21-64", "10-21", "5-10", "2-5", "1-2"]

# ---------------------------------------------------------------- F5 -------
# z scores, anomalous vs normal, 64-frame window, XD-Violence, per class.
# Transcribed from results/oscillatory_by_class.txt, the 64-frame block of
# the per-class diagnostic.  The bands are the ones the paper declares for the
# statistic, {1,3,6,12,24,33} FFT bins at W=64; the diagnostic was re-run on
# that set because it had been using a different bin set in bands 2--4.
F5 = {
    "fighting":     [-5.0, -7.3,  1.1,  7.8, 7.5],
    "shooting":     [-8.9, -6.9,  4.0,  9.2, 8.3],
    "riot":         [ 0.3, -3.2, -3.5,  2.6, 3.6],
    "abuse":        [-3.9, -3.0,  2.6,  4.4, 3.1],
    "car accident": [ 6.1, -5.6, -6.4, -1.7, -0.5],
    "explosion":    [ 1.5, -2.6, -2.4,  1.1, 0.7],
}
F5_ORDER = ["fighting", "shooting", "riot", "abuse", "car accident", "explosion"]


def fig5():
    M = np.array([F5[k] for k in F5_ORDER], dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    v = 8.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(BANDS, fontsize=8)
    ax.set_yticks(range(6))
    ax.set_yticklabels(F5_ORDER, fontsize=8)
    ax.set_xlabel("period band (frames)", fontsize=8)
    # no in-figure title: the LaTeX caption carries it (reviewer item F4)
    # mid-band column -- the only one that is negative for every class
    ax.add_patch(plt.Rectangle((0.5, -0.5), 1, 6, fill=False,
                               edgecolor="k", lw=1.6, ls="--"))
    for i in range(6):
        for j in range(5):
            ax.text(j, i, "%+.1f" % M[i, j], ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(M[i, j]) > 5.0 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    # imshow is genuine raster content: 600 dpi keeps it inside the IEEE
    # requirement for halftone/image art.
    save(fig, "F5_perclass_spectrum", dpi=600)


# ---------------------------------------------------------------- F6 -------
# per-band AUC measured on XD-Violence features (ledger §2.1)
F6_HZ = [(0.00, 0.88), (0.94, 1.82), (3.75, 4.63), (6.56, 7.50)]
F6_AUC = [0.461, 0.595, 0.747, 0.760]
# post-pooling Nyquist of the multi-scale branches
NYQ = {"8x": 0.94, "16x": 0.47}


def fig6():
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.array([0.5 * (a + b) for a, b in F6_HZ])
    ax.plot(x, F6_AUC, "o-", color=OKABE["blue"], lw=1.6, ms=6)
    for (a, b), y in zip(F6_HZ, F6_AUC):
        ax.hlines(y, a, b, color=OKABE["blue"], lw=3.0, alpha=0.35)
    # annotate the branch Nyquist limits at the top of the panel so they do
    # not collide with the "chance" baseline label or with each other
    for i, (name, f) in enumerate(NYQ.items()):
        ax.axvline(f, color=OKABE["orange"], ls="--", lw=1.2)
        ax.text(f + 0.08, 0.79 - i * 0.07, "%s Nyq.\n%.2f Hz" % (name, f),
                fontsize=7, color=OKABE["orange"], ha="left", va="top")
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax.text(0.02, 0.503, "chance", fontsize=7, color="0.45", va="bottom")
    ax.set_xlabel("frequency (Hz)", fontsize=8)
    ax.set_ylabel("per-band AUC", fontsize=8)
    ax.set_ylim(0.43, 0.80)
    # no in-figure title (F4)
    # F1: axis labels were colliding with the tick labels
    ax.tick_params(labelsize=8, pad=4)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    save(fig, "F6_band_auc")


# ------------------------------------------------------------ F7 / F8 ------
SEEDS = ["999999", "789", "2024", "1234"]
ARMS = {
    #            AP per seed                          AUC per seed
    "E1  (K=1)":    ([0.7074, 0.7177, 0.7358, 0.7123], [0.9189, 0.9181, 0.9198, 0.9180]),
    "E2  (1,8,16)": ([0.8227, 0.7446, 0.8114, 0.7898], [0.9418, 0.9278, 0.9404, 0.9359]),
    "E2b (1,2,4)":  ([0.7949, 0.7211, 0.7633, 0.7389], [0.9367, 0.9270, 0.9332, 0.9286]),
}
E2C_AP = [0.7957]   # single seed only
E2C_AUC = [0.9380]


def fig7():
    names = list(ARMS.keys()) + ["E2c (1,32,64)"]
    ap = [np.array(ARMS[k][0]) for k in ARMS] + [np.array(E2C_AP)]
    auc = [np.array(ARMS[k][1]) for k in ARMS] + [np.array(E2C_AUC)]

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1), sharex=True)
    for a, vals, ttl in ((ax[0], ap, "AP"), (ax[1], auc, "AUC")):
        xs = np.arange(len(names))
        means = [v.mean() for v in vals]
        errs = [v.std(ddof=1) if len(v) > 1 else 0.0 for v in vals]
        colors = [OKABE["sky"], OKABE["blue"], OKABE["orange"], OKABE["grey"]]
        a.bar(xs, means, 0.6, yerr=errs, capsize=4,
              color=colors, edgecolor="0.3", lw=0.5)
        for i, v in enumerate(vals):
            a.scatter(np.full(len(v), i) + np.linspace(-0.16, 0.16, len(v)),
                      v, s=16, color="k", zorder=3)
        if len(vals[-1]) == 1:
            a.text(len(names) - 1, means[-1], " 1 seed", fontsize=6.5,
                   ha="center", va="bottom", color="0.35")
        a.set_xticks(xs)
        a.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=7)
        a.set_ylabel(ttl, fontsize=9)
        a.tick_params(labelsize=8)
        a.grid(axis="y", alpha=0.25, lw=0.5)
    # panel markers stay; the descriptive sentence is the caption's job (F4)
    ax[0].set_title("(a) AP", fontsize=9)
    ax[1].set_title("(b) AUC", fontsize=9)
    fig.tight_layout()
    save(fig, "F7_multiscale_arms")


def fig8():
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for idx, (metric, ci) in enumerate([("AP", 0), ("AUC", 1)]):
        a = ax[idx]
        for name, (apv, aucv) in ARMS.items():
            v = np.array(apv if ci == 0 else aucv)
            x = np.arange(len(v))
            a.plot(x, v, "o-", lw=1.3, ms=5, label=name)
            a.hlines(v.mean(), -0.25, len(v) - 0.75, ls="--", lw=0.9,
                     color=a.lines[-1].get_color(), alpha=0.7)
        a.set_xticks(range(4))
        a.set_xticklabels(SEEDS, fontsize=7.5)
        a.set_xlabel("random seed", fontsize=8)
        a.set_ylabel(metric, fontsize=9)
        a.grid(alpha=0.25, lw=0.5)
        a.tick_params(labelsize=8)
        a.legend(fontsize=7, frameon=False)
    ax[0].set_title("(a) AP: sd 0.012-0.035", fontsize=9)
    ax[1].set_title("(b) AUC: sd 0.001-0.006", fontsize=9)
    fig.tight_layout()
    save(fig, "F8_seed_swing")


# ---------------------------------------------------------------- F9 -------
# (label, effect in AP, kind)  kind: 'point' or 'range'
F9 = [
    ("WaveSSM spread $-\\omega_0$ (1 seed)",        0.0420,  None),
    ("WaveSSM vs control (2 seeds)",               -0.0313, (-0.0399, -0.0227)),
    ("WPO on strong baseline",                     -0.0317,  None),
    ("WPO block-parallel",                         -0.0012,  None),
    ("weak baseline + band-pass",                   0.0336,  None),
    ("E2b $-$ E2 (paired, 4 seeds)",               -0.0376,  None),
    ("E2 $-$ E1 (architectural, 4 seeds)",          0.0738,  None),
]
NOISE_SD = 0.0345
NOISE_RANGE = 0.0781


def fig9():
    labels = [t for t, _, _ in F9]
    vals = [v for _, v, _ in F9]
    y = np.arange(len(F9))[::-1]

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.axvspan(-NOISE_SD, NOISE_SD, color=OKABE["yellow"], alpha=0.28, zorder=0)
    ax.axvline(0, color="0.35", lw=0.9)
    for yi, (lab, v, rng) in zip(y, F9):
        if rng is not None:
            ax.plot(rng, [yi, yi], color=OKABE["purple"], lw=3.2,
                    solid_capstyle="butt")
            ax.plot([v], [yi], "|", color=OKABE["purple"], ms=12, mew=2)
        else:
            col = OKABE["orange"] if v < 0 else OKABE["blue"]
            ax.plot([v], [yi], "o", color=col, ms=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("effect size (AP points)", fontsize=9)
    ax.set_xlim(-0.10, 0.10)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)
    # F1: the annotation used to sit on top of the first data row; give it its
    # own strip above the panel and an opaque background.
    ax.text(0.0, len(F9) - 0.45, "noise band  (sd = 0.0345)",
            fontsize=7.5, ha="center", va="bottom", color="#7d6608",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                      alpha=0.85))
    ax.set_ylim(-0.6, len(F9) + 0.15)
    # no in-figure title (F4); the caption now states the 1.2x exception
    fig.tight_layout()
    save(fig, "F9_effect_vs_noise")


if __name__ == "__main__":
    fig5(); fig6(); fig7(); fig8(); fig9()
