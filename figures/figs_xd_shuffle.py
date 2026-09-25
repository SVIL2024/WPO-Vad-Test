# -*- coding: utf-8 -*-
"""F19 -- shuffle control ON XD-Violence, i.e. on P1's own features.

    (a) d_orig vs d_shuf per band  -- shuffling collapses the effect
    (b) |d| comparison             -- 1.5% of the original size

Reads results/xd_shuffle.txt (no recompute needed).
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "_pylibs"))

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

OUT = os.path.join(REPO, "results", "F19_xd_shuffle.png")
TXT = os.path.join(REPO, "results", "xd_shuffle.txt")

BANDS = ["21-64 fr", "10-21 fr", "5-10 fr", "2-5 fr", "1-2 fr"]
MID = 1  # index of the 10-21 fr band


def parse():
    d_orig = d_shuf = None
    ratio = None
    for line in open(TXT, encoding="utf-8"):
        s = line.strip()
        if s.startswith("d_orig"):
            d_orig = np.array([float(x) for x in re.findall(r"-?\d+\.\d+", s)])
        elif s.startswith("d_shuf"):
            d_shuf = np.array([float(x) for x in re.findall(r"-?\d+\.\d+", s)])
        elif "ratio" in s:
            m = re.search(r"=\s*([\d.]+)", s)
            if m:
                ratio = float(m.group(1))
    assert d_orig is not None and len(d_orig) == 5, d_orig
    assert d_shuf is not None and len(d_shuf) == 5, d_shuf
    # sanity gate, same rule as the analysis script
    assert d_orig[MID] < 0, "mid band must be negative; got %r" % d_orig[MID]
    return d_orig, d_shuf, ratio


def main():
    d_orig, d_shuf, ratio = parse()
    x = np.arange(5)
    w = 0.38

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))

    # (a) per-band difference
    a = ax[0]
    a.axhline(0, color="0.35", lw=0.8)
    a.bar(x - w / 2, d_orig, w, color="#0072B2", label="original frame order")
    a.bar(x + w / 2, d_shuf, w, color="#999999", label="frame order shuffled (5x)")
    a.set_xticks(x)
    a.set_xticklabels(BANDS, rotation=20, ha="right", fontsize=8)
    a.set_ylabel("anomalous - normal", fontsize=9)
    a.set_title("(a) XD-Violence, 120 normal vs 120 anomalous clips",
                fontsize=9, pad=10)
    a.legend(fontsize=8, frameon=False)
    a.grid(axis="y", alpha=0.25, lw=0.5)
    a.tick_params(labelsize=8)
    # annotate the mid band, the one the paper argues about
    a.annotate("mid band\n%.4f" % d_orig[MID],
               xy=(MID - w / 2, d_orig[MID]),
               xytext=(MID - 0.95, d_orig[MID] * 1.35),
               fontsize=8, color="#0072B2",
               arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.9))

    # (b) magnitude
    b = ax[1]
    vals = [np.linalg.norm(d_orig), np.linalg.norm(d_shuf)]
    bars = b.bar([0, 1], vals, 0.5, color=["#0072B2", "#999999"])
    b.set_xticks([0, 1])
    b.set_xticklabels(["original", "shuffled"], fontsize=8)
    b.set_ylabel("|delta|  (5-band vector)", fontsize=9)
    b.set_title("(b) shuffling leaves %.1f%% of the effect" % (100 * ratio)
                if ratio else "(b) magnitude", fontsize=9)
    b.grid(axis="y", alpha=0.25, lw=0.5)
    b.tick_params(labelsize=8)
    for i, v in enumerate(vals):
        b.text(i, v + max(vals) * 0.03, "%.4f" % v, ha="center", fontsize=8)

    # no in-figure title (F4).  The old suptitle also leaked the internal
    # experiment code "P7", which meant nothing to a reader (F2).
    fig.tight_layout()
    save(fig, OUT)
    print("d_orig", np.round(d_orig, 4), "|d| =", round(float(np.linalg.norm(d_orig)), 4))
    print("d_shuf", np.round(d_shuf, 4), "|d| =", round(float(np.linalg.norm(d_shuf)), 4))
    print("ratio", ratio)


if __name__ == "__main__":
    main()
