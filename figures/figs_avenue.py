# -*- coding: utf-8 -*-
"""F20 -- CUHK Avenue, the cross-dataset replication (three panels).

  (a) PATCH view  : normal vs anomalous spectral shape
  (b) GLOBAL view : same for the 512-d pooled vector (the XD-comparable view)
  (c) mid-band z across every dataset/view we have -- the money panel.
      All negative => the finding is neither one dataset's nor one anomaly
      type's property.

Reads results/avenue_spectral.txt (no recompute needed).
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

OUT = os.path.join(REPO, "results", "F20_avenue_replication.png")
TXT = os.path.join(REPO, "results", "avenue_spectral.txt")

ROW = re.compile(r"^\s*([\d\-]+ fr)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-+][\d.]+)")


def parse():
    """-> {'patch': (labels, normal, anom, z), 'global': ...}"""
    out, cur, buf = {}, None, []
    for line in open(TXT, encoding="utf-8"):
        if line.startswith("=== PATCH"):
            cur = "patch"; buf = []
        elif line.startswith("=== GLOBAL"):
            if cur and buf:
                out[cur] = buf
            cur = "global"; buf = []
        m = ROW.match(line)
        if m and cur:
            buf.append((m.group(1).strip(), float(m.group(2)), float(m.group(3)),
                        float(m.group(5))))
    if cur and buf:
        out[cur] = buf
    return out


def main():
    d = parse()
    if "patch" not in d or "global" not in d:
        print("[FATAL] could not parse %s" % TXT)
        sys.exit(1)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    x = np.arange(len(d["patch"]))

    for ax, key, title in (
        (axes[0], "patch", "(a) Avenue — PATCH view"),
        (axes[1], "global", "(b) Avenue — GLOBAL view (XD-comparable)"),
    ):
        rows = d[key]
        lab = [r[0] for r in rows]
        nn = np.array([r[1] for r in rows])
        aa = np.array([r[2] for r in rows])
        zz = np.array([r[3] for r in rows])
        w = 0.38
        ax.bar(x - w / 2, nn, w, label="normal", color="#8fa8bf", edgecolor="k", lw=.5)
        ax.bar(x + w / 2, aa, w, label="anomalous", color="#c1533f", edgecolor="k", lw=.5)
        ax.set_xticks(x)
        ax.set_xticklabels(lab, fontsize=8)
        ax.set_ylabel("normalised band power")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", alpha=.3)
        # highlight the mid band
        ax.axvspan(0.5, 1.5, color="gold", alpha=.13, zorder=0)
        ax.text(1, ax.get_ylim()[1] * 0.97, "mid band\n10–21 fr", ha="center",
                va="top", fontsize=7.5, color="#8a6d00")
        for i, z in enumerate(zz):
            ax.text(i, max(nn[i], aa[i]) + ax.get_ylim()[1] * 0.02, "z=%+.2f" % z,
                    ha="center", fontsize=7.5,
                    color="#a00" if z < 0 else "#333")

    # ---- (c) mid-band z across datasets / views ----
    ax = axes[2]
    names = ["Avenue\nPATCH", "Avenue\nGLOBAL", "Ped2\nPATCH", "Ped2\nGLOBAL", "XD\n(6 classes)"]
    zs = [-4.57, -4.51, -6.97, -2.74, -6.08]
    note = ["motion-type\nJPEG", "scene-level\nanomaly", "persistent\nobject",
            "small object\n(pooled away)", "violence\n(per-class)"]
    cols = ["#c1533f", "#c1533f", "#5b7f9e", "#9aa7b1", "#5b7f9e"]
    ax.bar(np.arange(len(zs)), zs, .62, color=cols, edgecolor="k", lw=.5)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(np.arange(len(zs)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("mid-band z  (10–21 fr)")
    ax.set_title("(c) mid band is depleted everywhere", fontsize=10)
    ax.grid(axis="y", alpha=.3)
    for i, (z, n) in enumerate(zip(zs, note)):
        ax.text(i, z - 0.45 if z < 0 else z + 0.15, n, ha="center",
                va="top" if z < 0 else "bottom", fontsize=6.8, color="#444")
    ax.set_ylim(min(zs) - 2.2, 1.4)

    # no in-figure title (F4)
    fig.tight_layout()
    save(fig, OUT)


if __name__ == "__main__":
    main()
