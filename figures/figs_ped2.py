# -*- coding: utf-8 -*-
"""F11 and F12 for the paper: the Ped2 spatial-resolution check.

F11  two views (GLOBAL pooled vs PATCH resolved) of the same spectral-shape
     statistic on the same frames -- shows spatial resolution AMPLIFIES the
     mid-band deficit rather than revealing a hidden rhythm.

F12  the injection calibration -- shows the statistic CAN see a rhythm
     (dose-response, and the shift lands in the injected tone's own band).
     Everything in §2.5 rests on this figure; without it the null is
     uninterpretable.

Both are computed by premise_test/spectral.py so the figures and the tables can
never disagree.

NOTE: matplotlib is installed into _pylibs/ because the conda env's `packaging`
dist-info is broken and `pip install matplotlib` into the env fails with
OSError [Errno 2]. Keep that path first on sys.path.
"""
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # .../waveClipVad/ops
ROOT = os.path.dirname(HERE)                               # .../waveClipVad
sys.path.insert(0, os.path.join(ROOT, "_pylibs"))          # isolated matplotlib
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

OUT = os.environ.get("FIG_OUT", r"D:\program\waveClipVad\results_ped2")
MID = 1  # the 10-21 fr band: the one a behavioural rhythm would occupy


def save(fig, path):
    """Save as vector PDF, keeping the figure's exact figsize so the aspect
    ratio -- and hence the typeset layout -- is unchanged."""
    p = os.path.splitext(path)[0] + ".pdf"
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)
    return p


def fig11(R):
    labels = R["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=False)
    for ax, (tag, A, N) in zip(axes, [("(a) GLOBAL view (512-d pooled)", R["gA"], R["gN"]),
                                      ("(b) PATCH view (per-patch tokens)", R["pA"], R["pN"])]):
        x = np.arange(len(labels))
        m_n = N.mean(0)            # normal-window mean per band
        m_a = A.mean(0)            # anomalous-window mean per band
        ax.bar(x - 0.19, m_n, 0.38, label="normal", color="#c9d3e0", edgecolor="#5b6b80")
        ax.bar(x + 0.19, m_a, 0.38, label="anomalous", color="#d1495b", edgecolor="#7d1f2b")
        # highlight the band the whole argument is about
        ax.axvspan(MID - 0.5, MID + 0.5, color="#ffd166", alpha=0.25, zorder=0)
        for i in range(len(labels)):
            zz = S.zscore(N[:, i], A[:, i])
            ax.text(i, max(m_n[i], m_a[i]) + 0.022, "z=%+.2f" % zz,
                    ha="center", fontsize=7.5,
                    color="#7d1f2b" if i == MID else "#333")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(tag, fontsize=9)
        ax.set_ylabel("normalised band power")
        ax.legend(fontsize=8, frameon=False)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    # no in-figure title (F4): the caption carries it
    fig.tight_layout()
    return save(fig, os.path.join(OUT, "F11_ped2_two_view_spectrum.png"))


def fig12(R):
    labels = R["labels"]
    amps = [0.5, 1.0, 2.0]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
    for ax, per in zip(axes, (16, 8)):
        target = 1 if per == 16 else 2  # bin 64/16=4 -> band 1 ; 64/8=8 -> band 2
        for b in range(len(labels)):
            y = [R["cal"][per][a][b] for a in amps]
            ax.plot(amps, y, marker="o", lw=2.2 if b == target else 1.0,
                    color="#d1495b" if b == target else "#9aa5b1",
                    zorder=3 if b == target else 1,
                    label=("%s  <== injected" % labels[b]) if b == target else labels[b])
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_title("injected tone, period %d frames\n(its own band: %s)"
                     % (per, labels[target]), fontsize=9)
        ax.set_xlabel("amplitude (x natural fluctuation)")
        ax.set_xticks(amps)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("change in normalised band power")
    # no in-figure title (F4): the caption carries it
    fig.tight_layout()
    return save(fig, os.path.join(OUT, "F12_injection_calibration.png"))


def main():
    os.makedirs(OUT, exist_ok=True)
    print("recomputing via premise_test/spectral.py (single source of truth) ...")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        R = S.main()
    print(fig11(R))
    print(fig12(R))


if __name__ == "__main__":
    main()
