# -*- coding: utf-8 -*-
"""R1 (peer review): put the four mechanisms on one table.

The paper claims "None improves detection" but never shows the numbers.  This
recomputes, from the seed-234 score dumps that already exist, the frame-level
and video-level AUC/AP of every mechanism arm against the shared no-wave
control, so the claim can be checked rather than believed.

`logs_fetch/scores_xd_<arm>_s234.npz` holds, per video, one score per segment
(each segment covers REPEAT=16 raw frames) plus the frame-level ground truth.

Writes results/mechanisms.txt
"""
import glob
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LOGS = os.environ.get("XD_SCORE_DIR", os.path.join(REPO, "logs_fetch"))
REPEAT = 16
OUT = os.environ.get("MECH_OUT",
                     os.path.join(REPO, "results", "mechanisms.txt"))

# arm -> (mechanism label, what was varied)
# The arm -> mechanism mapping below was CORRECTED on 2026-09-24.  The first
# version of this table labelled e2a/e2b/e3a/e3b as "wave-propagation operator"
# and "resonators", which is wrong: reading the job scripts that produced the
# dumps (ops/_job_paperE2.sh, ops/_job_paperE3.sh) shows all four are WAVE
# FIELD arms that differ only in the fusion point and in whether the trunk is
# frozen.  A mislabelled table is worse than no table, so the labels now follow
# the argv each dump was produced with.
ARMS = [
    ("base", "(control) no-wave baseline", "A0 head, seed 234"),
    ("wf192", "(a) wave field, K=192", "field gain 0.1, from scratch"),
    ("wl2.5", "(c) PDE residual regulariser", "wave-loss lambda=2.5, from scratch"),
    ("e2a", "(a) wave field, evidence->logits1", "warm A0, trunk frozen"),
    ("e2b", "(a) wave field, evidence->logits1", "warm A0, joint"),
    ("e3a", "(a) wave field, feature gate", "warm A0, trunk frozen"),
    ("e3b", "(a) wave field, feature gate", "warm A0, joint"),
]

lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def load(name):
    path = os.path.join(LOGS, "scores_xd_%s_s234.npz" % name)
    if not os.path.isfile(path):
        return None, None, None
    d = np.load(path, allow_pickle=True)
    probs = d["probs"]
    gt = d["gt"].ravel()
    flat = np.concatenate([np.repeat(np.atleast_1d(p), REPEAT) for p in probs])
    if len(flat) != len(gt):
        sys.exit("%s: %d frames vs gt %d" % (name, len(flat), len(gt)))
    # per-video max score -> the video-level XD convention
    vmax = np.array([float(np.max(p)) for p in probs])
    # per-video label: does any frame in this video carry an anomaly?
    vlab = np.array([int(gt[o:o + len(p) * REPEAT].max() > 0)
                     for p, o in zip(probs, np.cumsum([0] + [len(q) * REPEAT for q in probs[:-1]]))])
    return flat, gt, (vmax, vlab)


def main():
    emit("=" * 74)
    emit("R1 -- the four mechanisms, measured on the same dumps (seed 234, XD)")
    emit("=" * 74)
    emit("  source: logs_fetch/scores_xd_<arm>_s234.npz")
    emit("  frame level: all frames pooled.  video level: max score per video,")
    emit("  video label = any anomalous frame (the XD headline convention).")
    emit("")

    rows = []
    ref = None
    for name, label, note in ARMS:
        flat, gt, vid = load(name)
        if flat is None:
            emit("  %-34s MISSING (%s)" % (label, name))
            continue
        fA, fP = roc_auc_score(gt, flat), average_precision_score(gt, flat)
        vmax, vlab = vid
        vA, vP = roc_auc_score(vlab, vmax), average_precision_score(vlab, vmax)
        if name == "base":
            ref = (fA, fP, vA, vP)
        rows.append((name, label, note, fA, fP, vA, vP, len(vmax)))

    if ref is None:
        sys.exit("the no-wave control dump is missing; nothing to compare against")

    emit("  %-34s %8s %8s %8s %8s   %s" %
         ("arm", "fAUC", "fAP", "vAUC", "vAP", "d vAP"))
    emit("  " + "-" * 70)
    for name, label, note, fA, fP, vA, vP, nv in rows:
        emit("  %-34s %8.4f %8.4f %8.4f %8.4f   %+8.4f   %s"
             % (label, fA, fP, vA, vP, vP - ref[3], note))
    emit("")
    emit("  %d videos per arm." % rows[0][7])
    emit("")
    emit("  Reading: every mechanism arm is compared with the SAME no-wave")
    emit("  control run at the SAME seed, so the difference is not a seed draw.")
    emit("  A single seed still cannot resolve differences below the seed sd")
    emit("  (0.0345 AP in our noisiest arm; Table 'mde'), so a small |d| here")
    emit("  excludes a large gain and nothing finer.")
    emit("")

    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
