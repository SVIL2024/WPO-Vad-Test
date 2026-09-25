# -*- coding: utf-8 -*-
"""Reproduce the multi-scale arm table and the paired same-seed comparison table.

Why this script exists
----------------------
The paper prints two tables whose evidence is a set of four-seed runs:

  tab:arms    -- per-arm AP and AUC, mean +- sd over four seeds
  tab:paired  -- four paired same-seed comparisons: mean +- sd of the paired
                 difference, t, p, and whether the row survives a
                 Holm-Bonferroni step-down over all four rows

The per-seed values ARE the record: they are the same arrays that
`figures/figs_main.py` plots (F7 / F8) and that
`resolution/five_seed_audit.py` reports the AP half of.  Nothing here re-runs
a training job.  What it does is recompute every printed cell FROM the
per-seed values and refuse to agree unless the paper's numbers come out --
the same contract `premise_test/cluster_recheck.py` uses for the statistic.

Why the check is written the way it is
--------------------------------------
The per-seed values are recorded to four decimals, and t and p are functions
of them, so the last printed digit of a t or a p is only determined up to the
recording precision.  A naive `round(derived, 4) == printed` would therefore
flag a difference that is not a defect: moving one input inside its own
+-0.00005 rounding cell moves p in the fourth decimal.  So the check asks the
question the data can actually answer -- is the printed value ATTAINABLE from
data that rounds to the recorded values? -- by re-running the test over that
cell and requiring the printed number to intersect the resulting interval.
The mean and the sd are compared exactly, because they carry no such
ambiguity.

The published numbers below are typed in as the HYPOTHESIS UNDER TEST.  They
are the only thing in this file not derived from the arrays.

Run from the repo root:
    python resolution/arms_paired.py
Prints its report to stdout and exits non-zero if any printed cell fails.
"""
import itertools
import sys

import numpy as np
from scipy import stats

# Four seeds, one epoch, fixed time base.  Pairing is by index: the i-th entry
# of every arm is the SAME seed, which is what makes the comparison paired.
SEEDS = ["999999", "789", "2024", "1234"]

# arm -> (AP per seed, AUC per seed)
ARMS = {
    "E1  (K=1)":    ([0.7074, 0.7177, 0.7358, 0.7123],
                     [0.9189, 0.9181, 0.9198, 0.9180]),
    "E2  (1,8,16)": ([0.8227, 0.7446, 0.8114, 0.7898],
                     [0.9418, 0.9278, 0.9404, 0.9359]),
    "E2b (1,2,4)":  ([0.7949, 0.7211, 0.7633, 0.7389],
                     [0.9367, 0.9270, 0.9332, 0.9286]),
}

# ---- tab:arms, as printed -------------------------------------------------
PUB_ARMS = {
    #  arm            metric   mean      sd
    "E1  (K=1)":    {"AP":  (0.7183, 0.0124), "AUC": (0.9187, 0.0008)},
    "E2  (1,8,16)": {"AP":  (0.7921, 0.0345), "AUC": (0.9365, 0.0063)},
    "E2b (1,2,4)":  {"AP":  (0.7546, 0.0320), "AUC": (0.9314, 0.0044)},
}

# ---- tab:paired, as printed ----------------------------------------------
# (label, metric, arm A, arm B, mean, sd, t, p as printed, survives)
PUB_PAIRED = [
    ("E2-E1 (AUC)",  "AUC", "E2  (1,8,16)", "E1  (K=1)",   0.0178, 0.0058, 6.17, "0.0086", "yes"),
    ("E2-E2b (AP)",  "AP",  "E2  (1,8,16)", "E2b (1,2,4)", 0.0376, 0.0139, 5.40, "0.0126", "yes"),
    ("E2-E1 (AP)",   "AP",  "E2  (1,8,16)", "E1  (K=1)",   0.0738, 0.0362, 4.08, "0.027",  "nominal"),
    ("E2-E2b (AUC)", "AUC", "E2  (1,8,16)", "E2b (1,2,4)", 0.0051, 0.0030, 3.35, "0.044",  "nominal"),
]

ALPHA = 0.05
EPS = 5e-5          # half-width of a four-decimal rounding cell
HALF = 1e-9         # tolerance for the mean/sd comparisons, which are exact
                    # AT THE FOUR DECIMALS THE PAPER PRINTS (see below)

fails = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("ok" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not ok:
        fails.append(name)


def attain_interval(x, y):
    """Range of t and p over the rounding cell of both arms' per-seed values."""
    ts, ps = [], []
    for dx in itertools.product((-EPS, EPS), repeat=len(x)):
        xx = np.array(x, dtype=float) + np.array(dx)
        for dy in itertools.product((-EPS, EPS), repeat=len(y)):
            yy = np.array(y, dtype=float) + np.array(dy)
            t, p = stats.ttest_rel(xx, yy)
            ts.append(t)
            ps.append(p)
    return min(ts), max(ts), min(ps), max(ps)


def printed_interval(value, decimals):
    """The interval a number printed to `decimals` places stands for."""
    half = 0.5 * 10.0 ** (-decimals)
    return value - half, value + half


def main():
    print("=" * 78)
    print("MULTI-SCALE ARMS -- %d seeds, 1 epoch, fixed time base" % len(SEEDS))
    print("=" * 78)
    print("  seeds: %s" % ", ".join(SEEDS))
    print("")

    for arm in ARMS:
        for metric, idx in (("AP", 0), ("AUC", 1)):
            a = np.array(ARMS[arm][idx], dtype=float)
            m, s = float(a.mean()), float(a.std(ddof=1))
            pm, ps = PUB_ARMS[arm][metric]
            print("  %-14s %-4s mean=%.4f sd=%.4f  (paper %.4f +- %.4f)"
                  % (arm, metric, m, s, pm, ps))
            check("%s %s mean" % (arm, metric), abs(round(m, 4) - pm) < HALF,
                  "derived %.4f" % m)
            check("%s %s sd" % (arm, metric), abs(round(s, 4) - ps) < HALF,
                  "derived %.4f" % s)

    print("")
    print("=" * 78)
    print("PAIRED SAME-SEED COMPARISONS -- df = %d" % (len(SEEDS) - 1))
    print("=" * 78)
    print("  %-14s %10s %8s %8s %10s  %s"
          % ("comparison", "diff", "t", "p", "p (paper)", "survives"))

    derived = []
    for label, metric, armA, armB, pm, ps, pt, pp, psurv in PUB_PAIRED:
        idx = 0 if metric == "AP" else 1
        x = ARMS[armA][idx]
        y = ARMS[armB][idx]
        d = np.array(x, dtype=float) - np.array(y, dtype=float)
        m, s = float(d.mean()), float(d.std(ddof=1))
        t, p = stats.ttest_rel(x, y)
        derived.append({"label": label, "p": float(p)})

        dec_p = len(pp.split(".")[1])
        tlo, thi, plo, phi = attain_interval(x, y)
        derived[-1]["p_lo"], derived[-1]["p_hi"] = plo, phi

        print("  %-14s %+10.4f %8.2f %8.4f %10s  %s"
              % (label, m, t, p, pp, psurv))
        check("%s diff mean" % label, abs(round(m, 4) - pm) < HALF, "derived %+.4f" % m)
        check("%s diff sd" % label, abs(round(s, 4) - ps) < HALF, "derived %.4f" % s)

        lo, hi = printed_interval(pt, 2)
        check("%s t attainable from 4-dp inputs" % label, not (thi < lo or tlo > hi),
              "t over the rounding cell [%.4f, %.4f], paper %.2f" % (tlo, thi, pt))

        lo, hi = printed_interval(float(pp), dec_p)
        check("%s p attainable from 4-dp inputs" % label, not (phi < lo or plo > hi),
              "p over the rounding cell [%.6f, %.6f], paper %s" % (plo, phi, pp))

        check("%s 4/4 same sign" % label, bool(np.all(np.sign(d) == np.sign(d[0]))),
              "signs %s" % "".join("+" if v > 0 else "-" for v in d))

    # ---- Holm-Bonferroni step-down over all four rows ---------------------
    print("")
    print("  Holm-Bonferroni step-down, alpha = %.2f over %d rows:"
          % (ALPHA, len(derived)))
    order = sorted(range(len(derived)), key=lambda i: derived[i]["p"])
    reject = True
    for rank, i in enumerate(order):
        thr = ALPHA / (len(derived) - rank)
        survives = reject and derived[i]["p"] <= thr
        if not survives:
            reject = False
        verdict = "yes" if survives else "nominal"
        want = PUB_PAIRED[i][8]
        print("    %-14s p=%.4f  threshold=%.5f  -> %-8s (paper %s)"
              % (derived[i]["label"], derived[i]["p"], thr, verdict, want))
        check("%s Holm verdict" % derived[i]["label"], verdict == want,
              "derived %s, paper %s" % (verdict, want))

    print("")
    print("  Is the verdict stable across the recording precision?  Re-test the")
    print("  step-down at the far edge of each row's rounding cell:")
    stop = None
    for rank, i in enumerate(order):
        thr = ALPHA / (len(derived) - rank)
        if stop is None and derived[i]["p_lo"] > thr:
            stop = (rank, i, thr)
            print("    %-14s smallest p=%.4f still exceeds threshold=%.5f"
                  % (derived[i]["label"], derived[i]["p_lo"], thr))
        elif stop is None:
            print("    %-14s largest  p=%.4f clears  threshold=%.5f  (margin %.1f%%)"
                  % (derived[i]["label"], derived[i]["p_hi"], thr,
                     100.0 * (1.0 - derived[i]["p_hi"] / thr)))
    check("the step-down stops at the same row for every input in the cell",
          stop is not None and stop[0] == 2,
          "stops at rank %s" % (stop[0] if stop else None))
    check("the two rows above the stop survive at their worst case",
          all(derived[order[r]]["p_hi"] <= ALPHA / (len(derived) - r)
              for r in (0, 1)))

    print("")
    if fails:
        print("FAILED %d check(s):" % len(fails))
        for f in fails:
            print("  -", f)
        return 1
    print("All printed cells of tab:arms and tab:paired reproduce from the")
    print("per-seed values, and the survive/nominal column does not turn on the")
    print("last recorded digit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
