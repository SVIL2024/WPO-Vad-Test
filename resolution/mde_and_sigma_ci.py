# -*- coding: utf-8 -*-
"""How uncertain is the seed sd, and does the conclusion survive it?

WHY THIS EXISTS
    Section 5 of the paper turns a measured seed sd into a rule: "you need n
    seeds to resolve delta".  Every sd in that table comes from 4 or 5 seeds,
    so the sd itself is a noisy estimate.  A reviewer will ask, correctly,
    whether the whole rule rests on four numbers.  This script answers that
    question in two parts instead of hand-waving it:

    (1) CONFIDENCE INTERVAL ON SIGMA.  (n-1)s^2 / sigma^2 ~ chi^2_{n-1}, so
        the 95% interval on sigma is s*sqrt(df/chi2_{.975}) to
        s*sqrt(df/chi2_{.025}).  At n=4 that interval is 6.6x wide, which is
        the honest size of the problem.

    (2) THE RATIO IS MUCH BETTER DETERMINED THAN THE LEVEL.  The claim that
        matters for the paper is comparative -- AP needs more seeds than AUC.
        AP and AUC are measured on the SAME runs, so their sds are correlated
        and their ratio is far tighter than two independent intervals would
        suggest.  We enumerate every one of the 5^5 = 3125 paired resamples of
        the five seeds, recompute both seed counts, and take the percentile
        interval of the ratio.  This is the right analysis and it is exact
        (no Monte Carlo error) at n=5.

    The two parts have opposite verdicts, which is the point: the absolute
    seed counts are soft, the AP-vs-AUC comparison is not.

Reads : results/field_baseline.txt (per-seed VadCLIP numbers, parsed),
        results/mde_table.txt (our arms).
Writes: results/sd_ci.txt

USAGE
    python resolution/mde_and_sigma_ci.py
"""
import itertools
import os
import re
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

ALPHA = 0.05
POWER = 0.80


def seeds_needed(sd, delta=0.01, nmax=6000):
    """Smallest n with MDE(n) <= delta, using t quantiles (not 2.9*sigma/sqrt n)."""
    for n in range(2, nmax):
        df = n - 1
        if (stats.t.ppf(1 - ALPHA / 2, df) + stats.t.ppf(POWER, df)) \
                * sd / np.sqrt(n) <= delta:
            return n
    return None


def sigma_ci(s, n):
    """95% interval on sigma from a sample sd s at n observations."""
    df = n - 1
    return (s * np.sqrt(df / stats.chi2.ppf(0.975, df)),
            s * np.sqrt(df / stats.chi2.ppf(0.025, df)))


def parse_per_seed():
    """Per-seed AUC/AP for UCF and XD, parsed from the audit file."""
    path = os.path.join(REPO, "results", "field_baseline.txt")
    if not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8", errors="replace").read()
    # take the CONSISTENT block only: the as-logged block uses a different AP
    # column on XD and is not comparable
    start = txt.index("(A) CONSISTENT")
    end = txt.index("(B) AS-LOGGED")
    blk = txt[start:end]
    # The two tables live in the same block and their rows look identical, so
    # each segment must be CUT at the next dataset header.  Slicing with
    # `split(ds, 1)[1]` silently swallows both tables for the first dataset
    # (10 rows instead of 5) and produces an sd that is wrong by 5x.
    marks = [(ds, blk.index(ds)) for ds in ("UCF-Crime", "XD-Violence")]
    marks.sort(key=lambda t: t[1])
    out = {}
    for i, (ds, pos) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(blk)
        seg = blk[pos:end]
        rows = re.findall(r"^\s+\d+\s+(0\.\d+)\s+(0\.\d+)\s*$", seg, re.M)
        if len(rows) != 5:
            raise RuntimeError("%s: parsed %d seed rows, expected 5" % (ds, len(rows)))
        a = np.array([float(r[0]) for r in rows])
        p = np.array([float(r[1]) for r in rows])
        out[ds] = (a, p)
    return out


def main():
    fh_path = os.path.join(REPO, "results", "sd_ci.txt")
    fh = open(fh_path, "w", encoding="utf-8")

    def W(s=""):
        print(s, flush=True)
        fh.write(s + "\n")

    W("Uncertainty of the seed-noise estimates"
      "  (95% interval, chi-square on sigma; paired enumeration on the ratio)")
    W("=" * 78)

    # ------------------------------------------------------------- our arms
    ours = [("ours E1  AP (1 ep)", 0.0124, 4),
            ("ours E2  AP (1 ep)", 0.0345, 4),
            ("ours E2b AP (1 ep)", 0.0320, 4)]
    W("")
    W("(1) INTERVAL ON sigma, our own arms (n = 4)")
    W("    %-20s %7s %22s %8s" % ("measurement", "sd", "95% CI on sigma", "width"))
    for name, s, n in ours:
        lo, hi = sigma_ci(s, n)
        W("    %-20s %7.4f  [%7.4f, %7.4f]  %6.2fx" % (name, s, lo, hi, hi / lo))
    W("    n = 4 gives a %.1fx-wide interval. The absolute seed counts inherit it."
      % (sigma_ci(0.0345, 4)[1] / sigma_ci(0.0345, 4)[0]))

    W("")
    W("    seeds needed to resolve a 1-point gain, propagated through that "
      "interval:")
    W("    %-20s %8s %18s" % ("measurement", "point", "95% CI"))
    for name, s, n in ours:
        lo, hi = sigma_ci(s, n)
        W("    %-20s %8s  [%5s, %6s]"
          % (name, seeds_needed(s), seeds_needed(lo), seeds_needed(hi)))
    W("    Read: even at the OPTIMISTIC end our arms need 7-32 seeds for one "
      "point;")
    W("    the field reports single-seed gains of 0.5-2 points.")

    # --------------------------------------------------- reference, per seed
    per = parse_per_seed()
    if per is None:
        W("")
        W("(2) could not parse per-seed values -- skipped")
        fh.close()
        return 1

    W("")
    W("(2) THE COMPARATIVE CLAIM: AP vs AUC, same seeds")
    W("    AP and AUC come from the same runs, so the ratio is resampled "
      "PAIRED")
    W("    (all 5^5 resamples enumerated; no Monte Carlo error).")
    W("")
    W("    %-12s %8s %20s %10s %10s" % ("dataset", "point", "95% CI on ratio",
                                        "P(ratio<1)", "P(ratio<2)"))
    for ds in ("UCF-Crime", "XD-Violence"):
        a, p = per[ds]
        sa, sp = float(a.std(ddof=1)), float(p.std(ddof=1))
        pt = seeds_needed(sp) / seeds_needed(sa)
        ratios = []
        for idx in itertools.product(range(len(a)), repeat=len(a)):
            ai, pi = a[list(idx)], p[list(idx)]
            if ai.std() < 1e-12 or pi.std() < 1e-12:
                continue
            na = seeds_needed(float(ai.std(ddof=1)))
            npi = seeds_needed(float(pi.std(ddof=1)))
            if na and npi:
                ratios.append(npi / na)
        r = np.array(ratios)
        W("    %-12s %7.1fx  [%5.1fx, %5.1fx]  %10.3f %10.3f"
          % (ds, pt, np.percentile(r, 2.5), np.percentile(r, 97.5),
             (r < 1).mean(), (r < 2).mean()))
        W("               (sd AUC %.4f, sd AP %.4f, n = %d seeds, %d resamples)"
          % (sa, sp, len(a), len(r)))
    W("")
    W("    Read: the DIRECTION is robust -- the probability that AP needs no "
      "more")
    W("    seeds than AUC is 0.003 (UCF) and 0.000 (XD). The MAGNITUDE is "
      "soft:")
    W("    the paper's 'three to seven times' should be read as a point "
      "estimate")
    W("    whose interval runs from about 1.3x to 6x.")

    fh.close()
    print("\nwrote", fh_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
