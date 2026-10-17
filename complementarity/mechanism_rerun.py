#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mechanism re-measurement at n=4 (the (b) and (c) rows of the mechanism table).

The mechanism screen in the paper is n=1 against a shared control, which cannot
distinguish a small gain from noise.  This script re-runs the two cheapest rows of
that screen as same-seed paired differences at four seeds against the default arm,
at the same one-epoch budget the screen used:

  (b) wave resonator bank    screen said -0.0317 AP
  (c) PDE residual, lambda=2.5   screen said +0.0000 AP

and reproduces Table 7 (tab:mechanisms), bottom block.

Two things the paper is careful about, repeated here so the numbers cannot be
quoted out of that context:

  * Neither row is distinguishable from its control, and we do **not** claim
    equivalence, because no equivalence margin was fixed before these runs.
    The report prints the screen value beside the measurement precisely so the
    gap between them is visible: the (b) screen value does not even keep its
    sign, which is a further instance of the paper's own thesis rather than a
    correction of the screen.
  * The state-space contrast (d) is the one positive-direction row and is still
    n=1.  It is deliberately absent from this script; see the paper.

Usage
-----
    python mechanism_rerun.py [--logs ../logs]

Exits non-zero unless every value quoted in the paper is reproduced.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seedstats import close, holm, logs_dir, paired, read_dir  # noqa: E402

TOL = 5e-4

# seeds used by the re-measurement; all four are a subset of the eight-seed arm
# run, so the paired control is available from the same logs
BATCH_SEEDS = (42, 234, 555, 789)

PAPER = {
    # name                    delta      sd        t      p      holm
    "wave resonator bank":   (0.0051, 0.0281, 0.36, 0.7415, 0.7804),
    "PDE residual":          (0.0047, 0.0094, 1.00, 0.3902, 0.7804),
}
PAPER_SCREEN = {
    # the n=1 value printed in the middle block of Table 7
    "wave resonator bank": -0.0317,
    "PDE residual": 0.0000,
}
# the two yardsticks the paper reads a mechanism delta against
YARDSTICK_PAPER = 0.0345     # published one-epoch frame AP sd
YARDSTICK_CONVERGED = 0.0133  # measured converged sd, Table 11


def main():
    ap = argparse.ArgumentParser(description="mechanism re-measurement at n=4")
    ap.add_argument("--logs", default=logs_dir())
    args = ap.parse_args()
    logs = args.logs

    out = []
    w = out.append
    failures = []

    default = read_dir(os.path.join(logs, "arms_n8"), r"e4b_default_e1_s(\d+)\.log$")
    wrb = read_dir(os.path.join(logs, "mechanism_n4"), r"br_p2_wrb_s(\d+)\.log$")
    pde = read_dir(os.path.join(logs, "mechanism_n4"), r"br_p2_pde_s(\d+)\.log$")

    w("=" * 78)
    w("MECHANISM RE-MEASUREMENT AT n=4  (one epoch, batch 64)")
    w("logs: %s" % os.path.normpath(logs))
    w("=" * 78)
    w("")
    w("control = the default (1,8,16) arm at the same four seeds, from the")
    w("eight-seed arm run; arm - control is a same-seed paired difference.")
    w("")

    ctrl = {s: default[s][0] for s in BATCH_SEEDS if s in default}
    if len(ctrl) < 2:
        w("NO CONTROL LOGS under %s -- cannot reproduce the table."
          % os.path.join(logs, "arms_n8"))
        failures.append("control logs missing")
        w("")
    else:
        w("  control AP: %s" % " ".join("%d:%.4f" % (s, ctrl[s]) for s in sorted(ctrl)))
        w("")

    arms = [("wave resonator bank", wrb), ("PDE residual", pde)]
    ps, rows = [], []
    for name, d in arms:
        if not d:
            failures.append("no logs for %s" % name)
            continue
        arm_vals = {s: d[s][0] for s in ctrl if s in d}
        try:
            delta, sd, t, p, n, common = paired(ctrl, arm_vals)
        except ValueError as exc:
            failures.append("%s: %s" % (name, exc))
            continue
        ps.append(p)
        rows.append((name, delta, sd, t, p, n, common))

    adj = holm(ps)
    w("-" * 78)
    w("TABLE 7, bottom block")
    w("-" * 78)
    w("  %-22s %9s %8s %7s %8s %8s %3s" %
      ("row", "delta", "sd", "t", "p", "p(Holm)", "n"))
    for (name, delta, sd, t, p, n, common), pa in zip(rows, adj):
        w("  %-22s %+9.4f %8.4f %7.2f %8.4f %8.4f %3d" %
          (name, delta, sd, t, p, pa, n))
        w("  %-22s seeds %s" % ("", " ".join(str(c) for c in common)))
        want = PAPER[name]
        for label, got, exp in (("delta", delta, want[0]), ("sd", sd, want[1]),
                                ("t", t, want[2]), ("p", p, want[3]),
                                ("p(Holm)", pa, want[4])):
            if not close(got, exp, 5e-3 if label == "t" else 5e-4):
                w("  FAIL %s %s: paper %.4f, got %.4f" % (name, label, exp, got))
                failures.append("%s %s" % (name, label))
        w("")

    if rows:
        w("-" * 78)
        w("SCREEN VALUE (n=1) BESIDE THE MEASUREMENT (n=4)")
        w("-" * 78)
        for name, delta, sd, t, p, n, _ in rows:
            w("  %-22s screen %+.4f  ->  measured %+.4f +/- %.4f   (%+.4f AP)"
              % (name, PAPER_SCREEN[name], delta, sd, delta - PAPER_SCREEN[name]))
        w("")
        w("  The (b) row does not keep its sign.  The two values are reported side")
        w("  by side rather than one replacing the other: replacing it would imply")
        w("  the screen was wrong when it was run, when what changed is the seed")
        w("  count.  That gap is an instance of the paper's own thesis.")
        w("")

    if rows and ctrl:
        w("-" * 78)
        w("SIZE AGAINST THE TWO YARDSTICKS")
        w("-" * 78)
        for name, delta, sd, t, p, n, _ in rows:
            w("  %-22s |delta|=%.4f = %.2f sigma (published %.4f), %.2f sigma (converged %.4f)"
              % (name, abs(delta), abs(delta) / YARDSTICK_PAPER, YARDSTICK_PAPER,
                 abs(delta) / YARDSTICK_CONVERGED, YARDSTICK_CONVERGED))
        w("")

    w("=" * 78)
    if failures:
        w("FAILED (%d): %s" % (len(failures), "; ".join(failures)))
    else:
        w("PASS  every value quoted in the paper was reproduced from the logs.")
    w("=" * 78)
    w("")
    w("NOT measured here, by design:")
    w("  * (d) SSM omega != 0 vs omega = 0, the one positive-direction row of")
    w("    Table 7, remains at its original single seed.  It needs batch 16")
    w("    (its state tensor is about 12x mamba2's) and was not re-run.")
    w("  * (a) the wave-field rows are NOT measured here, but they ARE measured:")
    w("    see complementarity/wavefield_rerun.py for the single-cell row,")
    w("    wavefield_factorial.py for the batch-size / fusion / gain factorial, and")
    w("    wavefield_warm.py for the four warm-start rows.  What is true is that")
    w("    they are not comparable to the rows above: the wave-field module and the")
    w("    multi-scale branches do not coexist in one source tree, so each family is")
    w("    measured against its own control in its own harness.")
    w("    An earlier version of this file said they 'cannot be re-measured at")
    w("    any seed count'; that was false and contradicted the paper.")

    text = "\n".join(out)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.stdout.write(text + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())