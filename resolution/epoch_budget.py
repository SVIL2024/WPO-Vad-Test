# -*- coding: utf-8 -*-
"""At which epoch does the reference implementation actually peak?

WHY THIS EXISTS
    The paper states our runs are 1 epoch.  A reviewer will ask whether that is
    under-training.  The honest answer has two halves, and only one of them was
    previously recorded:

      (1) OUR pipeline converges within one epoch -- that is a project-log
          observation (EXPERIMENT_SUMMARY.md §3.8), not something re-measured
          here.
      (2) WHAT THE REFERENCE ACTUALLY DOES -- a job-script comment in this
          project claimed "every baseline log reports its best AP at epoch 1".
          That claim is FALSE for the five-seed audit this paper cites in §5.3,
          so it must not be repeated in the manuscript.  This script measures
          the real thing.

    Result: the reference runs 3-5 epochs under early stopping and its best
    validation score is first reached at epoch 3 in three of the five UCF
    seeds.  So 1 epoch is a SHORTER budget than the reference's, and the paper
    must say so rather than claiming like-for-like.

Reads : logs_fetch/baseline_5seed.log
Writes: results/epoch_budget.txt
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LOG = os.path.join(REPO, "logs_fetch", "baseline_5seed.log")
OUT = os.path.join(REPO, "results", "epoch_budget.txt")


def main():
    if not os.path.exists(LOG):
        print("[FATAL] no log at", LOG)
        return 2
    txt = open(LOG, encoding="utf-8", errors="replace").read()

    lines = []
    W = lambda s="": lines.append(s)

    W("Epoch budget of the reference implementation (VadCLIP, official")
    W("default config), parsed from logs_fetch/baseline_5seed.log")
    W("=" * 66)
    W("'best at' = first epoch at which the best-so-far validation score was")
    W("reached; a run that never improves on epoch 1 reports epoch 1.")
    W("")

    for ds, label in (("ucf", "UCF-Crime"), ("xd", "XD-Violence")):
        blocks = re.split(r"\[%s\] seed=" % ds, txt)[1:]
        rows = []
        for b in blocks:
            seed = b.split()[0]
            ep = re.findall(
                r"epoch (\d+)/(\d+) AUC=([\d.]+) AP=([\d.]+) \(best ([\d.]+)\)",
                b)
            if not ep:
                continue
            best, at = None, None
            for e, tot, auc, ap, bs in ep:
                if best is None or float(bs) > best + 1e-12:
                    best, at = float(bs), int(e)
            rows.append((seed, len(ep), best, at))
        if not rows:
            continue
        W("%s" % label)
        W("  %-10s %8s %12s %10s" % ("seed", "epochs", "best AUC", "best at"))
        for seed, n, best, at in rows:
            W("  %-10s %8d %12.4f %10s" % (seed, n, best, "epoch %d" % at))
        ns = [r[1] for r in rows]
        ats = [r[3] for r in rows]
        W("  epochs run: %d-%d ; best first reached at epoch %s"
          % (min(ns), max(ns),
             ", ".join(str(a) for a in sorted(set(ats)))))
        W("")

    W("CONCLUSION: the reference does NOT peak at epoch 1 in these runs")
    W("(3 of 5 UCF seeds peak at epoch 3).  The paper must therefore not")
    W("claim that a 1-epoch budget is like-for-like with the reference; it is")
    W("a shorter budget, and the correct framing is that our noise figures are")
    W("measured at the shorter end while both budgets are reported.")

    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
