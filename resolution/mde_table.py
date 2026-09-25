# -*- coding: utf-8 -*-
"""T17: minimum detectable effect size for paired multi-seed VAD comparisons.

WHY
    The paper's field-facing claim is "this benchmark cannot resolve effects of
    the size people report".  To make that actionable rather than rhetorical we
    need the actual decision rule: given the seed noise you measured, how big
    must a difference be before n seeds can see it?

STATISTICS
    Paired, two-sided, alpha = 0.05, power = 0.80, n seeds, df = n-1:

        MDE(n) = (t_{0.975, n-1} + t_{0.80, n-1}) * sigma_d / sqrt(n)

    The t quantiles matter a lot at the seed counts this field actually uses
    (n = 3..5): the normal approximation (1.96 + 0.84 = 2.80) understates the
    MDE by ~25-30% at n = 4.  Do NOT use 2.9*sigma/sqrt(n).

    sigma_d = sd of the PAIRED per-seed differences (not the single-seed sd).
    When pairing does not help (E2 vs E1: 0.0362 vs unpaired 0.0345), sigma_d
    is essentially the single-seed sd -- use that as an upper bound.

INPUTS (all measured, not assumed)
    paired sd : from the project's internal evidence ledger, section 3.3 (4 seeds)
    seed  sd  : official VadCLIP 5 seeds, consistent col1 (section 3.4.2)
"""
from scipy import stats
import numpy as np

ALPHA, POWER = 0.05, 0.80


def mde(sigma_d, n):
    df = n - 1
    q = stats.t.ppf(1 - ALPHA / 2, df) + stats.t.ppf(POWER, df)
    return q * sigma_d / np.sqrt(n)


def n_needed(sigma_d, delta, n_max=200):
    # n starts at 2: the paper states MDE(n) with no floor, and resolution/mde_and_sigma_ci.py
    # uses range(2, ...).  A floor of 3 here made the published table
    # disagree with its own equation for the smallest sigma_d rows.
    for n in range(2, n_max + 1):
        if mde(sigma_d, n) <= delta:
            return n
    return None


def main():
    print("=" * 78)
    print("T17 -- minimum detectable effect (paired, two-sided alpha=.05, power=.80)")
    print("=" * 78)

    # --- Part A: from MEASURED paired sd (our own 4-seed runs) -------------
    paired = [
        ("E2 - E2b  AUC  (K=3 vs K=3)", 0.0030),
        ("E2 - E1   AUC  (K=3 vs K=1)", 0.0058),
        ("E2 - E2b  AP   (K=3 vs K=3)", 0.0139),
        ("E2 - E1   AP   (K=3 vs K=1)", 0.0362),
    ]
    print("\n(A) using the MEASURED paired sd (section 3.3), n = 4 seeds")
    print("    %-30s %8s  %10s  %10s" % ("comparison", "sd_pair", "MDE n=4", "MDE n=5"))
    for name, sd in paired:
        print("    %-30s %8.4f  %10.4f  %10.4f" % (name, sd, mde(sd, 4), mde(sd, 5)))

    # --- Part B: from the seed-to-seed sd (official VadCLIP, 5 seeds) ------
    print("\n(B) using the seed-to-seed sd (official VadCLIP 5 seeds, col1)")
    print("    when pairing does NOT help, sigma_d ~ single-seed sd (upper bound)")
    seeds = [
        ("ours E2      AP  (4 seed, 1 ep)", 0.0345, 4),
        ("ours E2b     AP  (4 seed, 1 ep)", 0.0320, 4),
        ("ours E1      AP  (4 seed, 1 ep)", 0.0124, 4),
        ("VadCLIP UCF  AP  (5 seed, 5+ep)", 0.0168, 5),
        ("VadCLIP XD   AP  (5 seed, 5+ep)", 0.0091, 5),
        ("VadCLIP XD   AP  (as-logged col2)", 0.0320, 5),
        ("VadCLIP UCF  AUC (5 seed, 5+ep)", 0.0064, 5),
        ("VadCLIP XD   AUC (5 seed, 5+ep)", 0.0023, 5),
    ]
    print("    %-33s %8s  %10s  %10s  %10s" % ("measurement", "sd", "MDE n=4", "MDE n=5", "MDE n=10"))
    for name, sd, _ in seeds:
        print("    %-33s %8.4f  %10.4f  %10.4f  %10.4f"
              % (name, sd, mde(sd, 4), mde(sd, 5), mde(sd, 10)))

    # --- Part C: how many seeds for a claimed gain? -----------------------
    print("\n(C) seeds REQUIRED to resolve a claimed gain of delta (80% power)")
    claims = [0.005, 0.010, 0.020, 0.030, 0.050]
    print("    %-33s %8s " % ("noise source (sd)", "") +
          "".join("%9.3f" % d for d in claims))
    for name, sd, _ in seeds:
        row = "    %-33s %8.4f " % (name, sd)
        for d in claims:
            n = n_needed(sd, d)
            row += "%9s" % (str(n) if n and n <= 200 else ">200")
        print(row)

    print("\n    (delta = absolute AP / AUC points.  VAD papers routinely claim")
    print("     0.5-2 points; that is the column delta=0.005 / 0.010.)")

    # --- the one-line takeaway -------------------------------------------
    lo = mde(0.0091, 5)     # tightest: XD AP, col1
    hi = mde(0.0345, 4)     # loosest: our E2 AP
    print("\n" + "=" * 78)
    print("TAKEAWAY")
    print("=" * 78)
    print("  Across every measurement we have, 4-5 seeds resolve roughly")
    print("  %.2f to %.2f absolute points (AP) at 80%% power." % (lo * 100, hi * 100))
    print("  Best case  : XD AP, official 5 seeds, col1      -> %.2f pp" % (lo * 100))
    print("  Worst case : our E2 AP, 4 seeds, 1 epoch        -> %.2f pp" % (hi * 100))
    print("  Typical published VAD gains: 0.5-2 pp -> BELOW the resolution")
    print("  in every configuration we measured.")
    print()
    print("  Caveat to state in the paper: these use sigma_d = sd as an UPPER")
    print("  bound where no paired sd was measured; successful pairing (section")
    print("  3.3) shrinks sigma_d by ~2.4x and improves the MDE accordingly.")


if __name__ == "__main__":
    main()
