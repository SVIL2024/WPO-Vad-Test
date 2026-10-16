#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decimation arm: separate the pooling kernel from the branch factors.

Reproduces Table 12 (tab:decimate) and the lower block of Table 9 (tab:paired) of
the paper, from the raw per-seed training logs in ``logs/``.

The design, in one paragraph.  The default arm pools with
``F.avg_pool1d(kernel=f, stride=f)``; the fine arm changes the factors.  That
contrast moves the stride and the pooling kernel at once.  The decimated arm
keeps the factors at 1,8,16 and replaces the boxcar with a pick of every f-th
frame, so the stride, the output sampling rate, the output length, the mask
convention and all weights match the default arm, and the kernel is the only
candidate left.

What this script deliberately does NOT establish:  ``avg_pool1d`` with
kernel == stride is the standard *anti-aliased* downsample, so its boxcar is at
once an average over f consecutive frames and a low-pass that suppresses
fold-over above the new Nyquist rate.  Removing it removes the averaging AND
admits aliasing.  The contrast therefore attributes the difference to the kernel;
separating mixing from anti-aliasing needs a windowed-sinc resampler, which was
not run.  The paper says so in Section 5.3 and in the limitations.  Do not quote
the ratio below as "cross-frame mixing is the cause".

Usage
-----
    python resolution/decimation_arm.py [--logs logs]

Exits non-zero unless every value quoted in the paper is reproduced.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seedstats import close, holm, logs_dir, mean_sd, paired, read_dir, t_sf  # noqa: E402

# Everything the paper prints for this experiment.  The script fails if any of it
# does not come back out of the logs.
PAPER_ARMS = {
    "default (1,8,16) pooled": (0.7961, 0.0255, 0.9370),
    "fine (1,2,4) pooled": (0.7705, 0.0262, 0.9329),
    "1,8,16 decimated": (0.7798, 0.0154, 0.9306),
}
PAPER_PAIRED = {
    # label                   delta     sd        t      p      p(Holm)
    # Labels read as (second) - (first); the value is that difference, so a
    # negative number means the second-named arm scored below the first-named one.
    "decimated - default":   (-0.0163, 0.0142, -3.25, 0.0140, 0.0297),
    "fine - default":        (-0.0256, 0.0207, -3.51, 0.0099, 0.0297),
    "fine - decimated":      (-0.0093, 0.0155, -1.70, 0.1333, 0.1333),
}
PAPER_DEFAULT_SINGLE_N8 = (0.0757, 0.0272, 7.87, 0.0001)
PAPER_KERNEL_SHARE = 64.0     # kernel-only cost as a % of the published cost


def main():
    ap = argparse.ArgumentParser(description="decimation arm reproduction")
    ap.add_argument("--logs", default=logs_dir())
    args = ap.parse_args()
    logs = args.logs

    out = []
    w = out.append
    failures = []

    w("=" * 78)
    w("DECIMATION ARM  (one epoch, batch 64, single 11 GB card)")
    w("logs: %s" % os.path.normpath(logs))
    w("=" * 78)
    w("")

    arms_dir = os.path.join(logs, "arms_n8")
    default = read_dir(arms_dir, r"e4b_default_e1_s(\d+)\.log$")
    fine = read_dir(arms_dir, r"e4b_fine_e1_s(\d+)\.log$")
    single = read_dir(arms_dir, r"e4b_single_e1_s(\d+)\.log$")
    decimated = read_dir(os.path.join(logs, "decimation_n8"), r"br_p1_decim_s(\d+)\.log$")

    for nm, d in (("default", default), ("fine", fine), ("single", single),
                  ("decimated", decimated)):
        if not d:
            failures.append("no %s logs under %s" % (nm, logs))

    seeds = sorted(set(default) & set(fine) & set(decimated))
    w("shared seeds across default / fine / decimated: %s  (n=%d)" % (seeds, len(seeds)))
    w("")

    # --- alignment gate ----------------------------------------------------
    w("-" * 78)
    w("ALIGNMENT GATE  (the control is only meaningful if the decimated run")
    w("differs from the default in the pooling mode and nothing else the log shows)")
    w("-" * 78)
    cfg = {}
    for nm, d in (("default", default), ("fine", fine), ("decimated", decimated)):
        vals = sorted({v[2] for v in d.values() if v[2]})
        cfg[nm] = vals
        w("  %-10s %s" % (nm, " | ".join(vals) if vals else "(no [Config] line)"))
    dcfg = (cfg.get("decimated") or [""])[0]
    pcfg = (cfg.get("default") or [""])[0]
    dfac = re.match(r"scales=([\d,]+)", dcfg)
    pfac = re.match(r"scales=([\d,]+)", pcfg)
    if dfac and pfac:
        if dfac.group(1) != pfac.group(1):
            w("  FAIL factors differ: default %s vs decimated %s"
              % (pfac.group(1), dfac.group(1)))
            failures.append("factors differ between default and decimated")
        else:
            w("  ok   factors identical (%s) -- only the mode differs" % dfac.group(1))
        mode = dcfg.split("(")[-1].rstrip(")") or "?"
        if mode == "decimate":
            w("  ok   decimated arm reports pooling mode '%s'" % mode)
        else:
            w("  FAIL decimated arm reports pooling mode '%s', expected 'decimate'" % mode)
            failures.append("decimated mode is %r" % mode)
    else:
        w("  could not read a multi_scale config fragment from the logs")
        failures.append("config fragment unreadable")
    w("")

    # --- Table 12 ----------------------------------------------------------
    w("-" * 78)
    w("TABLE 12  three arms over the shared seeds")
    w("-" * 78)
    w("  %-26s %7s %7s %8s %4s" % ("arm", "AP", "sd", "AUC", "n"))
    got_arms = {}
    for nm, d in (("default (1,8,16) pooled", default),
                  ("fine (1,2,4) pooled", fine),
                  ("1,8,16 decimated", decimated)):
        if not d:
            continue
        ss = [s for s in seeds if s in d]
        aps = [d[s][0] for s in ss]
        aucs = [d[s][1] for s in ss if d[s][1] is not None]
        am, asd = mean_sd(aps)
        um = mean_sd(aucs)[0] if len(aucs) > 1 else float("nan")
        got_arms[nm] = am
        w("  %-26s %7.4f %7.4f %8.4f %4d" % (nm, am, asd, um, len(aps)))
        w("  %-26s %s" % ("", " ".join("%.4f" % v for v in aps)))
        want = PAPER_ARMS[nm]
        for label, got, exp in (("AP", am, want[0]), ("sd", asd, want[1]),
                                ("AUC", um, want[2])):
            if not close(got, exp):
                w("  FAIL %s %s: paper %.4f, got %.4f" % (nm, label, exp, got))
                failures.append("%s %s" % (nm, label))
    w("")

    # --- paired contrasts --------------------------------------------------
    w("-" * 78)
    w("PAIRED CONTRASTS  (same-seed differences, df = n-1, Holm over the three)")
    w("-" * 78)
    series = {"default": {s: default[s][0] for s in seeds},
              "fine": {s: fine[s][0] for s in seeds},
              "decimated": {s: decimated[s][0] for s in seeds},
              "single": {s: single[s][0] for s in seeds if s in single}}
    combos = [("default", "decimated"), ("default", "fine"), ("decimated", "fine")]
    ps, rows = [], []
    for a, b in combos:
        try:
            # value is b - a, and the label says so, so a negative number means
            # the second-named arm scored below the first-named one.
            delta, sd, t, p, n, _ = paired(series[a], series[b])
        except (ValueError, KeyError):
            continue
        ps.append(p)
        rows.append(("%s - %s" % (b, a), delta, sd, t, p, n))
    adj = holm(ps)
    w("  %-22s %9s %8s %7s %8s %8s %3s" %
      ("contrast", "delta", "sd", "t", "p", "p(Holm)", "n"))
    for (label, delta, sd, t, p, n), pa in zip(rows, adj):
        w("  %-22s %+9.4f %8.4f %7.2f %8.4f %8.4f %3d" %
          (label, delta, sd, t, p, pa, n))
        want = PAPER_PAIRED[label]
        for nm_, got, exp in (("delta", delta, want[0]), ("sd", sd, want[1]),
                              ("t", t, want[2]), ("p", p, want[3]),
                              ("p(Holm)", pa, want[4])):
            if not abs(got - exp) <= (5e-3 if nm_ == "t" else 5e-4):
                w("  FAIL %s %s: paper %.4f, got %.4f" % (label, nm_, exp, got))
                failures.append("%s %s" % (label, nm_))
    w("")

    # --- the ratio the paper quotes ---------------------------------------
    deltas = {r[0]: r[1] for r in rows}
    if "decimated - default" in deltas and "fine - default" in deltas:
        share = 100.0 * deltas["decimated - default"] / deltas["fine - default"]
        w("-" * 78)
        w("RATIO QUOTED IN SECTION 5.3")
        w("-" * 78)
        w("  kernel-only %.4f / published %.4f = %.1f%%      paper %.0f%%"
          % (deltas["decimated - default"], deltas["fine - default"],
             share, PAPER_KERNEL_SHARE))
        if not abs(share - PAPER_KERNEL_SHARE) <= 1.0:
            failures.append("kernel share %.1f%% vs paper %.0f%%"
                            % (share, PAPER_KERNEL_SHARE))
        w("")
        w("  A ratio of two measured differences, NOT an additive decomposition.")
        w("  The boxcar is also the anti-alias filter, so this attributes the")
        w("  difference to the kernel and not to frame averaging alone.")
        w("")

    # --- Table 9 lower block, default minus single ------------------------
    try:
        # single is the control here, so the value comes out as default - single
        delta, sd, t, p, n, _ = paired(series["single"], series["default"])
    except (ValueError, KeyError):
        delta = None
    if delta is not None:
        w("-" * 78)
        w("TABLE 9, lower block   default - single-branch (AP), n=%d" % n)
        w("-" * 78)
        w("  %+.4f +/- %.4f   t=%.2f   p=%.4f" % (delta, sd, t, p))
        w("  paper %+.4f +/- %.4f   t=%.2f   p=%.4f" % PAPER_DEFAULT_SINGLE_N8)
        for nm_, got, exp in (("delta", delta, PAPER_DEFAULT_SINGLE_N8[0]),
                              ("sd", sd, PAPER_DEFAULT_SINGLE_N8[1]),
                              ("t", t, PAPER_DEFAULT_SINGLE_N8[2]),
                              ("p", p, PAPER_DEFAULT_SINGLE_N8[3])):
            if not abs(got - exp) <= (5e-3 if nm_ == "t" else 5e-4):
                failures.append("default-single %s" % nm_)
        w("")

    w("=" * 78)
    if failures:
        w("FAILED (%d): %s" % (len(failures), "; ".join(failures)))
    else:
        w("PASS  every value quoted in the paper was reproduced from the logs.")
    w("=" * 78)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.stdout.write("\n".join(out) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())