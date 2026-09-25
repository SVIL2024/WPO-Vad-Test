# -*- coding: utf-8 -*-
"""Parse logs_fetch/baseline_5seed.log into a clean table, under BOTH
column conventions.

WHY TWO CONVENTIONS
    Every epoch in the raw log prints four numbers:
        AUC1 / AP1
        AUC2 / AP2
        ... then the summary line:  epoch k AUC=<x> AP=<y>
    _audit_baseline_log.py (not in this release) showed the summary line is built as
        UCF: (AUC1, AP1)   -- 22/22 epochs
        XD:  (AUC1, AP2)   -- 18/18 epochs
    i.e. the two datasets were summarised under DIFFERENT conventions.  The
    original run's own headline "XD AP sd = 0.0339" therefore mixes columns
    across datasets and must not be quoted as a cross-dataset fact.

    This script reports:
      (a) CONSISTENT  -- col1 for both datasets (the only apples-to-apples
          comparison we can make from this log)
      (b) AS-LOGGED   -- exactly what the original run's summary printed, for
          provenance / auditability

Usage:
    python resolution/five_seed_audit.py
"""
import re
from collections import defaultdict

import numpy as np

LOG = os.environ.get("BASELINE_LOG", r"D:\program\waveClipVad\logs_fetch\baseline_5seed.log")
DS_RE = re.compile(r"^\[(?P<ds>ucf|xd)\] seed=(?P<seed>\d+) training")
RAW_RE = re.compile(r"^AUC(?P<i>[12]):\s+(?P<auc>[\d.]+)\s+"
                    r"AP(?P<j>[12]):\s*(?P<ap>[\d.]+)")
SUM_RE = re.compile(r"^\s*epoch\s+\d+/\d+\s+AUC=(?P<auc>[\d.]+)\s+"
                    r"AP=(?P<ap>[\d.]+)")
SEEDS = ("42", "234", "555", "789", "999999")


def parse():
    cur_ds = cur_seed = None
    pending = {}
    col1 = defaultdict(list)
    col2 = defaultdict(list)
    aslogged = defaultdict(list)

    with open(LOG) as f:
        for line in f:
            m = DS_RE.match(line)
            if m:
                cur_ds, cur_seed = m.group("ds"), m.group("seed")
                pending = {}
                continue
            m = RAW_RE.match(line)
            if m and cur_ds:
                pending[m.group("i")] = (float(m.group("auc")),
                                          float(m.group("ap")))
                continue
            m = SUM_RE.match(line)
            if m and cur_ds:
                aslogged[(cur_ds, cur_seed)].append(
                    (float(m.group("auc")), float(m.group("ap"))))
                if "1" in pending:
                    col1[(cur_ds, cur_seed)].append(pending["1"])
                if "2" in pending:
                    col2[(cur_ds, cur_seed)].append(pending["2"])
                pending = {}
    return col1, col2, aslogged


def stats(col, ds, pick_best_by="auc"):
    """Best epoch per seed, then mean +/- sd across seeds."""
    aucs, aps = [], []
    for s in SEEDS:
        pairs = col[(ds, s)]
        if not pairs:
            continue
        best = max(pairs, key=lambda t: t[0 if pick_best_by == "auc" else 1])
        aucs.append(best[0]); aps.append(best[1])
    return np.array(aucs), np.array(aps)


def block(name, col, note=""):
    print("\n" + "=" * 80)
    print("%s   %s" % (name, note))
    print("=" * 80)
    for ds, pretty in (("ucf", "UCF-Crime"), ("xd", "XD-Violence")):
        a, p = stats(col, ds)
        print("\n  %s" % pretty)
        print("    seed     AUC     AP")
        for s, ai, pi in zip(SEEDS, a, p):
            print("    %-8s  %.4f   %.4f" % (s, ai, pi))
        print("    ------------------------------------------")
        print("    mean      %.4f   %.4f" % (a.mean(), p.mean()))
        print("    sd        %.4f   %.4f" % (a.std(ddof=1), p.std(ddof=1)))
        print("    range     %.4f   %.4f" % (a.max() - a.min(), p.max() - p.min()))
        print("    AP/AUC sd ratio = %.1fx" % (p.std(ddof=1) / a.std(ddof=1)))
    return a, p


def main():
    col1, col2, aslogged = parse()

    print("=" * 80)
    print("FIELD-LEVEL 5-SEED BASELINE  --  logs_fetch/baseline_5seed.log")
    print("=" * 80)
    print("\n  AUDIT (_audit_baseline_log.py (not in this release)): the run's own summary line is")
    print("  built as  UCF=(AUC1,AP1)  but  XD=(AUC1,AP2).  Two different")
    print("  conventions across datasets -> do NOT quote 'XD AP sd' as a")
    print("  cross-dataset fact without saying which column it came from.")

    block("(A) CONSISTENT  -- col1 for BOTH datasets", col1,
          "(the only apples-to-apples reading)")
    block("(B) AS-LOGGED  -- exactly what the run printed", aslogged,
          "(UCF=col1, XD=col1-AUC + col2-AP)")

    print("\n" + "=" * 80)
    print("OUR INTERNAL (4 seeds, 1 epoch, fixed-timebase)  -- paper section 3.1")
    print("=" * 80)
    e1 = np.array([0.7074, 0.7177, 0.7358, 0.7123])
    e2 = np.array([0.8227, 0.7446, 0.8114, 0.7898])
    e2b = np.array([0.7949, 0.7211, 0.7633, 0.7389])
    print("  E1  (K=1) AP mean=%.4f sd=%.4f range=%.4f"
          % (e1.mean(), e1.std(ddof=1), e1.max() - e1.min()))
    print("  E2  (K=3) AP mean=%.4f sd=%.4f range=%.4f"
          % (e2.mean(), e2.std(ddof=1), e2.max() - e2.min()))
    print("  E2b (K=3) AP mean=%.4f sd=%.4f range=%.4f"
          % (e2b.mean(), e2b.std(ddof=1), e2b.max() - e2b.min()))

    print("\n" + "=" * 80)
    print("CROSS-PLATFORM AP sd, consistent col1 convention")
    print("=" * 80)
    rows = []
    for nm, arr in (("ours E1  (4 seed, 1 ep)", e1),
                    ("ours E2  (4 seed, 1 ep)", e2),
                    ("ours E2b (4 seed, 1 ep)", e2b)):
        rows.append((nm, arr.mean(), arr.std(ddof=1)))
    for ds, pretty in (("ucf", "VadCLIP UCF (5 seed, 5+ ep)"),
                       ("xd", "VadCLIP XD  (5 seed, 5+ ep)")):
        _, p = stats(col1, ds)
        rows.append((pretty, p.mean(), p.std(ddof=1)))
    print("  %-28s %10s %10s" % ("config", "AP mean", "AP sd"))
    for nm, m, s in rows:
        print("  %-28s %10.4f %10.4f" % (nm, m, s))
    print("\n  ALL five independent measurements show AP sd in 0.009 - 0.035.")
    print("  Ours (1 epoch) is the noisiest; VadCLIP (5+ epochs, early stop) is")
    print("  tighter.  The P3 claim 'single-seed AP is unreliable' holds on all")
    print("  of them -- but the earlier '0.0339 matches our 0.0345 digit for digit'")
    print("  line was a column-mismatch artefact and has been withdrawn.")


if __name__ == "__main__":
    main()
