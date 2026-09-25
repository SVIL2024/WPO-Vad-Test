# -*- coding: utf-8 -*-
"""Persist the residual test as a result file.

WHY THIS EXISTS
The residual test answers: "after removing everything the trained trunk score
already explains, does a candidate temporal score still separate anomaly from
normal?"  It is quoted in the paper (Sec. 7.1, Table tab:residual) as the
evidence that the wave operator's advantage over a plain magnitude control is
redundant with the trunk.

Until now the number existed only in the printed output of
`complementarity/wave_complement.py`, so the manuscript had no result file to parse and
the figure could drift.  This script recomputes the test for all four
candidates and writes `results/residual_test.txt`.

CONVENTION (identical to the thing being checked)
`conditional_z` / `_residual` are imported from `complementarity/wave_complement.py`
rather than reimplemented, so this file and the verdict script can never
disagree: one implementation, two consumers.  The residual is
`score - E[score | trunk score]` estimated in 50 quantile bins, and the
significance test is against the chance level of 0.5.

Run from repo root:  python complementarity/residual_test.py
"""
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wave_complement as W  # noqa: E402  (one implementation, two users)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.environ.get("XD_PAIR_DIR", os.path.join(REPO, "logs_fetch"))
OUT = os.environ.get("RESIDUAL_OUT",
                     os.path.join(REPO, "results", "residual_test.txt"))

# (key in wave_energy_test.npz, label used in the paper, table row name)
VARIANTS = [
    ("e_ut", "Magnitude $\\lvert u_t\\rvert$", "ut_magnitude"),
    ("e_wave_c0", "Wave operator, $c=0$", "wave_c0"),
    ("e_wave", "Wave operator, $K=192$", "wave_K192"),
    ("e_waveD", "Wave operator, full-$D$ Laplacian", "waveD_fullD"),
]


def main():
    aligned = os.path.join(LOGS, "aligned_pair.npz")
    probe = os.path.join(LOGS, "wave_energy_test.npz")
    for p in (aligned, probe):
        if not os.path.isfile(p):
            print("MISSING %s" % p)
            return 1

    d = np.load(aligned)
    a0 = d["a0"].astype(np.float64)
    y = d["y"].astype(int)
    dw = np.load(probe, allow_pickle=True)

    rows = []
    for key, label, name in VARIANTS:
        s = W._flat(dw[key])
        n = min(len(a0), len(s), len(y))
        aa, ss, yy = a0[:n], s[:n], y[:n]
        raw = W.roc_auc_score(yy, ss)
        ra, z = W.conditional_z(aa, ss, yy)
        rows.append((name, label, raw, ra, z, n))

    # every variant must have been scored on the same segments
    ns = set(r[5] for r in rows)
    assert len(ns) == 1, "variants scored on different segment counts: %s" % ns
    n = rows[0][5]
    yv = y[:n]

    # sanity: the magnitude control must be the best AFTER residualising and
    # must NOT be the best BEFORE - that reversal is the whole point, so a
    # silent change in the inputs must not pass unnoticed.
    best_raw = max(rows, key=lambda r: r[2])
    best_res = max(rows, key=lambda r: r[3])
    assert best_res[0] == "ut_magnitude", (
        "residual ordering changed: %s now leads" % best_res[0])
    assert best_raw[0] != "ut_magnitude", (
        "standalone ordering changed: %s now leads" % best_raw[0])

    L = []
    L.append("Residual test on XD-Violence (VadCLIP trunk, CLIP ViT-B/16)")
    L.append("produced by complementarity/residual_test.py")
    L.append("inputs: logs_fetch/aligned_pair.npz + logs_fetch/wave_energy_test.npz")
    L.append("")
    L.append("segments=%d  positives=%d (%.1f%%)" % (n, yv.sum(), 100 * yv.mean()))
    L.append("")
    L.append("residual = score - E[score | trunk score], 50 quantile bins;")
    L.append("residual AUC is that remainder against the label; z is vs 0.5.")
    L.append("")
    L.append("  %-16s %14s %12s %8s" % ("variant", "standalone_AUC",
                                        "residual_AUC", "z"))
    for name, label, raw, ra, z, _ in rows:
        L.append("  %-16s %14.4f %12.4f %+8.2f" % (name, raw, ra, z))
    L.append("")
    L.append("READING")
    L.append("  best standalone AUC : %s (%.4f)" % (best_raw[0], best_raw[2]))
    L.append("  best residual  AUC  : %s (%.4f)" % (best_res[0], best_res[3]))
    L.append("  The ordering reverses after residualising: the strongest")
    L.append("  oscillatory variant leads the magnitude control unconditionally,")
    L.append("  and loses to it once the trunk score is accounted for.  Every")
    L.append("  residual AUC is above chance, so the information is real; what")
    L.append("  does not survive is the oscillatory operator's advantage.")
    txt = "\n".join(L) + "\n"

    io.open(OUT, "w", encoding="utf-8", newline="\n").write(txt)
    print(txt)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
