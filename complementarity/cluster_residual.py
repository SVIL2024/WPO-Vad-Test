# -*- coding: utf-8 -*-
"""R2, second half: the residual test with the VIDEO as the resampling unit.

WHY
    `results/residual_test.txt` reports z = +27.60 for the magnitude
    control on 145,701 segments.  Those segments come from 800 videos, about
    182 per video, and segments inside one video are not independent draws:
    the closed-form z (Hanley--McNeil SE) assumes they are, so it is an upper
    bound on the evidence, which is what the paper's Table caption says.
    This script replaces that admission with a number.

WHAT IT DOES
    1. Rebuilds the segment -> video map exactly the way `_diag_align_check.py`
       built the aligned pair: match the per-video segment-count sequences to
       locate the one video the probe dropped, and ABORT unless exactly one
       removal makes the match exact.  Without this the cluster unit would be
       a guess.
    2. CONTROL: recomputes the published segment-level residual AUC and z and
       compares them with the values in `results/residual_test.txt`
       (parsed, not retyped).  Aborts on any mismatch, so the "after" numbers
       cannot come from a different statistic.
    3. CLUSTER BOOTSTRAP: resample the VIDEOS with replacement, pool their
       segments, and recompute the AUC of the SAME residual scores.  The
       residual is built once on the full sample (the 50 quantile bins are a
       property of the score construction, not of the resample); the bootstrap
       measures the sampling variability of the AUC itself.  Reports the 95%
       interval, the SE ratio SE_boot/SE_Hanley--McNeil (how much the published
       z is inflated by pooling correlated segments), and the design effect,
       which is that ratio squared -- the ratio of variances, not of standard
       errors.  Calling the SE ratio itself the design effect is a factor-of-two
       error in the exponent, and it was made here first and copied into the
       manuscript.

The residual machinery and the AUC function are imported from
`complementarity/wave_complement.py`, never reimplemented: one implementation, three
consumers, so the convention cannot drift.

Writes results/cluster_residual.txt
"""
import io
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LOGS = os.environ.get("XD_PAIR_DIR", os.path.join(REPO, "logs_fetch"))
RES = os.environ.get("RESULTS_DIR", os.path.join(REPO, "results"))
sys.path.insert(0, HERE)

import wave_complement as W  # noqa: E402  (one implementation, three users)

OUT = os.environ.get("CLUSTER_RESID_OUT",
                     os.path.join(RES, "cluster_residual.txt"))
BOOT = 2000
SEED = 20260924

# (key in wave_energy_test.npz, row name as printed in residual_test.txt)
VARIANTS = [
    ("e_ut", "ut_magnitude"),
    ("e_wave_c0", "wave_c0"),
    ("e_wave", "wave_K192"),
    ("e_waveD", "waveD_fullD"),
]

lines = []


def emit(s=""):
    print(s, flush=True)
    lines.append(s)


def published():
    """residual_test.txt -> {row name: (residual AUC, z)} for the control."""
    txt = io.open(os.path.join(RES, "residual_test.txt"),
                  encoding="utf-8").read()
    out = {}
    for name, _raw, ra, z in re.findall(
            r"^\s{2}(\S+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)\s*$", txt, re.M):
        out[name] = (float(ra), float(z))
    if len(out) != len(VARIANTS):
        sys.exit("residual_test.txt: parsed %d rows, expected %d"
                 % (len(out), len(VARIANTS)))
    return out


def video_ids():
    """One video id per segment of the aligned pair, or abort."""
    d0 = np.load(os.path.join(LOGS, "scores_xd_base_s234.npz"),
                 allow_pickle=True)
    dw = np.load(os.path.join(LOGS, "wave_energy_test.npz"), allow_pickle=True)
    A = np.array([len(np.atleast_1d(p)) for p in d0["probs"]])
    key = next(c for c in ("e_wave_c0", "e_wave", "e_waveD", "e_ut")
               if c in dw.files)
    Wn = np.array([len(np.atleast_1d(v)) for v in dw[key]])

    perfect = [k for k in range(len(A))
               if len(np.concatenate([A[:k], A[k + 1:]])) == len(Wn)
               and bool((np.concatenate([A[:k], A[k + 1:]]) == Wn).all())]
    if len(perfect) != 1:
        sys.exit("segment-count match is not unique: videos %s all fit"
                 % perfect)
    skip = perfect[0]

    ids = np.concatenate([np.full(n, i) for i, n in enumerate(A) if i != skip])
    m = np.load(os.path.join(LOGS, "aligned_pair.npz"))["a0"].shape[0]
    if len(ids) != m:
        sys.exit("video map has %d entries but the aligned pair has %d"
                 % (len(ids), m))
    return ids, skip, len(A)


def main():
    emit("=" * 74)
    emit("R2 -- residual test at the segment unit vs the video unit")
    emit("=" * 74)

    ids, skip, n_vid = video_ids()
    uniq = np.unique(ids)
    emit("  videos in the trunk dump        : %d" % n_vid)
    emit("  video dropped by the probe      : #%d" % skip)
    emit("  videos behind the residual test : %d (map exact, %d segments,"
         % (len(uniq), len(ids)))
    emit("  %.1f per video)" % (len(ids) / float(len(uniq))))

    d = np.load(os.path.join(LOGS, "aligned_pair.npz"))
    a0 = d["a0"].astype(np.float64)
    y = d["y"].astype(int)
    dw = np.load(os.path.join(LOGS, "wave_energy_test.npz"), allow_pickle=True)
    pub = published()

    emit("")
    emit("  CONTROL: reproduce the published segment-level numbers")
    resid = {}
    for key, name in VARIANTS:
        s = W._flat(dw[key])[:len(a0)]
        r = W._residual(a0, s)
        auc = W.roc_auc_score(y, r)
        _, se_hm = W.auc_se(y, r)
        z = (auc - 0.5) / se_hm
        exp_ra, exp_z = pub[name]
        if abs(auc - exp_ra) > 1e-4 or abs(z - exp_z) > 0.01:
            sys.exit("  [ABORT] %s: published (%.4f, %+.2f), recomputed "
                     "(%.4f, %+.2f)" % (name, exp_ra, exp_z, auc, z))
        resid[name] = r
        emit("    %-14s residual AUC %.4f  z %+6.2f   matches the published row"
             % (name, auc, z))

    idx = {v: np.where(ids == v)[0] for v in uniq}
    rng = np.random.default_rng(SEED)

    emit("")
    emit("  CLUSTER BOOTSTRAP: resample the %d videos with replacement, pool"
         % len(uniq))
    emit("  their segments, recompute the AUC of the same residual scores")
    emit("  (%d resamples).  SE ratio = bootstrap SE / Hanley-McNeil SE --"
         % BOOT)
    emit("  how much the published z is inflated.  deff is the design effect,")
    emit("  the VARIANCE ratio, i.e. the SE ratio squared.  z_video is the")
    emit("  same AUC against the clustered SE.")
    emit("")
    emit("  %-14s %10s %8s %25s %9s %9s %8s %9s %7s"
         % ("variant", "resid AUC", "z seg", "95% bootstrap CI (AUC)",
            "SE boot", "SE HM", "z video", "SE ratio", "deff"))

    summary = {}
    for key, name in VARIANTS:
        r = resid[name]
        auc = W.roc_auc_score(y, r)
        _, se_hm = W.auc_se(y, r)
        boot = np.empty(BOOT, dtype=np.float64)
        for b in range(BOOT):
            pick = uniq[rng.integers(0, len(uniq), len(uniq))]
            rows = np.concatenate([idx[v] for v in pick])
            boot[b] = W.roc_auc_score(y[rows], r[rows])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        se_b = float(np.std(boot, ddof=1))
        z_vid = (auc - 0.5) / se_b if se_b > 1e-12 else float("nan")
        de = se_b / se_hm if se_hm > 1e-12 else float("nan")
        deff = de ** 2
        emit("  %-14s %10.4f %+8.2f   [%+.4f, %+.4f] %9.5f %9.5f %+8.2f %8.1fx %6.0fx"
             % (name, auc, pub[name][1], lo, hi, se_b, se_hm, z_vid, de, deff))
        summary[name] = dict(auc=auc, z_seg=pub[name][1], lo=lo, hi=hi,
                             se_b=se_b, se_hm=se_hm, z_vid=z_vid, de=de,
                             deff=deff)

    emit("")
    emit("READING")
    emit("  The residual AUCs are identical by construction -- only the standard")
    emit("  error moves, and it moves by the SE ratio, so the published z")
    emit("  is inflated by that factor.  The paper's claim about this table is")
    emit("  the ORDERING (the magnitude control leads after residualising, and")
    emit("  the strongest oscillatory variant loses to it), not the magnitude of")
    emit("  any z; the ordering is a property of the point estimates and so is")
    emit("  untouched by the unit.")
    worst = max(summary, key=lambda k: summary[k]["de"])
    emit("  Largest SE ratio: %s, z %+.2f -> %+.2f (%.1fx, i.e. deff %.0fx)."
         % (worst, summary[worst]["z_seg"], summary[worst]["z_vid"],
            summary[worst]["de"], summary[worst]["deff"]))
    los = min(summary[k]["de"] for k in summary)
    his = max(summary[k]["de"] for k in summary)
    emit("  SE ratio range %.1f-%.1fx  ->  deff range %.0f-%.0fx."
         % (los, his, los ** 2, his ** 2))
    still = sorted(k for k in summary if summary[k]["z_vid"] > 3.0)
    emit("  Variants that keep z > 3 at the video unit: %s."
         % (", ".join(still) if still else "none"))
    emit("")

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
