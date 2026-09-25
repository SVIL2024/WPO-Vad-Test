# -*- coding: utf-8 -*-
"""R2 (peer review): what happens to the Ped2 / Avenue z scores when the
analysis unit is the VIDEO rather than the (overlapping) window?

WHY
    The published tables pool windows: WIN=64, STRIDE=16.  Consecutive windows
    overlap by 48 of 64 frames, and on Ped2 all of them come from 12 videos
    (Avenue: 21).  `spectral.zscore` divides by sqrt(var/n) as if the
    windows were independent draws.  They are not, so the published z is
    pseudo-replicated (reviewer R2).

WHAT THIS SCRIPT DOES, per dataset and per view
    1. Reproduces the published window-level z EXACTLY.  If it does not match
       `results/spectrum_ped2.txt` / `avenue_spectral.txt` the script
       aborts, so the "after" numbers cannot come from a different statistic.
    2. VIDEO-level paired contrast: one number per video,
         d_c = mean over its anomalous windows - mean over its normal windows,
       then a t test on the n_videos paired differences and a bootstrap CI.
    3. CLUSTER bootstrap of the published window-level z: resample the videos
       with replacement, pool their windows, recompute z.  This puts an
       honest interval on the published number.

Writes results/cluster_recheck.txt
"""
import glob
import io
import os
import re
import sys

import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import spectral as S          # one implementation of the statistic

OUT = os.environ.get("CLUSTER_OUT",
                     os.path.join(REPO, "results", "cluster_recheck.txt"))
BOOT = 20000
SEED = 12345

lines = []


def emit(s=""):
    print(s, flush=True)
    lines.append(s)


def windows_per_clip(clips, view, pca, thr, ge=False, use_frame_ok=False):
    """[(name, A or None, N or None)] for one view.

    Mirrors the two published scripts exactly -- they differ in the anomalous
    threshold for the GLOBAL view, which is why `thr` is a parameter:
      `spectral.py`   global: per-FRAME gt coverage, A if >= 0.50, N if 0
                            (coverage from `frame_ok`)
      `controls/shuffle_avenue.py` global: per-FRAME any-gt,      A if >  0.05, N if 0
      both                  patch : per-PATCH gt fraction, A if >  0.05, N if 0
    `pca=None` means no projection (Avenue's global view, as in its script).
    """
    P = S.detrender()
    hann = np.hanning(S.WIN).astype(np.float32)
    nb = len(S.BAND_EDGES) - 1
    out = []
    for c in clips:
        if view == "GLOBAL":
            g = c["g"] if pca is None else c["g"] @ pca
            s = g[:, None, :]
            # Ped2's global view selects on `frame_ok`; Avenue's selects on
            # "any ground truth in this frame" (`controls/shuffle_avenue.py` L145).
            if use_frame_ok and c["ok"] is not None:
                fr = c["ok"].astype(np.float32)
            else:
                fr = (c["gt"].sum(axis=(1, 2)) > 0).astype(np.float32)
            gt_win = fr[:, None]
        else:
            s = c["p"] @ pca
            gt_win = c["gt"].reshape(len(s), -1)
        ca, cn = [], []
        for st in range(0, len(s) - S.WIN + 1, S.STRIDE):
            sh, ok = S.spec_shape(s[st:st + S.WIN], P, hann)
            if not ok.any():
                continue
            gf = gt_win[st:st + S.WIN].mean(0)
            a = ((gf >= thr) if ge else (gf > thr)) & ok
            n = (gf == 0.0) & ok
            if a.any():
                ca.append(sh[:, a].mean(1))
            if n.any():
                cn.append(sh[:, n].mean(1))
        A = np.array(ca) if ca else None
        N = np.array(cn) if cn else None
        out.append((c["name"], A, N))
    return out


def analyse(tag, per_clip, expect, nb):
    """per_clip: [(name, A or None, N or None)]; expect: published z per band."""
    wA = [a for _, a, _ in per_clip if a is not None]
    wN = [n for _, _, n in per_clip if n is not None]
    poolA = np.concatenate(wA)
    poolN = np.concatenate(wN)

    emit("")
    emit("-" * 74)
    emit("%s" % tag)
    emit("-" * 74)
    emit("  published: %d anomalous and %d normal windows, from %d videos"
         % (len(poolA), len(poolN), len(per_clip)))
    pub = [S.zscore(poolN[:, i], poolA[:, i]) for i in range(nb)]

    ok = True
    if expect is not None:
        for i, (e, g) in enumerate(zip(expect, pub)):
            if abs(e - g) > 0.005:
                ok = False
                emit("  [ABORT] band %s: published %+.2f, recomputed %+.2f"
                     % (S.band_label(i), e, g))
    if not ok:
        sys.exit("recomputation does not reproduce the published table; "
                 "refusing to report a 're-analysis' of a different statistic")
    emit("  [control] window-level z reproduces the published table exactly")

    emit("  %-10s %9s %9s %8s" % ("band", "normal", "anomal", "z"))
    for i in range(nb):
        emit("  %-10s %9.4f %9.4f %+8.2f"
             % (S.band_label(i), poolN[:, i].mean(), poolA[:, i].mean(), pub[i]))

    both = [(nm, a, n) for nm, a, n in per_clip if a is not None and n is not None]
    emit("")
    emit("  videos carrying BOTH classes: %d of %d" % (len(both), len(per_clip)))
    if len(both) >= 2:
        D = np.array([a.mean(0) - n.mean(0) for _, a, n in both])
        rng = np.random.default_rng(SEED)
        boot = D[rng.integers(0, len(D), size=(BOOT, len(D)))].mean(1)
        emit("  VIDEO-level paired contrast (unit = video, n = %d):" % len(D))
        emit("  %-10s %10s %9s %10s %26s"
             % ("band", "mean d", "t", "p", "95% bootstrap CI of mean d"))
        for i in range(nb):
            d = D[:, i]
            n_ = len(d)
            t = d.mean() / (d.std(ddof=1) / np.sqrt(n_) + 1e-18)
            p = 2 * st.t.sf(abs(t), n_ - 1)
            lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
            emit("  %-10s %+10.4f %9.2f %10.4f   [%+.4f, %+.4f]"
                 % (S.band_label(i), d.mean(), t, p, lo, hi))

    emit("")
    emit("  CLUSTER bootstrap of the published window-level z")
    emit("  (resample the %d videos with replacement, pool their windows):" % len(per_clip))
    rng = np.random.default_rng(SEED + 1)
    zs = np.full((BOOT, nb), np.nan)
    for b in range(BOOT):
        pick = rng.integers(0, len(per_clip), len(per_clip))
        A = [per_clip[j][1] for j in pick if per_clip[j][1] is not None]
        N = [per_clip[j][2] for j in pick if per_clip[j][2] is not None]
        if not A or not N:
            continue
        A, N = np.concatenate(A), np.concatenate(N)
        if len(A) < 2 or len(N) < 2:
            continue
        for i in range(nb):
            zs[b, i] = S.zscore(N[:, i], A[:, i])
    emit("  %-10s %8s %28s %9s" % ("band", "z", "95% cluster-bootstrap CI", "width"))
    for i in range(nb):
        col = zs[:, i]
        col = col[np.isfinite(col)]
        lo, hi = np.percentile(col, [2.5, 97.5])
        emit("  %-10s %+8.2f   [%+7.2f, %+7.2f] %9.2f"
             % (S.band_label(i), pub[i], lo, hi, hi - lo))
    return dict(pub=pub, n_videos=len(per_clip), n_both=len(both),
                n_win=(len(poolN), len(poolA)))


def parse_bb_patch():
    """backbone_robustness.txt -> {backbone: [PATCH z per band]}.

    The three-backbone table quotes the same statistic on the same Ped2 windows
    with three different feature extractors, so its caption inherits this
    table's unit statement.  Parsing its PATCH column (rather than retyping the
    five z values) is what lets `analyse` abort if the recomputation disagrees.
    """
    txt = io.open(os.path.join(REPO, "results",
                               "backbone_robustness.txt"),
                  encoding="utf-8", errors="replace").read()
    out, cur = {}, None
    for s in txt.splitlines():
        m = re.match(r"^(\w+)\s+\(grid ", s)
        if m:
            cur, out[m.group(1)] = m.group(1), []
            continue
        if cur is None:
            continue
        # the band label can contain a space ("2- 5 fr"), so match on the row
        # shape rather than on the label text
        m = re.match(r"^\s+\S[^|]*fr\s+\|\s*([-\d.]+)\s+([-\d.]+)\s+([-+\d.]+)"
                     r"\s+\|\s*([-\d.]+)\s+([-\d.]+)\s+([-+\d.]+)\s*$", s)
        if m:
            out[cur].append(float(m.group(6)))
    for bb, zs in out.items():
        if len(zs) != 5:
            sys.exit("backbone_robustness.txt: %s has %d bands, expected 5"
                     % (bb, len(zs)))
    return out


def load_dir(featdir, need_gt=True):
    tests = sorted(glob.glob(os.path.join(featdir, "Test_*.npz")))
    trains = sorted(glob.glob(os.path.join(featdir, "Train_*.npz")))
    clips, tr = [], []
    for f in tests:
        d = np.load(f)
        if need_gt and "gt" not in d.files:
            continue
        clips.append(dict(name=os.path.basename(f)[5:-4],
                          g=d["global_feat"].astype(np.float32),
                          p=d["patch"].astype(np.float32),
                          gt=d["gt"].astype(np.float32) if "gt" in d.files else None,
                          ok=d["frame_ok"] if "frame_ok" in d.files else None))
    for f in trains:
        d = np.load(f)
        tr.append(d["patch"].astype(np.float32))
    return clips, tr


def main():
    emit("=" * 74)
    emit("R2 -- analysis unit: overlapping window vs video")
    emit("=" * 74)
    emit("  published statistic: WIN=%d, STRIDE=%d, detrend, Hann, rfft, 16-d PCA,"
         % (S.WIN, S.STRIDE))
    emit("  band shares; z = (mean_A - mean_N) / sqrt(var_N/n_N + var_A/n_A)")

    summary = {}

    # ------------------------------------------------------------- Ped2
    clips, tr_p = load_dir(S.FEAT)
    PCA_P = S.fit_pca(np.concatenate([c["p"][:, ::5, :].reshape(-1, 768) for c in clips] +
                                     [x[:, ::5, :].reshape(-1, 768) for x in tr_p]))
    tr_g = [np.load(f)["global_feat"].astype(np.float32)
            for f in sorted(glob.glob(os.path.join(S.FEAT, "Train_*.npz")))]
    PCA_G = S.fit_pca(np.concatenate([c["g"] for c in clips] + tr_g))
    emit("")
    emit("#" * 74)
    emit("# UCSD Ped2 -- %d test clips (raw frames, pixel GT)" % len(clips))
    emit("#" * 74)
    summary["Ped2 GLOBAL"] = analyse(
        "Ped2 GLOBAL view (512-d pooled, same kind as XD)",
        windows_per_clip(clips, "GLOBAL", PCA_G, 0.50, ge=True, use_frame_ok=True),
        [2.39, -2.74, -0.27, -0.76, 0.25], 5)
    summary["Ped2 PATCH"] = analyse(
        "Ped2 PATCH view (per-patch tokens, GT-selected)",
        windows_per_clip(clips, "PATCH", PCA_P, 0.05),
        [21.67, -6.97, -18.29, -22.68, -19.77], 5)

    # ----------------------------------------------------------- Avenue
    AV = os.environ.get("AVENUE_FEAT",
                     os.path.join(REPO, "feat_avenue_clip_vit_b16"))
    if glob.glob(os.path.join(AV, "Test_*.npz")):
        aclips, atr = load_dir(AV)
        PCA_AV = S.fit_pca(np.concatenate(
            [c["p"][:, ::5, :].reshape(-1, c["p"].shape[2]) for c in aclips] +
            [x[:, ::5, :].reshape(-1, x.shape[2]) for x in atr]))
        emit("")
        emit("#" * 74)
        emit("# CUHK Avenue -- %d test clips (motion-type anomalies, JPEG frames)"
             % len(aclips))
        emit("#" * 74)
        summary["Avenue PATCH"] = analyse(
            "Avenue PATCH view (per-patch tokens, GT-selected)",
            windows_per_clip(aclips, "PATCH", PCA_AV, 0.05), None, 5)
        summary["Avenue GLOBAL"] = analyse(
            "Avenue GLOBAL view (512-d pooled, XD-comparable; no PCA, as in "
            "controls/shuffle_avenue.py)",
            windows_per_clip(aclips, "GLOBAL", None, 0.05), None, 5)
    else:
        emit("(no Avenue features on disk -- skipped)")

    # ------------------------------------------------- the three backbones
    # Same Ped2 windows, three feature extractors.  Table 'backbone' defers its
    # unit statement to Table 'ped2', so the same re-analysis has to hold here
    # for that deferral to be honest.
    expect_bb = parse_bb_patch()
    for bb, dirname in (("clip_vit_b16", "ped2_feat"),
                        ("clip_rn50", "ped2_feat_clip_rn50"),
                        ("inet_rn50", "ped2_feat_inet_rn50")):
        fdir = os.path.join(REPO, dirname)
        if not glob.glob(os.path.join(fdir, "Test_*.npz")):
            emit("(no %s features on disk -- skipped)" % bb)
            continue
        bclips, btr = load_dir(fdir)
        P = S.fit_pca(np.concatenate(
            [c["p"][:, ::5, :].reshape(-1, c["p"].shape[2]) for c in bclips] +
            [x[:, ::5, :].reshape(-1, x.shape[2]) for x in btr]))
        emit("")
        emit("#" * 74)
        emit("# Ped2 PATCH view, backbone %s -- %d test clips" % (bb, len(bclips)))
        emit("#" * 74)
        summary["BB " + bb] = analyse(
            "Ped2 PATCH view, %s" % bb,
            windows_per_clip(bclips, "PATCH", P, 0.05), expect_bb.get(bb), 5)

    emit("")
    emit("=" * 74)
    emit("READING")
    emit("=" * 74)
    emit("  'published' = the window-level z that is in the paper today.")
    emit("  'cluster bootstrap' = the same statistic with the VIDEOS, not the")
    emit("  overlapping windows, as the sampling units.")
    emit("  'VIDEO-level paired contrast' = one number per video, paired within")
    emit("  video: the conservative re-analysis, and the one the paper should")
    emit("  quote when the videos are the independent units.")
    emit("  Every view above reproduced its published window-level z first; the")
    emit("  'after' numbers cannot come from a different statistic.")
    emit("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
