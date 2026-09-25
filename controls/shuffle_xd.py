# -*- coding: utf-8 -*-
"""P7 (temporal-order control) repeated ON THE XD FEATURES THAT P1 USED.

WHY
    P7 on Ped2 showed |d_shuf|/|d_orig| = 0.026: the spectral-shape statistic
    really does depend on frame ORDER, not just on the marginal distribution of
    the features.  But Ped2 is a different dataset AND a different extraction
    (we re-encoded raw TIFF frames ourselves).  A reviewer can therefore say:
    "fine, but that says nothing about the XD measurement your paper actually
    leads with."

    This script runs the same control directly on XD-Violence, using the very
    same features and the very same statistic as P1
    (_diag_is_anomaly_oscillatory.py (not in this release): WIN=64, STRIDE=32, 16-d PCA, detrend
    + Hann + rfft + 5 bands).  If the ratio is small here too, the "temporal"
    reading is established on P1's own data.

DESIGN
    d_orig  = mean shape(anomalous clips) - mean shape(normal clips)   (5 bands)
    d_shuf  = same, but each clip's FRAME AXIS is permuted first.
    Pre-registered reading (same as Ped2):
        ratio < 0.3  -> the statistic measures temporal structure
        ratio > 0.8  -> it only measures the marginal distribution

SANITY GATE (learned the hard way on Ped2, see results/superseded/shuffle_ped2_v1_SUPERSEDED.txt)
    Before drawing any conclusion, check that d_orig reproduces P1's published
    direction: the mid band (10-21 fr) must be NEGATIVE.  If this pipeline
    cannot reproduce P1, the shuffle verdict is meaningless and we abort.

USAGE
    python controls/shuffle_xd.py
"""
import io
import os
import re
import sys
import zipfile

import numpy as np

ZIP = os.environ.get("XD_FEATURES_ZIP", r"D:\dataset\XDTrainClipFeatures.zip")
WIN = 64
STRIDE = 32
BAND_EDGES = [1, 3, 6, 12, 24, 33]
N_PER_CLASS = 120
N_SHUFFLE = 5
PCA_DIM = 16
PCA = None

LABEL = "bins %2d-%2d (period %2d-%2d fr)"


def band_labels():
    return [LABEL % (BAND_EDGES[i], BAND_EDGES[i + 1] - 1,
                     WIN // BAND_EDGES[i + 1], WIN // BAND_EDGES[i])
            for i in range(len(BAND_EDGES) - 1)]


# ---------------------------------------------------------------------- data
def load_index(z):
    vids = {}
    for n in z.namelist():
        if not n.endswith(".npy"):
            continue
        m = re.match(r"^(.*)__(\d+)$", os.path.basename(n)[:-4])
        if m:
            vids.setdefault(m.group(1), []).append((int(m.group(2)), n))
    for v in vids.values():
        v.sort()
    return vids


def is_normal(vid):
    return "_label_A" in vid


def load_video(z, entries):
    # MUST close each member: `z.open(n).read()` leaves the handle open, and
    # after a few thousand members Windows runs out and every later read throws
    # -- which the caller's `except: continue` would then silently swallow.
    out = []
    for _, n in entries:
        with z.open(n) as fh:
            out.append(np.load(io.BytesIO(fh.read())).astype(np.float32))
    return np.concatenate(out, axis=0)


# ------------------------------------------------------------------- metrics
def shape_vector(x):
    """Normalised spectral shape averaged over windows. Returns (5,)."""
    s = x @ PCA
    T = s.shape[0]
    if T < WIN + STRIDE:
        return None
    hann = np.hanning(WIN).astype(np.float32)[:, None]
    acc, nw = np.zeros(len(BAND_EDGES) - 1), 0
    for st in range(0, T - WIN + 1, STRIDE):
        w = s[st:st + WIN]
        t = np.arange(WIN, dtype=np.float32)
        A = np.stack([np.ones_like(t), t], axis=1)
        w = w - A @ np.linalg.lstsq(A, w, rcond=None)[0]
        P = (np.abs(np.fft.rfft(w * hann, axis=0)) ** 2).mean(1)
        tot = P[1:].sum()
        if tot <= 1e-12:
            continue
        P = P / tot
        for i in range(len(BAND_EDGES) - 1):
            acc[i] += P[BAND_EDGES[i]:BAND_EDGES[i + 1]].sum()
        nw += 1
    return acc / nw if nw else None


def main():
    global PCA
    z = zipfile.ZipFile(ZIP)
    vids = load_index(z)
    normals = sorted(v for v in vids if is_normal(v))
    anoms = sorted(v for v in vids if not is_normal(v))
    print("[P7-XD] %d normal / %d anomalous videos in zip" % (len(normals), len(anoms)),
          flush=True)

    rng = np.random.default_rng(0)
    sel_n = list(rng.choice(normals, min(N_PER_CLASS, len(normals)), replace=False))
    sel_a = list(rng.choice(anoms, min(N_PER_CLASS, len(anoms)), replace=False))

    # ---- PCA on a MIXED sample (normal + anomalous), matching P1's fit ----
    print("[P7-XD] fitting %d-d PCA on %d clips ..." % (PCA_DIM, len(sel_n) + len(sel_a)),
          flush=True)
    X, tot = [], 0
    for v in sel_n + sel_a:
        try:
            x = load_video(z, vids[v])
        except Exception:
            continue
        X.append(x[::3])
        tot += x[::3].shape[0]
        if tot > 200000:
            break
    X = np.concatenate(X, axis=0).astype(np.float64)
    X -= X.mean(0, keepdims=True)
    w, V = np.linalg.eigh(X.T @ X / len(X))
    PCA = V[:, ::-1][:, :PCA_DIM].astype(np.float32)
    print("[P7-XD] PCA done on %d frames" % tot, flush=True)

    # ---- load the two samples into memory (small: (T,512) fp32) ----
    def build(sel, tag):
        out = []
        for v in sel:
            try:
                x = load_video(z, vids[v])
            except Exception:
                continue
            if x.shape[0] >= WIN + STRIDE:
                out.append(x)
        print("[P7-XD] %s: %d clips usable" % (tag, len(out)), flush=True)
        return out

    N = build(sel_n, "normal")
    A_ = build(sel_a, "anomalous")

    def delta(shuffle_seed=None):
        rs = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
        mn, ma, na, nb = None, None, 0, 0
        acc_n = np.zeros(len(BAND_EDGES) - 1)
        acc_a = np.zeros(len(BAND_EDGES) - 1)
        for x in N:
            if rs is not None:
                x = x[rs.permutation(x.shape[0])]
            s = shape_vector(x)
            if s is None:
                continue
            acc_n += s; na += 1
        for x in A_:
            if rs is not None:
                x = x[rs.permutation(x.shape[0])]
            s = shape_vector(x)
            if s is None:
                continue
            acc_a += s; nb += 1
        mn = acc_n / max(na, 1)
        ma = acc_a / max(nb, 1)
        return ma - mn, na, nb

    d_orig, na, nb = delta(None)
    print("\n[P7-XD] n_windows-clips: normal=%d anomalous=%d" % (na, nb), flush=True)
    print("[P7-XD] ORIGINAL delta (anom - normal):", flush=True)
    for lab, v in zip(band_labels(), d_orig):
        print("    %s   %+0.4f" % (lab, v), flush=True)

    # ---- SANITY GATE: must reproduce P1's mid-band depletion ----
    MID = 1  # 10-21 fr band
    if d_orig[MID] >= 0:
        print("\n[FATAL] mid-band delta is %+0.4f (>= 0): this pipeline does NOT "
              "reproduce P1 -- aborting, do not draw any conclusion."
              % d_orig[MID], flush=True)
        sys.exit(1)
    print("[P7-XD] sanity OK: mid-band delta %+0.4f is negative, matches P1."
          % d_orig[MID], flush=True)

    # ---- shuffles ----
    ds = []
    for k in range(N_SHUFFLE):
        d, _, _ = delta(1000 + k)
        ds.append(d)
        print("[P7-XD] shuffle %d: %s" % (k, "  ".join("%+0.4f" % v for v in d)),
              flush=True)
    ds = np.stack(ds)
    d_shuf = ds.mean(0)
    nrm = np.linalg.norm(d_orig) + 1e-12
    cos = float(d_orig @ d_shuf / (nrm * (np.linalg.norm(d_shuf) + 1e-12)))
    ratio = float(np.linalg.norm(d_shuf) / nrm)

    print("\n" + "=" * 70, flush=True)
    print("P7 on XD-Violence: does the statistic depend on temporal order?", flush=True)
    print("=" * 70, flush=True)
    print("  cos(d_orig, d_shuf_mean)      = %+0.3f" % cos, flush=True)
    print("  |d_shuf_mean| / |d_orig|      = %0.3f" % ratio, flush=True)
    print("  |d_orig|                      = %0.4f" % np.linalg.norm(d_orig), flush=True)
    print("  |d_shuf_mean|                 = %0.4f" % np.linalg.norm(d_shuf), flush=True)
    verdict = ("temporal structure -- 'temporal' framing OK" if ratio < 0.3
               else "largely a marginal-distribution property -- 'temporal' "
                    "wording is MISLEADING" if ratio > 0.8 else "inconclusive band")
    print("  verdict: %s" % verdict, flush=True)
    print("\n  pre-registered: ratio < 0.3 -> temporal; ratio > 0.8 -> not temporal.",
          flush=True)
    print("  Ped2 reference (P7): ratio = 0.026, cos = -0.16", flush=True)

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                             "results"), exist_ok=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "results", "xd_shuffle.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("P7 shuffle control on XD-Violence (P1's own features)\n")
        fh.write("n: normal=%d anomalous=%d clips, %d shuffles\n\n" % (na, nb, N_SHUFFLE))
        fh.write("d_orig  : " + "  ".join("%+0.4f" % v for v in d_orig) + "\n")
        fh.write("d_shuf  : " + "  ".join("%+0.4f" % v for v in d_shuf) + "\n\n")
        for i, lab in enumerate(band_labels()):
            fh.write("  %s  orig %+0.4f  shuf %+0.4f\n" % (lab, d_orig[i], d_shuf[i]))
        fh.write("\ncos = %+0.3f\nratio |d_shuf|/|d_orig| = %0.3f\n" % (cos, ratio))
    print("\nwrote %s" % out, flush=True)


if __name__ == "__main__":
    main()
