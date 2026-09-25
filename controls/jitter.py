# -*- coding: utf-8 -*-
"""Experiment J2 -- can encoder jitter REPRODUCE the P1 anomaly signature?

Experiment J (controls/jitter_v1.py) established the jitter FLOOR: an
imperceptible sigma = 2/255 pixel perturbation produces high-frequency feature
power comparable to (or larger than) the real signal's.  That alone does NOT
falsify P1 -- what matters is the SIGN of the shape shift.

The P1 signature (PATCH view, Ped2) is:
    band        21-64   10-21    5-10    2- 5    1- 2
    real delta  +0.1269 -0.0253 -0.0399 -0.0404 -0.0213
i.e. the SLOWEST band goes UP and every faster band goes DOWN.

Jitter is temporally WHITE: it adds power proportional to the number of FFT
bins in each band, i.e. mostly to the fast bands (12 bins at 2-5 fr, 9 bins at
1-2 fr, only 2 bins at 21-64 fr).  After renormalising to sum 1 that must push
the fast bands UP and the slow band DOWN -- the OPPOSITE sign to P1.

If that prediction holds, jitter cannot explain the finding; in fact it biases
AGAINST it, so the measured mid/fast-band deficits are conservative.

This script measures the jitter-induced shape shift directly by encoding the
same all-normal TRAIN frames clean and with additive pixel noise, then reports
the cosine similarity between the jitter shift and the real P1 shift.

PRE-REGISTERED READING
    cos < 0   -> jitter moves the shape the OPPOSITE way to the anomaly
                 signature; P1's deficits are conservative.  (Expected.)
    cos > 0.7 -> jitter reproduces the anomaly signature; P1 must be withdrawn
                 as an encoder artefact.
"""
import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)


# The statistic exists in exactly one place (premise_test/spectral.py).  This
# script is run as `python <dir>/<name>.py`, so the directory holding that
# module is not on sys.path by default; add it rather than keep a second copy
# of the statistic here.
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _d in ("premise_test", "controls"):
    _p = _os.path.normpath(_os.path.join(_HERE, "..", _d))
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)

import spectral as S
import extract_backbones as E

RAW = os.environ.get("PED2_RAW", r"D:\dataset\UCSD\UCSDped2")
BATCH = 32
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
TRAIN_CLIPS = ["Train001", "Train002", "Train003"]
SIGMAS = [0.5, 1.0, 2.0, 4.0]


def main():
    import clip
    model, pre = clip.load("ViT-B/16", device="cpu", download_root=os.path.join(REPO, "_ckpt"))
    model.eval().float()
    std_t = torch.tensor(CLIP_STD).view(1, 3, 1, 1)

    frames = []
    for c in TRAIN_CLIPS:
        frames += sorted(glob.glob(os.path.join(RAW, "Train", c, "*.tif")))
    print("[J2] %d train frames from %s" % (len(frames), TRAIN_CLIPS), flush=True)

    t0 = time.time()
    Xg, Xp = [], []
    for i in range(0, len(frames), BATCH):
        ch = frames[i:i + BATCH]
        x = torch.stack([pre(Image.open(p).convert("RGB")) for p in ch])
        with torch.no_grad():
            g, sp = E.forward("clip_vit_b16", model, x)
        Xg.append(g.numpy().astype(np.float32))
        Xp.append(E.as_tokens(sp).numpy().astype(np.float32))
    Xg = np.concatenate(Xg)
    Xp = np.concatenate(Xp)
    # guard against the (B,M,C) vs (B,C,M) mix-up: ViT must give 196 patches
    assert Xp.shape[1:] == (196, 768), "patch axis order wrong: %s" % (Xp.shape,)
    print("[J2] clean encoded %s in %.0fs" % (Xp.shape, time.time() - t0), flush=True)

    P = S.detrender()
    hann = np.hanning(S.WIN).astype(np.float32)
    nb = len(S.BAND_EDGES) - 1

    # PCA fitted on the CLEAN patches, reused for every noise level so the
    # comparison is apples-to-apples
    flat = Xp.reshape(-1, Xp.shape[2])
    pca = S.fit_pca(flat[::max(1, len(flat) // 50000)])

    def shape(series):
        """Paper statistic: (T,M,C) -> (n_band,) averaged over windows."""
        acc = np.zeros(nb)
        n = 0
        for st in range(0, len(series) - S.WIN + 1, S.STRIDE):
            sh, ok = S.spec_shape(series[st:st + S.WIN] @ pca, P, hann)
            if not ok.any():
                continue
            acc += sh[:, ok].mean(1)
            n += 1
        return acc / max(n, 1)

    base = shape(Xp)
    print("\n[J2] clean shape      %s" % "".join("%10.4f" % v for v in base))

    rows = {}
    for s in SIGMAS:
        Yg, Yp = [], []
        rng = np.random.default_rng(0)
        for i in range(0, len(frames), BATCH):
            ch = frames[i:i + BATCH]
            x = torch.stack([pre(Image.open(p).convert("RGB")) for p in ch])
            nz = torch.tensor(rng.normal(0.0, s / 255.0, size=x.shape).astype(np.float32))
            with torch.no_grad():
                g, sp = E.forward("clip_vit_b16", model, x + nz / std_t)
            Yg.append(g.numpy().astype(np.float32))
            Yp.append(E.as_tokens(sp).numpy().astype(np.float32))
        Yp = np.concatenate(Yp)
        d = shape(Yp) - base
        rows[s] = d
        print("[J2] sigma=%4.1f/255 delta %s" % (s, "".join("%+10.4f" % v for v in d)),
              flush=True)

    # ---- the real P1 signature ---------------------------------------------
    # single source of truth: reuse the cached PATCH windows that produced
    # results/backbone_robustness.txt, so this can never drift from the
    # numbers that go into the paper.
    z = np.load(os.path.join(REPO, "results", "_bb_cache.npz"), allow_pickle=True)
    real = (z["clip_vit_b16_pA"].mean(0) - z["clip_vit_b16_pN"].mean(0)).astype(np.float64)
    print("\n[J2] REAL P1 delta    %s" % "".join("%+10.4f" % v for v in real))

    print("\n" + "=" * 78)
    print("J2: can jitter reproduce the P1 anomaly signature?")
    print("=" * 78)
    print("  %-12s %10s %10s   %s" % ("sigma(/255)", "cos", "|d|/|real|", "verdict"))
    nr = np.linalg.norm(real)
    for s in SIGMAS:
        d = rows[s]
        c = float(d @ real / (np.linalg.norm(d) * nr + 1e-12))
        v = ("OPPOSITE sign - P1 conservative" if c < 0 else
             "SAME sign - P1 at risk" if c > 0.7 else "weakly aligned")
        print("  %-12.1f %10.3f %10.3f   %s" % (s, c, np.linalg.norm(d) / nr, v))
    print("\n  cos < 0 means jitter moves the spectral shape the OPPOSITE way to")
    print("  the anomaly signature, so the measured mid/fast-band DEFICITS are")
    print("  conservative lower bounds, not artefacts.")
    print("\n  total %.1f min" % ((time.time() - t0) / 60))

    # cache so the figure can never disagree with this table
    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    np.savez(os.path.join(REPO, "results", "_jitter2_cache.npz"),
             base=base, real=real,
             sigmas=np.array(SIGMAS),
             deltas=np.stack([rows[s] for s in SIGMAS]),
             labels=np.array([S.band_label(i) for i in range(nb)]))
    print("[J2] cached -> results/_jitter2_cache.npz")


if __name__ == "__main__":
    main()
