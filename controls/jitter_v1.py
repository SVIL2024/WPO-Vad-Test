# -*- coding: utf-8 -*-
"""Experiment J -- how much of the high-frequency band is ENCODER JITTER?

WHY
    Every spectral result in this project is measured on frozen per-frame CLIP
    features.  CLIP has NO temporal modelling, so the frame-to-frame variation
    we call "scene dynamics" could in part be the encoder reacting to
    imperceptible input changes (compression artefacts, resampling, sensor
    noise).  If so, the "excess power at <=5 frames = roughness" reading of P1
    is partly an artefact.

DESIGN (incremental perturbation)
    Encode the SAME Ped2 frames twice: once clean, once with additive Gaussian
    noise of sigma (in 0-255 pixel units) applied AFTER normalisation.
    sigma = 1..4 / 255 is invisible.  The difference
        J = f(x + noise) - f(x)
    is pure encoder sensitivity: the scene is identical, only an imperceptible
    input perturbation changed.  Comparing band power of J against band power
    of X gives the jitter floor per band.

PRE-REGISTERED READING (written before looking at the numbers)
    r_b = P_J(band b) / P_X(band b)
      r_b < ~0.20 -> roughness is predominantly scene dynamics; P1's reading
                     stands as written.
      r_b > ~0.50 -> a large share of the high band is encoder sensitivity;
                     P1's roughness reading must be qualified.
    EITHER WAY the 0.6-6.2% tone bound of section 2.6.2 is UNAFFECTED: the
    power-matched calibration injects a tone ON TOP of the real (already
    jitter-contaminated) signal and measures the INCREMENTAL shift, so it
    already contains the jitter floor.  This experiment can qualify the
    INTERPRETATION; it cannot overturn the BOUND.

    Reporting rule: report both the all-dimension and the 16-d-PCA view.  If
    the two disagree materially, report INCONCLUSIVE rather than picking the
    nicer one.
"""
import argparse
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
INET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def encode(bb, model, x):
    with torch.no_grad():
        g, sp = E.forward(bb, model, x)
    return (g.numpy().astype(np.float32),
            E.as_tokens(sp).numpy().astype(np.float32))


def band_power(series, P, hann, pca=None):
    """Absolute band power of `series` (T, M, C), averaged over windows and
    over the (patch, channel) axes.  Returns (n_band,) ABSOLUTE power -- not
    normalised to 1, because the J/X ratio needs absolute magnitudes."""
    if pca is not None:
        series = series @ pca
    T, M, C = series.shape
    acc = np.zeros(len(S.BAND_EDGES) - 1, np.float64)
    n = 0
    for st in range(0, T - S.WIN + 1, S.STRIDE):
        w = series[st:st + S.WIN].reshape(S.WIN, M * C)
        w = (P @ w).reshape(S.WIN, M, C) * hann[:, None, None]
        pw = (np.abs(np.fft.rfft(w, axis=0)) ** 2).mean(2).mean(1)   # (33,)
        for i in range(len(S.BAND_EDGES) - 1):
            acc[i] += pw[S.BAND_EDGES[i]:S.BAND_EDGES[i + 1]].sum()
        n += 1
    return acc / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip_vit_b16",
                    choices=["clip_vit_b16", "clip_rn50", "inet_rn50"])
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--sigmas", default="1,2,4",
                    help="noise sigma in 0-255 pixel units, comma separated")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",")]
    model, pre, grid = E.load_backbone(args.backbone)
    std = CLIP_STD if args.backbone.startswith("clip") else INET_STD
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    print("[jitter] backbone=%s grid=%d clips=%d sigmas=%s"
          % (args.backbone, grid, args.clips, sigmas), flush=True)
    print("[jitter] channel std = %s" % (std,), flush=True)

    P = S.detrender()
    hann = np.hanning(S.WIN).astype(np.float32)
    nb = len(S.BAND_EDGES) - 1

    tests = sorted(glob.glob(os.path.join(RAW, "Test", "*")))
    tests = [d for d in tests if os.path.isdir(d) and not d.endswith("_gt")]
    tests = tests[:args.clips]

    acc_X = {"g": np.zeros(nb), "p": np.zeros(nb)}
    acc_J = {s: {"g": np.zeros(nb), "p": np.zeros(nb)} for s in sigmas}

    t_all = time.time()
    for d in tests:
        name = os.path.basename(d)
        frames = sorted(glob.glob(os.path.join(d, "*.tif")))
        x = torch.stack([pre(Image.open(p).convert("RGB")) for p in frames])
        Xg, Xp = encode(args.backbone, model, x)
        if args.backbone == "clip_vit_b16":
            assert Xp.shape[1:] == (196, 768), "patch axis order wrong: %s" % (Xp.shape,)

        # 16-d PCA fitted on THIS clip (the paper statistic uses 16 dims)
        flat = Xp.reshape(-1, Xp.shape[2])
        pca = S.fit_pca(flat[::max(1, len(flat) // 50000)])

        acc_X["g"] += band_power(Xg[:, None, :], P, hann, pca=None)
        acc_X["p"] += band_power(Xp, P, hann, pca=pca)

        for s in sigmas:
            rng = np.random.default_rng(args.seed)
            Jg_all, Jp_all = [], []
            for i in range(0, len(x), BATCH):
                xb = x[i:i + BATCH]
                nz = torch.tensor(
                    rng.normal(0.0, s / 255.0, size=xb.shape).astype(np.float32))
                with torch.no_grad():
                    g2, sp2 = E.forward(args.backbone, model, xb + nz / std_t)
                Jg_all.append(g2.numpy().astype(np.float32) - Xg[i:i + BATCH])
                Jp_all.append(E.as_tokens(sp2).numpy().astype(np.float32)
                              - Xp[i:i + BATCH])
            Jg = np.concatenate(Jg_all)
            Jp = np.concatenate(Jp_all)
            acc_J[s]["g"] += band_power(Jg[:, None, :], P, hann, pca=None)
            acc_J[s]["p"] += band_power(Jp, P, hann, pca=pca)
        print("[jitter] %s T=%d  %.0fs" % (name, len(frames), time.time() - t_all),
              flush=True)

    print("\n" + "=" * 78)
    print("ENCODER JITTER FLOOR   backbone=%s   %d clips   noise sigma in /255"
          % (args.backbone, len(tests)))
    print("=" * 78)
    for view in ("g", "p"):
        vname = "GLOBAL (all dims)" if view == "g" else "PATCH (16-d PCA)"
        print("\n  %s" % vname)
        print("    %-10s %13s %s" % ("band", "P_X (abs)",
                                     "".join("%13s" % ("r(s=%g)" % s) for s in sigmas)))
        for b in range(nb):
            row = "".join("%13.4f" % (acc_J[s][view][b] / (acc_X[view][b] + 1e-12))
                          for s in sigmas)
            print("    %-10s %13.4g %s" % (S.band_label(b), acc_X[view][b], row))
    print("\n  r = jitter power / real signal power in the same band.")
    print("  Pre-registered reading: r < 0.2 -> roughness is scene dynamics;")
    print("  r > 0.5 -> roughness is largely encoder sensitivity (qualify P1).")
    print("  Either way the 2.6.2 tone bound is unaffected: the calibration")
    print("  measures an INCREMENTAL shift on the real, jittered signal.")
    print("\n  total %.1f min" % ((time.time() - t_all) / 60))


if __name__ == "__main__":
    main()
