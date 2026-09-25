# -*- coding: utf-8 -*-
"""Experiment J3 -- does realistic COMPRESSION reproduce the P1 signature?

WHY
    Section 2.7.3 flags that P1 was originally measured on XD-Violence, which
    is H.264-compressed, while the jitter control (J1/J2) was run on UCSD Ped2,
    which is UNCOMPRESSED TIFF.  Compression artefacts are exactly the kind of
    imperceptible input perturbation J1 showed the encoder is sensitive to, and
    unlike i.i.d. Gaussian noise they vary from frame to frame in a
    content-dependent way.  So the XD result is more exposed than the Ped2 one
    -- but by how much?

DESIGN
    Take the same all-normal Ped2 TRAIN frames, apply a JPEG round-trip at
    quality q (a cheap, reproducible stand-in for compression artefacts),
    re-encode with CLIP, and measure the shift of the 5-band spectral shape
    relative to the clean encoding.  Compare against:
        the real P1 anomaly signature   (cos, magnitude ratio)
        the Gaussian-noise jitter shift (already measured in J2)

LIMIT (state this in the paper)
    JPEG is INTRA-frame only.  H.264 is inter-frame: its artefacts are
    motion-compensated and therefore temporally correlated in a way JPEG's are
    not.  So this is a PROXY -- a lower bound on the artefact magnitude, not a
    faithful simulation of H.264.  A finding of "small" therefore bounds the
    risk but does not eliminate it; a finding of "large" is decisive.

PRE-REGISTERED READING
    |delta_q| / |delta_real| < 0.3 for q >= 75  -> compression at realistic
        quality moves the spectrum much less than the anomaly effect; soften
        the 2.7.3 caveat.
    |delta_q| / |delta_real| > 1 for realistic q -> compression alone can
        produce an effect as large as the anomaly signature; the XD measurement
        must be presented as unreliable.
    cos(delta_q, delta_real) > 0.7 -> compression mimics the anomaly
        signature; P1 on XD must be withdrawn.
"""
import glob
import io
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
TRAIN_CLIPS = ["Train001", "Train002", "Train003"]
QUALITIES = [95, 75, 50, 30]


def jpeg_roundtrip(pil_rgb, q):
    buf = io.BytesIO()
    pil_rgb.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def load_frames():
    frames = []
    for c in TRAIN_CLIPS:
        frames += sorted(glob.glob(os.path.join(RAW, "Train", c, "*.tif")))
    return frames


def main():
    import clip
    model, pre = clip.load("ViT-B/16", device="cpu",
                           download_root=os.path.join(REPO, "_ckpt"))
    model.eval().float()

    frames = load_frames()
    print("[J3] %d train frames, qualities=%s" % (len(frames), QUALITIES), flush=True)

    P = S.detrender()
    hann = np.hanning(S.WIN).astype(np.float32)
    nb = len(S.BAND_EDGES) - 1

    def encode_paths(paths_or_imgs):
        G, Pp = [], []
        for i in range(0, len(paths_or_imgs), BATCH):
            ch = paths_or_imgs[i:i + BATCH]
            if isinstance(ch[0], str):
                ims = [Image.open(p).convert("RGB") for p in ch]
            else:
                ims = ch
            x = torch.stack([pre(im) for im in ims])
            with torch.no_grad():
                g, sp = E.forward("clip_vit_b16", model, x)
            G.append(g.numpy().astype(np.float32))
            Pp.append(E.as_tokens(sp).numpy().astype(np.float32))
        return np.concatenate(G), np.concatenate(Pp)

    t0 = time.time()
    Xg, Xp = encode_paths(frames)
    assert Xp.shape[1:] == (196, 768), "patch axis order wrong: %s" % (Xp.shape,)
    print("[J3] clean encoded %s in %.0fs" % (Xp.shape, time.time() - t0), flush=True)

    flat = Xp.reshape(-1, Xp.shape[2])
    pca = S.fit_pca(flat[::max(1, len(flat) // 50000)])

    def shape(series):
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
    print("\n[J3] clean shape        %s" % "".join("%10.4f" % v for v in base), flush=True)

    deltas = {}
    for q in QUALITIES:
        ims = [jpeg_roundtrip(Image.open(p).convert("RGB"), q) for p in frames]
        _, Yp = encode_paths(ims)
        d = shape(Yp) - base
        deltas[q] = d
        print("[J3] JPEG q=%-3d delta  %s" % (q, "".join("%+10.4f" % v for v in d)),
              flush=True)

    # ---- reference signatures ----------------------------------------------
    z = np.load(os.path.join(REPO, "results", "_jitter2_cache.npz"), allow_pickle=True)
    real = z["real"]
    jit_sig = [float(s) for s in z["sigmas"]]
    dj = z["deltas"][jit_sig.index(2.0)]

    print("\n[J3] REAL P1 delta      %s" % "".join("%+10.4f" % v for v in real))
    print("[J3] jitter(2/255)      %s" % "".join("%+10.4f" % v for v in dj))

    nr = np.linalg.norm(real)
    print("\n" + "=" * 80)
    print("J3: can realistic compression reproduce the P1 signature?")
    print("=" * 80)
    print("  %-10s %8s %12s %10s" % ("JPEG q", "cos", "|d|/|real|", "verdict"))
    for q in QUALITIES:
        d = deltas[q]
        c = float(d @ real / (np.linalg.norm(d) * nr + 1e-12))
        r = np.linalg.norm(d) / nr
        v = ("MIMICS anomaly - withdraw" if c > 0.7 else
             "same order as real effect" if r > 1.0 else
             "small vs real effect" if r < 0.3 else "moderate")
        print("  %-10d %8.3f %12.3f   %s" % (q, c, r, v))
    print("\n  proxy limit: JPEG is intra-frame; H.264 artefacts are")
    print("  motion-compensated and temporally correlated -- treat as a bound.")
    print("\n  total %.1f min" % ((time.time() - t0) / 60))

    np.savez(os.path.join(REPO, "results", "_compress_cache.npz"),
             base=base, real=real, jitter=dj,
             qualities=np.array(QUALITIES),
             deltas=np.stack([deltas[q] for q in QUALITIES]),
             labels=np.array([S.band_label(i) for i in range(nb)]))
    print("[J3] cached -> results/_compress_cache.npz")


if __name__ == "__main__":
    main()
