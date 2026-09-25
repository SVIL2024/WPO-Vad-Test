# -*- coding: utf-8 -*-
"""Ped2 feature extraction for SEVERAL backbones (backbone-robustness check).

WHY
    Every spectral result in this project (P1 on XD, P5 on Ped2) was measured
    inside ONE feature space: frozen CLIP ViT-B/16, applied frame by frame.
    That invites two distinct objections:
      (a) ARCHITECTURE: is the "no mid-band rhythm" finding specific to ViT
          patch tokens?
      (b) OBJECTIVE / per-frame noise: CLIP has NO temporal modelling, so all
          temporal structure in the feature sequence comes from either the real
          scene dynamics or from per-frame encoder jitter (compression, motion
          blur, aliasing). Our "excess at <=5 frames = roughness" reading could
          therefore be partly ENCODER NOISE rather than signal.
    Three backbones separate these:
      clip_vit_b16  CLIP objective, ViT arch, 14x14 grid, 768-d   (baseline)
      clip_rn50     CLIP objective, CNN arch,   7x7  grid, 2048-d (isolates arch)
      inet_rn50     ImageNet objective, CNN,    7x7  grid, 2048-d (isolates objective)

USAGE
    python controls/extract_backbones.py clip_rn50 --limit-test
Saves one .npz per clip into ped2_feat_<bb>/ with:
    global_feat (T,D)  patch (T,M,C)  gt (T,grid,grid)  frame_ok (T,)  grid (scalar)
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import uniform_filter1d

ROOT = os.environ.get("PED2_RAW", r"D:\dataset\UCSD\UCSDped2")
CK = os.environ.get("CLIP_CKPT", r"D:\program\waveClipVad\_ckpt")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BATCH = 32


def load_backbone(name, device="cpu"):
    if name == "clip_vit_b16":
        import clip
        model, pre = clip.load("ViT-B/16", device=device, download_root=CK)
        model.eval().float()
        return model, pre, 14

    if name == "clip_rn50":
        import clip
        model, pre = clip.load("RN50", device=device, download_root=CK)
        model.eval().float()
        return model, pre, 7

    if name == "inet_rn50":
        import torchvision
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet50(weights=w)
        model.eval().float()
        return model, w.transforms(), 7

    raise ValueError(name)


def as_tokens(sp):
    """Normalise a spatial tensor to (B, M, C).

    BUG THIS FIXES: the ViT branch already returns (B, M, C) -- 3-D -- so the
    usual `flatten(2).permute(0, 2, 1)` idiom (which is meant for a 4-D conv
    map (B, C, H, W)) silently TRANSPOSES it to (B, C, M).  For ViT that swaps
    196 patches with 768 channels.  Only permute when the tensor really is 4-D.
    """
    if sp.dim() == 4:
        return sp.flatten(2).permute(0, 2, 1)
    return sp


def forward(name, model, x):
    """Return (global (B,D), spatial (B,M,C))."""
    if name.startswith("clip_vit"):
        v = model.visual
        g = model.encode_image(x)
        h = v.conv1(x).flatten(2).permute(0, 2, 1)
        cls = v.class_embedding.view(1, 1, -1).expand(h.size(0), 1, -1)
        h = torch.cat([cls, h], 1) + v.positional_embedding
        h = v.transformer(v.ln_pre(h).permute(1, 0, 2)).permute(1, 0, 2)
        return g, v.ln_post(h[:, 1:, :])

    if name == "clip_rn50":
        # CLIP's ResNet tower has NO `proj` (only the ViT tower does):
        # encode_image == visual(x) == attnpool(layer4(...))  -> 1024-d.
        v = model.visual

        def stem(x):
            x = v.relu1(v.bn1(v.conv1(x)))
            x = v.relu2(v.bn2(v.conv2(x)))
            x = v.relu3(v.bn3(v.conv3(x)))
            return v.avgpool(x)

        f = v.layer4(v.layer3(v.layer2(v.layer1(stem(x)))))
        return v.attnpool(f), f

    if name == "inet_rn50":
        m = model
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        g = m.avgpool(x).flatten(1)
        return g, x

    raise ValueError(name)


def disk_free_gb(path):
    import shutil
    return shutil.disk_usage(path).free / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone", choices=["clip_vit_b16", "clip_rn50", "inet_rn50"])
    ap.add_argument("--root",
                    default=os.environ.get(
                        "PED2_RAW", r"D:\dataset\UCSD\UCSDped2"),
                    help="raw-frame dataset root; must contain Test/ and Train/ subdirs")
    ap.add_argument("--limit-test", action="store_true",
                    help="only the 12 Test clips + Train001 (for calibration)")
    ap.add_argument("--out-suffix", default="",
                    help="optional suffix on the output dir name "
                         "(default: derived from --root basename)")
    args = ap.parse_args()

    bb = args.backbone
    root_name = os.path.basename(os.path.normpath(args.root))
    suffix = args.out_suffix or root_name.lower().replace(" ", "_")
    out_dir = os.path.join(REPO, "feat_%s_%s" % (suffix, bb))
    os.makedirs(out_dir, exist_ok=True)

    model, pre, grid = load_backbone(bb)
    print("[%s] grid=%d -> %s" % (bb, grid, out_dir), flush=True)
    print("[%s] raw root = %s" % (bb, args.root), flush=True)
    free_gb = disk_free_gb(out_dir)
    print("[%s] disk free under output = %.1f GB" % (bb, free_gb), flush=True)
    if free_gb < 2.0:
        print("[FATAL] less than 2 GB free -- refusing to start", flush=True)
        sys.exit(2)

    jobs = []
    for d in sorted(glob.glob(os.path.join(args.root, "Test", "*"))):
        if os.path.isdir(d) and not d.endswith("_gt"):
            jobs.append(("Test", d))
    if args.limit_test:
        jobs.append(("Train", os.path.join(args.root, "Train", "Train001")))
    else:
        for d in sorted(glob.glob(os.path.join(args.root, "Train", "*"))):
            if os.path.isdir(d) and not d.endswith("_gt"):
                jobs.append(("Train", d))

    t_all = time.time()
    for split, d in jobs:
        name = os.path.basename(d)
        frames = sorted(glob.glob(os.path.join(d, "*.tif")))
        out = os.path.join(out_dir, "%s_%s.npz" % (split, name))
        if os.path.exists(out):
            print("[skip] %s" % name, flush=True)
            continue
        gtdir = os.path.join(ROOT, split, name + "_gt")
        have_gt = os.path.isdir(gtdir)

        G, P, M, F = [], [], [], []
        t0 = time.time()
        for i in range(0, len(frames), BATCH):
            ch = frames[i:i + BATCH]
            x = torch.stack([pre(Image.open(p).convert("RGB")) for p in ch])
            with torch.no_grad():
                g, sp = forward(bb, model, x)
            sp = as_tokens(sp)          # -> (B, M, C), no-op for ViT
            G.append(g.numpy())
            P.append(sp.numpy())
            if have_gt:
                ms, ok = [], []
                for p in ch:
                    idx = int(os.path.splitext(os.path.basename(p))[0])
                    mp = os.path.join(gtdir, "%03d.bmp" % idx)
                    if os.path.exists(mp):
                        a = np.asarray(Image.open(mp).convert("L"), np.float32) / 255.0
                        hh, ww = a.shape
                        a = a[:hh // grid * grid, :ww // grid * grid]
                        a = a.reshape(grid, hh // grid, grid, ww // grid).mean(axis=(1, 3))
                        ms.append(a); ok.append(True)
                    else:
                        ms.append(np.zeros((grid, grid), np.float32)); ok.append(False)
                M.append(np.stack(ms)); F.append(np.array(ok))

        dd = dict(global_feat=np.concatenate(G).astype(np.float16),
                  patch=np.concatenate(P).astype(np.float16),
                  grid=np.int64(grid))
        if have_gt:
            dd["gt"] = np.concatenate(M).astype(np.float16)
            dd["frame_ok"] = np.concatenate(F)
        np.savez_compressed(out, **dd)
        nz = int((dd["gt"].sum(axis=(1, 2)) > 0).sum()) if have_gt else -1
        print("[%s] T=%d anom_frames=%d  %.0fs" % (name, len(frames), nz, time.time() - t0), flush=True)

    print("DONE %s in %.1f min" % (bb, (time.time() - t_all) / 60))


if __name__ == "__main__":
    main()
