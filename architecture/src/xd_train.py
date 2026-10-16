import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import os
import random

from model import CLIPVAD
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
from wave_loss import wave_source_loss
import xd_option

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def train(model, train_loader, test_loader, args, label_map: dict, device):
    model.to(device)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    # --- E2: warm start from a converged trunk ---------------------------
    # strict=False on purpose: when the checkpoint predates the wave branch its
    # new parameters simply keep their own initialisation. Everything that IS
    # present must match, so the run really starts from the control's solution.
    if getattr(args, 'init_ckpt', ''):
        _ck = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        _sd = _ck['model_state_dict'] if isinstance(_ck, dict) and 'model_state_dict' in _ck else _ck
        _missing, _unexpected = model.load_state_dict(_sd, strict=False)
        print(f"[InitCkpt] {args.init_ckpt}")
        print(f"[InitCkpt] {len(_missing)} key(s) absent from the checkpoint "
              f"(kept at fresh init), {len(_unexpected)} unused")
        if len(_missing):
            print(f"[InitCkpt]   e.g. {_missing[:4]}")
        # A previous WAVE run can masquerade as the control: a dead branch
        # scores identically, so its checkpoint shares the control's filename
        # prefix and carries all-NaN wave tensors. Without this the branch is
        # dead before step 0 and the run measures nothing.
        _repaired = model.sanitize_wave_branch()
        if _repaired:
            print(f"[InitCkpt] WARNING: the warm start was poisoned "
                  f"({_repaired} tensor(s)); the wave branch starts from a "
                  f"fresh init, so this run is NOT a warm-started comparison.")

    # --- E2: optionally train only the wave branch ------------------------
    if getattr(args, 'freeze_trunk', 0):
        for _n, _p in model.named_parameters():
            _p.requires_grad = ('wave' in _n)
        _tr = sum(_p.numel() for _p in model.parameters() if _p.requires_grad)
        print(f"[Freeze] trunk frozen; {_tr} trainable parameter(s), all wave-*")

    # The wave branch is randomly initialised on top of a converged trunk, so
    # it needs a larger step than the pretrained weights (same lesson as the
    # WRB P/Q group in src_sgl). NOTE the two filters are complementary -
    # 'wave' in n / 'wave' not in n - otherwise a parameter lands in both
    # groups and AdamW raises "some parameters appear in more than one
    # parameter group".
    _wave = [_p for _n, _p in model.named_parameters() if 'wave' in _n and _p.requires_grad]
    _base = [_p for _n, _p in model.named_parameters() if 'wave' not in _n and _p.requires_grad]
    _groups = [{'params': _base, 'lr': args.lr}]
    if _wave:
        _groups.append({'params': _wave,
                        'lr': args.lr * getattr(args, 'wave_field_lr_scale', 1.0),
                        'weight_decay': 0.0})
        print(f"[Optim] 2 param groups: {sum(p.numel() for p in _base)} base @"
              f"{args.lr}, {sum(p.numel() for p in _wave)} wave @"
              f"{args.lr * getattr(args, 'wave_field_lr_scale', 1.0)}")
    optimizer = torch.optim.AdamW(_groups, lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0
    best_model_path = None

    # Tag every checkpoint with the wave-loss config so runs are identifiable
    # (mirrors src_sgl's wpo_tag convention).
    wave_tag = ""
    if args.wave_loss > 0:
        wave_tag = f"_wl{args.wave_loss}a{args.wave_alpha}"
    if getattr(args, 'wave_field', 0) and args.wave_field > 0:
        wave_tag += (f"_wf{args.wave_field}g{args.wave_field_gain}"
                     f"{'_fz' if getattr(args, 'freeze_trunk', 0) else ''}")
        # E3: distinguish the two fusion points in the filename. A dead branch
        # scores exactly like the control (that is how a wave run's checkpoint
        # once got mistaken FOR the control - see sanitize_wave_branch), so the
        # tag is the only thing telling these runs apart.
        if getattr(args, 'wave_fusion', 'logits') == 'feature':
            wave_tag += "_ft"
    print(f"[Wave] fusion={getattr(args, 'wave_fusion', 'logits')}")

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        loss_totalw = 0
        for i, item in enumerate(train_loader):
            step = 0
            visual_feat, text_labels, feat_lengths = item
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2 = model(visual_feat, None, prompt_text, feat_lengths) 

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()

            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 6

            loss = loss1 + loss2 + loss3 * 1e-4

            # Wave-source sparsity prior on the per-frame score. `logits1` is the
            # (B, T, 1) binary anomaly score that the AP metric is computed from,
            # so the prior acts exactly where the metric looks. Off by default.
            if args.wave_loss > 0:
                loss_wave = wave_source_loss(logits1, lengths=feat_lengths,
                                             alpha=args.wave_alpha)
                loss_totalw += loss_wave.item()
                loss = loss + loss_wave * args.wave_loss

            optimizer.zero_grad()
            loss.backward()
            # ---- keep the new branch's gradients finite ------------------
            # A randomly-initialised auxiliary branch can produce one enormous
            # gradient and NaN its own parameters; after that every entry shares
            # the NaN and the branch is dead for the rest of the run while
            # still "training" (that is what silently zeroed E2a's evidence
            # channel - the run's numbers were then the control's, exactly).
            # The trunk is deliberately NOT touched: the baseline has to stay
            # bit-identical.
            if _wave:
                for _p in _wave:
                    if _p.grad is not None:
                        torch.nan_to_num(_p.grad, nan=0.0, posinf=0.0,
                                         neginf=0.0, out=_p.grad)
                torch.nn.utils.clip_grad_norm_(_wave, max_norm=1.0)
            optimizer.step()
            step += i * train_loader.batch_size
            if step % 4800 == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                if args.wave_loss > 0:
                    print('   wave: ', loss_totalw / (i+1), '| lambda: ', args.wave_loss)
                
        scheduler.step()
        # Heartbeat: if the wave branch's non-finite guard ever fires, say so
        # per epoch instead of once. A silent guard turns a broken run into a
        # run that just reproduces the control and looks fine.
        if getattr(model, "wave_field", None) is not None:
            # Report the evidence MAGNITUDE, not just "is it finite". E2a's
            # branch was perfectly finite and still changed the reported
            # probability by 8e-5 - alive in every log line we had, yet with
            # no lever at all. gain x |evidence| is the number that actually
            # reaches the model, so print both.
            _ev = getattr(model, "_ev_last", None)
            _evs = (f"  |evidence|max={float(_ev.abs().max()):.4g}"
                    if _ev is not None else "")
            _eff = (f"  gain*|ev|={float(model.wave_gain) * float(_ev.abs().max()):.4g}"
                    if _ev is not None else "")
            print(f"[WaveField] epoch {e + 1}: "
                  f"non-finite energy entries so far="
                  f"{getattr(model, '_wave_bad', 0)}"
                  f"  gain={float(model.wave_gain):.5f}{_evs}{_eff}")
        AUC, AP, mAP = test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)

        if AP > ap_best:
            ap_best = AP
            # Publication bar: a sub-threshold checkpoint is dead weight on the
            # server disk, so only persist runs that clear --save-min-ap.
            if AP >= args.save_min_ap:
                new_path = (f"model/xd_best_ap{AP:.4f}_auc{AUC:.4f}_ep{e + 1}"
                            f"{wave_tag}.pt")
                torch.save({'epoch': e,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'ap': ap_best}, new_path)
                # keep only the newest above-bar checkpoint
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = new_path
                print(f"[Save] {new_path} (AP {AP:.4f} >= {args.save_min_ap:.4f})")
            else:
                print(f"[NoSave] AP {AP:.4f} < --save-min-ap {args.save_min_ap:.4f}"
                      f" - checkpoint not written (best so far {ap_best:.4f})")

        # Keep the BEST weights in memory, not the last epoch's.
        if best_model_path:
            model.load_state_dict(torch.load(best_model_path,
                                             weights_only=False)['model_state_dict'])

    if best_model_path and os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, weights_only=False)
        torch.save(ckpt['model_state_dict'], args.model_path)
        print(f"Final model saved to {args.model_path} (Best AP: {ap_best:.4f})")
    else:
        print(f"[NoSave] no model written: best AP {ap_best:.4f} < "
              f"--save-min-ap {args.save_min_ap:.4f}")

    # The inference-time physics decoder (phase D) needs the RAW per-frame score
    # trajectories. Dumping them here - from whatever weights `model` now holds
    # (the best ones if the disk gate let a checkpoint through, the last epoch's
    # otherwise) - means the decoder ablation never depends on whether a
    # checkpoint was written. One extra test pass costs ~1 min; the npz is ~9 MB
    # against ~200 MB for a checkpoint.
    if args.dump_prob:
        test(model, test_loader, args.visual_length, prompt_text, gt,
             gtsegments, gtlabels, device, dump_path=args.dump_prob)
        print(f"[Dump] raw score trajectories -> {args.dump_prob}")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    nw = args.num_workers
    train_dataset = XDDataset(args.visual_length, args.train_list, False, label_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=(device == "cuda"),
                              persistent_workers=(nw > 0))

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=max(1, min(4, nw)),
                             pin_memory=(device == "cuda"))

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device,
                    wave_field=args.wave_field, wave_gain=args.wave_field_gain,
                    wave_fusion=getattr(args, 'wave_fusion', 'logits'))
    train(model, train_loader, test_loader, args, label_map, device)