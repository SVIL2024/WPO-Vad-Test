import argparse
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import random

from mamba_vad_dual_model import SingleMambaVAD
from wave_tv import wave_source_loss, TimeVaryingWaveBank
from xd_test_sgl import test
from utils.dataset import XDDataset
from utils.log_utils import setup_logging
from utils.tools import get_prompt_text, get_batch_label
import xd_option


def CLAS2(logits, labels, lengths, device, label_smooth=0.1):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    if label_smooth > 0:
        labels = labels * (1 - label_smooth) + label_smooth * 0.5
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for i in range(logits.shape[0]):
        k = max(1, int(lengths[i] / 16 + 1))
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=k, largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)
    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss


def CLASM(logits, labels, lengths, device):
    """Per-class top-k multi-class MIL (the dual branch's multi-class loss).

    logits: (B, T, K) per-frame class scores (from class prototypes here, from
    text similarity in the dual branch). Each class independently top-k-pools
    over frames, then the video-level class distribution is matched against the
    normalised multi-hot video labels.
    """
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    for i in range(logits.shape[0]):
        k = max(1, int(lengths[i] / 16 + 1))
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=k, largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)


def normal_prototype_loss(h_pooled, labels, margin=0.5, device='cuda'):
    h_norm = F.normalize(h_pooled, dim=-1)
    labels_binary = (labels[:, 0] == 1).float()
    h_normal = h_norm[labels_binary == 1]
    h_anomaly = h_norm[labels_binary == 0]
    if len(h_normal) < 1 or len(h_anomaly) < 1:
        return torch.tensor(0.0, device=device), h_norm.mean(dim=0)
    normal_proto = F.normalize(h_normal.mean(dim=0), dim=-1)
    sim_normal = h_normal @ normal_proto
    loss_normal = (1 - sim_normal).mean()
    sim_anomaly = h_anomaly @ normal_proto
    loss_anomaly = F.relu(sim_anomaly + margin).mean()
    return loss_normal + loss_anomaly, normal_proto.detach()


def augment_features(x, lengths, noise_std=0.1, mask_len=24, p_mask=0.5):
    """Feature-level augmentation for pre-extracted CLIP sequences (mask-safe).

    (a) relative Gaussian noise on valid frames — attacks memorisation of the
    fixed 4k training videos (each epoch replays identical features otherwise);
    (b) a random contiguous temporal span zeroed with probability p_mask
    (SpecAugment-style) — forces the top-k MIL selection to spread over many
    frames instead of a fixed few. Padding frames are left untouched.
    """
    B, T, C = x.shape
    device = x.device
    vmask = torch.arange(T, device=device)[None, :] < lengths[:, None]      # (B, T)
    std = x.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    noise = noise_std * std * torch.randn_like(x)
    x = torch.where(vmask.unsqueeze(-1), x + noise, x)
    span = torch.randint(max(8, mask_len // 2), mask_len + 1, (B,), device=device)
    start = ((lengths - span).clamp(min=0).float() * torch.rand(B, device=device)).long()
    idx = torch.arange(T, device=device)[None, :]
    tmask = (idx >= start[:, None]) & (idx < (start + span)[:, None]) & vmask
    keep = (torch.rand(B, device=device) < p_mask)[:, None]
    # condition needs a trailing channel dim to broadcast over (B, T, C)
    x = torch.where((tmask & keep).unsqueeze(-1), torch.zeros_like(x), x)
    return x


def train(model, train_loader, test_loader, args, label_map, device):
    os.makedirs('model', exist_ok=True)
    model.to(device)
    t_start = time.time()

    # The wave operator's (P, Q, gain) scalars are second-order params: they
    # freeze at the base lr, so they get their own group - but they must NOT
    # land in the base groups too, or AdamW dies with "some parameters appear in
    # more than one parameter group" (exactly what killed the first WRB chain).
    # Where the time-varying wave bank lives depends on --wave-tv-where:
    # 'pre'/'fusion' -> model.wave_tv, but 'block' nests one inside every Mamba
    # block. Walk the module tree so the entropy term and the diagnostics work
    # for all three. (Must be built here, not in main(): train() is a
    # module-level function and cannot see main's locals.)
    _wave_banks = [m for m in model.modules()
                   if isinstance(m, TimeVaryingWaveBank)]
    for _b in _wave_banks:
        _b.ent_weight = float(args.wave_ent)

    # Identify wave params BY MODULE IDENTITY, not by name. With
    # --wave-tv-where block the bank is built by wpo_factory and lands at
    # 'blocks.N.wpo.*', so a name match on 'wave_tv' misses it - and 'wpo' in n
    # then sweeps (P, Q, gain) AND the gate conv into the WPO group at
    # wpo_lr_scale=100x, 10x hotter than the intended wave_lr_scale=10x.
    _WAVE_IDS, _WAVE_SCALAR_IDS = set(), set()
    for _b in _wave_banks:
        for _p in _b.parameters():
            _WAVE_IDS.add(id(_p))
        for _nm in ('log_P', 'log_Q', 'gain'):
            _p = getattr(_b, _nm, None)
            if _p is not None:
                _WAVE_SCALAR_IDS.add(id(_p))

    def _wave_scalar(n, p):
        return id(p) in _WAVE_SCALAR_IDS

    def _any_wave(n, p):
        return id(p) in _WAVE_IDS

    def _wave_ssm_p(n, p):
        # WaveSSM's (lam, omega) are the whole point of the layer: they are the
        # decay rate and the frequency. Same failure mode as the WPO (P, Q) —
        # under default weight decay they stall at init — so they get their own
        # group with wd=0.
        return n.endswith('log_lam') or n.endswith('log_omega')

    def _plain(n, p):
        return ('wpo' not in n and 'wrb' not in n and not _any_wave(n, p)
                and not _wave_ssm_p(n, p))

    if args.mamba_lr_scale < 1.0:
        param_groups = [
            {'params': [p for n, p in model.named_parameters()
                        if 'mamba' in n and 'wpo' not in n and 'wrb' not in n
                        and not _any_wave(n, p) and not _wave_ssm_p(n, p)],
             'lr': args.lr * args.mamba_lr_scale},
            {'params': model.frame_position_embeddings.parameters(),
             'lr': args.lr * args.mamba_lr_scale},
            {'params': [p for n, p in model.named_parameters()
                        if 'mamba' not in n and 'frame_position' not in n
                        and 'wpo' not in n and 'wrb' not in n
                        and not _any_wave(n, p) and not _wave_ssm_p(n, p)]},
        ]
    else:
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if _plain(n, p)]},
        ]
    wrb_params = [p for n, p in model.named_parameters() if 'wrb' in n]
    if wrb_params:
        # The resonator's (P, Q) are the whole point of the module — they must
        # actually move. Same treatment as the WPO group (the src/ lesson: with
        # the base lr and default wd these second-order params freeze at init).
        param_groups.append({'params': wrb_params, 'lr': args.lr * args.wpo_lr_scale,
                             'weight_decay': 0.0})
    wpo_params = [p for n, p in model.named_parameters()
                  if 'wpo' in n and not _any_wave(n, p)]
    if wpo_params:
        # src/ lesson: with default wd and base lr the wave params' net task
        # gradient ~ 0 and they freeze at init — give them their own group.
        param_groups.append({'params': wpo_params, 'lr': args.lr * args.wpo_lr_scale,
                             'weight_decay': 0.0})
    wave_scalars = [p for n, p in model.named_parameters() if _wave_scalar(n, p)]
    if wave_scalars:
        param_groups.append({'params': wave_scalars,
                             'lr': args.lr * args.wave_lr_scale,
                             'weight_decay': 0.0})
    # The rest of the bank (gate convs, LayerNorm) is ordinary: base lr, default
    # wd. Without this group they would be excluded from every group above and
    # silently never receive a gradient step.
    wave_other = [p for n, p in model.named_parameters()
                  if _any_wave(n, p) and not _wave_scalar(n, p)]
    if wave_other:
        param_groups.append({'params': wave_other})
    # Direction A: WaveSSM's (lam, omega). These two fields ARE the physics —
    # decay rate and natural frequency. Zero weight decay so they cannot be
    # dragged back to their init by the decoupled term.
    wave_ssm_params = [p for n, p in model.named_parameters() if _wave_ssm_p(n, p)]
    if wave_ssm_params:
        param_groups.append({'params': wave_ssm_params,
                             'lr': args.lr * args.wave_ssm_lr_scale,
                             'weight_decay': 0.0})

    # Every parameter must land in exactly ONE group: missing ones silently
    # freeze, duplicated ones make AdamW raise. Cheap enough to assert.
    _seen, _dup = set(), []
    for _g in param_groups:
        for _p in _g['params']:
            if id(_p) in _seen:
                _dup.append(_p.shape)
            _seen.add(id(_p))
    _missing = [n for n, p in model.named_parameters() if id(p) not in _seen]
    if _dup or _missing:
        raise RuntimeError(f"param groups broken: {len(_dup)} duplicated, "
                           f"{len(_missing)} missing (e.g. {_missing[:3]})")

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    # EMA of floating weights: evaluated/saved instead of the raw weights, to
    # counter the peak-early-then-decline pattern observed in every run.
    ema_state = None
    if args.ema:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                     if v.is_floating_point()}
        print(f"[EMA] enabled, decay={args.ema_decay} over {len(ema_state)} tensors")

    def ema_update():
        with torch.no_grad():
            for k, v in model.state_dict().items():
                e = ema_state.get(k)
                if e is not None:
                    e.mul_(args.ema_decay).add_(v.detach(), alpha=1.0 - args.ema_decay)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    best_ep = 0
    best_model_path = None
    normal_proto_ema = None
    wpo_tag = "_wpoms" if args.wpo_multiscale else ("_wpo" if args.use_wpo else "")
    if args.wpo_multiscale and args.wpo_band_mode == 'fixed':
        wpo_tag += "_fxband"
    if args.adaptive_fusion:
        wpo_tag += "_af"
    if args.wpo_parallel:
        wpo_tag += "_par"
    if args.wpo_adaptive_params:
        wpo_tag += "_apa"
    if args.wpo_motion_cond and args.wpo_adaptive_params:
        wpo_tag += "_mcond"
    if args.ema:
        wpo_tag += "_ema"
    if (args.mamba_version != 'mamba1' or args.mamba_layers != 1
            or args.d_state != 16 or args.dropout != 0.0):
        # keep checkpoint filenames self-describing for non-default architectures
        wpo_tag += f"_mv{args.mamba_version}l{args.mamba_layers}ds{args.d_state}do{args.dropout}"
    if args.future_weight != 0.1:
        wpo_tag += f"_fw{args.future_weight}"
    if args.aug:
        wpo_tag += "_aug"
    if args.eval_steps > 0:
        wpo_tag += f"_es{args.eval_steps}"
    if args.class_proto:
        wpo_tag += "_cpem"
        if args.cp_weight != 0.1:
            wpo_tag += f"w{args.cp_weight}"
    if args.proto_weight != 0.1:
        wpo_tag += f"_pw{args.proto_weight}"
    if args.use_wrb:
        wpo_tag += f"_wrb{args.wrb_kind[:3]}P{args.wrb_periods.replace(',', '-')}" \
                   f"Q{args.wrb_q.replace(',', '-')}g{args.wrb_groups}{args.wrb_where[:1]}"
        if args.wrb_adaptive:
            wpo_tag += "_wa"
            if args.wrb_motion_cond:
                wpo_tag += "mc"
    # NB: outside the use_wrb block - the wave-loss / wave-operator runs carry no
    # WRB, and without this the checkpoint name would not record their config.
    if args.wave_tv:
        # k/size are part of the name: the whole point of chain 6 is that the
        # (P, Q) tiling depends on them, and W1's 'wtvp' already exists with
        # the old hand-written 8,16,32,64 / 2,3,4,6 pair.
        wpo_tag += f"_wtvk{args.wave_tv_k}s{args.wave_tv_size}"
        if args.wave_tv_where != 'pre':
            wpo_tag += args.wave_tv_where[:1]
        if abs(args.wave_tv_gain - 0.1) > 1e-9:
            wpo_tag += f"g{args.wave_tv_gain}"
    if args.wave_loss > 0:
        wpo_tag += f"_wl{args.wave_loss}a{args.wave_alpha}"
    if args.wave_ent > 0:
        wpo_tag += f"_we{args.wave_ent}"
    no_improve = 0
    last_auc, last_ap = 0.0, 0.0
    num_batches = max(len(train_loader), 1)

    def run_eval(e, step_label):
        """Evaluate (EMA weights when enabled), save on improvement, update the
        early-stop counter. step_label: intra-epoch step number, or None for the
        epoch-end evaluation. Returns True when training should stop."""
        nonlocal ap_best, best_ep, best_model_path, no_improve, last_auc, last_ap
        ema_backup = None
        if ema_state is not None:
            ema_backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                          if k in ema_state}
            model.load_state_dict(ema_state, strict=False)

        AUC, AP, _ = test(model, test_loader, args.visual_length,
                          gt, gtsegments, gtlabels, device)
        last_auc, last_ap = AUC, AP

        if AP > ap_best + args.early_stop_min_delta:
            ap_best = AP
            best_ep = e + 1
            no_improve = 0
            # Only persist checkpoints that clear the publication bar — a
            # sub-threshold model is dead weight on the server disk.
            if AP >= args.save_min_ap:
                ep_tag = f"_ep{e+1}" + (f"s{step_label}" if step_label else "")
                new_path = f"model/xd_best_ap{AP:.4f}_auc{AUC:.4f}{ep_tag}{wpo_tag}.pt"
                # model currently holds the EMA weights that produced this metric —
                # the checkpoint must match the evaluated numbers
                torch.save({'model_state_dict': model.trainable_state_dict(),
                            'normal_proto_ema': normal_proto_ema}, new_path)
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = new_path
            else:
                print(f"[NoSave] AP {AP:.4f} < --save-min-ap {args.save_min_ap:.4f} "
                      f"- checkpoint not written (best so far {ap_best:.4f})")
        else:
            no_improve += 1

        if ema_backup is not None:
            model.load_state_dict(ema_backup, strict=False)

        if step_label:
            print(f"[Eval] epoch {e+1} step {step_label} | AUC:{AUC:.4f} AP:{AP:.4f} "
                  f"| BEST AP:{ap_best:.4f}(e{best_ep})")
        if no_improve >= args.early_stop_patience:
            print(f"[EarlyStop] No improvement for {no_improve} evals "
                  f"(patience={args.early_stop_patience}). Best AP: {ap_best:.4f}")
            return True
        return False

    for e in range(args.max_epoch):
        model.train()
        t_epoch = time.time()
        loss_cls_total = 0
        loss_proto_total = 0
        loss_future_total = 0
        loss_cp_total = 0.0
        loss_wave_total = 0.0
        stop = False
        i = -1

        for i, item in enumerate(train_loader):
            visual_feat, text_labels, feat_lengths = item
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)

            if args.aug:
                visual_feat = augment_features(visual_feat, feat_lengths)

            text_onehot = get_batch_label(text_labels, prompt_text, label_map).to(device)
            outputs = model(visual_feat, lengths=feat_lengths)

            # Classification loss (CLAS2) — label_smooth comes from the CLI so the
            # logged RunCommand reflects the actual behaviour (the function default
            # of 0.1 used to apply silently while the log claimed 0.0).
            loss2 = CLAS2(outputs['logits1'], text_onehot, feat_lengths, device,
                          label_smooth=args.label_smooth)
            loss_cls_total += loss2.item()
            loss = loss2

            # Normal prototype contrastive loss
            h_fused = outputs['hidden_state_fused']
            lengths = feat_lengths
            mask = torch.arange(h_fused.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
            h_pooled = (h_fused * mask.unsqueeze(-1).float()).sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)
            loss_proto, normal_proto = normal_prototype_loss(h_pooled, text_onehot, device=device)
            loss_proto_total += loss_proto.item()
            loss = loss + loss_proto * args.proto_weight

            # Future score prediction bridge
            logits_detach = outputs['logits1'].detach()
            loss_future = F.mse_loss(outputs['score_future'], logits_detach)
            loss_future_total += loss_future.item()
            loss = loss + loss_future * args.future_weight

            # Wave-equation source loss on the frame scores: the score behaves
            # like a damped string, events are sparse sources. Punishes isolated
            # one-frame spikes ~40x harder (per anomaly frame) than a real
            # multi-frame event, and adds no capacity at all.
            if args.wave_loss > 0:
                loss_wave = wave_source_loss(outputs['logits1'], lengths=feat_lengths,
                                             alpha=args.wave_alpha)
                loss_wave_total += loss_wave.item()
                loss = loss + loss_wave * args.wave_loss

            # Maximise the MUTUAL INFORMATION between the selected band and
            # (sample, time), MI = H_marginal - H_perframe. V1 measured
            # gbar=[0,0,0,1,0] with MI ~ 0: the gate had collapsed onto one
            # kernel, i.e. a CONSTANT one-hot gate - which is just as linear as
            # a constant uniform one, so it does not escape the LTI collapse.
            # Sharpening (minimising H_perframe) would make that worse; what we
            # need is a gate whose choice depends on where we are in the clip.
            if args.wave_ent > 0:
                for _b in _wave_banks:
                    _mi = _b.gate_mi()
                    if torch.is_tensor(_mi):
                        loss = loss - args.wave_ent * _mi / max(len(_wave_banks), 1)

            # Class-prototype multi-class MIL (restores the 7-class supervision).
            # EMA anchors: update the buffer from this batch (no grad), then the
            # CLASM loss pulls FEATURES toward the slowly-moving class anchors —
            # gradient reaches the encoder only, never the anchors themselves.
            if args.class_proto:
                model.update_class_proto(h_pooled.detach(), text_onehot)
                loss_mil = CLASM(outputs['logits_proto'], text_onehot, feat_lengths, device)
                loss = loss + loss_mil * args.cp_weight
                loss_cp_total += loss_mil.item()

            if normal_proto_ema is None:
                normal_proto_ema = normal_proto
            else:
                normal_proto_ema = F.normalize(0.999 * normal_proto_ema + 0.001 * normal_proto, dim=-1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if ema_state is not None:
                ema_update()

            # Intra-epoch evaluation: the true peak may lie inside epoch 1 (every
            # run peaks at e1-e3, so end-of-epoch evals can overshoot it).
            if args.eval_steps > 0 and (i + 1) % args.eval_steps == 0:
                stop = run_eval(e, i + 1)
                model.train()
                if stop:
                    break

        scheduler.step()
        epoch_time = time.time() - t_epoch
        avg_cls = loss_cls_total / num_batches
        avg_proto = loss_proto_total / num_batches
        avg_future = loss_future_total / num_batches

        # Epoch-end evaluation, unless an intra-epoch eval just ran at the final
        # step (eval_steps=0 means epoch-end only — the historical behaviour).
        if not stop and not (args.eval_steps > 0 and (i + 1) % args.eval_steps == 0):
            stop = run_eval(e, None)
            model.train()

        # One analysis-ready line per epoch: losses, metrics, best-so-far, lr,
        # and the live WPO parameters (gate/alpha/c) for improvement analysis.
        lr_now = optimizer.param_groups[0]['lr']
        wpo_str = " ".join(f"{n.split('.')[-1]}={p.item():.4f}"
                           for n, p in model.named_parameters()
                           if 'wpo' in n and p.numel() == 1)
        cp_str = f" CP:{loss_cp_total / num_batches:.4f}" if args.class_proto else ""
        wrb_str = f" | WRB {model.wrb_describe()}" if args.use_wrb else ""
        wtv_str = ""
        if _wave_banks:
            wtv_str = " | " + _wave_banks[0].describe()
        print(f"[Epoch {e+1}/{args.max_epoch} | {epoch_time:.0f}s] "
              f"Cls:{avg_cls:.4f} Proto:{avg_proto:.4f} Future:{avg_future:.4f}{cp_str} "
              f"| AUC:{last_auc:.4f} AP:{last_ap:.4f} | BEST AP:{ap_best:.4f}(e{best_ep}) "
              f"| lr:{lr_now:.2e} | WPO {wpo_str or 'off'}{wrb_str}{wtv_str}")

        if stop:
            break

    if best_model_path and os.path.exists(best_model_path):
        final = f"model/xd_final_ap{ap_best:.4f}{wpo_tag}.pt"
        # Final artifact must be the BEST weights, not the last epoch's
        # (early stopping leaves the in-memory model at a later, worse epoch).
        ckpt = torch.load(best_model_path, weights_only=False)
        torch.save({'model_state_dict': ckpt['model_state_dict'],
                    'normal_proto_ema': ckpt.get('normal_proto_ema')}, final)
        print(f"Final model saved to {final} (Best AP: {ap_best:.4f})")
    else:
        # Below the publication bar: keep nothing, the log line is the result.
        print(f"[NoSave] no model written: best AP {ap_best:.4f} < "
              f"--save-min-ap {args.save_min_ap:.4f}")
    total_time = time.time() - t_start
    print(f"Total training time: {total_time:.0f}s ({total_time/60:.1f}min)")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[xd_option.parser], add_help=False)
    parser.add_argument('--d-state', default=8, type=int)
    parser.add_argument('--d-conv', default=4, type=int)
    parser.add_argument('--expand', default=2, type=int)
    parser.add_argument('--mamba-layers', default=2, type=int)
    parser.add_argument('--mlp-scale', default=2, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--mamba-version', default='mamba2', choices=['mamba1', 'mamba2'])

    # How a pooled multi-scale branch is produced.  'pool' is the historical
    # F.avg_pool1d(kernel=f, stride=f); 'decimate' keeps every f-th frame, so the
    # stride (temporal scale spacing) and the aliasing are identical and only the
    # cross-frame averaging is removed.  That isolates the confound Reviewer 1 #3
    # raised against the (1,8,16) vs (1,2,4) contrast.  Default 'pool' leaves every
    # existing run bit-identical.
    parser.add_argument('--ms-pool-mode', default='pool', choices=['pool', 'decimate'],
                        help='multi-scale branch construction: boxcar average (default) '
                             'or same-stride decimation')
    parser.add_argument('--mamba2-headdim', default=64, type=int)
    parser.add_argument('--mamba2-ngroups', default=1, type=int)
    # ---- direction A: second-order (complex-diagonal) SSM -------------------
    parser.add_argument('--wave-ssm', action='store_true',
                        help='replace the Mamba mixer with WaveSSM (complex diagonal '
                             'A = -lam + i*omega, i.e. a damped oscillator)')
    parser.add_argument('--wave-omega-init', default='spread',
                        help="'spread' = log-spread resonator bank; a float = every "
                             "slot starts at that frequency; 0 = FREEZE omega at zero, "
                             "which makes WaveSSM an exact real-diagonal SSM (ablation)")
    parser.add_argument('--wave-checkpoint-scan', action='store_true',
                        help='recompute the selective scan in backward to save memory')
    parser.add_argument('--wave-state-chunk', default=1, type=int,
                        help='state channels per scan pass; memory/speed knob only')
    parser.add_argument('--wave-ssm-lr-scale', default=1.0, type=float,
                        help='lr multiplier for WaveSSM log_lam / log_omega')
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--multi-scale-factors', default='1,8,16',
                        help='pooling granularities (legacy pooling path) and kernel '
                             'sizes (--wpo-band-mode fixed); Plan B band scales use '
                             '--wpo-band-c instead')
    parser.add_argument('--weight-decay', default=0.0, type=float)
    parser.add_argument('--mamba-lr-scale', default=1.0, type=float)
    parser.add_argument('--label-smooth', default=0.0, type=float)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--use-wpo', action='store_true',
                        help='identity-anchored TemporalWPO inside each Mamba block')
    parser.add_argument('--wpo-lr-scale', default=100.0, type=float,
                        help='lr multiplier for the WPO alpha/c/gate params')
    parser.add_argument('--wpo-alpha-init', default=0.3, type=float)
    parser.add_argument('--wpo-c-init', default=0.8, type=float)
    parser.add_argument('--wpo-c-max', default=20.0, type=float)
    parser.add_argument('--wpo-multiscale', action='store_true',
                        help='Plan B: WPO band-split scales replace avg_pool multi-scale')
    parser.add_argument('--wpo-band-c', default='0.5,3.0',
                        help='comma-separated c_init per band (drives band diversity)')
    parser.add_argument('--wpo-band-mode', default='learnable', choices=['learnable', 'fixed'],
                        help='Plan B band modules: learnable TemporalWPO vs parameter-free '
                             'FixedBandFilter (control arm for the wave operator)')
    parser.add_argument('--adaptive-fusion', action='store_true',
                        help='per-frame adaptive scale fusion (identity-anchored, '
                             'requires --multi-scale)')
    parser.add_argument('--wpo-parallel', action='store_true',
                        help='WPO deviation bypasses the SSM and is added to the '
                             'residual stream directly (WaveFormer-style parallel '
                             'gating instead of serial pre-filtering)')
    parser.add_argument('--wpo-adaptive-params', action='store_true',
                        help='video-level conditioned wave parameters: a zero-init '
                             'hypernetwork predicts per-video (alpha, c) offsets '
                             'from the pooled video context (FFTNet-style adaptive '
                             'spectral filtering, minimal dimensionality)')
    parser.add_argument('--wpo-motion-cond', action='store_true',
                        help='condition the adaptive wave parameters on pooled '
                             '|x_t - x_t-1| motion descriptors as well as appearance '
                             '(requires --wpo-adaptive-params)')
    parser.add_argument('--ema', action='store_true',
                        help='evaluate/save an exponential moving average of the '
                             'weights (targets the peak-early-then-decline pattern)')
    parser.add_argument('--ema-decay', default=0.999, type=float)
    parser.add_argument('--future-weight', default=0.1, type=float,
                        help='weight of the future-score prediction loss; 0 disables '
                             'it (the loss kept rising in every run, adding gradient '
                             'noise without benefit)')
    parser.add_argument('--aug', action='store_true',
                        help='feature-level augmentation: relative Gaussian noise + '
                             'random temporal span masking on valid frames')
    parser.add_argument('--eval-steps', default=0, type=int,
                        help='evaluate every N training steps within an epoch '
                             '(0 = epoch-end evaluation only)')
    parser.add_argument('--class-proto', action='store_true',
                        help='learnable class-prototype multi-class MIL: restores the '
                             'K-class supervision (K=classes_num) with data-learned '
                             'anchors replacing the dual branch text prompts')
    parser.add_argument('--cp-weight', default=0.1, type=float,
                        help='weight of the class-prototype MIL loss (the anchors '
                             'are EMA buffers, so this gradient reaches only the '
                             'features; 1.0 was tested and distorted training)')
    parser.add_argument('--proto-weight', default=0.1, type=float,
                        help='weight of the normal-prototype contrastive loss; the '
                             'original 0.8248 run used 4.0 (40x the old default)')
    # ---- Wave resonator bank (driven damped-wave band-pass views) ----
    parser.add_argument('--use-wrb', action='store_true',
                        help='enable the wave resonator bank: band-pass views whose DC '
                             'gain is 1/sqrt(1+Q^2), i.e. it CAN suppress the static '
                             'background (the old free-decay WPO provably cannot)')
    parser.add_argument('--wrb-kind', default='resonator',
                        choices=['resonator', 'diffusion', 'free'],
                        help='resonator = forced hyperbolic (ours) | diffusion = forced '
                             'parabolic control | free = initial-value hyperbolic (old WPO)')
    parser.add_argument('--wrb-periods', default='64,16',
                        help='resonant periods in frames, one per band')
    parser.add_argument('--wrb-q', default='4,4', help='quality factor per band')
    parser.add_argument('--wrb-groups', default=1, type=int,
                        help='channel groups with independent (P, Q)')
    parser.add_argument('--wrb-adaptive', action='store_true',
                        help='per-video (dlogP, dlogQ) offsets from a zero-init '
                             'hyper-network (makes the bank selective, not LTI)')
    parser.add_argument('--wrb-motion-cond', action='store_true',
                        help='condition the adaptive offsets on pooled |dx| motion too')
    parser.add_argument('--wrb-where', default='fusion', choices=['fusion', 'block', 'both'],
                        help='fusion = extra multi-scale scales (cheap) | block = '
                             'inside each Mamba block | both')
    parser.add_argument('--wrb-gain-init', default=1.0, type=float,
                        help='initial per-band gain at the fusion insertion point')
    parser.add_argument('--wave-tv', action='store_true',
                        help='time-varying wave operator (wave_tv.py): K damped-'
                             'oscillator depthwise convs + a per-frame softmax gate')
    parser.add_argument('--wave-tv-where', default='pre',
                        choices=['pre', 'fusion', 'block'],
                        help='pre = on the CLIP features before the encoder '
                             '(parameter-matched) | fusion = extra scale | block = '
                             'inside every Mamba block')
    parser.add_argument('--wave-tv-periods', default='',
                        help="resonant period (frames) of each fixed kernel. "
                             "''/auto -> log-spaced tiling in [pmin, size/2] "
                             "(see wave_tv.log_period_tiling)")
    parser.add_argument('--wave-tv-q', default='',
                        help="quality factor of each kernel. ''/auto -> "
                             "Q = pi*size/(4P), i.e. Q DECREASES with P")
    parser.add_argument('--wave-tv-k', default=5, type=int,
                        help='number of bands when P is auto-tiled')
    parser.add_argument('--wave-tv-pmin', default=4.0, type=float,
                        help='shortest period (frames) of the auto tiling')
    parser.add_argument('--wave-ent', default=0.0, type=float,
                        help='weight of the gate MUTUAL-INFORMATION bonus: '
                             'maximise MI = H_marginal - H_perframe, i.e. make '
                             'the band choice depend on position in the clip. '
                             'MI = 0 (constant gate) is the linear/collapsed '
                             'case; sharpening alone would make it worse')
    parser.add_argument('--wave-tv-size', default=33, type=int,
                        help='kernel length in frames')
    parser.add_argument('--wave-tv-gain', default=0.1, type=float,
                        help='initial gain (identity-anchored)')
    parser.add_argument('--wave-loss', default=0.0, type=float,
                        help='weight of the wave-equation source loss on the frame '
                             'scores: L1 on  s_tt + alpha*s_t  (no extra capacity)')
    parser.add_argument('--wave-alpha', default=0.25, type=float,
                        help='damping alpha of the score-domain wave equation')
    parser.add_argument('--wave-lr-scale', default=10.0, type=float,
                        help='lr multiplier for the wave (P, Q, gain) scalars - they '
                             'are second-order params and freeze at the base lr, but '
                             '100x (like --wpo-lr-scale) makes them chase noise')
    parser.add_argument('--early-stop-patience', default=2, type=int,
                        help='stop after N consecutive evals without improvement')
    parser.add_argument('--early-stop-min-delta', default=0.0, type=float,
                        help='minimum AP gain that counts as an improvement')
    parser.add_argument('--save-min-ap', default=0.85, type=float,
                        help='disk policy: write a checkpoint ONLY when AP reaches '
                             'this bar (paper bar on XD-Violence). Early stopping and '
                             'logging still track the true best AP; 0 disables gating')
    args = parser.parse_args()
    args.lr = float(args.lr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)
    # One log file per run (timestamped, never overwritten) — needed for
    # per-experiment inspection when writing the paper.
    wpo_tag = "_wpoms" if args.wpo_multiscale else ("_wpo" if args.use_wpo else "")
    if args.wpo_multiscale and args.wpo_band_mode == 'fixed':
        wpo_tag += "_fxband"
    if args.adaptive_fusion:
        wpo_tag += "_af"
    if args.wpo_parallel:
        wpo_tag += "_par"
    if args.wpo_adaptive_params:
        wpo_tag += "_apa"
    if args.wpo_motion_cond and args.wpo_adaptive_params:
        wpo_tag += "_mcond"
    if args.ema:
        wpo_tag += "_ema"
    if (args.mamba_version != 'mamba1' or args.mamba_layers != 1
            or args.d_state != 16 or args.dropout != 0.0):
        # keep checkpoint filenames self-describing for non-default architectures
        wpo_tag += f"_mv{args.mamba_version}l{args.mamba_layers}ds{args.d_state}do{args.dropout}"
    if args.future_weight != 0.1:
        wpo_tag += f"_fw{args.future_weight}"
    if args.aug:
        wpo_tag += "_aug"
    if args.eval_steps > 0:
        wpo_tag += f"_es{args.eval_steps}"
    if args.class_proto:
        wpo_tag += "_cpem"
        if args.cp_weight != 0.1:
            wpo_tag += f"w{args.cp_weight}"
    if args.proto_weight != 0.1:
        wpo_tag += f"_pw{args.proto_weight}"
    if args.use_wrb:
        wpo_tag += f"_wrb{args.wrb_kind[:3]}P{args.wrb_periods.replace(',', '-')}" \
                   f"Q{args.wrb_q.replace(',', '-')}g{args.wrb_groups}{args.wrb_where[:1]}"
        if args.wrb_adaptive:
            wpo_tag += "_wa"
            if args.wrb_motion_cond:
                wpo_tag += "mc"
    # NB: outside the use_wrb block - the wave-loss / wave-operator runs carry no
    # WRB, and without this the checkpoint name would not record their config.
    if args.wave_tv:
        # k/size are part of the name: the whole point of chain 6 is that the
        # (P, Q) tiling depends on them, and W1's 'wtvp' already exists with
        # the old hand-written 8,16,32,64 / 2,3,4,6 pair.
        wpo_tag += f"_wtvk{args.wave_tv_k}s{args.wave_tv_size}"
        if args.wave_tv_where != 'pre':
            wpo_tag += args.wave_tv_where[:1]
        if abs(args.wave_tv_gain - 0.1) > 1e-9:
            wpo_tag += f"g{args.wave_tv_gain}"
    if args.wave_loss > 0:
        wpo_tag += f"_wl{args.wave_loss}a{args.wave_alpha}"
    if args.wave_ent > 0:
        wpo_tag += f"_we{args.wave_ent}"
    setup_logging(f"train_xd_sgl{wpo_tag}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    print(f"[RunCommand] python xd_train_sgl.py --seed {args.seed} --batch-size {args.batch_size} "
          f"--lr {args.lr} --mamba-layers {args.mamba_layers} --d-state {args.d_state} "
          f"--expand {args.expand} --mlp-scale {args.mlp_scale} --mamba-version {args.mamba_version} "
          f"--multi-scale {args.multi_scale} --multi-scale-factors '{args.multi_scale_factors}' "
          f"--bidirectional {args.bidirectional} --use-wpo {args.use_wpo} "
          f"--wpo-lr-scale {args.wpo_lr_scale} --wpo-c-max {args.wpo_c_max} "
          f"--wpo-multiscale {args.wpo_multiscale} --wpo-band-c '{args.wpo_band_c}' "
          f"--wpo-band-mode {args.wpo_band_mode} "
          f"--adaptive-fusion {args.adaptive_fusion} "
          f"--wpo-parallel {args.wpo_parallel} "
          f"--wpo-adaptive-params {args.wpo_adaptive_params} "
          f"--wpo-motion-cond {args.wpo_motion_cond} "
          f"--ema {args.ema} --ema-decay {args.ema_decay} "
          f"--wave-ssm {args.wave_ssm} --wave-omega-init '{args.wave_omega_init}' "
          f"--future-weight {args.future_weight} "
          f"--aug {args.aug} --eval-steps {args.eval_steps} "
          f"--class-proto {args.class_proto} --cp-weight {args.cp_weight} "
          f"--proto-weight {args.proto_weight} "
          f"--use-wrb {args.use_wrb} --wrb-kind {args.wrb_kind} "
          f"--wrb-periods '{args.wrb_periods}' --wrb-q '{args.wrb_q}' "
          f"--wrb-groups {args.wrb_groups} --wrb-where {args.wrb_where} "
          f"--wrb-adaptive {args.wrb_adaptive} --wrb-motion-cond {args.wrb_motion_cond} "
          f"--wave-tv {args.wave_tv} --wave-tv-where {args.wave_tv_where} "
          f"--wave-tv-periods '{args.wave_tv_periods}' --wave-tv-q '{args.wave_tv_q}' "
          f"--wave-tv-size {args.wave_tv_size} --wave-tv-gain {args.wave_tv_gain} "
          f"--wave-tv-k {args.wave_tv_k} --wave-tv-pmin {args.wave_tv_pmin} "
          f"--wave-loss {args.wave_loss} --wave-alpha {args.wave_alpha} "
          f"--wave-ent {args.wave_ent} "
          f"--dropout {args.dropout} --weight-decay {args.weight_decay} "
          f"--label-smooth {args.label_smooth} "
          f"--early-stop-patience {args.early_stop_patience} "
          f"--early-stop-min-delta {args.early_stop_min_delta}")

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting',
                      'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident',
                      'G': 'explosion'})

    train_dataset = XDDataset(args.visual_length, args.train_list, False,
                              label_map, is_random=args.fixed_timebase)
    # num_workers: XDDataset.__getitem__ does one np.load per video, so with the
    # default num_workers=0 the GPU starves - measured on this project at
    # 106 s/batch (GPU 0%) vs 4 s/batch (GPU 76%) with num_workers=8, a ~26x
    # difference. Without this a 3-epoch run costs hours instead of minutes.
    #
    # NOTE: this only changes how fast batches are produced, not which batches
    # are produced (that is fixed by shuffle + seed), so it does not affect the
    # metric. Both arms of the multi-seed ablation use the same setting, so the
    # WITHIN-seed delta stays comparable to the historical seed-234 numbers.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=8, pin_memory=True,
                              persistent_workers=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=4, pin_memory=True)

    mamba_kwargs = {}
    if args.mamba_version == 'mamba2':
        mamba_kwargs = {'mamba2_headdim': args.mamba2_headdim, 'mamba2_ngroups': args.mamba2_ngroups}
    if args.wave_ssm:
        # overrides mamba_version entirely: the mixer becomes a complex-diagonal
        # 2nd-order SSM. omega_init=0 == real-diagonal baseline (exact ablation).
        mamba_kwargs.update({'wave_ssm': True,
                             'wave_omega_init': args.wave_omega_init,
                             'wave_checkpoint_scan': args.wave_checkpoint_scan,
                             'wave_state_chunk': args.wave_state_chunk})

    model = SingleMambaVAD(
        visual_width=args.visual_width, visual_length=args.visual_length,
        mamba_layers=args.mamba_layers, d_state=args.d_state,
        d_conv=args.d_conv, expand=args.expand,
        dropout=args.dropout,
        num_class=args.classes_num,
        device=device, mamba_version=args.mamba_version,
        multi_scale=args.multi_scale,
        multi_scale_factors=args.multi_scale_factors,
        ms_pool_mode=args.ms_pool_mode,
        bidirectional=args.bidirectional,
        mlp_scale=args.mlp_scale,
        use_wpo=args.use_wpo, wpo_alpha_init=args.wpo_alpha_init,
        wpo_c_init=args.wpo_c_init, wpo_c_max=args.wpo_c_max,
        wpo_multiscale=args.wpo_multiscale, wpo_band_c=args.wpo_band_c,
        wpo_band_mode=args.wpo_band_mode,
        adaptive_fusion=args.adaptive_fusion, wpo_parallel=args.wpo_parallel,
        wpo_adaptive_params=args.wpo_adaptive_params, wpo_motion_cond=args.wpo_motion_cond,
        use_class_proto=args.class_proto,
        use_wrb=args.use_wrb, wrb_kind=args.wrb_kind, wrb_periods=args.wrb_periods,
        wrb_q=args.wrb_q, wrb_groups=args.wrb_groups, wrb_adaptive=args.wrb_adaptive,
        wrb_motion_cond=args.wrb_motion_cond, wrb_where=args.wrb_where,
        wrb_gain_init=args.wrb_gain_init,
        wave_tv=args.wave_tv, wave_tv_where=args.wave_tv_where,
        wave_tv_periods=args.wave_tv_periods, wave_tv_q=args.wave_tv_q,
        wave_tv_size=args.wave_tv_size, wave_tv_gain=args.wave_tv_gain,
        wave_tv_k=args.wave_tv_k, wave_tv_pmin=args.wave_tv_pmin,
        **mamba_kwargs)

    if args.multi_scale and args.use_wpo and args.wpo_multiscale:
        ms_str = f"bands={args.wpo_band_c}({args.wpo_band_mode})"
    elif args.multi_scale:
        ms_str = f"scales={args.multi_scale_factors}({args.ms_pool_mode})"
    else:
        ms_str = "off"
    print(f"[Config] SingleMambaVAD XD, "
          f"mamba_layers={args.mamba_layers}, d_state={args.d_state}, "
          f"mlp_scale={args.mlp_scale}, mamba_version={args.mamba_version}, "
          f"multi_scale={ms_str}, bidirectional={args.bidirectional}, "
          f"use_wpo={args.use_wpo}"
          + (f" (wpo_lr_scale={args.wpo_lr_scale}, c_max={args.wpo_c_max}, "
             f"band_mode={args.wpo_band_mode})"
             if args.use_wpo else ""))

    train(model, train_loader, test_loader, args, label_map, device)
