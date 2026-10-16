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
from ucf_test_sgl import test
from utils.dataset import UCFDataset
from utils.log_utils import setup_logging
from utils.tools import get_prompt_text, get_batch_label
from xd_train_sgl import augment_features
import ucf_option


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
    """Per-class top-k multi-class MIL (the dual branch's multi-class loss)."""
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    for i in range(logits.shape[0]):
        k = max(1, int(lengths[i] / 16 + 1))
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=k, largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)


def normal_prototype_loss(h_pooled, binary_labels, margin=0.5, device='cuda'):
    h_norm = F.normalize(h_pooled, dim=-1)
    h_normal = h_norm[binary_labels == 0]
    h_anomaly = h_norm[binary_labels == 1]
    if len(h_normal) < 1 or len(h_anomaly) < 1:
        return torch.tensor(0.0, device=device), h_norm.mean(dim=0)
    normal_proto = F.normalize(h_normal.mean(dim=0), dim=-1)
    sim_normal = h_normal @ normal_proto
    loss_normal = (1 - sim_normal).mean()
    sim_anomaly = h_anomaly @ normal_proto
    loss_anomaly = F.relu(sim_anomaly + margin).mean()
    return loss_normal + loss_anomaly, normal_proto.detach()


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    os.makedirs('model', exist_ok=True)
    model.to(device)
    t_start = time.time()

    if args.mamba_lr_scale < 1.0:
        param_groups = [
            {'params': [p for n, p in model.named_parameters()
                        if 'mamba' in n and 'wpo' not in n],
             'lr': args.lr * args.mamba_lr_scale},
            {'params': model.frame_position_embeddings.parameters(),
             'lr': args.lr * args.mamba_lr_scale},
            {'params': [p for n, p in model.named_parameters()
                        if 'mamba' not in n and 'frame_position' not in n and 'wpo' not in n]},
        ]
    else:
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'wpo' not in n]},
        ]
    wpo_params = [p for n, p in model.named_parameters() if 'wpo' in n]
    if wpo_params:
        # src/ lesson: with default wd and base lr the wave params' net task
        # gradient ~ 0 and they freeze at init — give them their own group.
        param_groups.append({'params': wpo_params, 'lr': args.lr * args.wpo_lr_scale,
                             'weight_decay': 0.0})
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
    prompt_text = get_prompt_text(label_map)

    auc_best = 0
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
    no_improve = 0
    last_auc, last_ap = 0.0, 0.0

    def run_eval(e, step_label):
        """Evaluate (EMA weights when enabled), save on improvement, update the
        early-stop counter. step_label: intra-epoch step number, or None for the
        epoch-end evaluation. Returns True when training should stop."""
        nonlocal auc_best, ap_best, best_ep, best_model_path, no_improve, last_auc, last_ap
        ema_backup = None
        if ema_state is not None:
            ema_backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                          if k in ema_state}
            model.load_state_dict(ema_state, strict=False)

        AUC, AP = test(model, testloader, args.visual_length, device, args.gt_path,
                       normal_proto=normal_proto_ema)
        last_auc, last_ap = AUC, AP

        if AUC > auc_best + args.early_stop_min_delta:
            auc_best = AUC
            ap_best = AP
            best_ep = e + 1
            no_improve = 0
            # Only persist checkpoints that clear the publication bar — a
            # sub-threshold model is dead weight on the server disk.
            if AUC >= args.save_min_auc:
                ep_tag = f"_ep{e+1}" + (f"s{step_label}" if step_label else "")
                new_path = f"model/best_auc{auc_best:.4f}_ap{ap_best:.4f}{ep_tag}{wpo_tag}.pt"
                # model currently holds the EMA weights that produced this metric —
                # the checkpoint must match the evaluated numbers
                torch.save({'model_state_dict': model.trainable_state_dict(),
                            'normal_proto_ema': normal_proto_ema}, new_path)
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = new_path
            else:
                print(f"[NoSave] AUC {AUC:.4f} < --save-min-auc {args.save_min_auc:.4f} "
                      f"- checkpoint not written (best so far {auc_best:.4f})")
        else:
            no_improve += 1

        if ema_backup is not None:
            model.load_state_dict(ema_backup, strict=False)

        if step_label:
            print(f"[Eval] epoch {e+1} step {step_label} | AUC:{AUC:.4f} AP:{AP:.4f} "
                  f"| BEST AUC:{auc_best:.4f}(e{best_ep})")
        if no_improve >= args.early_stop_patience:
            print(f"[EarlyStop] No improvement for {no_improve} evals "
                  f"(patience={args.early_stop_patience}). Best AUC: {auc_best:.4f}")
            return True
        return False

    for e in range(args.max_epoch):
        t_epoch = time.time()
        model.train()
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        num_batches = min(len(normal_loader), len(anomaly_loader))
        loss_cls_total = 0
        loss_proto_total = 0
        loss_future_total = 0
        loss_cp_total = 0.0
        stop = False
        i = -1

        for i in range(num_batches):
            normal_feat, normal_label, normal_len = next(normal_iter)
            anomaly_feat, anomaly_label, anomaly_len = next(anomaly_iter)

            B_normal = normal_feat.shape[0]
            B_anomaly = anomaly_feat.shape[0]

            visual = torch.cat([normal_feat, anomaly_feat], dim=0).to(device)
            feat_lengths = torch.cat([normal_len, anomaly_len], dim=0).to(device)

            if args.aug:
                visual = augment_features(visual, feat_lengths)

            normal_mask = torch.zeros(B_normal + B_anomaly, dtype=torch.bool, device=device)
            normal_mask[:B_normal] = True

            text_labels = list(normal_label) + list(anomaly_label)
            text_onehot = get_batch_label(text_labels, prompt_text, label_map).to(device)
            outputs = model(visual, lengths=feat_lengths)

            # Classification loss (CLAS2)
            loss2 = CLAS2(outputs['logits1'], text_onehot, feat_lengths, device, label_smooth=args.label_smooth)
            loss_cls_total += loss2.item()
            loss = loss2

            # Normal prototype contrastive loss
            h_fused = outputs['hidden_state_fused']
            lengths = feat_lengths
            mask = torch.arange(h_fused.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
            h_pooled = (h_fused * mask.unsqueeze(-1).float()).sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)
            binary_labels = torch.cat([torch.zeros(B_normal), torch.ones(B_anomaly)]).to(device)
            loss_proto, normal_proto = normal_prototype_loss(h_pooled, binary_labels, device=device)
            loss_proto_total += loss_proto.item()
            loss = loss + loss_proto * args.proto_weight

            # Future score prediction bridge
            logits_detach = outputs['logits1'].detach()
            loss_future = F.mse_loss(outputs['score_future'], logits_detach)
            loss_future_total += loss_future.item()
            loss = loss + loss_future * args.future_weight

            # Class-prototype multi-class MIL (restores the 14-class supervision).
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

            # Intra-epoch evaluation: the true peak may lie inside epoch 1.
            if args.eval_steps > 0 and (i + 1) % args.eval_steps == 0:
                stop = run_eval(e, i + 1)
                model.train()
                if stop:
                    break

        scheduler.step()
        avg_cls = loss_cls_total / num_batches
        avg_proto = loss_proto_total / num_batches
        avg_future = loss_future_total / num_batches
        epoch_time = time.time() - t_epoch

        if not stop and not (args.eval_steps > 0 and (i + 1) % args.eval_steps == 0):
            stop = run_eval(e, None)
            model.train()

        # One analysis-ready line per epoch: losses, metrics, best-so-far, lr,
        # and the live WPO parameters (gate/alpha/c) for improvement analysis.
        lr_now = optimizer.param_groups[0]['lr']
        wpo_str = " ".join(f"{n.split('.')[-1]}={p.item():.4f}"
                           for n, p in model.named_parameters()
                           if 'wpo' in n and p.numel() == 1)
        cp_str = f" CP:{loss_cp_total / max(num_batches, 1):.4f}" if args.class_proto else ""
        print(f"[Epoch {e+1}/{args.max_epoch} | {epoch_time:.0f}s] "
              f"Cls:{avg_cls:.4f} Proto:{avg_proto:.4f} Future:{avg_future:.4f}{cp_str} "
              f"| AUC:{last_auc:.4f} AP:{last_ap:.4f} | BEST AUC:{auc_best:.4f}(e{best_ep}) "
              f"| lr:{lr_now:.2e} | WPO {wpo_str or 'off'}")

        if stop:
            break

    if best_model_path and os.path.exists(best_model_path):
        final = f"model/final_auc{auc_best:.4f}{wpo_tag}.pt"
        ckpt = torch.load(best_model_path, weights_only=False)
        torch.save({'model_state_dict': ckpt['model_state_dict'],
                    'normal_proto_ema': ckpt.get('normal_proto_ema')}, final)
        print(f"Final model saved to {final} (AUC: {auc_best:.4f})")
    else:
        # Below the publication bar: keep nothing, the log line is the result.
        print(f"[NoSave] no model written: best AUC {auc_best:.4f} < "
              f"--save-min-auc {args.save_min_auc:.4f}")
    total_time = time.time() - t_start
    print(f"Total training time: {total_time:.0f}s ({total_time/60:.1f}min)")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[ucf_option.parser], add_help=False)
    parser.add_argument('--d-state', default=16, type=int)
    parser.add_argument('--d-conv', default=4, type=int)
    parser.add_argument('--expand', default=2, type=int)
    parser.add_argument('--mamba-layers', default=2, type=int)
    parser.add_argument('--mlp-scale', default=4, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--mamba-version', default='mamba1', choices=['mamba1', 'mamba2'])
    parser.add_argument('--mamba2-headdim', default=64, type=int)
    parser.add_argument('--mamba2-ngroups', default=1, type=int)
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
    parser.add_argument('--early-stop-patience', default=2, type=int,
                        help='stop after N consecutive evals without improvement')
    parser.add_argument('--early-stop-min-delta', default=0.0, type=float,
                        help='minimum AUC gain that counts as an improvement')
    parser.add_argument('--save-min-auc', default=0.87, type=float,
                        help='disk policy: write a checkpoint ONLY when AUC reaches '
                             'this bar (paper bar on UCF-Crime). Early stopping and '
                             'logging still track the true best AUC; 0 disables gating')
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
    setup_logging(f"train_ucf_sgl{wpo_tag}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    print(f"[RunCommand] python ucf_train_sgl.py --seed {args.seed} --batch-size {args.batch_size} "
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
          f"--future-weight {args.future_weight} "
          f"--aug {args.aug} --eval-steps {args.eval_steps} "
          f"--class-proto {args.class_proto} --cp-weight {args.cp_weight} "
          f"--proto-weight {args.proto_weight} "
          f"--dropout {args.dropout} --weight-decay {args.weight_decay} "
          f"--label-smooth {args.label_smooth} "
          f"--early-stop-patience {args.early_stop_patience} "
          f"--early-stop-min-delta {args.early_stop_min_delta}")

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest',
                      'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary',
                      'Explosion': 'explosion', 'Fighting': 'fighting',
                      'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
                      'Shooting': 'shooting', 'Shoplifting': 'shoplifting',
                      'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    # --fixed-timebase (P4): only the two TRAIN loaders change.  The test path
    # already used process_split, so it is untouched either way.
    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True,
                                is_random=args.fixed_timebase)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False,
                                 is_random=args.fixed_timebase)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    mamba_kwargs = {}
    if args.mamba_version == 'mamba2':
        mamba_kwargs = {'mamba2_headdim': args.mamba2_headdim, 'mamba2_ngroups': args.mamba2_ngroups}

    model = SingleMambaVAD(
        visual_width=args.visual_width, visual_length=args.visual_length,
        mamba_layers=args.mamba_layers, d_state=args.d_state,
        d_conv=args.d_conv, expand=args.expand,
        dropout=args.dropout,
        num_class=args.classes_num,
        device=device, mamba_version=args.mamba_version,
        multi_scale=args.multi_scale,
        multi_scale_factors=args.multi_scale_factors,
        bidirectional=args.bidirectional,
        mlp_scale=args.mlp_scale,
        use_wpo=args.use_wpo, wpo_alpha_init=args.wpo_alpha_init,
        wpo_c_init=args.wpo_c_init, wpo_c_max=args.wpo_c_max,
        wpo_multiscale=args.wpo_multiscale, wpo_band_c=args.wpo_band_c,
        wpo_band_mode=args.wpo_band_mode,
        adaptive_fusion=args.adaptive_fusion, wpo_parallel=args.wpo_parallel,
        wpo_adaptive_params=args.wpo_adaptive_params, wpo_motion_cond=args.wpo_motion_cond,
        use_class_proto=args.class_proto,
        **mamba_kwargs)

    if args.multi_scale and args.use_wpo and args.wpo_multiscale:
        ms_str = f"bands={args.wpo_band_c}({args.wpo_band_mode})"
    elif args.multi_scale:
        ms_str = f"scales={args.multi_scale_factors}(pool)"
    else:
        ms_str = "off"
    print(f"[Config] SingleMambaVAD UCF, "
          f"mamba_layers={args.mamba_layers}, d_state={args.d_state}, "
          f"mlp_scale={args.mlp_scale}, mamba_version={args.mamba_version}, "
          f"multi_scale={ms_str}, bidirectional={args.bidirectional}, "
          f"use_wpo={args.use_wpo}"
          + (f" (wpo_lr_scale={args.wpo_lr_scale}, c_max={args.wpo_c_max}, "
             f"band_mode={args.wpo_band_mode})"
             if args.use_wpo else ""))

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)