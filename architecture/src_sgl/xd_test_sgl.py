import argparse
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from mamba_vad_dual_model import SingleMambaVAD
from utils.dataset import XDDataset
import xd_option
from wave_decode import wave_decode


def test(model, testdataloader, maxlen, gt, gtsegments, gtlabels, device,
         decode=None, dump_path=''):
    model.to(device)
    model.eval()

    all_prob = []
    all_raw = []          # pre-decode scores, for the offline mu/alpha sweep

    with torch.no_grad():
        for item in testdataloader:
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length

            if visual.dim() == 2:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            num_chunks = int(length / maxlen) + 1
            chunk_lens = []
            remaining = length
            for _ in range(num_chunks):
                chunk_l = min(maxlen, remaining)
                chunk_lens.append(chunk_l)
                remaining -= chunk_l

            lengths_tensor = torch.tensor(chunk_lens, device=device)
            outputs = model(visual, lengths=lengths_tensor)

            # Classifier score (primary)
            logits1 = torch.sigmoid(outputs['logits1']).squeeze(-1)
            cls_chunks = [logits1[j, :chunk_lens[j]] for j in range(num_chunks) if chunk_lens[j] > 0]
            prob = torch.cat(cls_chunks)[:len_cur] if cls_chunks else torch.zeros(len_cur, device=device)

            # Inference-time physics decoder: project the WHOLE video's score
            # trajectory onto the nearest one with a sparse wave source. Done on
            # the full video (not per chunk) so evidence can propagate across
            # chunk boundaries - that long-range reach is exactly what the
            # local training-time loss cannot do.
            all_raw.append(prob.detach().cpu())

            if decode is not None and decode.get('mu', 0.0) > 0:
                prob = wave_decode(prob, alpha=decode['alpha'], mu=decode['mu'],
                                   rho=decode.get('rho', 1.0),
                                   iters=decode.get('iters', 40))

            all_prob.append(prob.cpu())

    # Video-level AUC/AP
    prob_all = torch.cat(all_prob).numpy()
    prob_all = np.repeat(prob_all, 16)[:len(gt)]
    gt = np.array(gt)[:len(prob_all)]

    if dump_path:
        # Dump the RAW (undecoded) per-video trajectories so the mu/alpha sweep
        # can be redone offline in seconds without touching the GPU again. Uses
        # all_raw, NOT all_prob, so a dump taken with --wave-decode-mu > 0 still
        # records the un-decoded scores the sweep expects.
        # MUST build an explicit 1-D object array: if every video happens to have
        # the SAME length, np.array(list, dtype=object) silently collapses to a
        # 2-D array whose elements are object-dtype ROWS, and torch.tensor()
        # then rejects them.
        _arr = np.empty(len(all_raw), dtype=object)
        for _i, _p in enumerate(all_raw):
            _arr[_i] = _p.numpy()
        np.savez(dump_path, probs=_arr, gt=np.array(gt), repeat=16)

    ROC = roc_auc_score(gt, prob_all)
    AP = average_precision_score(gt, prob_all)
    print(f"  AUC: {ROC:.4f} | AP: {AP:.4f}")

    return ROC, AP, 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[xd_option.parser], add_help=False)
    parser.add_argument('--d-state', default=8, type=int)
    parser.add_argument('--d-conv', default=4, type=int)
    parser.add_argument('--expand', default=2, type=int)
    parser.add_argument('--mamba-layers', default=2, type=int)
    parser.add_argument('--mlp-scale', default=2, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--mamba-version', default='mamba2', choices=['mamba1', 'mamba2'])
    parser.add_argument('--mamba2-headdim', default=64, type=int)
    parser.add_argument('--mamba2-ngroups', default=1, type=int)
    parser.add_argument('--pred-stride', default=1, type=int)
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--multi-scale-factors', default='1,8,16',
                        help='pooling granularities (legacy pooling path) and kernel '
                             'sizes (--wpo-band-mode fixed); Plan B band scales use '
                             '--wpo-band-c instead')
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--use-wpo', action='store_true')
    parser.add_argument('--wpo-alpha-init', default=0.3, type=float)
    parser.add_argument('--wpo-c-init', default=0.8, type=float)
    parser.add_argument('--wpo-c-max', default=20.0, type=float)
    parser.add_argument('--wpo-multiscale', action='store_true')
    parser.add_argument('--wpo-band-mode', default='learnable', choices=['learnable', 'fixed'])
    parser.add_argument('--adaptive-fusion', action='store_true')
    parser.add_argument('--wpo-parallel', action='store_true')
    parser.add_argument('--wpo-adaptive-params', action='store_true')
    parser.add_argument('--wpo-motion-cond', action='store_true')
    parser.add_argument('--class-proto', action='store_true')
    parser.add_argument('--wave-decode-mu', default=0.0, type=float,
                        help='inference-time physics decoder strength; 0 = off')
    parser.add_argument('--wave-decode-alpha', default=0.25, type=float)
    parser.add_argument('--wave-decode-rho', default=1.0, type=float)
    parser.add_argument('--wave-decode-iters', default=40, type=int)
    parser.add_argument('--dump-prob', default='',
                        help='if set, save raw per-video scores + gt to this npz')
    args = parser.parse_args()
    args.lr = float(args.lr)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting',
                      'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident',
                      'G': 'explosion'})

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
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
        wpo_multiscale=args.wpo_multiscale, wpo_band_mode=args.wpo_band_mode,
        adaptive_fusion=args.adaptive_fusion, wpo_parallel=args.wpo_parallel,
        wpo_adaptive_params=args.wpo_adaptive_params, wpo_motion_cond=args.wpo_motion_cond,
        use_class_proto=args.class_proto,
        **mamba_kwargs)

    ckpt = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    t_start = time.time()
    test(model, test_loader, args.visual_length,
         gt, gtsegments, gtlabels, device,
         decode={'mu': args.wave_decode_mu, 'alpha': args.wave_decode_alpha,
                 'rho': args.wave_decode_rho, 'iters': args.wave_decode_iters},
         dump_path=args.dump_prob)
    elapsed = time.time() - t_start
    print(f"  Test time: {elapsed:.0f}s")
