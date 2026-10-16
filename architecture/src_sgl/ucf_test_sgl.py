import argparse
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from mamba_vad_dual_model import SingleMambaVAD
from utils.dataset import UCFDataset
import ucf_option


def test(model, testdataloader, maxlen, device, gt_path,
         normal_proto=None):
    model.to(device)
    model.eval()
    all_prob = []

    with torch.no_grad():
        for item in testdataloader:
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length
            visual = visual.to(device)

            num_chunks = int(length / maxlen) + 1
            chunk_lens = []
            remaining = length
            for _ in range(num_chunks):
                chunk_l = min(maxlen, remaining)
                chunk_lens.append(chunk_l)
                remaining -= chunk_l

            if visual.dim() == 2:
                visual = visual.unsqueeze(0)

            lengths_tensor = torch.tensor(chunk_lens, device=device)
            outputs = model(visual, lengths=lengths_tensor)

            # Classifier score (primary)
            logits1 = torch.sigmoid(outputs['logits1']).squeeze(-1)
            cls_chunks = [logits1[j, :chunk_lens[j]] for j in range(num_chunks) if chunk_lens[j] > 0]
            prob = torch.cat(cls_chunks)[:len_cur] if cls_chunks else torch.zeros(len_cur, device=device)

            all_prob.append(prob.cpu())

    prob_all = torch.cat(all_prob).numpy()
    prob_all = np.repeat(prob_all, 16)
    gt = np.load(gt_path)
    ROC = roc_auc_score(gt, prob_all)
    AP = average_precision_score(gt, prob_all)
    print(f"  AUC: {ROC:.4f} | AP: {AP:.4f}")

    return ROC, AP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[ucf_option.parser], add_help=False)
    parser.add_argument('--d-state', default=16, type=int)
    parser.add_argument('--d-conv', default=4, type=int)
    parser.add_argument('--expand', default=2, type=int)
    parser.add_argument('--mamba-layers', default=2, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--mamba-version', default='mamba1', choices=['mamba1', 'mamba2'])
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
    args = parser.parse_args()
    args.lr = float(args.lr)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest',
                      'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary',
                      'Explosion': 'explosion', 'Fighting': 'fighting',
                      'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
                      'Shooting': 'shooting', 'Shoplifting': 'shoplifting',
                      'Stealing': 'stealing', 'Vandalism': 'vandalism'})

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
        use_wpo=args.use_wpo, wpo_alpha_init=args.wpo_alpha_init,
        wpo_c_init=args.wpo_c_init, wpo_c_max=args.wpo_c_max,
        wpo_multiscale=args.wpo_multiscale, wpo_band_mode=args.wpo_band_mode,
        adaptive_fusion=args.adaptive_fusion, wpo_parallel=args.wpo_parallel,
        wpo_adaptive_params=args.wpo_adaptive_params, wpo_motion_cond=args.wpo_motion_cond,
        use_class_proto=args.class_proto,
        **mamba_kwargs)

    ckpt = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    normal_proto_ema = ckpt.get('normal_proto_ema')

    t_start = time.time()
    test(model, test_loader, args.visual_length, device, args.gt_path,
         normal_proto=normal_proto_ema)
    elapsed = time.time() - t_start
    print(f"  Test time: {elapsed:.0f}s")
