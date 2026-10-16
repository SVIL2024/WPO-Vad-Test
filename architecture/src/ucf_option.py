import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='model/model_ucf.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='model/checkpoint.pth')
parser.add_argument('--batch-size', default=64, type=int)
# Same reasoning as xd_option: the dataset np.loads one file per video, so
# num_workers=0 leaves the GPU idle for most of every batch.
parser.add_argument('--num-workers', default=8, type=int,
                    help='DataLoader worker processes; 0 = load in the main '
                         'process (very slow here)')
parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy')

parser.add_argument('--lr', default=2e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[4, 8])

# Wave-source sparsity prior (see src/wave_loss.py). Default 0.0 = OFF, so the
# original VadCLIP baseline stays bit-identical unless you opt in.
parser.add_argument('--wave-loss', default=0.0, type=float,
                    help='weight of the wave-source L1 loss; 0 disables it')
parser.add_argument('--wave-alpha', default=0.25, type=float,
                    help='damping coefficient alpha in s_tt + alpha*s_t = f')

# Feature-field wave operator (src/wave_field.py). See the same block in
# xd_option.py for the rationale. 0 = disabled -> baseline bit-identical.
parser.add_argument('--wave-field', default=0, type=int,
                    help='number of learned wave modes K; 0 disables the branch')
parser.add_argument('--wave-field-gain', default=0.1, type=float,
                    help='initial fusion weight of the wave evidence channel')
# Checkpoint gate: runs below this AUC are NOT written to disk (the server disk
# cannot hold every run). Set 0.0 to disable and save unconditionally.
parser.add_argument('--save-min-auc', default=0.86, type=float,
                    help='only persist a checkpoint when AUC >= this value')

# Inference-time physics decoder (src/wave_decode.py). Test-time only: no
# retraining, no new parameters. Default mu=0.0 keeps it switched off.
parser.add_argument('--wave-decode-mu', default=0.0, type=float,
                    help='decoder strength; 0 disables decoding')
parser.add_argument('--wave-decode-alpha', default=0.25, type=float)
parser.add_argument('--wave-decode-rho', default=1.0, type=float)
parser.add_argument('--wave-decode-iters', default=40, type=int)
parser.add_argument('--wave-decode-on', default='p1', choices=['p1', 'p2'],
                    help='which branch to decode; UCF reports ROC1 (p1)')
parser.add_argument('--dump-prob', default='',
                    help='if set, save raw per-video scores + gt to this npz')