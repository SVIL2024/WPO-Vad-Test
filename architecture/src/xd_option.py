import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=1, type=int)
parser.add_argument('--attn-window', default=64, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=7, type=int)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='model/model_xd.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='model/checkpoint.pth')
parser.add_argument('--batch-size', default=96, type=int)
# The dataset __getitem__ does an np.load per video, so with the DataLoader
# default (num_workers=0) every batch stalls the GPU on disk I/O - measured
# 106 s/batch and ~0% GPU util on the AutoDL box, i.e. ~74 min per epoch.
# >0 workers overlap the loads with the forward/backward.
parser.add_argument('--num-workers', default=8, type=int,
                    help='DataLoader worker processes; 0 = load in the main '
                         'process (very slow here)')
parser.add_argument('--train-list', default='list/xd_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/xd_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment.npy')
parser.add_argument('--gt-label-path', default='list/gt_label.npy')

parser.add_argument('--lr', default=1e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[3, 6, 10])

# Wave-source sparsity prior (see src/wave_loss.py). Default 0.0 = OFF, so the
# original VadCLIP baseline stays bit-identical unless you opt in.
parser.add_argument('--wave-loss', default=0.0, type=float,
                    help='weight of the wave-source L1 loss; 0 disables it')
parser.add_argument('--wave-alpha', default=0.25, type=float,
                    help='damping coefficient alpha in s_tt + alpha*s_t = f')

# Feature-field wave operator (src/wave_field.py). Unlike --wave-loss, which
# penalises the source on the SCALAR score (measured inert: |A s| is 0.33% of
# |s|), this one acts on the CLIP feature field where |A u| is ~69% of |u| and
# feeds the source energy back in as an evidence channel. 0 = disabled, which
# leaves the baseline bit-identical.
parser.add_argument('--wave-field', default=0, type=int,
                    help='number of learned wave modes K; 0 disables the branch')
parser.add_argument('--wave-field-gain', default=0.1, type=float,
                    help='initial fusion weight of the wave evidence channel')
parser.add_argument('--wave-field-lr-scale', default=1.0, type=float,
                    help='lr multiplier for the wave branch; it is randomly '
                         'initialised on top of a converged trunk, so it needs '
                         'a bigger step than the pretrained weights')

# --- E3: WHERE the wave evidence is fused ---------------------------------
# E2 measured both fusion arms against A0 (AP2 0.8490380) and both missed the
# +0.005 bar by orders of magnitude: E2a (trunk frozen) +0.0000255, E2b (joint)
# -0.0184. The branch was provably ALIVE (_wave_bad=0, gain 0.02 -> 0.071), and
# AUC1 - the very path the evidence feeds - went UP in both arms (+0.0049 in
# E2b) while the reported AP2 fell. So the signal is real; the lever is not.
# Adding a per-video z-scored term to the attention logits barely moves a
# weighted average, so E3 modulates the FEATURES being pooled instead.
parser.add_argument('--wave-fusion', default='logits', choices=['logits', 'feature'],
                    help='logits: add the evidence to the attention logits '
                         '(E1/E2 behaviour). feature: gate the visual features '
                         'with (1 + gain*tanh(evidence)) before pooling (E3).')

# --- E2: warm-start a converged trunk and train the wave branch on top -----
# Motivation (2026-09-16): E1 trained the wave branch JOINTLY from a random
# init and lost ~3 AP against the A0 control (0.8182 vs 0.8490, two runs).
# A randomly-initialised branch injects a random direction into the attention
# pooling from step 0, so the trunk is damaged before the branch has learned
# anything. Starting from the converged A0 checkpoint with a near-zero fusion
# gain means step 0 IS A0, so the experiment has a floor instead of a cliff.
parser.add_argument('--init-ckpt', default='',
                    help='warm-start weights; loaded with strict=False so a '
                         'branch that does not exist in the checkpoint keeps '
                         'its own init')
parser.add_argument('--freeze-trunk', default=0, type=int,
                    help='1 = train only parameters whose name contains '
                         '"wave"; the pretrained trunk stays fixed')
# Checkpoint gate: runs below this AP are NOT written to disk (the server disk
# cannot hold every run). Set 0.0 to disable and save unconditionally.
parser.add_argument('--save-min-ap', default=0.83, type=float,
                    help='only persist a checkpoint when AP >= this value')

# Inference-time physics decoder (src/wave_decode.py). Test-time only: it needs
# no retraining and adds no parameters. Default mu=0.0 keeps it switched off.
parser.add_argument('--wave-decode-mu', default=0.0, type=float,
                    help='decoder strength; 0 disables decoding')
parser.add_argument('--wave-decode-alpha', default=0.25, type=float)
parser.add_argument('--wave-decode-rho', default=1.0, type=float)
parser.add_argument('--wave-decode-iters', default=40, type=int)
parser.add_argument('--wave-decode-on', default='p2', choices=['p1', 'p2'],
                    help='which branch to decode; p2 is the metric-driving score')
parser.add_argument('--dump-prob', default='',
                    help='if set, save raw per-video scores + gt to this npz')