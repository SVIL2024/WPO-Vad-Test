import argparse

parser = argparse.ArgumentParser(description='Single Mamba VAD - XD')
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
parser.add_argument('--train-list', default='../list/xd_CLIP_rgb.csv')
parser.add_argument('--test-list', default='../list/xd_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='../list/gt.npy')
parser.add_argument('--gt-segment-path', default='../list/gt_segment.npy')
parser.add_argument('--gt-label-path', default='../list/gt_label.npy')
parser.add_argument('--lr', default=1e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[3, 6, 10])
# Train/test time-base mismatch (see utils/tools.py).
#   False (default, inherited): `uniform_extract` squeezes the WHOLE clip into
#     `visual_length` frames, so one model frame = clip_seconds/visual_length.
#     Clip durations span 0.13 s .. 2163 s, so the time base varies ~10x.
#   True: take a contiguous `visual_length`-frame window at the native 15 fps,
#     exactly what `process_split` already does at TEST time.  One model frame is
#     always 1/15 s, so train and test finally share a time axis and a fixed
#     omega means a fixed physical frequency.
# store_true, NOT type=bool: with type=bool the string "False" evaluates to
# True, so `--fixed-timebase False` would silently turn the fix ON.
parser.add_argument('--fixed-timebase', default=False, action='store_true')
