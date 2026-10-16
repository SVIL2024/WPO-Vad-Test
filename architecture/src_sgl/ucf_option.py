import argparse

parser = argparse.ArgumentParser(description='Single Mamba VAD - UCF')
parser.add_argument('--seed', default=555, type=int)

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
parser.add_argument('--train-list', default='../list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='../list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='../list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='../list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='../list/gt_label_ucf.npy')

parser.add_argument('--lr', default=2e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[4, 8])

# P4 fix (ported from xd_option.py 2026-09-17).
# Off  -> reproduce upstream VadCLIP: train with process_feat (whole-clip
#         squeeze -> time base varies ~10x across the set), test with
#         process_split (native-fps windows).  Train/test time bases disagree.
# On   -> train on one contiguous native-fps window (random_extract), i.e. the
#         same time base the test path already uses.
# Default False so existing UCF numbers stay reproducible; pass
# --fixed-timebase to run the corrected arm.
parser.add_argument('--fixed-timebase', default=False, action='store_true')
