import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools

class UCFDataset(data.Dataset):
    # `is_random` is misleadingly named (see XDDataset): it does NOT shuffle.
    # It selects how a clip longer than `clip_dim` is reduced to `clip_dim` frames:
    #   False -> tools.uniform_extract : squeeze the WHOLE clip in, so one model
    #            frame = clip_seconds/clip_dim, which varies ~10x across the set.
    #   True  -> tools.random_extract  : one CONTIGUOUS clip_dim-frame window at
    #            the native fps, i.e. the same time base that process_split
    #            already uses at TEST time.  Exposed as --fixed-timebase.
    #
    # This is the UCF port of the P4 fix.  Upstream VadCLIP trains UCF with
    # process_feat (whole-clip squeeze, variable time base) and tests with
    # process_split (native-fps windows) -- the same train/test time base
    # asymmetry already documented for XD.  Default False = reproduce upstream.
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict,
                 normal: bool = False, is_random: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        self.is_random = is_random
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim,
                                                           is_random=self.is_random)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length

class XDDataset(data.Dataset):
    # `is_random` is misleadingly named: it does NOT shuffle.  It selects how a
    # clip longer than `clip_dim` is reduced to `clip_dim` frames.
    #   False -> tools.uniform_extract : squeeze the WHOLE clip in, so one model
    #            frame = clip_seconds/clip_dim, which varies ~10x across the set.
    #   True  -> tools.random_extract  : one CONTIGUOUS clip_dim-frame window at
    #            the native 15 fps, i.e. the same time base that process_split
    #            already uses at TEST time.  Exposed as --fixed-timebase.
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, is_random: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.is_random = is_random
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim, is_random=self.is_random)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length