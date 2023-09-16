# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

def myopia_level(se):
    if se > 0.75:
        return 0
    if se > -0.5:
        return 1
    if se > -6.0:
        return 2
    return 3


class OCTDataset(Dataset):
    def __init__(self, img_path, labels_file, transform):

        self.img_path = img_path
        self.labels = pd.read_csv(labels_file)
        self.transform = transform

    def __getitem__(self, item):
        oct_dir = self.labels.loc[item, 'oct_dir']
        oct_name = os.path.basename(oct_dir)
        oct_dir = os.path.join(self.img_path, oct_dir)
        label = myopia_level(self.labels.loc[item, 'se'])
        assert len(os.listdir(oct_dir)) == 12
        oct_imgs = None
        for i in range(12):
            oct_img_name = os.path.join(oct_dir, oct_name + '_' + str(i) + '.png')
            oct_img = Image.open(oct_img_name)
            oct_img = self.transform(oct_img)
            if oct_imgs is None:
                oct_imgs = oct_img
            else:
                oct_imgs = torch.cat((oct_imgs, oct_img), dim=0)

        return oct_imgs, label

    def __len__(self):
        return len(self.labels.index)
