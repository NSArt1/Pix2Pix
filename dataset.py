import os

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Pix2PixDataset(data.Dataset):
    def __init__(self, root_dir, transform=None, is_train=True, to_leak=0.0, swap=False, path = 'train'):
        self.root_dir = root_dir
        self.transform = transform
        self.swap = swap
        self.is_train = is_train
        self.path_to_train_files = os.path.join(root_dir, 'train')
        self.path_to_val_files = os.path.join(root_dir, 'val')

        train_files = os.listdir(self.path_to_train_files)
        val_files = os.listdir(self.path_to_val_files)

        leak_count = int(to_leak * len(val_files))

        if is_train:
            self.path = self.path_to_train_files
            self.files = train_files + val_files[:leak_count]
            self.leak_prefix = self.path_to_val_files  
            self.leak_files = val_files[:leak_count]
        else:
            self.path = self.path_to_val_files
            self.files = val_files[leak_count:]

    def __getitem__(self, idx):
        filename = self.files[idx]

        if self.is_train and filename in getattr(self, "leak_files", []):
            img_path = os.path.join(self.leak_prefix, filename)
        else:
            img_path = os.path.join(self.path, filename)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img_x = img.crop((0, 0, w // 2, h))
        img_y = img.crop((w // 2, 0, w, h))

        if self.transform is not None:
            transformed = self.transform(image=np.array(img_x), image0=np.array(img_y))
            img_x = transformed["image"]
            img_y = transformed["image0"]

        if self.swap:
            return img_y, img_x
        return img_x, img_y

    def __len__(self):
        return len(self.files)