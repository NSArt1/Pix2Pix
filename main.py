
import os

import numpy as np
import torch
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
# from torchvision import transforms
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import Pix2PixDataset
from model import Pix2pix
transform_maps = A.Compose([
    A.Resize(286, 286),
    A.RandomCrop(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
], additional_targets={'image0': 'image'})

root_datasets = {
    'edges2shoes': '/content/drive/MyDrive/datasets/edges2shoes',
    'edges2bags': '/content/drive/MyDrive/datasets/edges2handbags2',
    'facades': '/content/drive/MyDrive/datasets/facades',
    'maps': '/content/drive/MyDrive/datasets/maps',
    'cityscapes': '/content/drive/MyDrive/datasets/cityscapes',

}

root_dir = root_datasets['maps']

train_dataset = Pix2PixDataset(root_dir, transform=transform_maps,  is_train=True, to_leak=0.9,swap=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
lr_decay_start_epoch=200
N_EPOCHS = 100
weight_decay=0
model = Pix2pix(num_epochs=N_EPOCHS, gan_loss_type = 'MSE', device=device).to(device)
model.train_model(train_loader, num_epochs=N_EPOCHS)
torch.save(model.state_dict(), '/content/drive/MyDrive/yoursdir/'+f'/model_weights_{N_EPOCHS}.pth')