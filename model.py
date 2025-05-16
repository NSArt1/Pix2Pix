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

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1), ##doesnt change HxW, only add channels
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        self.dropout = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        res = self.conv_block(x)
        x = self.maxpool(res)
        x = self.dropout(x)

        return res, x
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dropout = nn.Dropout(dropout)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, y):
        x = self.upsample(x)
        x = self.dropout(x)
        # print("Inside UpSample: ", x.shape, y.shape)
        x = torch.cat((x, y), dim = 1) #concat by channels
        # print("Inside UpSample: ", x.shape, y.shape)
        x = self.conv_block(x)
        # print("x.shape Inside UpSample: ", x.shape)
        return x

class UNet(nn.Module):
    def __init__(self, depth=5, d=64, dropout=0.1):
        super().__init__()

        self.depth = depth
        self.dropout = dropout

        self.down_depth = depth
        self.up_depth = depth

        down_channels = [3] + [d * 2 ** i for i in range(self.down_depth - 1)] # [3, 64, 128, 256, 512]
        self.down_blocks = nn.ModuleList([DownSampleBlock(in_c, out_c, self.dropout)
            for in_c, out_c in zip(down_channels[:-1], down_channels[1:])
        ])

        up_channels = [d * 2 ** (i - 1) for i in range(self.up_depth, 0, -1)]
        up_channels.append(up_channels[-1]) #[1024, 512, 256, 128, 64, 64]
        self.up_blocks = nn.ModuleList([UpSampleBlock(in_c, out_c, self.dropout)
            for in_c, out_c in zip(up_channels[:-2], up_channels[2:])
        ]) #1024-256, 512 - 256, 128 -64

        mid_channels = [down_channels[-1],
                        down_channels[-1] // 2,
                        up_channels[0] // 2] #512 256 512
        self.mid_block = nn.Sequential(
            nn.Conv2d(mid_channels[0], mid_channels[1], 3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels[1]),
            nn.ReLU(),
            nn.Conv2d(mid_channels[1], mid_channels[2], 3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels[2]),
            nn.ReLU(),
        )

        self.conv1x1 = nn.Conv2d(up_channels[-1], 3, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        shortcuts = []
        for i in range(len(self.down_blocks)):
            res, x = self.down_blocks[i](x)
            shortcuts.append(res)
           

        x = self.mid_block(x)
        

        for i in range(len(self.up_blocks)):
            
            x = self.up_blocks[i](x, shortcuts[-i - 1])
        x = self.tanh(self.conv1x1(x))

        return x


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super().__init__()
        kw = 4  
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class Pix2pix(nn.Module):
    def __init__(self, alpha=100.0, lr=2e-4, betas=(0.5, 0.999), weight_decay=0.0, lr_decay_start_epoch=None, num_epochs=None, gan_loss_type = 'BCE',device="cuda"):
        super().__init__()

        self.generator = UNet()
        self.discriminator = Discriminator(input_nc=6)
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.lr_decay_start_epoch = lr_decay_start_epoch
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.device = device
        if gan_loss_type == 'BCE':
          self.criterion_GAN = nn.BCEWithLogitsLoss()
          print('BCE')
        else:
          print("L2")
          self.criterion_GAN = nn.MSELoss()

        self.criterion_L1 = nn.L1Loss()

        self.generator.to(device)
        self.discriminator.to(device)
        self.losses_D =[]
        self.losses_G =[]
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

    def forward(self, input_image):
        return self.generator(input_image)

    def generator_loss(self, input_image, real_image):
        fake_image = self.generator(input_image)
        fake_pair = torch.cat((input_image, fake_image), dim=1)
        pred_fake = self.discriminator(fake_pair)

        target_real = torch.full(pred_fake.shape, 0.9, device=self.device)
        loss_GAN = self.criterion_GAN(pred_fake, target_real)
        loss_L1 = self.criterion_L1(fake_image, real_image) * self.alpha
        loss_G = loss_GAN + loss_L1

        return loss_G, fake_image

    def discriminator_loss(self, input_image, real_image, fake_image):
        fake_pair = torch.cat((input_image, fake_image.detach()), dim=1)
        real_pair = torch.cat((input_image, real_image), dim=1)

        pred_real = self.discriminator(real_pair)
        pred_fake = self.discriminator(fake_pair)

        target_real = torch.full(pred_real.shape, 0.9, device=self.device)
        target_fake = torch.zeros_like(pred_fake, device=self.device)

        loss_real = self.criterion_GAN(pred_real, target_real)
        loss_fake = self.criterion_GAN(pred_fake, target_fake)

        loss_D = 0.5 * (loss_real + loss_fake)
        return loss_D

    def adjust_learning_rate(self, epoch):
        if self.lr_decay_start_epoch is not None and epoch >= self.lr_decay_start_epoch:
            factor = 1 - (epoch - self.lr_decay_start_epoch) / (self.num_epochs - self.lr_decay_start_epoch)
            lr = self.lr * factor
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr

    def train_model(self, dataloader, num_epochs=100):
        self.train()
        for epoch in tqdm(range(num_epochs)):
            epoch_loss_G, epoch_loss_D = 0.0, 0.0
            i = 0
            for input_image, real_image in tqdm(dataloader):
                input_image = input_image.to(self.device)
                real_image = real_image.to(self.device)

                # update generator
                requires_grad(self.discriminator, False)
                self.optimizer_G.zero_grad()
                loss_G, fake_image = self.generator_loss(input_image, real_image)
                loss_G.backward()
                self.optimizer_G.step()

                # update discriminator
                requires_grad(self.discriminator, True)
                self.optimizer_D.zero_grad()
                loss_D = self.discriminator_loss(input_image, real_image, fake_image)
                loss_D.backward()
                self.optimizer_D.step()

                epoch_loss_G += loss_G.item()
                epoch_loss_D += loss_D.item()

                
                i += 1

            avg_G = epoch_loss_G / len(dataloader)
            avg_D = epoch_loss_D / len(dataloader)
            self.losses_G.append(avg_G)
            self.losses_D.append(avg_D)
            tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] Loss_G: {avg_G:.4f} | Loss_D: {avg_D:.4f}")

    def train_l1(self, dataloader, num_epochs=100, save_path=None, vis_every=10):

      self.train()
      for epoch in tqdm(range(num_epochs)):
          self.adjust_learning_rate(epoch)

      
          if epoch == self.lr_decay_start_epoch:
              self.set_weight_decay(1e-5)

          epoch_loss_G= 0.0
          i = 0
          for input_image, real_image in tqdm(dataloader):
              input_image = input_image.to(self.device)
              real_image = real_image.to(self.device)

              requires_grad(self.discriminator, False)
              self.optimizer_G.zero_grad()
              loss_G, fake_image = self.generator_loss(input_image, real_image)
              loss_G.backward()
              self.optimizer_G.step()

 

              epoch_loss_G += loss_G.item()
         
              i += 1

          avg_G = epoch_loss_G / len(dataloader)

          self.losses_G.append(avg_G)

          tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] Loss_G: {avg_G:.4f}")