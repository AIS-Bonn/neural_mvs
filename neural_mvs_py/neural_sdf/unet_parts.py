""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ResnetBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(mid_channels),
        #     # nn.ReLU(inplace=True),
        #     torch.sort(dim=1),
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(out_channels),
        #     # nn.ReLU(inplace=False)
        #     torch.sort(dim=1)
        # )
        self.c1=nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0)
        # self.b1= nn.BatchNorm2d(mid_channels)
        self.b1=nn.GroupNorm(num_groups=int(mid_channels), num_channels=mid_channels)
        self.r1=nn.ReLU(inplace=True)
        self.c2=nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)
        # self.b2=nn.BatchNorm2d(out_channels)
        self.b2=nn.GroupNorm(num_groups=int(mid_channels), num_channels=mid_channels)
        self.r2=nn.ReLU(inplace=False)

    def forward(self, x):
        identity=x
        # x=self.double_conv(x)

        x=self.c1(x)
        x=self.b1(x)
        # x, indices = torch.sort(x, dim=1)
        # x= self.r1(x)
        x= F.gelu(x)
        # x= torch.tanh(x)
        x=self.c2(x)
        x=self.b2(x)
        # x, indices = torch.sort(x, dim=1)
        # x= self.r2(x)
        x= F.gelu(x)
        # x= torch.tanh(x)

        if(x.shape[1]==identity.shape[1]):
            x+=identity
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
