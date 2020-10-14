""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         factor = 2 if bilinear else 1

#         self.inc = DoubleConv(n_channels, 32)
#         self.down1 = Down(32, 64)
#         self.down2 = Down(64, 128)
#         self.down3 = Down(128, 256 //factor)
#         # self.down4 = Down(512, 1024 // factor)
#         # self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(256, 128 // factor, bilinear)
#         self.up3 = Up(128, 64 // factor, bilinear)
#         self.up4 = Up(64, 32, bilinear)
#         self.outc = OutConv(32, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         # x5 = self.down4(x4)
#         # x = self.up1(x5, x4)
#         x = self.up2(x4, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         factor = 2 if bilinear else 1

#         self.inc = ResnetBlock(n_channels, 32)

#         self.c1 = ResnetBlock(32, 64)
#         self.c2 = ResnetBlock(64, 64)
#         self.c3 = ResnetBlock(64, 64)
#         # self.c4 = ResnetBlock(64, 64)
#         # self.down1 = Down(32, 64)
#         # self.down2 = Down(64, 128)
#         # self.down3 = Down(128, 256 //factor)
#         # # self.down4 = Down(512, 1024 // factor)
#         # # self.up1 = Up(1024, 512 // factor, bilinear)
#         # self.up2 = Up(256, 128 // factor, bilinear)
#         # self.up3 = Up(128, 64 // factor, bilinear)
#         # self.up4 = Up(64, 32, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x = self.inc(x)
#         x = self.c1(x)
#         # x = self.c2(x)
#         # x = self.c3(x)
#         # x = self.c4(x)

#         logits = self.outc(x)
#         return logits




class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        self.nr_layers=2
        hidden_channels=64
        self.first_layer = nn.Linear(n_channels, hidden_channels)

        self.layers=torch.nn.ModuleList([])
        self.norms=torch.nn.ModuleList([])
        for i in range(self.nr_layers):
            self.layers.append( nn.Linear(hidden_channels, hidden_channels) )
            self.norms.append( nn.BatchNorm2d(hidden_channels) )

        # self.l1 = nn.Linear(n_channels, 8)
        # self.l2 = nn.Linear(8, 8)
        self.last_linear = nn.Linear(hidden_channels, n_classes)


        # self.seam_layer = nn.Linear(32, 1)
        # self.mean_layer = nn.Linear(32, 2)

    def forward(self, x):
        x=x.permute(0,2,3,1) # from NCHW to NHWC

        x = self.first_layer(x)

        # identity=x
        # # x= F.gelu(x)
        # x= torch.tanh(x)
        # x = self.l2(x)
        # x+=identity
        # # x= F.gelu(x)
        # x= torch.tanh(x)

        # x = self.l3(x)
        # identity=x
        # x= F.gelu(x)
        # x = self.l4(x)
        # x+=identity
        # x= F.gelu(x)

        #with loops 
        for i in range(self.nr_layers):
            identity=x

            # x=x.permute(0,3,1,2) # NHWC to NCHW
            # x=self.norms[i](x)
            # x=x.permute(0,2,3,1) # from NCHW to NHWC


            x=self.layers[i](x)
            x= F.gelu(x)
            # x= torch.tanh(x)
            x+=identity

        logits = self.last_linear(x)
        # seams = self.seam_layer(x)
        # mean = self.mean_layer(x)
        # print("logits has hsape ", logits.shape)
        logits=logits.permute(0,3,1,2) # NHWC to NCHW
        # seams=seams.permute(0,3,1,2) # NHWC to NCHW
        # mean=mean.permute(0,3,1,2) # NHWC to NCHW
        seams=logits
        mean=logits
        return logits, seams, mean
