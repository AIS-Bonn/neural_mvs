import torch
from neural_mvs.modules import *
# from latticenet_py.lattice.lattice_modules import *
# import torchvision.models as models
from easypbr import mat2tensor

import torch
import torch.nn as nn
import torchvision

from collections import OrderedDict

from torchmeta.modules.conv import *
from torchmeta.modules.module import *
from torchmeta.modules.utils import *
from torchmeta.modules import (MetaModule, MetaSequential)
from torchmeta.modules.utils import get_subdict
from meta_modules import *

from functools import reduce
from torch.nn.modules.module import _addindent

from neural_mvs.nerf_utils import *


##Network with convgnrelu so that the coord added by concat coord are not destroyed y the gn in gnreluconv
class VAE_2(torch.nn.Module):
    def __init__(self):
        super(VAE_2, self).__init__()

        self.start_nr_channels=32
        # self.start_nr_channels=4
        # self.z_size=256
        # self.z_size=16
        self.z_size=128
        self.nr_downsampling_stages=3
        self.nr_blocks_down_stage=[2,2,2]
        self.nr_upsampling_stages=3
        self.nr_blocks_up_stage=[1,1,1]

        # self.concat_coord=ConcatCoord()

        #start with a normal convolution
        self.first_conv = torch.nn.Conv2d(3, self.start_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() 
        # self.first_conv = torch.nn.utils.weight_norm( torch.nn.Conv2d(3, self.start_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() )
        cur_nr_channels=self.start_nr_channels
        

        #cnn for encoding
        self.blocks_down_per_stage_list=torch.nn.ModuleList([])
        self.coarsens_list=torch.nn.ModuleList([])
        for i in range(self.nr_downsampling_stages):
            self.blocks_down_per_stage_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_down_stage[i]):
                # cur_nr_channels+=2 #because we concat the coords
                self.blocks_down_per_stage_list[i].append( ResnetBlock(cur_nr_channels, dilations=[1,1], biases=[True,True], with_dropout=False) )
            nr_channels_after_coarsening=int(cur_nr_channels*2)
            # self.coarsens_list.append( ConvGnRelu(nr_channels_after_coarsening, kernel_size=2, stride=2, padding=0, dilation=1, bias=False, with_dropout=False, transposed=False).cuda() )
            self.coarsens_list.append( Block(nr_channels_after_coarsening, kernel_size=2, stride=2, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() )
            cur_nr_channels=nr_channels_after_coarsening
            # cur_nr_channels+=2 #because we concat the coords

        # linear layer to get to the z and the sigma
        # self.gn_before_fc=None
        self.fc_mu= None
        self.fc_sigma= None

        #linear layer to get from the z_size to whatever size it had before the fully connected layer
        self.d1=None

        ##cnn for decoding
        # cur_nr_channels= self.z_size #when decoding we decode the z which is a tensor of N, ZSIZE, 1, 1
        self.blocks_up_per_stage_list=torch.nn.ModuleList([])
        self.finefy_list=torch.nn.ModuleList([])
        for i in range(self.nr_upsampling_stages):
            self.blocks_up_per_stage_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_up_stage[i]):
                # cur_nr_channels+=2 #because we concat the coords
                self.blocks_up_per_stage_list[i].append( ResnetBlock(cur_nr_channels, dilations=[1,1], biases=[True,True], with_dropout=False) )
            nr_channels_after_finefy=int(cur_nr_channels/2)
            # self.finefy_list.append( GnGeluConv(nr_channels_after_finefy, kernel_size=4, stride=2, padding=1, dilation=1, bias=False, with_dropout=False, transposed=True).cuda() )
            # self.finefy_list.append( GnConv(nr_channels_after_finefy, kernel_size=2, stride=2, padding=0, dilation=1, bias=False, with_dropout=False, transposed=True).cuda() )
            # self.finefy_list.append( ConvRelu(nr_channels_after_finefy, kernel_size=2, stride=2, padding=0, dilation=1, bias=False, with_dropout=False, transposed=True).cuda() )
            self.finefy_list.append( Block(nr_channels_after_finefy, kernel_size=2, stride=2, padding=0, dilation=1, bias=True, with_dropout=False, transposed=True).cuda() )
            cur_nr_channels=nr_channels_after_finefy
            # cur_nr_channels+=2 #because we concat the coords

        #last conv to regress the color 
        # self.last_conv=GnGeluConv(3, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False).cuda()
        # cur_nr_channels+=2 #because we concat the coords
        self.last_conv= torch.nn.Conv2d(cur_nr_channels, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() 
        # self.last_conv=torch.nn.utils.weight_norm( torch.nn.Conv2d(cur_nr_channels, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() )
        self.tanh=torch.nn.Tanh()
        self.relu=torch.nn.ReLU(inplace=True)


    def forward(self, x):

        #reshape x from H,W,C to 1,C,H,W
        # x = x.permute(2,0,1).unsqueeze(0).contiguous()
        # print("x input has shape, ",x.shape)


        #first conv
        # x = self.concat_coord(x)
        x = self.first_conv(x)
        x=gelu(x)

        #encode 
        # TIME_START("down_path")
        for i in range(self.nr_downsampling_stages):
            # print("DOWNSAPLE ", i, " with x of shape ", x.shape)

            #resnet blocks
            for j in range(self.nr_blocks_down_stage[i]):
                # x = self.concat_coord(x)
                x = self.blocks_down_per_stage_list[i][j] (x) 

            #now we do a downsample
            # x = self.concat_coord(x)
            x = self.coarsens_list[i] ( x )
            # x = self.concat_coord(x)
        # TIME_END("down_path")

        # x_shape_before_fcs=x.shape[0]
        x_shape_before_bottlneck=x.shape
        if (self.fc_mu is None):
            # self.gn_before_fc = torch.nn.GroupNorm(x_shape_before_bottlneck[1], x_shape_before_bottlneck[1]).cuda()
            self.fc_mu= torch.nn.Linear(x.flatten().shape[0], self.z_size).cuda()
            # self.fc_sigma= torch.nn.Linear(x.shape[0], self.z_size).cuda()
        # x=self.gn_before_fc(x)
        x=x.flatten()
        # print("x shape before the fc layers is", x.shape)
        mu=self.fc_mu(x)
        # mu=self.relu(mu)
        # sigma=self.fc_sigma(x)
        # print("mu has size", mu.shape)
        # print("sigma has size", mu.shape)

        #sample 
        # TODO
        x=mu

        #decode
        if (self.d1 is None):
            self.d1= torch.nn.Linear(self.z_size, np.prod(x_shape_before_bottlneck) ).cuda()
        x=self.d1(x) #to make it the same size as before the bottleneck

        # x_shape_before_fcs
        x = x.view(x_shape_before_bottlneck)
        for i in range(self.nr_downsampling_stages):
            # print("UPSAMPLE ", i, " with x of shape ", x.shape)

            #resnet blocks
            for j in range(self.nr_blocks_up_stage[i]):
                # x = self.concat_coord(x)
                x = self.blocks_up_per_stage_list[i][j] (x) 

            #now we do a upscale with tranposed conv
            # x = self.concat_coord(x)
            x = self.finefy_list[i] ( x )
            # x = self.concat_coord(x)
        # print("x output after upsampling has shape, ",x.shape)

        #regress color with one last conv that has 3 channels
        # x = self.concat_coord(x)
        x=self.last_conv(x)
        x=self.tanh(x)
        # print("x output before reshaping has shape, ",x.shape)

        


        #reshape x from 1,C,H,W to H,W,C
        # x = x.squeeze(0).permute(1,2,0).contiguous()
        # print("x output has shape, ",x.shape)
        return x




class VAE_tiling(torch.nn.Module):
    def __init__(self):
        super(VAE_tiling, self).__init__()

        self.first_time=True

        self.start_nr_channels=32
        # self.start_nr_channels=4
        # self.z_size=256
        # self.z_size=16
        self.z_size=256
        self.nr_downsampling_stages=5
        self.nr_blocks_down_stage=[2,2,2,2,2]
        # self.nr_upsampling_stages=3
        # self.nr_blocks_up_stage=[1,1,1]
        self.nr_decoder_layers=3
        # self.pos_encoding_elevated_channels=128
        # self.pos_encoding_elevated_channels=2
        self.pos_encoding_elevated_channels=26
        # self.pos_encoding_elevated_channels=0

        # self.concat_coord=ConcatCoord()

        #start with a normal convolution
        # self.first_conv = torch.nn.utils.weight_norm( torch.nn.Conv2d(3, self.start_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() )
        self.first_conv = torch.nn.Conv2d(3, self.start_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() 
        cur_nr_channels=self.start_nr_channels
        

        #cnn for encoding
        self.blocks_down_per_stage_list=torch.nn.ModuleList([])
        self.coarsens_list=torch.nn.ModuleList([])
        for i in range(self.nr_downsampling_stages):
            self.blocks_down_per_stage_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_down_stage[i]):
                # cur_nr_channels+=2 #because we concat the coords
                self.blocks_down_per_stage_list[i].append( ResnetBlock(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True,True], with_dropout=False, do_norm=True) )
            nr_channels_after_coarsening=int(cur_nr_channels*2)
            # self.coarsens_list.append( ConvGnRelu(nr_channels_after_coarsening, kernel_size=2, stride=2, padding=0, dilation=1, bias=False, with_dropout=False, transposed=False).cuda() )
            self.coarsens_list.append( Block(nr_channels_after_coarsening, kernel_size=2, stride=2, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True).cuda() )
            cur_nr_channels=nr_channels_after_coarsening
            # cur_nr_channels+=2 #because we concat the coords

        # linear layer to get to the z and the sigma
        # self.gn_before_fc=None
        self.fc_mu= None
        self.fc_sigma= None

        #linear layer to get from the z_size to whatever size it had before the fully connected layer
        self.d1=None

        
        # self.coord_elevate=Block( out_channels=self.pos_encoding_elevated_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        # self.coord_elevate2=Block( out_channels=self.pos_encoding_elevated_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 

        ##cnn for decoding
        cur_nr_channels= self.z_size #when decoding we decode the z which is a tensor of N, ZSIZE, 1, 1
        # self.blocks_decoder_stage_list=torch.nn.ModuleList([])
        # for i in range(self.nr_decoder_layers):
            # self.blocks_decoder_stage_list.append( Block(cur_nr_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() )

        #last conv to regress the color 
        # self.last_conv=torch.nn.utils.weight_norm( torch.nn.Conv2d(cur_nr_channels, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() )
        # self.last_conv= torch.nn.Conv2d(cur_nr_channels, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() 
        self.tanh=torch.nn.Tanh()
        self.relu=torch.nn.ReLU(inplace=True)


        #test for impving the tiling vae
        self.attention_regresor=Block(activ=torch.sigmoid, out_channels=self.z_size+self.pos_encoding_elevated_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        # self.attention_regresor=Block(activ=torch.sigmoid, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        self.bias_regresor=Block(out_channels=self.z_size+self.pos_encoding_elevated_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        # self.bias_regresor=Block(out_channels=self.z_size+self.pos_encoding_elevated_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        # self.bk1=Block(out_channels=self.z_size+self.pos_encoding_elevated_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        # self.bk2=Block(out_channels=self.z_size+self.pos_encoding_elevated_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        # self.bk3=Block(out_channels=self.z_size+self.pos_encoding_elevated_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        # self.bk4=Block(out_channels=self.z_size+self.pos_encoding_elevated_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        # self.bk5=Block(out_channels=self.z_size+self.pos_encoding_elevated_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        # self.bk6=Block(out_channels=self.z_size+self.pos_encoding_elevated_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        # self.r1= ResnetBlock(int(self.z_size/4)+self.pos_encoding_elevated_channels, kernel_size=1, stride=1, padding=0, dilations=[1,1], biases=[True,True], with_dropout=False)
        # self.r1= ResnetBlock(self.z_size+self.pos_encoding_elevated_channels, kernel_size=1, stride=1, padding=0, dilations=[1,1], biases=[True,True], with_dropout=False)
        self.r1= ResnetBlock(activ=torch.sin, out_channels=1024+self.pos_encoding_elevated_channels, kernel_size=1, stride=1, padding=0, dilations=[1,1], biases=[True,True], with_dropout=False)
        self.bk1=Block(activ=torch.sin, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        self.r2= ResnetBlock(activ=torch.sin, out_channels=128, kernel_size=1, stride=1, padding=0, dilations=[1,1], biases=[True,True], with_dropout=False)
        self.bk2=Block(activ=torch.sin, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        # self.r3= ResnetBlock(128, kernel_size=1, stride=1, padding=0, dilations=[1,1], biases=[True,True], with_dropout=False)
        # self.bk3=Block(out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        # self.bk3=Block(activ=torch.sin, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        # self.bk4=Block(activ=torch.sin, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False).cuda() 
        self.rgb_regresor=torch.nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True).cuda() 


        self.upsample = torch.nn.Upsample(size=128, mode='bilinear').cuda()

    def forward(self, x):

        #reshape x from H,W,C to 1,C,H,W
        # x = x.permute(2,0,1).unsqueeze(0).contiguous()
        # print("x input has shape, ",x.shape)

        height=x.shape[2]
        width=x.shape[3]
        # print("height is ", height)
        # print("width is ", width)

        # image_height=height
        # image_width=width
        # y_coords = 2.0 * torch.arange(image_height).unsqueeze(
        #     1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        # x_coords = 2.0 * torch.arange(image_width).unsqueeze(
        #     0).expand(image_height, image_width) / (image_width - 1.0) - 1.0
        # coords = torch.stack((y_coords, x_coords), dim=0).float()
        # coords=coords.unsqueeze(0).to("cuda")
        # pos_encoding=positional_encoding(coords, num_encoding_functions=6, log_sampling=False)
        # x=torch.cat( [x,pos_encoding], dim=1)
        # x=torch.cat( [x,coords], dim=1)

        #first conv
        # x = self.concat_coord(x)
        x = self.first_conv(x)
        x=self.relu(x)

        #encode 
        # TIME_START("down_path")
        for i in range(self.nr_downsampling_stages):
            # print("DOWNSAPLE ", i, " with x of shape ", x.shape)

            #resnet blocks
            for j in range(self.nr_blocks_down_stage[i]):
                # x = self.concat_coord(x)
                x = self.blocks_down_per_stage_list[i][j] (x) 

            #now we do a downsample
            # x = self.concat_coord(x)
            x = self.coarsens_list[i] ( x )
            # x = self.concat_coord(x)
        # TIME_END("down_path")

        # x_shape_before_fcs=x.shape[0]
        # x_shape_before_bottlneck=x.shape
        # if (self.fc_mu is None):
            # self.gn_before_fc = torch.nn.GroupNorm(x_shape_before_bottlneck[1], x_shape_before_bottlneck[1]).cuda()
            # self.fc_mu= torch.nn.Linear(x.flatten().shape[0], self.z_size).cuda()
            # self.fc_sigma= torch.nn.Linear(x.shape[0], self.z_size).cuda()
        # x=self.gn_before_fc(x)
        # x=x.flatten()
        # print("x shape before the fc layers is", x.shape)
        # mu=self.fc_mu(x)
        # mu=self.relu(mu)
        # sigma=self.fc_sigma(x)
        # print("mu has size", mu.shape)
        # print("sigma has size", mu.shape)

        #sample 
        # TODO
        # x=mu


        #try to reshape it into an image and upsample bilinearly
        # x=x.view(1,int(self.z_size/4),2,2)
        # x=self.upsample(x)


        #try to reshape it into an image and upsample bilinearly
        # x=x.view(1,int(self.z_size/4),2,2)
        x=self.upsample(x)
        # print("x has shape ", x.shape)



        # TILE
        # x=x.view(1,self.z_size,1,1)
        # x = x.expand(-1, -1, height, width)
        # print("x repreated is ", x.shape)


        #compute the coords 
        image_height=height
        image_width=width
        y_coords = 2.0 * torch.arange(image_height).unsqueeze(
            1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(image_width).unsqueeze(
            0).expand(image_height, image_width) / (image_width - 1.0) - 1.0
        coords = torch.stack((y_coords, x_coords), dim=0).float()
        coords=coords.unsqueeze(0).to("cuda")

        # print("coords is ", coords)
        # coords=(coords+1)*0.5
        # print("coords has shaoe ", coords.shape)
        pos_encoding=positional_encoding(coords, num_encoding_functions=6, log_sampling=False)
        # print("pos_encoding has shaoe ", pos_encoding.shape)

        # pos_encoding_elevated=self.coord_elevate(pos_encoding)
        # pos_encoding_elevated=self.coord_elevate2(pos_encoding_elevated)

        x=torch.cat( [x,pos_encoding], dim=1)
        # x=torch.cat( [x,pos_encoding_elevated], dim=1)
        # x=torch.cat( [x,coords], dim=1)
        # print("pos_encoding_elevated is", pos_encoding_elevated.shape)
        # print("pos_encoding is", pos_encoding.shape)
        # print("x has hspae" , x.shape)

        # each feature column of X is now required to create te correct output color a a weight matrix is multiplied with each feature at each pixel
        # however, since the weight matrix is the same at every position, that means that the Z vector is required to have enough information to reconstruct every pixel in a sorta global manner 
        # so the gradient that gets pushed back into the Z vector might both say (please make the z vector more appropriate for rendering a white pixel and simultanously a black pixel)
        # however the whole z vector need not be as important for the regressing each pixel, imagine that the Z vector is 512 dimensional, only a small part of it is required to regress the top left part of the image and another small part is required for the right bottom part of the image
        # therefore we need some sort of attention over the vector
        #ATTENTION mechanism: 
        # z_coord= [z_size, coord]
        # z_coord-> attention(size of 512)
        # z_coord-> bias(size of 512)
        # z_attenuated = z_coord * attention
        # rgb = w*z_atenuated + bias (the w is a z_size X 3 the same at every pixel but is gets attenuated by the attenion
        #SPATIAL weight 
        #an alternative would be to learn for every pixel a Z_size x 3 matrix and a bias (size 3) and that will be directly the weight and bias matrix



        ##ATTENTION mechanism deocding: DOESNT MAKE ANY DIFFERENCE
        # z_and_coord=x
        # attention_spatial=self.attention_regresor(z_and_coord)
        # bias_spatial=self.bias_regresor(z_and_coord)
        # z_attenuated=z_and_coord*attention_spatial + bias_spatial
        # x=z_attenuated
        # x=z_and_coord

        # identity=x
        # x=self.bk1(x)
        # x=self.bk2(x)
        # x+=identity
        # identity=x
        # x=self.bk3(x)
        # x=self.bk4(x)
        # x+=identity
        # identity=x
        # x=self.bk5(x)
        # x=self.bk6(x)
        # x+=identity

        # x=-19
        # print("x mean is ", x.mean())
        # print("x std is ", x.std())
        x*=30
        x=self.r1(x)
        x=self.bk1(x)
        x=self.r2(x)
        x=self.bk2(x)
        # x=self.r3(x)
        # x=self.bk3(x)
        x=self.rgb_regresor(x)
        # x=self.rgb_regresor(z_and_coord)
        x=self.tanh(x)




        # if self.first_time: 
        #     with torch.set_grad_enabled(False):
        #         self.first_time=False
        #         torch.nn.init.zeros_(self.attention_regresor.conv.weight) 
        #         torch.nn.init.zeros_(self.bias_regresor.conv.weight) 





        #decode
        # if (self.d1 is None):
            # self.d1= torch.nn.Linear(self.z_size, np.prod(x_shape_before_bottlneck) ).cuda()
        # x=self.d1(x) #to make it the same size as before the bottleneck

        # # x_shape_before_fcs
        # # x = x.view(x_shape_before_bottlneck
        # identity=None
        # for i in range(self.nr_decoder_layers):
        #     x = self.blocks_decoder_stage_list[i] ( x )
        #     if i==0:
        #         identity=x
        #     print("x has hspae" , x.shape)
        #     # x = self.concat_coord(x)
        # # print("x output after upsampling has shape, ",x.shape)
        # x+=identity

        # #regress color with one last conv that has 3 channels
        # # x = self.concat_coord(x)
        # x=self.last_conv(x)
        # x=self.tanh(x)
        # # print("x output before reshaping has shape, ",x.shape)

        


        #reshape x from 1,C,H,W to H,W,C
        # x = x.squeeze(0).permute(1,2,0).contiguous()
        # print("x output has shape, ",x.shape)
        return x







#spatial broadcaster 
# https://github.com/dfdazac/vaesbd/blob/master/model.py
class VAE(nn.Module):
    """Variational Autoencoder with spatial broadcast decoder.
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{in}, H_{in}, W_{in})`
    """
    def __init__(self, im_size, decoder='sbd'):
        super(VAE, self).__init__()
        enc_convs = [nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=4, stride=2, padding=1)]
        enc_convs.extend([nn.Conv2d(in_channels=64, out_channels=64,
                                    kernel_size=4, stride=2, padding=1)
                          for i in range(3)])
        self.enc_convs = nn.ModuleList(enc_convs)

        self.fc = nn.Sequential(nn.Linear(in_features=4096, out_features=256),
                                nn.ReLU(),
                                nn.Linear(in_features=256, out_features=256))

        if decoder == 'deconv':
            self.dec_linear = nn.Linear(in_features=128, out_features=256)
            dec_convs = [nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                            kernel_size=4, stride=2, padding=1)
                         for i in range(4)]
            self.dec_convs = nn.ModuleList(dec_convs)
            self.decoder = self.deconv_decoder
            self.last_conv = nn.ConvTranspose2d(in_channels=64, out_channels=3,
                                                kernel_size=4, stride=2,
                                                padding=1)

        elif decoder == 'sbd':
            # Coordinates for the broadcast decoder
            self.im_size = im_size
            x = torch.linspace(0, 1, im_size)
            y = torch.linspace(0, 1, im_size)
            x_grid, y_grid = torch.meshgrid(x, y)
            # print("x_Grid is ", x_grid)
            # print("y_Grid is ", y_grid)
            # print("x_grid is ", x_grid.shape)
            x_grid=x_grid.view(1,1,128,128)
            y_grid=y_grid.view(1,1,128,128)
            x_grid=positional_encoding(x_grid, num_encoding_functions=6, log_sampling=False)
            y_grid=positional_encoding(y_grid, num_encoding_functions=6, log_sampling=False)
            # print("x_grid is ", x_grid.shape)
            # print("x_Grid is ", x_grid)
            # print("y_Grid is ", y_grid)
            # Add as constant, with extra dims for N and C
            # self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
            # self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))
            self.register_buffer('x_grid', x_grid.view((1, 13, 128, 128)))
            self.register_buffer('y_grid', y_grid.view((1, 13, 128, 128)))

            # dec_convs = [nn.Conv2d(in_channels=12, out_channels=64,
            # dec_convs = [nn.Conv2d(in_channels=36, out_channels=64,
            dec_convs = [nn.Conv2d(in_channels=154, out_channels=64,
                                   kernel_size=3, padding=1),
                         nn.Conv2d(in_channels=64, out_channels=64,
                                   kernel_size=3, padding=1)]
            self.dec_convs = nn.ModuleList(dec_convs)
            self.decoder = self.sb_decoder
            self.last_conv = nn.Conv2d(in_channels=64, out_channels=3,
                                       kernel_size=3, padding=1)

    def encoder(self, x):
        batch_size = x.size(0)
        for module in self.enc_convs:
            x = F.relu(module(x))

        x = x.view(batch_size, -1)
        x = self.fc(x)

        return torch.chunk(x, 2, dim=1)

    def deconv_decoder(self, z):
        x = F.relu(self.dec_linear(z)).view(-1, 64, 2, 2)
        for module in self.dec_convs:
            x = F.relu(module(x))
        x = self.last_conv(x)

        return x

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sb_decoder(self, z):
        batch_size = z.size(0)
        # View z as 4D tensor to be tiled across new H and W dimensions
        # Shape: NxDx1x1
        z = z.view(z.shape + (1, 1))

        # Tile across to match image size
        # Shape: NxDx64x64
        z = z.expand(-1, -1, self.im_size, self.im_size)

        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x64x64
        x = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                       self.y_grid.expand(batch_size, -1, -1, -1), z), dim=1)

        for module in self.dec_convs:
            x = F.relu(module(x))
        x = self.last_conv(x)

        return x

    def forward(self, x):
        batch_size = x.size(0)
        mu, logvar = self.encoder(x)
        z=mu
        # z = self.sample(mu, logvar)
        x_rec = self.decoder(z)

        return x_rec



class Encoder2D(torch.nn.Module):
    def __init__(self, z_size):
        super(Encoder2D, self).__init__()

        self.first_time=True

        self.start_nr_channels=32
        self.z_size=z_size
        self.nr_downsampling_stages=5
        self.nr_blocks_down_stage=[2,2,2,2,2]
        #
        self.first_conv = torch.nn.Conv2d(5, self.start_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() 
        cur_nr_channels=self.start_nr_channels
        

        #cnn for encoding
        self.blocks_down_per_stage_list=torch.nn.ModuleList([])
        self.coarsens_list=torch.nn.ModuleList([])
        for i in range(self.nr_downsampling_stages):
            self.blocks_down_per_stage_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_down_stage[i]):
                # cur_nr_channels+=2 #because we concat the coords
                self.blocks_down_per_stage_list[i].append( ResnetBlock(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True,True], with_dropout=False, do_norm=True) )
            nr_channels_after_coarsening=int(cur_nr_channels*2)
            # self.coarsens_list.append( ConvGnRelu(nr_channels_after_coarsening, kernel_size=2, stride=2, padding=0, dilation=1, bias=False, with_dropout=False, transposed=False).cuda() )
            self.coarsens_list.append( Block(in_channels=cur_nr_channels, out_channels=nr_channels_after_coarsening, kernel_size=2, stride=2, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True).cuda() )
            cur_nr_channels=nr_channels_after_coarsening
            # cur_nr_channels+=2 #because we concat the coords

        # linear layer to get to the z and the sigma
        # self.gn_before_fc=None
        self.fc_mu= None
        self.fc_sigma= None

        #linear layer to get from the z_size to whatever size it had before the fully connected layer
        self.d1=None

        
       
        self.tanh=torch.nn.Tanh()
        self.relu=torch.nn.ReLU(inplace=True)

        self.concat_coord=ConcatCoord()


    def forward(self, x):

        x=self.concat_coord(x)
       
        #first conv
        x = self.first_conv(x)
        x=self.relu(x)

        #encode 
        # TIME_START("down_path")
        for i in range(self.nr_downsampling_stages):
            # print("DOWNSAPLE ", i, " with x of shape ", x.shape)

            #resnet blocks
            for j in range(self.nr_blocks_down_stage[i]):
                x = self.blocks_down_per_stage_list[i][j] (x) 

            #now we do a downsample
            x = self.coarsens_list[i] ( x )
        # TIME_END("down_path")

        # x_shape_before_fcs=x.shape[0]
        x_shape_before_bottlneck=x.shape
        if (self.fc_mu is None):
            # self.gn_before_fc = torch.nn.GroupNorm(x_shape_before_bottlneck[1], x_shape_before_bottlneck[1]).cuda()
            self.fc_mu= torch.nn.Linear(x.flatten().shape[0], self.z_size).cuda()
            # self.fc_sigma= torch.nn.Linear(x.shape[0], self.z_size).cuda()
        # x=self.gn_before_fc(x)
        x=x.flatten()
        # print("x shape before the fc layers is", x.shape)
        mu=self.fc_mu(x)
        # mu=self.relu(mu)
        # sigma=self.fc_sigma(x)
        print("mu has size", mu.shape)
        print("mu has min max", mu.min(), " ", mu.max())
        # print("sigma has size", mu.shape)

        #sample 
        # TODO
        x=mu

        return x



class Encoder(torch.nn.Module):
    def __init__(self, z_size):
        super(Encoder, self).__init__()

        ##params 
        # self.nr_points_z=128
        self.z_size=z_size


        #activations
        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()
        self.gelu=torch.nn.GELU()

        #layers
        resnet = torchvision.models.resnet50(pretrained=True)
        modules=list(resnet.children())[:-1]
        self.resnet=nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = True







        self.start_nr_channels=32
        # self.start_nr_channels=4
        self.nr_downsampling_stages=6
        self.nr_blocks_down_stage=[2,2,2,2,2,2,2]
        self.nr_channels_after_coarsening_per_layer=[64,64,128,128,256,256,512,512,512,1024,1024]
        # self.nr_upsampling_stages=3
        # self.nr_blocks_up_stage=[1,1,1]
        self.nr_decoder_layers=3
        # self.pos_encoding_elevated_channels=128
        # self.pos_encoding_elevated_channels=2
        self.pos_encoding_elevated_channels=26
        # self.pos_encoding_elevated_channels=0


        #make my own resnet so that is can take a coordconv
        self.concat_coord=ConcatCoord()

        #start with a normal convolution
        self.first_conv = torch.nn.Conv2d(5, self.start_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() 
        # self.first_conv = torch.nn.utils.weight_norm( torch.nn.Conv2d(3, self.start_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() )
        cur_nr_channels=self.start_nr_channels

        #cnn for encoding
        self.blocks_down_per_stage_list=torch.nn.ModuleList([])
        self.coarsens_list=torch.nn.ModuleList([])
        for i in range(self.nr_downsampling_stages):
            self.blocks_down_per_stage_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_down_stage[i]):
                cur_nr_channels+=2 #because we concat the coords
                self.blocks_down_per_stage_list[i].append( ResnetBlock(cur_nr_channels, 3, 1, 1, dilations=[1,1], biases=[True,True], with_dropout=False) )
            # nr_channels_after_coarsening=int(cur_nr_channels*2)
            nr_channels_after_coarsening=self.nr_channels_after_coarsening_per_layer[i]
            print("nr_channels_after_coarsening is ", nr_channels_after_coarsening)
            # self.coarsens_list.append( ConvGnRelu(nr_channels_after_coarsening, kernel_size=2, stride=2, padding=0, dilation=1, bias=False, with_dropout=False, transposed=False).cuda() )
            cur_nr_channels+=2 #because we concat the coords
            self.coarsens_list.append( BlockForResnet(cur_nr_channels, nr_channels_after_coarsening, kernel_size=2, stride=2, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False ).cuda() )
            cur_nr_channels=nr_channels_after_coarsening
            # cur_nr_channels+=2 #because we concat the coords





        # self.z_to_3d=None
        self.to_z=None


    def forward(self, x):
        # for p in self.resnet.parameters():
        #     p.requires_grad = False

        # print("encoder x input is ", x.min(), " ", x.max())
        # z=self.resnet(x) # z has size 1x512x1x1



        # first conv
        x = self.concat_coord(x)
        x = self.first_conv(x)
        x=self.relu(x)

        #encode 
        # TIME_START("down_path")
        for i in range(self.nr_downsampling_stages):
            # print("DOWNSAPLE ", i, " with x of shape ", x.shape)
            #resnet blocks
            for j in range(self.nr_blocks_down_stage[i]):
                x = self.concat_coord(x)
                x = self.blocks_down_per_stage_list[i][j] (x) 

            #now we do a downsample
            x = self.concat_coord(x)
            x = self.coarsens_list[i] ( x )
        # TIME_END("down_path")
        z=x
        print("z after encoding has shape ", z.shape)



        return z





class SirenNetwork(MetaModule):
    def __init__(self, in_channels, out_channels):
        super(SirenNetwork, self).__init__()

        self.first_time=True

        self.nr_layers=4
        self.out_channels_per_layer=[128, 128, 128, 128, 16]
        # self.out_channels_per_layer=[20, 20, 20, 20, 20]

        # #cnn for encoding
        # self.layers=torch.nn.ModuleList([])
        # for i in range(self.nr_layers):
        #     is_first_layer=i==0
        #     self.layers.append( Block(activ=torch.sin, out_channels=self.channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=is_first_layer).cuda() )
        # self.rgb_regresor=Block(activ=torch.tanh, out_channels=3, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda() 

        cur_nr_channels=in_channels

        self.net=torch.nn.ModuleList([])
        # self.net.append( MetaSequential( Block(activ=torch.sin, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[0], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=True).cuda() ) )
        # cur_nr_channels=self.out_channels_per_layer[0]

        for i in range(self.nr_layers):
            is_first_layer=i==0
            self.net.append( MetaSequential( BlockSiren(activ=torch.sin, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=is_first_layer).cuda() ) )
            # self.net.append( MetaSequential( ResnetBlock(activ=torch.sin, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=False, is_first_layer=False).cuda() ) )
            # cur_nr_channels=self.out_channels_per_layer[i] * 2 #because we also added a relu
            cur_nr_channels=self.out_channels_per_layer[i]  #when we do NOT add a relu
        self.net.append( MetaSequential(BlockSiren(activ=None, in_channels=cur_nr_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda()  ))

        # self.net = MetaSequential(*self.net)

        self.pca=PCA()

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # print("siren entry x is ", x.shape )

        # print("params of sirenent is ", params)

        #reshape x from H,W,C to 1,C,H,W
        # x = x.permute(2,0,1).unsqueeze(0).contiguous()
        # print("x input has shape, ",x.shape)

        height=x.shape[2]
        width=x.shape[3]
        # print("height is ", height)
        # print("width is ", width)

        image_height=height
        image_width=width
        y_coords = 2.0 * torch.arange(image_height).unsqueeze(
            1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(image_width).unsqueeze(
            0).expand(image_height, image_width) / (image_width - 1.0) - 1.0
        coords = torch.stack((y_coords, x_coords), dim=0).float()
        coords=coords.unsqueeze(0).to("cuda")
        # pos_encoding=positional_encoding(coords, num_encoding_functions=6, log_sampling=False)
        # x=torch.cat( [x,pos_encoding], dim=1)
        # x=torch.cat( [x,coords], dim=1)

        x=coords
        # print("x coords for siren is ", x.shape)
        # print("x which is actually coords is ", x)
        # print("as input to siren x is  " , x.mean().item() , " std ", x.std().item(), " min: ", x.min().item(),  "max ", x.max().item() )
     

        # print ("running siren")
        # x=self.net(x, params=get_subdict(params, 'net'))
        for i in range(len(self.net)):
            
            #if we are at the last layer, we get the features and pca them
            if i==len(self.net)-1:
                orig=x
                print("last layer", x.shape)
                nr_channels=x.shape[1]
                height=x.shape[2]
                weight=x.shape[3]
                x=x.permute(0,2,3,1).contiguous() #from N,C,H,W to N,H,W,C
                x=x.view(-1,nr_channels)
                c=self.pca.forward(x)
                c=c.view(height,width,3)
                c=c.unsqueeze(0).permute(0,3,1,2).contiguous() #from N,H,W,C to N,C,H,W
                print("c is ", c.shape)
                c_mat=tensor2mat(c)
                Gui.show(c_mat, "c_mat")
                #switch back 
                x=orig

                #just show the first pixel and the last pixel
                first=x[:,:,0:1,0:1]
                last=x[:,:,84:85,56:57]
                # print("first", first)
                # print("last", last)
                print("first and last feature is ", first.shape)
                two=torch.cat([first,last],2)
                diff=(first-last).norm()
                print("diff", diff)
                # print("the two feature of the pixels side by side is ", two)

            x=self.net[i](x, None)

            if i==len(self.net)-1:
                #now we got the color
                #just show the first pixel and the last pixel
                first=x[:,:,0:1,0:1]
                last=x[:,:,84:85,56:57]
                print("first", first)
                print("last", last)
                diff=(first-last).norm()
                print("diff", diff)

            # x=self.net[i](x, None)
        # print("finished siren")
        # exit(1)

       
        return x



#this siren net receives directly the coordinates, no need to make a concat coord or whatever
class SirenNetworkDirect(MetaModule):
    def __init__(self, in_channels, out_channels):
        super(SirenNetworkDirect, self).__init__()

        self.first_time=True

        self.nr_layers=6
        self.out_channels_per_layer=[128, 128, 128, 128, 128, 128]
        # self.out_channels_per_layer=[100, 100, 100, 100, 100]
        # self.out_channels_per_layer=[256, 256, 256, 256, 256]
        # self.out_channels_per_layer=[256, 128, 64, 32, 16]
        # self.out_channels_per_layer=[256, 256, 256, 256, 256]

        # #cnn for encoding
        # self.layers=torch.nn.ModuleList([])
        # for i in range(self.nr_layers):
        #     is_first_layer=i==0
        #     self.layers.append( Block(activ=torch.sin, out_channels=self.channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=is_first_layer).cuda() )
        # self.rgb_regresor=Block(activ=torch.tanh, out_channels=3, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda() 

        cur_nr_channels=in_channels

        self.net=torch.nn.ModuleList([])
        # self.net.append( MetaSequential( Block(activ=torch.sin, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[0], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=True).cuda() ) )
        # cur_nr_channels=self.out_channels_per_layer[0]
    
        self.position_embedders=torch.nn.ModuleList([])


        self.position_embedder=( MetaSequential( 
            Block(activ=torch.relu, in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            Block(activ=torch.relu, in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            # Block(activ=torch.relu, in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            # Block(activ=torch.relu, in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            Block(activ=None, in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            ) )
        # cur_nr_channels=128+3

        for i in range(self.nr_layers):
            is_first_layer=i==0
            self.net.append( MetaSequential( BlockSiren(activ=torch.sin, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=is_first_layer).cuda() ) )
            # self.net.append( MetaSequential( ResnetBlock(activ=torch.sin, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=False, is_first_layer=False).cuda() ) )

            # self.position_embedders.append( MetaSequential( 
            #     Block(activ=torch.sigmoid, in_channels=in_channels, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            #     Block(activ=None, in_channels=self.out_channels_per_layer[i], out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            #     ) )

            # if i<self.nr_layers-4:
            if i!=self.nr_layers:
            # if i==0:
                # cur_nr_channels=self.out_channels_per_layer[i]+ in_channels*10
                # cur_nr_channels=self.out_channels_per_layer[i]+ in_channels*30 #when repeating the raw coordinates a bit
                # cur_nr_channels=self.out_channels_per_layer[i]+ self.out_channels_per_layer[i] #when using a positional embedder for the raw coords
                cur_nr_channels=self.out_channels_per_layer[i]+ 128+3 #when using a positional embedder for the raw coords
            else:
                cur_nr_channels=self.out_channels_per_layer[i]
            # if i!=0:
                # cur_nr_channels+=self.out_channels_per_layer[0]

            # cur_nr_channels=self.out_channels_per_layer[i]
        # self.net.append( MetaSequential(Block(activ=torch.sigmoid, in_channels=cur_nr_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda()  ))
        self.net.append( MetaSequential(BlockSiren(activ=None, in_channels=cur_nr_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda()  ))

        # self.net = MetaSequential(*self.net)



    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())


        #the x in this case is Nx3 but in order to make it run fine with the 1x1 conv we make it a Nx3x1x1
        nr_points=x.shape[0]
        # x=x.view(nr_points,3,1,1)

        # X comes as nr_points x 3 matrix but we want adyacen rays to be next to each other. So we put the nr of the ray in the batch
        # x=x.contiguous()
        # x=x.view(71,107,30,3)
        # x=x.view(1,-1,71,107,30)
        # print("x is ", x.shape)
        x=x.permute(2,3,0,1).contiguous() #from 71,107,30,3 to 30,3,71,107


        # print ("running siren")
        # x=self.net(x, params=get_subdict(params, 'net'))

        x_raw_coords=x
        x_first_layer=None
        position_input=self.position_embedder(x_raw_coords)
        position_input=torch.cat([position_input,x_raw_coords],1)

        for i in range(len(self.net)):
            # print("x si ", x.shape)
            x=self.net[i](x)
            # if i==0:
            #     x_first_layer=x

            # print("x has shape ", x.shape, " x_raw si ", x_raw_coords.shape)
            
            # if i<len(self.net)-5: #if it's any layer except the last one
            if i!=len(self.net)-1: #if it's any layer except the last one
            # if i == 0:
                # print("cat", i)
                # positions=x_raw_coords*2**i
                # encoding=[]
                # for func in [torch.sin, torch.cos]:
                #     encoding.append(func(positions))
                # encoding.append(x_raw_coords)
                # position_input=torch.cat(encoding, 1)
                # position_input=x_raw_coords*self.out_channels_per_layer[i]*0.5
                # position_input=x_raw_coords.repeat(1,30,1,1)

                # position_input=self.position_embedders[i](x_raw_coords)
                # position_input=self.position_embedder(x_raw_coords)
                x=torch.cat([position_input,x],1)
            # if i!=0 and i!=len(self.net)-1:
                # x=torch.cat([x_first_layer,x],1)
                # x=x+position_input
        # print("finished siren")

        x=x.permute(2,3,0,1).contiguous() #from 30,nr_out_channels,71,107 to  71,107,30,3
        x=x.view(nr_points,-1,1,1)
       
        return x


#this siren net receives directly the coordinates, no need to make a concat coord or whatever
#This one does soemthin like densenet so each layers just append to a stack
class SirenNetworkDense(MetaModule):
    def __init__(self, in_channels, out_channels):
        super(SirenNetworkDense, self).__init__()

        self.first_time=True

        self.nr_layers=5
        self.out_channels_per_layer=[128, 64, 64, 64, 64]
        # self.out_channels_per_layer=[100, 100, 100, 100, 100]
        # self.out_channels_per_layer=[256, 256, 256, 256, 256]


        cur_nr_channels=in_channels

        self.net=torch.nn.ModuleList([])
        # self.net.append( MetaSequential( Block(activ=torch.sin, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[0], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=True).cuda() ) )
        # cur_nr_channels=self.out_channels_per_layer[0]

        for i in range(self.nr_layers):
            is_first_layer=i==0
            self.net.append( MetaSequential( BlockSiren(activ=torch.sin, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=is_first_layer).cuda() ) )
            # self.net.append( MetaSequential( ResnetBlock(activ=torch.sin, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=False, is_first_layer=False).cuda() ) )
            cur_nr_channels= cur_nr_channels + self.out_channels_per_layer[i]
        # self.net.append( MetaSequential(Block(activ=torch.sigmoid, in_channels=cur_nr_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda()  ))
        self.net.append( MetaSequential(BlockSiren(activ=None, in_channels=cur_nr_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda()  ))

        # self.net = MetaSequential(*self.net)



    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())


        #the x in this case is Nx3 but in order to make it run fine with the 1x1 conv we make it a Nx3x1x1
        nr_points=x.shape[0]
        # x=x.view(nr_points,3,1,1)

        # X comes as nr_points x 3 matrix but we want adyacen rays to be next to each other. So we put the nr of the ray in the batch
        # x=x.contiguous()
        # x=x.view(71,107,30,3)
        # x=x.view(1,-1,71,107,30)
        print("x is ", x.shape)
        x=x.permute(2,3,0,1).contiguous() #from 71,107,30,3 to 30,3,71,107


        # print ("running siren")
        # x=self.net(x, params=get_subdict(params, 'net'))

        x_raw_coords=x
        x_first_layer=None

        stack=[]
        stack.append(x)

        for i in range(len(self.net)):

            input = torch.cat(stack,1)
            x=self.net[i](input)
            stack.append(x)
          

        x=x.permute(2,3,0,1).contiguous() #from 30,3,71,107 to  71,107,30,3
        x=x.view(nr_points,-1,1,1)
       
        return x


#this siren net receives directly the coordinates, no need to make a concat coord or whatever
class NerfDirect(MetaModule):
    def __init__(self, in_channels, out_channels):
        super(NerfDirect, self).__init__()

        self.first_time=True

        self.nr_layers=5
        self.out_channels_per_layer=[128, 128, 128, 128, 128, 128, 128]
        # self.out_channels_per_layer=[100, 100, 100, 100, 100]

        # #cnn for encoding
        # self.layers=torch.nn.ModuleList([])
        # for i in range(self.nr_layers):
        #     is_first_layer=i==0
        #     self.layers.append( Block(activ=torch.sin, out_channels=self.channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=is_first_layer).cuda() )
        # self.rgb_regresor=Block(activ=torch.tanh, out_channels=3, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda() 

        self.initial_channels=in_channels

        cur_nr_channels=in_channels

        self.net=[]
        # self.net.append( MetaSequential( Block(activ=torch.sin, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[0], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=True).cuda() ) )
        # cur_nr_channels=self.out_channels_per_layer[0]

        for i in range(self.nr_layers):
            is_first_layer=i==0
            self.net.append( MetaSequential( Block( in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=is_first_layer).cuda() ) )
            # self.net.append( MetaSequential( ResnetBlock(activ=torch.sin, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=False, is_first_layer=False).cuda() ) )
            cur_nr_channels=self.out_channels_per_layer[i]
        self.net.append( MetaSequential(Block(activ=None, in_channels=cur_nr_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda()  ))

        self.net = MetaSequential(*self.net)



    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())


        #the x in this case is Nx3 but in order to make it run fine with the 1x1 conv we make it a Nx3x1x1
        nr_points=x.shape[0]
        x=x.view(nr_points,-1,1,1)


        # print ("running siren")
        x=self.net(x, params=get_subdict(params, 'net'))
        # print("finished siren")

        x=x.view(nr_points,-1,1,1)

        # print("nerf output has min max", x.min().item(), x.mean().item(), "mean ", x.mean() )
       
        return x





class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.first_time=True

        #params
        # self.z_size=512
        self.z_size=256
        # self.z_size=2048
        self.nr_points_z=256
        self.num_encodings=10
        # self.siren_out_channels=64
        # self.siren_out_channels=32
        self.siren_out_channels=4

        #activ
        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()


        # self.encoder=Encoder2D(self.z_size)
        self.encoder=Encoder(self.z_size)
        # self.siren_net = SirenNetwork(in_channels=3, out_channels=4)
        # self.siren_net = SirenNetworkDirect(in_channels=3, out_channels=4)
        # self.siren_net = SirenNetworkDirect(in_channels=3+3*self.num_encodings*2, out_channels=self.siren_out_channels)
        # self.siren_net = SirenNetworkDense(in_channels=3+3*self.num_encodings*2, out_channels=4)
        self.nerf_net = NerfDirect(in_channels=3+3*self.num_encodings*2, out_channels=4)
        # self.hyper_net = HyperNetwork(hyper_in_features=self.nr_points_z*3*2, hyper_hidden_layers=1, hyper_hidden_features=512, hypo_module=self.siren_net)
        self.hyper_net = HyperNetwork(hyper_in_features=self.nr_points_z*3*2, hyper_hidden_layers=1, hyper_hidden_features=512, hypo_module=self.nerf_net)


        self.z_to_z3d = torch.nn.Sequential(
            torch.nn.Linear( self.z_size , self.z_size).to("cuda"),
            torch.nn.ReLU(),
            torch.nn.Linear( self.z_size , self.nr_points_z*3).to("cuda")
        )

        self.z_to_zapp = torch.nn.Sequential(
            torch.nn.Linear( self.z_size , self.z_size).to("cuda"),
            torch.nn.ReLU(),
            torch.nn.Linear( self.z_size , self.nr_points_z*3).to("cuda")
        )


        # #from the features of the siren we predict the rgb
        # self.rgb_pred=torch.nn.ModuleList([])
        # self.nr_rgb_pred=1
        # self.rgb_pred.append(
        #     Block(activ=gelu, in_channels=self.siren_out_channels-1, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        # )
        # for i in range(self.nr_rgb_pred):
        #     self.rgb_pred.append(
        #         Block(activ=gelu, in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     )
        # self.rgb_pred.append(
        #     Block(activ=None, in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        # )

        self.leaned_pe=LearnedPE(in_channels=3, num_encoding_functions=self.num_encodings, logsampling=True)

       

      
    def forward(self, guide, x, all_imgs_poses_cam_world_list, gt_tf_cam_world, gt_K, novel=False):

        # nr_imgs=x.shape[0]

        # # print("encoding")
        # TIME_START("encoding")
        # z=self.encoder(x)
        # TIME_END("encoding")
        # # print("encoder outputs z", z.shape)

        # TIME_START("rotate")
        # #make z into a nr_imgs x z_size
        # z=z.view(nr_imgs, self.z_size)

        # #make z into a 3D thing
        # z3d=self.z_to_z3d(z)
        # z3d=z3d.reshape(nr_imgs, self.nr_points_z, 3)
        # # z3d=self.sigmoid(z3d)
        # # print("after making 3d z is ", z.shape)

        # #rotate everything into the same world frame
        # R_world_cam_all_list=[]
        # t_world_cam_all_list=[]
        # for i in range(nr_imgs):
        #     tf_world_cam= all_imgs_poses_cam_world_list[i].inverse()
        #     R=torch.from_numpy(tf_world_cam.linear()).to("cuda")
        #     t=torch.from_numpy(tf_world_cam.translation()).to("cuda")
        #     R_world_cam_all_list.append(R.unsqueeze(0))
        #     t_world_cam_all_list.append(t.view(1,1,3) )
        # R_world_cam_all=torch.cat(R_world_cam_all_list, 0) 
        # t_world_cam_all=torch.cat(t_world_cam_all_list, 0) 
        # # print("R_world_cam_all is ", R_world_cam_all.shape)
        # # print("before rotatin z3d is", z3d)
        # z3d=torch.matmul(z3d, R_world_cam_all.transpose(1,2))  + t_world_cam_all
        # # print("aftering rotatin z3d is", z3d)
        # # print("after multiplying z3d is ", z3d.shape)



        # #get also an apperence vector and append it
        # zapp=self.z_to_zapp(z)
        # zapp=zapp.view(nr_imgs, self.nr_points_z, -1)
        # # print("zapp has size ", zapp.shape)
        # # print("z3d has size ", z3d.shape)
        # z=torch.cat([zapp, z3d], 2)
        # # DO NOT use the zapp
        # # z=z3d

        # #aggregate all the z from every camera now expressed in world coords, into one z vector
        # # z3d=z3d.mean(0)
        # # z=z.mean(0)
        # z,_=z.max(0)
        # # print("after agregating z3d is ", z3d.shape)
        # TIME_END("rotate")

        # #reduce it so that the hypernetwork makes smaller weights for siren
        # # z=z/10 #IF the image is too noisy we need to reduce the range for this because the smaller, the smaller the siren weight will be


        # # print("z has shape ", z.shape)
        # # z=z.reshape(1,self.z_size)
        # z=z.reshape(1,-1)
        # # print("encoder has size", z.shape )
        # # print("running hypernet")
        # TIME_START("hyper")
        # siren_params=self.hyper_net(z)
        # TIME_END("hyper")
        # # print("finished hypernet")
        # # print("computer params of siren")
        # # print("sirenparams", siren_params)
        # # print("x shape is ", x.shape)
        # # x.unsqueeze_(0)
        # # x=self.siren_net(x, params=siren_params)
        # # x=self.siren_net(x, params=None)


        TIME_START("full_siren")
        #siren has to receive some 3d points as query, The 3d points are located along rays so we can volume render and compare with the image itself
        #most of it is from https://github.com/krrish94/nerf-pytorch/blob/master/tiny_nerf.py
        # print("x has shape", x.shape)
        height=x.shape[2]
        width=x.shape[3]
        fx=gt_K[0,0] ### 
        fy=gt_K[1,1] ### 
        cx=gt_K[0,2] ### 
        cy=gt_K[1,2] ### 
        tform_cam2world =torch.from_numpy( gt_tf_cam_world.inverse().matrix() )
        #vase
        # near_thresh=0.7
        # far_thresh=1.2
        #socrates
        near_thresh=0.9
        far_thresh=1.7
        # depth_samples_per_ray=100
        depth_samples_per_ray=60
        # depth_samples_per_ray=30
        chunksize=512*512
        # chunksize=1024*1024

        # Get the "bundle" of rays through all image pixels.
        TIME_START("ray_bundle")
        ray_origins, ray_directions = get_ray_bundle(
            height, width, fx,fy,cx,cy, tform_cam2world, novel=novel
        )
        # ###DEBUG ray origin
        # rays_o_mesh=Mesh()
        # rays_o_mesh.V=ray_origins.view(-1, 3).numpy()
        # rays_o_mesh.m_vis.m_show_points=True
        # rays_o_mesh.m_vis.m_point_size=20
        # Scene.show(rays_o_mesh, "rays_o_mesh")
        # #debug ray direction
        # rays_points=ray_origins+ray_directions*1.0
        # rays_p_mesh=Mesh()
        # rays_p_mesh.V=rays_points.view(-1, 3).numpy()
        # rays_p_mesh.m_vis.m_show_points=True
        # rays_p_mesh.m_vis.m_point_size=4
        # Scene.show(rays_p_mesh, "rays_p_mesh")
        TIME_END("ray_bundle")

        TIME_START("sample")
        # Sample query points along each ray
        query_points, depth_values = compute_query_points_from_rays(
            ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray, randomize=True
        )

        TIME_END("sample")

        # "Flatten" the query points.
        # print("query points is ", query_points.shape)
        flattened_query_points = query_points.reshape((-1, 3))
        # print("flattened_query_points is ", flattened_query_points.shape)

        # TIME_START("pos_encode")
        # flattened_query_points = positional_encoding(flattened_query_points, num_encoding_functions=self.num_encodings, log_sampling=True)
        flattened_query_points=self.leaned_pe(flattened_query_points.to("cuda") )
        # print("flattened_query_points after pe", flattened_query_points.shape)
        # flattened_query_points=flattened_query_points.view(height,width,depth_samples_per_ray,-1 )
        # print("flatened_query_pointss is ", flatened_query_pointss.shape)
        # TIME_END("pos_encode")

        # batches = get_minibatches(flattened_query_points, chunksize=chunksize)
        # predictions = []
        # nr_batches=0
        # TIME_START("siren_batches")
        # for batch in batches:
        #     # print("batch is ", batch.shape)
        #     nr_batches+=1
        #     # predictions.append( self.siren_net(batch.to("cuda"), params=siren_params) )
        #     # batch=batch-0.4
        #     # batch=((batch+1.92)/2.43)*2.0-1.0
        #     # print("batch has mean min max ", batch.mean(), batch.min(), batch.max() )
        #     # predictions.append( self.siren_net(batch.to("cuda"), params=siren_params ) )
        #     predictions.append( self.siren_net(batch.to("cuda") )-3.0 )
        #     # predictions.append( self.nerf_net(batch.to("cuda"), params=siren_params ) )
        #     # predictions.append( self.nerf_net(batch.to("cuda") ) )
        #     # if not Scene.does_mesh_with_name_exist("rays"):
        #     # if not novel:
        #     #     rays_mesh=Mesh()
        #     #     rays_mesh.V=batch.numpy()
        #     #     rays_mesh.m_vis.m_show_points=True
        #     #     Scene.show(rays_mesh, "rays_mesh")
        #     # if novel:
        #     #     rays_mesh=Mesh()
        #     #     rays_mesh.V=batch.numpy()
        #     #     rays_mesh.m_vis.m_show_points=True
        #     #     Scene.show(rays_mesh, "rays_mesh_novel")
        # print(" nr batch ", nr_batches, " / ", len(batches))
        # # print("got nr_batches ", nr_batches)
        # TIME_END("siren_batches")
        # radiance_field_flattened = torch.cat(predictions, dim=0)

        # if not novel:
        #     rays_mesh=Mesh()
        #     rays_mesh.V=query_points.reshape((-1, 3)).numpy()
        #     rays_mesh.m_vis.m_show_points=True
        #     Scene.show(rays_mesh, "rays_mesh_novel")

        # radiance_field_flattened = self.siren_net(query_points.to("cuda") )-3.0 
        # radiance_field_flattened = self.siren_net(query_points.to("cuda") )
        # flattened_query_points/=2.43
        # radiance_field_flattened = self.siren_net(flattened_query_points.to("cuda") )
        # radiance_field_flattened = self.siren_net(query_points.to("cuda"), params=siren_params )
        radiance_field_flattened = self.nerf_net(flattened_query_points.to("cuda") ) 
        # radiance_field_flattened = self.nerf_net(flattened_query_points.to("cuda"), params=siren_params ) 
        radiance_field_flattened=radiance_field_flattened.view(-1,self.siren_out_channels)


        # "Unflatten" to obtain the radiance field.
        unflattened_shape = list(query_points.shape[:-1]) + [self.siren_out_channels]
        radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_predicted, depth_map, acc_map = render_volume_density(
        # rgb_predicted, depth_map, acc_map = render_volume_density_nerfplusplus(
            radiance_field, ray_origins.to("cuda"), depth_values.to("cuda"), self.siren_out_channels
        )

        # print("rgb predicted has shpae ", rgb_predicted.shape)
        # rgb_predicted=rgb_predicted.view(1,3,height,width)
        rgb_predicted=rgb_predicted.permute(2,0,1).unsqueeze(0).contiguous()
        # print("depth map size is ", depth_map.shape)
        depth_map=depth_map.unsqueeze(0).unsqueeze(0).contiguous()
        # depth_map_mat=tensor2mat(depth_map)
        TIME_END("full_siren")

        # print("rgb_predicted is ", rgb_predicted.shape)


        #when rgb predicted is actually just a bunch of features like 64 features per pixel then we regress from them the rgb
        #GUIDE cannot be the rgb image because when we render from a novel view we don't have the gt for that one
        # guide=rgb_predicted
        # print("rgb_predicted out is ", rgb_predicted.shape)
        # print("guide out is ", guide.shape)

        # for i in range(self.nr_rgb_pred+2):
        #     rgb_predicted=self.rgb_pred[i](rgb_predicted )
        # print("rgb_predicted out is ", rgb_predicted.shape)



        #NEW LOSSES
        new_loss=0
        #make the opacity be low all around
        sigma_a = torch.sigmoid(radiance_field[..., 3])
        loss_sigma_a=sigma_a.mean()
        new_loss+=loss_sigma_a




        return rgb_predicted, depth_map, acc_map, new_loss






        return x



class PCA(Function):
    # @staticmethod
    def forward(ctx, sv): #sv corresponds to the slices values, it has dimensions nr_positions x val_full_dim

        # http://agnesmustar.com/2017/11/01/principal-component-analysis-pca-implemented-pytorch/


        X=sv.detach().cpu()#we switch to cpu because svd for gpu needs magma: No CUDA implementation of 'gesdd'. Install MAGMA and rebuild cutorch (http://icl.cs.utk.edu/magma/) at /opt/pytorch/aten/src/THC/generic/THCTensorMathMagma.cu:191
        k=3
        print("x is ", X.shape)
        X_mean = torch.mean(X,0)
        print("x_mean is ", X_mean.shape)
        X = X - X_mean.expand_as(X)

        U,S,V = torch.svd(torch.t(X)) 
        C = torch.mm(X,U[:,:k])
        print("C has shape ", C.shape)
        print("C min and max is ", C.min(), " ", C.max() )
        C-=C.min()
        C/=C.max()
        print("after normalization C min and max is ", C.min(), " ", C.max() )

        return C

#https://github.com/pytorch/pytorch/issues/2001
def summary(self,file=sys.stderr):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if p is not None:
                total_params += reduce(lambda x, y: x * y, p.shape)
                # if(p.grad==None):
                #     print("p has no grad", name)
                # else:
                #     print("p has gradnorm ", name ,p.grad.norm() )

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stderr:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
            for name, p in model._parameters.items():
                if hasattr(p, 'grad'):
                    if(p.grad==None):
                        # print("p has no grad", name)
                        main_str+="p no grad"
                    else:
                        # print("p has gradnorm ", name ,p.grad.norm() )
                        main_str+= "\n" + name + " p has grad norm " + str(p.grad.norm())
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(self)
    if file is not None:
        print(string, file=file)
    return count