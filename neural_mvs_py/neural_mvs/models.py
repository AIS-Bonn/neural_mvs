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

#latticenet 
from latticenet_py.lattice.lattice_wrapper import LatticeWrapper
from latticenet_py.lattice.diceloss import GeneralizedSoftDiceLoss
from latticenet_py.lattice.lovasz_loss import LovaszSoftmax
from latticenet_py.lattice.models import LNN

from latticenet_py.callbacks.callback import *
from latticenet_py.callbacks.viewer_callback import *
from latticenet_py.callbacks.visdom_callback import *
from latticenet_py.callbacks.state_callback import *
from latticenet_py.callbacks.phase import *
from latticenet_py.lattice.lattice_modules import *


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

        # #layers
        # resnet = torchvision.models.resnet18(pretrained=True)
        # modules=list(resnet.children())[:-1]
        # self.resnet=nn.Sequential(*modules)
        # for p in self.resnet.parameters():
        #     p.requires_grad = True







        self.start_nr_channels=64
        # self.start_nr_channels=4
        self.nr_downsampling_stages=4
        self.nr_blocks_down_stage=[2,2,2,2,2,2,2]
        # self.nr_channels_after_coarsening_per_layer=[64,64,128,128,256,256,512,512,512,1024,1024]
        self.nr_channels_after_coarsening_per_layer=[128,128,256,512,512]
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
                self.blocks_down_per_stage_list[i].append( ResnetBlock2D(cur_nr_channels, 3, 1, 1, dilations=[1,1], biases=[True,True], with_dropout=False) )
            # nr_channels_after_coarsening=int(cur_nr_channels*2)
            nr_channels_after_coarsening=self.nr_channels_after_coarsening_per_layer[i]
            print("nr_channels_after_coarsening is ", nr_channels_after_coarsening)
            # self.coarsens_list.append( ConvGnRelu(nr_channels_after_coarsening, kernel_size=2, stride=2, padding=0, dilation=1, bias=False, with_dropout=False, transposed=False).cuda() )
            cur_nr_channels+=2 #because we concat the coords
            self.coarsens_list.append( BlockForResnet(cur_nr_channels, nr_channels_after_coarsening, kernel_size=3, stride=2, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False ).cuda() )
            # self.coarsens_list.append( BlockPAC(cur_nr_channels, nr_channels_after_coarsening, kernel_size=3, stride=2, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False ).cuda() )
            cur_nr_channels=nr_channels_after_coarsening
            # cur_nr_channels+=2 #because we concat the coords





        # self.z_to_3d=None
        self.to_z=None


    def forward(self, x):
        # for p in self.resnet.parameters():
        #     p.requires_grad = False

        # print("encoder x input is ", x.min(), " ", x.max())
        # z=self.resnet(x) # z has size 1x512x1x1

        guide=x
        # first conv
        x = self.concat_coord(x)
        x = self.first_conv(x)
        x=self.relu(x)

        #encode 
        # TIME_START("down_path")
        for i in range(self.nr_downsampling_stages):
            # print("downsampling stage ", i)
            # print("DOWNSAPLE ", i, " with x of shape ", x.shape)
            #resnet blocks
            for j in range(self.nr_blocks_down_stage[i]):
                x = self.concat_coord(x)
                x = self.blocks_down_per_stage_list[i][j] (x, x) 

            #now we do a downsample
            x = self.concat_coord(x)
            x = self.coarsens_list[i] ( x )
            # print("x has shape ", x.shape)
        # TIME_END("down_path")
        z=x
        # print("z after encoding has shape ", z.shape)



        return z

class EncoderLNN(torch.nn.Module):
    def __init__(self, z_size):
        super(EncoderLNN, self).__init__()

        #a bit more control
        self.nr_downsamples=4
        self.nr_blocks_down_stage=[2,2,2,2]
        self.pointnet_layers=[16,32,64]
        self.start_nr_filters=32
        experiment="none"
        #check that the experiment has a valid string
        valid_experiment=["none", "slice_no_deform", "pointnet_no_elevate", "pointnet_no_local_mean", "pointnet_no_elevate_no_local_mean", "splat", "attention_pool"]
        if experiment not in valid_experiment:
            err = "Experiment " + experiment + " is not valid"
            sys.exit(err)





        self.distribute=DistributeLatticeModule(experiment) 
        print("pointnet layers is ", self.pointnet_layers)
        self.point_net=PointNetModule( self.pointnet_layers, self.start_nr_filters, experiment)  




        #####################
        # Downsampling path #
        #####################
        self.resnet_blocks_per_down_lvl_list=torch.nn.ModuleList([])
        self.coarsens_list=torch.nn.ModuleList([])
        corsenings_channel_counts = []
        cur_channels_count=self.start_nr_filters
        for i in range(self.nr_downsamples):
            
            #create the resnet blocks
            self.resnet_blocks_per_down_lvl_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_down_stage[i]):
                self.resnet_blocks_per_down_lvl_list[i].append( ResnetBlock(cur_channels_count, [1,1], [False,False], False) )
            nr_channels_after_coarsening=int(cur_channels_count*2)
            # print("adding bnReluCorsen which outputs nr of channels ", nr_channels_after_coarsening )
            self.coarsens_list.append( GnReluCoarsen(nr_channels_after_coarsening)) #is still the best one because it can easily learn the versions of Avg and Blur. and the Max version is the worse for some reason
            cur_channels_count=nr_channels_after_coarsening

      

    def forward(self, cloud):

        positions=torch.from_numpy(cloud.V.copy() ).float().to("cuda")
        values=torch.from_numpy(cloud.C.copy() ).float().to("cuda")
        values=torch.cat( [positions,values],1 )


        ls=LatticePy()
        ls.create("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/config/train.cfg", "splated_lattice")

        with torch.set_grad_enabled(False):
            distributed, indices=self.distribute(ls, positions, values)

        lv, ls=self.point_net(ls, distributed, indices)

        # print("lv at the beggining is ", lv.shape)

        
        for i in range(self.nr_downsamples):

            #resnet blocks
            for j in range(self.nr_blocks_down_stage[i]):
                lv, ls = self.resnet_blocks_per_down_lvl_list[i][j] ( lv, ls) 
            #now we do a downsample
            lv, ls = self.coarsens_list[i] ( lv, ls)

        #from a lv of Nx256 we want a vector of just 256
        # print("lv at the final is ", lv.shape)

        z=lv.mean(dim=0)


        return z


class SpatialLNN(torch.nn.Module):
    def __init__(self ):
        super(SpatialLNN, self).__init__()

        #a bit more control
        self.pointnet_layers=[16,32,64]
        self.start_nr_filters=64
        experiment="none"
        #check that the experiment has a valid string
        valid_experiment=["none", "slice_no_deform", "pointnet_no_elevate", "pointnet_no_local_mean", "pointnet_no_elevate_no_local_mean", "splat", "attention_pool"]
        if experiment not in valid_experiment:
            err = "Experiment " + experiment + " is not valid"
            sys.exit(err)




        self.distribute=DistributeLatticeModule(experiment) 
        print("pointnet layers is ", self.pointnet_layers)
        self.point_net=PointNetModule( self.pointnet_layers, self.start_nr_filters, experiment)  




        #####################
        # Downsampling path #
        #####################
        self.resnets=torch.nn.ModuleList([])
        for i in range(8):
            self.resnets.append( ResnetBlock(self.start_nr_filters, [1,1], [False,False], False) )
         
      

    def forward(self, cloud):

        positions=torch.from_numpy(cloud.V.copy() ).float().to("cuda")
        values=torch.from_numpy(cloud.C.copy() ).float().to("cuda")
        values=torch.cat( [positions,values],1 )


        ls=Lattice.create("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/config/train.cfg", "splated_lattice")

        with torch.set_grad_enabled(False):
            distributed, indices, weights=self.distribute(ls, positions, values)

        lv, ls=self.point_net(ls, distributed, indices)

        
        for i in range(len(self.resnets)):
            lv, ls = self.resnets[i]( lv, ls) 



        return lv, ls




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
        # nr_points=x.shape[0]
        # x=x.view(nr_points,3,1,1)

        # X comes as nr_points x 3 matrix but we want adyacen rays to be next to each other. So we put the nr of the ray in the batch
        # x=x.contiguous()
        # x=x.view(71,107,30,3)
        # x=x.view(1,-1,71,107,30)
        # print("x is ", x.shape)
        x=x.permute(2,3,0,1).contiguous() #from 71,107,30,3 to 30,3,71,107
        # print("x is ", x.shape)


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
        # x=x.view(nr_points,-1,1,1)
       
        return x


#this siren net receives directly the coordinates, no need to make a concat coord or whatever
#Has as first layer a learned PE
class SirenNetworkDirectPE(MetaModule):
    def __init__(self, in_channels, out_channels):
        super(SirenNetworkDirectPE, self).__init__()

        self.first_time=True

        self.nr_layers=4
        self.out_channels_per_layer=[128, 128, 128, 128, 128, 128]
        # self.out_channels_per_layer=[256, 128, 128, 128, 128, 128]
        # self.out_channels_per_layer=[100, 100, 100, 100, 100]
        # self.out_channels_per_layer=[256, 256, 256, 256, 256]
        # self.out_channels_per_layer=[256, 128, 64, 32, 16]
        # self.out_channels_per_layer=[256, 256, 256, 256, 256]
        # self.out_channels_per_layer=[256, 64, 64, 64, 64, 64]
        # self.out_channels_per_layer=[256, 32, 32, 32, 32, 32]

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
    
        # self.position_embedders=torch.nn.ModuleList([])

        num_encodings=11
        # self.learned_pe=LearnedPE(in_channels=in_channels, num_encoding_functions=num_encodings, logsampling=True)
        # cur_nr_channels=in_channels + in_channels*num_encodings*2
        #new leaned pe with gaussian random weights as in  Fourier Features Let Networks Learn High Frequency 
        self.learned_pe=LearnedPEGaussian(in_channels=in_channels, out_channels=256, std=5)
        cur_nr_channels=256+in_channels
        #combined PE  and gaussian
        # self.learned_pe=LearnedPEGaussian2(in_channels=in_channels, out_channels=256, std=5, num_encoding_functions=num_encodings, logsampling=True)
        # cur_nr_channels=256+in_channels +    in_channels + in_channels*num_encodings*2
        learned_pe_channels=cur_nr_channels
        self.skip_pe_point=999

        # self.position_embedder=( MetaSequential( 
        #     Block(activ=torch.relu, in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     Block(activ=torch.relu, in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     # Block(activ=torch.relu, in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     # Block(activ=torch.relu, in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     Block(activ=None, in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     ) )
        # # cur_nr_channels=128+3

        #we add also the point features 
        cur_nr_channels+=64

        for i in range(self.nr_layers):
            is_first_layer=i==0
            self.net.append( MetaSequential( BlockSiren(activ=torch.relu, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=is_first_layer).cuda() ) )
            # self.net.append( MetaSequential( ResnetBlock(activ=torch.sin, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=False, is_first_layer=False).cuda() ) )

            # self.position_embedders.append( MetaSequential( 
            #     Block(activ=torch.sigmoid, in_channels=in_channels, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            #     Block(activ=None, in_channels=self.out_channels_per_layer[i], out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            #     ) )

            # # if i<self.nr_layers-4:
            # if i!=self.nr_layers:
            # # if i==0:
            #     # cur_nr_channels=self.out_channels_per_layer[i]+ in_channels*10
            #     # cur_nr_channels=self.out_channels_per_layer[i]+ in_channels*30 #when repeating the raw coordinates a bit
            #     # cur_nr_channels=self.out_channels_per_layer[i]+ self.out_channels_per_layer[i] #when using a positional embedder for the raw coords
            #     # cur_nr_channels=self.out_channels_per_layer[i]+ 128+3 #when using a positional embedder for the raw coords
            # else:
            #     cur_nr_channels=self.out_channels_per_layer[i]
            # if i!=0:
                # cur_nr_channels+=self.out_channels_per_layer[0]

            #at some point concat back the learned pe
            if i==self.skip_pe_point:
                cur_nr_channels=self.out_channels_per_layer[i]+learned_pe_channels
            else:
                cur_nr_channels=self.out_channels_per_layer[i]
        # self.net.append( MetaSequential(Block(activ=torch.sigmoid, in_channels=cur_nr_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda()  ))
        # self.net.append( MetaSequential(BlockSiren(activ=None, in_channels=cur_nr_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda()  ))

        # self.net = MetaSequential(*self.net)

        self.pred_sigma_and_feat=MetaSequential(
            # BlockSiren(activ=torch.relu, in_channels=cur_nr_channels, out_channels=cur_nr_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            BlockSiren(activ=None, in_channels=cur_nr_channels, out_channels=cur_nr_channels+1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            )
        num_encoding_directions=4
        self.learned_pe_dirs=LearnedPE(in_channels=3, num_encoding_functions=num_encoding_directions, logsampling=True)
        dirs_channels=3+ 3*num_encoding_directions*2
        # new leaned pe with gaussian random weights as in  Fourier Features Let Networks Learn High Frequency 
        # self.learned_pe_dirs=LearnedPEGaussian(in_channels=in_channels, out_channels=64, std=10)
        # dirs_channels=64
        cur_nr_channels=cur_nr_channels+dirs_channels
        self.pred_rgb=MetaSequential( 
            BlockSiren(activ=torch.relu, in_channels=cur_nr_channels, out_channels=cur_nr_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            BlockSiren(activ=None, in_channels=cur_nr_channels, out_channels=3, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda()    
            )



    def forward(self, x, ray_directions, point_features, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())


        if len(x.shape)!=4:
            print("SirenDirectPE forward: x should be a H,W,nr_points,3 matrix so 4 dimensions but it actually has ", x.shape, " so the lenght is ", len(x.shape))
            exit(1)

        height=x.shape[0]
        width=x.shape[1]
        nr_points=x.shape[2]

        #from 71,107,30,3  to Nx3
        x=x.view(-1,3)
        x=self.learned_pe(x, params=get_subdict(params, 'learned_pe') )
        x=x.view(height, width, nr_points, -1 )

        #also make the direcitons into image 
        ray_directions=ray_directions.view(-1,3)
        # print("ray_directions is ", ray_directions.shape )
        ray_directions=F.normalize(ray_directions, p=2, dim=1)
        ray_directions=self.learned_pe_dirs(ray_directions, params=get_subdict(params, 'learned_pe_dirs'))
        ray_directions=ray_directions.view(height, width, -1)
        ray_directions=ray_directions.permute(2,0,1).unsqueeze(0)
        # print("ray_directions is ", ray_directions.shape )


        #append to x the point_features
        # print("before the x has shape", x.shape)
        point_features=point_features.view(height, width, nr_points, -1 )
        x=torch.cat([x,point_features],3)
        # print("after the x has shape", x.shape)


        #the x in this case is Nx3 but in order to make it run fine with the 1x1 conv we make it a Nx3x1x1
        # nr_points=x.shape[0]
        # x=x.view(nr_points,3,1,1)

        # X comes as nr_points x 3 matrix but we want adyacen rays to be next to each other. So we put the nr of the ray in the batch
        # x=x.contiguous()
        # x=x.view(71,107,30,3)
        # x=x.view(1,-1,71,107,30)
        # print("x is ", x.shape)
        x=x.permute(2,3,0,1).contiguous() #from 71,107,30,3 to 30,3,71,107
        # print("x is ", x.shape)

        learned_pe_out=x


        # print ("running siren")
        # x=self.net(x, params=get_subdict(params, 'net'))

        x_raw_coords=x
        x_first_layer=None
        # position_input=self.position_embedder(x_raw_coords)
        # position_input=torch.cat([position_input,x_raw_coords],1)

        for i in range(len(self.net)):
            # print("x si ", x.shape)
            # print("params is ", params)
            # x=self.net[i](x, params=get_subdict(params, 'net['+str(i)+"]"  )  )
            x=self.net[i](x, params=get_subdict(params, 'net.'+str(i)  )  )
            # if i==0:
            #     x_first_layer=x

            if i==self.skip_pe_point:
                x=torch.cat([learned_pe_out,x],1)


            # print("x has shape ", x.shape, " x_raw si ", x_raw_coords.shape)
            
            # if i<len(self.net)-5: #if it's any layer except the last one
            # if i!=len(self.net)-1: #if it's any layer except the last one
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
                # x=torch.cat([position_input,x],1)
            # if i!=0 and i!=len(self.net)-1:
                # x=torch.cat([x_first_layer,x],1)
                # x=x+position_input
        # print("finished siren")

        #predict the sigma and a feature vector for the rest of things
        sigma_and_feat=self.pred_sigma_and_feat(x,  params=get_subdict(params, 'pred_sigma_and_feat'))
        #get the feature vector for the rest of things and concat it with the direction
        sigma_a=torch.relu( sigma_and_feat[:,0:1, :, :] ) #first channel is the sigma
        feat=torch.relu( sigma_and_feat[:,1:sigma_and_feat.shape[1], :, : ] )
        # print("sigma and feat is ", sigma_and_feat.shape)
        # print(" feat is ", feat.shape)
        ray_directions=ray_directions.repeat(feat.shape[0],1,1,1) #repeat as many times as samples that you have in a ray
        feat_and_dirs=torch.cat([feat, ray_directions], 1)
        #predict rgb
        # rgb=torch.sigmoid(  (self.pred_rgb(feat_and_dirs,  params=get_subdict(params, 'pred_rgb') ) +1.0)*0.5 )
        rgb=torch.sigmoid(  self.pred_rgb(feat_and_dirs,  params=get_subdict(params, 'pred_rgb') )  )
        #concat 
        # print("rgb is", rgb.shape)
        # print("sigma_a is", sigma_a.shape)
        x=torch.cat([rgb, sigma_a],1)

        x=x.permute(2,3,0,1).contiguous() #from 30,nr_out_channels,71,107 to  71,107,30,4
        # x=x.view(nr_points,-1,1,1)

        # rgb = torch.sigmoid( x[:,:,:, 0:3] )
        # sigma_a = torch.relu( x[:,:,:, 3:4] )

        # x=torch.cat([rgb,sigma_a],3)
        
       
        return x


#this siren net receives directly the coordinates, no need to make a concat coord or whatever
#Has as first layer a learned PE It's a bit of a trimmed version
class SirenNetworkDirectPETrim(MetaModule):
    def __init__(self, in_channels, out_channels):
        super(SirenNetworkDirectPETrim, self).__init__()

        self.first_time=True

        self.nr_layers=4
        self.out_channels_per_layer=[128, 128, 128, 128, 128, 128]
        # self.out_channels_per_layer=[100, 100, 100, 100, 100]
        # self.out_channels_per_layer=[256, 256, 256, 256, 256]
        # self.out_channels_per_layer=[256, 128, 64, 32, 16]
        # self.out_channels_per_layer=[256, 256, 256, 256, 256]
        # self.out_channels_per_layer=[256, 64, 64, 64, 64, 64]
        # self.out_channels_per_layer=[256, 32, 32, 32, 32, 32]

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
    
        # self.position_embedders=torch.nn.ModuleList([])

        num_encodings=11
        # self.learned_pe=LearnedPE(in_channels=in_channels, num_encoding_functions=num_encodings, logsampling=True)
        # cur_nr_channels=in_channels + in_channels*num_encodings*2
        #new leaned pe with gaussian random weights as in  Fourier Features Let Networks Learn High Frequency 
        self.learned_pe=LearnedPEGaussian(in_channels=in_channels, out_channels=256, std=5)
        cur_nr_channels=256+in_channels
        #combined PE  and gaussian
        # self.learned_pe=LearnedPEGaussian2(in_channels=in_channels, out_channels=256, std=5, num_encoding_functions=num_encodings, logsampling=True)
        # cur_nr_channels=256+in_channels +    in_channels + in_channels*num_encodings*2
        learned_pe_channels=cur_nr_channels
        # self.skip_pe_point=2

        # self.position_embedder=( MetaSequential( 
        #     Block(activ=torch.relu, in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     Block(activ=torch.relu, in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     # Block(activ=torch.relu, in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     # Block(activ=torch.relu, in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     Block(activ=None, in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     ) )
        # # cur_nr_channels=128+3

        for i in range(self.nr_layers):
            is_first_layer=i==0
            self.net.append( MetaSequential( BlockSiren(activ=torch.relu, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=is_first_layer).cuda() ) )
            # self.net.append( MetaSequential( ResnetBlock(activ=torch.sin, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=False, is_first_layer=False).cuda() ) )

            # self.position_embedders.append( MetaSequential( 
            #     Block(activ=torch.sigmoid, in_channels=in_channels, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            #     Block(activ=None, in_channels=self.out_channels_per_layer[i], out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            #     ) )

            # # if i<self.nr_layers-4:
            # if i!=self.nr_layers:
            # # if i==0:
            #     # cur_nr_channels=self.out_channels_per_layer[i]+ in_channels*10
            #     # cur_nr_channels=self.out_channels_per_layer[i]+ in_channels*30 #when repeating the raw coordinates a bit
            #     # cur_nr_channels=self.out_channels_per_layer[i]+ self.out_channels_per_layer[i] #when using a positional embedder for the raw coords
            #     # cur_nr_channels=self.out_channels_per_layer[i]+ 128+3 #when using a positional embedder for the raw coords
            # else:
            #     cur_nr_channels=self.out_channels_per_layer[i]
            # if i!=0:
                # cur_nr_channels+=self.out_channels_per_layer[0]

            #at some point concat back the learned pe
            # if i==self.skip_pe_point:
            #     cur_nr_channels=self.out_channels_per_layer[i]+learned_pe_channels
            # else:
            cur_nr_channels=self.out_channels_per_layer[i]
        # self.net.append( MetaSequential(Block(activ=torch.sigmoid, in_channels=cur_nr_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda()  ))
        self.net.append( MetaSequential(BlockSiren(activ=None, in_channels=cur_nr_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda()  ))

        # self.net = MetaSequential(*self.net)

        # self.pred_sigma_and_feat=MetaSequential(
        #     # BlockSiren(activ=torch.relu, in_channels=cur_nr_channels, out_channels=cur_nr_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     BlockSiren(activ=None, in_channels=cur_nr_channels, out_channels=cur_nr_channels+1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     )
        # num_encoding_directions=4
        # self.learned_pe_dirs=LearnedPE(in_channels=3, num_encoding_functions=num_encoding_directions, logsampling=True)
        # dirs_channels=3+ 3*num_encoding_directions*2
        # # new leaned pe with gaussian random weights as in  Fourier Features Let Networks Learn High Frequency 
        # # self.learned_pe_dirs=LearnedPEGaussian(in_channels=in_channels, out_channels=64, std=10)
        # # dirs_channels=64
        # cur_nr_channels=cur_nr_channels+dirs_channels
        # self.pred_rgb=MetaSequential( 
        #     BlockSiren(activ=torch.sin, in_channels=cur_nr_channels, out_channels=cur_nr_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     BlockSiren(activ=None, in_channels=cur_nr_channels, out_channels=3, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda()    
        #     )



    def forward(self, x, ray_directions, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())


        if len(x.shape)!=4:
            print("SirenDirectPE forward: x should be a H,W,nr_points,3 matrix so 4 dimensions but it actually has ", x.shape, " so the lenght is ", len(x.shape))
            exit(1)

        height=x.shape[0]
        width=x.shape[1]
        nr_points=x.shape[2]

        #from 71,107,30,3  to Nx3
        x=x.view(-1,3)
        x=self.learned_pe(x, params=get_subdict(params, 'learned_pe') )
        x=x.view(height, width, nr_points, -1 )

        # #also make the direcitons into image 
        # ray_directions=ray_directions.view(-1,3)
        # # print("ray_directions is ", ray_directions.shape )
        # ray_directions=F.normalize(ray_directions, p=2, dim=1)
        # ray_directions=self.learned_pe_dirs(ray_directions, params=get_subdict(params, 'learned_pe_dirs'))
        # ray_directions=ray_directions.view(height, width, -1)
        # ray_directions=ray_directions.permute(2,0,1).unsqueeze(0)
        # # print("ray_directions is ", ray_directions.shape )


        #the x in this case is Nx3 but in order to make it run fine with the 1x1 conv we make it a Nx3x1x1
        # nr_points=x.shape[0]
        # x=x.view(nr_points,3,1,1)

        # X comes as nr_points x 3 matrix but we want adyacen rays to be next to each other. So we put the nr of the ray in the batch
        # x=x.contiguous()
        # x=x.view(71,107,30,3)
        # x=x.view(1,-1,71,107,30)
        # print("x is ", x.shape)
        x=x.permute(2,3,0,1).contiguous() #from 71,107,30,3 to 30,3,71,107
        # print("x is ", x.shape)

        learned_pe_out=x


        # print ("running siren")
        # x=self.net(x, params=get_subdict(params, 'net'))

        x_raw_coords=x
        x_first_layer=None
        # position_input=self.position_embedder(x_raw_coords)
        # position_input=torch.cat([position_input,x_raw_coords],1)

        for i in range(len(self.net)):
            # print("x si ", x.shape)
            # print("params is ", params)
            # x=self.net[i](x, params=get_subdict(params, 'net['+str(i)+"]"  )  )
            x=self.net[i](x, params=get_subdict(params, 'net.'+str(i)  )  )
            # if i==0:
            #     x_first_layer=x

            # if i==self.skip_pe_point:
            #     x=torch.cat([learned_pe_out,x],1)


            # print("x has shape ", x.shape, " x_raw si ", x_raw_coords.shape)
            
            # if i<len(self.net)-5: #if it's any layer except the last one
            # if i!=len(self.net)-1: #if it's any layer except the last one
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
                # x=torch.cat([position_input,x],1)
            # if i!=0 and i!=len(self.net)-1:
                # x=torch.cat([x_first_layer,x],1)
                # x=x+position_input
        # print("finished siren")

        # #predict the sigma and a feature vector for the rest of things
        # sigma_and_feat=self.pred_sigma_and_feat(x,  params=get_subdict(params, 'pred_sigma_and_feat'))
        # #get the feature vector for the rest of things and concat it with the direction
        # sigma_a=torch.relu( sigma_and_feat[:,0:1, :, :] ) #first channel is the sigma
        # feat=torch.sin( sigma_and_feat[:,1:sigma_and_feat.shape[1], :, : ] )
        # # print("sigma and feat is ", sigma_and_feat.shape)
        # # print(" feat is ", feat.shape)
        # ray_directions=ray_directions.repeat(feat.shape[0],1,1,1) #repeat as many times as samples that you have in a ray
        # feat_and_dirs=torch.cat([feat, ray_directions], 1)
        # #predict rgb
        # rgb=torch.sigmoid( self.pred_rgb(feat_and_dirs,  params=get_subdict(params, 'pred_rgb') ) )
        # #concat 
        # # print("rgb is", rgb.shape)
        # # print("sigma_a is", sigma_a.shape)
        # x=torch.cat([rgb, sigma_a],1)

        # x=x.permute(2,3,0,1).contiguous() #from 30,nr_out_channels,71,107 to  71,107,30,4
        # # x=x.view(nr_points,-1,1,1)

        rgb = torch.sigmoid( x[:,0:3,:, :] )
        sigma_a = torch.relu( x[:,3:4,:, :] )
        x=torch.cat([rgb, sigma_a],1)
        x=x.permute(2,3,0,1).contiguous() #from 30,nr_out_channels,71,107 to  71,107,30,4
        # # x=x.view(nr_points,-1,1,1)

        # x=torch.cat([rgb,sigma_a],3)
        
       
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


class DecoderTo2D(torch.nn.Module):
    def __init__(self, z_size, out_channels):
        super(DecoderTo2D, self).__init__()

        self.z_size=z_size
        self.out_channels=out_channels
        
        # self.nr_upsamples=8 # 256
        self.nr_upsamples=5 # 32
        self.channels_per_stage=[z_size, int(z_size/2), int(z_size/4), int(z_size/4), int(z_size/8), int(z_size/16), int(z_size/16), int(z_size/32), int(z_size/32), int(z_size/32)    ]
        self.upsample_list=torch.nn.ModuleList([])
        self.conv_list=torch.nn.ModuleList([])
        cur_nr_channels=z_size
        for i in range(self.nr_upsamples):
            # self.upsample_list.append(  torch.nn.ConvTranspose2d(in_channels=cur_nr_channels, out_channels=int(cur_nr_channels/2), kernel_size=2, stride=2)  )
            # cur_nr_channels=int(cur_nr_channels/2)
            self.upsample_list.append(  torch.nn.ConvTranspose2d(in_channels=cur_nr_channels, out_channels=self.channels_per_stage[i], kernel_size=2, stride=2)  )
            cur_nr_channels=self.channels_per_stage[i]
            if cur_nr_channels<4:
                print("the cur nr of channels is too low to compute a good output with the final conv, we might want to have at least 8 feature maps")
                exit(1)
            self.conv_list.append(  ResnetBlock2D( out_channels=cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True,True], with_dropout=False, do_norm=True   ) )

        self.final_conv= BlockPAC(in_channels=cur_nr_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, activ=None, is_first_layer=False )
      

    def forward(self, z):

        x=z.view(1,self.z_size,1,1)

        for i in range(self.nr_upsamples):
            x=self.upsample_list[i](x)
            x=self.conv_list[i](x,x)
            # print("x for stage ", i, " has shape ", x.shape )

        # print("final x after decoding is ", x.shape)
        x=self.final_conv(x,x)

        # print("final x after decoding is ", x.shape)

        return x





class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.first_time=True

        #params
        self.z_size=512
        # self.z_size=128
        # self.z_size=256
        # self.z_size=2048
        self.nr_points_z=128
        self.num_encodings=10
        # self.siren_out_channels=64
        # self.siren_out_channels=32
        self.siren_out_channels=4

        #activ
        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()


        # self.encoder=Encoder2D(self.z_size)
        # self.encoder=Encoder(self.z_size)
        self.encoder=EncoderLNN(self.z_size)
        self.decoder=DecoderTo2D(self.z_size, 6)

        self.spatial_lnn=SpatialLNN()
        self.slice=SliceLatticeModule()

        # self.z_size+=3 #because we add the direcitons
        # self.siren_net = SirenNetwork(in_channels=3, out_channels=4)
        # self.siren_net = SirenNetworkDirect(in_channels=3, out_channels=4)
        # self.siren_net = SirenNetworkDirect(in_channels=3, out_channels=3)
        # self.siren_net = SirenNetworkDirect(in_channels=3+3*self.num_encodings*2, out_channels=self.siren_out_channels)
        self.siren_net = SirenNetworkDirectPE(in_channels=3, out_channels=self.siren_out_channels)
        # self.siren_net = SirenNetworkDirectPETrim(in_channels=3, out_channels=self.siren_out_channels)
        # self.siren_net = SirenNetworkDense(in_channels=3+3*self.num_encodings*2, out_channels=4)
        # self.nerf_net = NerfDirect(in_channels=3+3*self.num_encodings*2, out_channels=4)
        # self.hyper_net = HyperNetwork(hyper_in_features=self.nr_points_z*3*2, hyper_hidden_layers=1, hyper_hidden_features=512, hypo_module=self.siren_net)
        # self.hyper_net = HyperNetworkPrincipledInitialization(hyper_in_features=self.nr_points_z*3*2, hyper_hidden_layers=1, hyper_hidden_features=512, hypo_module=self.siren_net)
        self.hyper_net = HyperNetworkPrincipledInitialization(hyper_in_features=self.z_size, hyper_hidden_layers=1, hyper_hidden_features=512, hypo_module=self.siren_net)
        # self.hyper_net = HyperNetwork(hyper_in_features=3468, hyper_hidden_layers=3, hyper_hidden_features=512, hypo_module=self.siren_net)
        # self.hyper_net = HyperNetworkIncremental(hyper_in_features=self.nr_points_z*3*2, hyper_hidden_layers=1, hyper_hidden_features=512, hypo_module=self.siren_net)
        # self.hyper_net = HyperNetwork(hyper_in_features=self.nr_points_z*3*2, hyper_hidden_layers=1, hyper_hidden_features=512, hypo_module=self.nerf_net)


        # self.z_to_z3d = torch.nn.Sequential(
        #     # torch.nn.Linear( self.z_size , self.z_size).to("cuda"),
        #     # torch.nn.ReLU(),
        #     # torch.nn.Linear( self.z_size , self.nr_points_z*3).to("cuda")
        #     BlockLinear(  in_channels=self.z_size, out_channels=self.z_size,  bias=True,  activ=torch.relu ),
        #     BlockLinear(  in_channels=self.z_size, out_channels= self.nr_points_z*3,  bias=True,  activ=None )
        # )

        # self.z_to_zapp = torch.nn.Sequential(
        #     # torch.nn.Linear( self.z_size , self.z_size).to("cuda"),
        #     # torch.nn.ReLU(),
        #     # torch.nn.Linear( self.z_size , self.nr_points_z*3).to("cuda")
        #     BlockLinear(  in_channels=self.z_size, out_channels=self.z_size,  bias=True,  activ=torch.relu ),
        #     BlockLinear(  in_channels=self.z_size, out_channels= self.nr_points_z*3,  bias=True,  activ=None )
        # )

        cam_params=9 + 3 + 3+ 1
        self.cam_embedder = torch.nn.Sequential(
            BlockLinear(  in_channels=cam_params, out_channels=64,  bias=True,  activ=torch.relu ),
            # BlockLinear(  in_channels=64, out_channels=64,  bias=True,  activ=torch.relu ),
            BlockLinear(  in_channels=64, out_channels=64,  bias=True,  activ=None )
        )
        self.z_with_cam_embedder = torch.nn.Sequential(
            BlockLinear(  in_channels=self.z_size+64, out_channels=self.z_size,  bias=True,  activ=torch.relu ),
            # BlockLinear(  in_channels=self.z_size, out_channels=self.z_size,  bias=True,  activ=torch.relu ),
            BlockLinear(  in_channels=self.z_size, out_channels=self.z_size,  bias=True,  activ=None )
        )
        # self.z_scene_embedder = torch.nn.Sequential(
        #     BlockLinear(  in_channels=self.z_size*6, out_channels=self.z_size*3,  bias=True,  activ=torch.relu ),
        #     BlockLinear(  in_channels=self.z_size*3, out_channels=self.z_size,  bias=True,  activ=torch.relu ),
        #     BlockLinear(  in_channels=self.z_size, out_channels=self.z_size,  bias=True,  activ=None )
        # )


        # # cur_nr_channels=self.nr_points_z*3*2    *6 #the z for all images
        # # cur_nr_channels=self.nr_points_z*3*2   +4 #the z for all images together dist translation and distance
        # cur_nr_channels=3468 # when putting the whole image as z
        # channels_aggregate=[1024,512,512]
        # self.aggregate_layers=torch.nn.ModuleList([])
        # for i in range(2):
        #     # print("cur nr channes", cur_nr_channels)
        #     # self.aggregate_layers.append( BlockLinear(  in_channels=cur_nr_channels, out_channels=channels_aggregate[i],  bias=True,  activ=torch.relu ) )
        #     self.aggregate_layers.append( BlockLinear(  in_channels=cur_nr_channels, out_channels=cur_nr_channels,  bias=True,  activ=torch.relu ) )
        #     # cur_nr_channels= channels_aggregate[i]
        # self.aggregate_layers.append( BlockLinear(  in_channels=cur_nr_channels, out_channels=512,  bias=True,  activ=None) )



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

        # self.leaned_pe=LearnedPE(in_channels=3, num_encoding_functions=self.num_encodings, logsampling=True)

        self.iter=0
       

      
    def forward(self, gt_frame, frames_for_encoding, all_imgs_poses_cam_world_list, gt_tf_cam_world, gt_K, depth_min, depth_max, use_ray_compression, novel=False):

 

        # nr_imgs=x.shape[0]

        mesh=Mesh()
        for i in range(len(frames_for_encoding)):
            mesh.add( frames_for_encoding[i].cloud )
        mesh.random_subsample(0.7)
        mesh.m_vis.m_show_points=True
        mesh.m_vis.set_color_pervertcolor()
        Scene.show(mesh, "cloud")


        #pass the mesh through the lattice 
        lv, ls = self.spatial_lnn(mesh)
       





        TIME_START("full_siren")
        #siren has to receive some 3d points as query, The 3d points are located along rays so we can volume render and compare with the image itself
        #most of it is from https://github.com/krrish94/nerf-pytorch/blob/master/tiny_nerf.py
        # print("x has shape", x.shape)
        height=gt_frame.height
        width=gt_frame.width
        fx=gt_K[0,0] ### 
        fy=gt_K[1,1] ### 
        cx=gt_K[0,2] ### 
        cy=gt_K[1,2] ### 
        tform_cam2world =torch.from_numpy( gt_tf_cam_world.inverse().matrix() ).to("cuda")
        #vase
        # near_thresh=0.7
        # far_thresh=1.2
        #socrates
        near_thresh=depth_min
        far_thresh=depth_max
        # if novel:
            # near_thresh=near_thresh+0.1
            # far_thresh=far_thresh-0.1
        # depth_samples_per_ray=100
        # depth_samples_per_ray=60
        depth_samples_per_ray=40
        # depth_samples_per_ray=40
        # depth_samples_per_ray=30
        chunksize=512*512
        # chunksize=1024*1024

        # Get the "bundle" of rays through all image pixels.
        TIME_START("ray_bundle")
        ray_origins, ray_directions = get_ray_bundle(
            height, width, fx,fy,cx,cy, tform_cam2world, novel=novel
        )
       
        TIME_END("ray_bundle")

        TIME_START("sample")
      

        #just set the two tensors to the min and max 
        near_thresh_tensor= torch.ones([1,1,height,width], dtype=torch.float32, device=torch.device("cuda")) 
        far_thresh_tensor= torch.ones([1,1,height,width], dtype=torch.float32, device=torch.device("cuda")) 
        near_thresh_tensor.fill_(depth_min)
        far_thresh_tensor.fill_(depth_max)

        query_points, depth_values = compute_query_points_from_rays2(
            ray_origins, ray_directions, near_thresh_tensor, far_thresh_tensor, depth_samples_per_ray, randomize=True
        )

        TIME_END("sample")

        # "Flatten" the query points.
        # print("query points is ", query_points.shape)
        flattened_query_points = query_points.reshape((-1, 3))
        # print("flattened_query_points is ", flattened_query_points.shape)

        # TIME_START("pos_encode")
        # flattened_query_points = positional_encoding(flattened_query_points, num_encoding_functions=self.num_encodings, log_sampling=True)
        # flattened_query_points=self.leaned_pe(flattened_query_points.to("cuda") )
        # print("flattened_query_points after pe", flattened_query_points.shape)
        flattened_query_points=flattened_query_points.view(height,width,depth_samples_per_ray,-1 )
        # print("flatened_query_pointss is ", flatened_query_pointss.shape)
        # TIME_END("pos_encode")


        #slice from lattice 
        flattened_query_points_for_slicing= flattened_query_points.view(-1,3)
        point_features=self.slice(lv, ls, flattened_query_points_for_slicing)


        # print("self, iter is ",self.iter, )

        # radiance_field_flattened = self.siren_net(query_points.to("cuda") )-3.0 
        # radiance_field_flattened = self.siren_net(query_points.to("cuda") )
        # flattened_query_points/=2.43
        radiance_field_flattened = self.siren_net(flattened_query_points.to("cuda"), ray_directions, point_features ) #radiance field has shape height,width, nr_samples,4
        # radiance_field_flattened = self.siren_net(flattened_query_points.to("cuda"), ray_directions, params=siren_params )
        # radiance_field_flattened = self.siren_net(query_points.to("cuda"), params=siren_params )
        # radiance_field_flattened = self.nerf_net(flattened_query_points.to("cuda") ) 
        # radiance_field_flattened = self.nerf_net(flattened_query_points.to("cuda"), params=siren_params ) 

        #debug
        if not novel and self.iter%100==0:
            rays_mesh=Mesh()
            rays_mesh.V=query_points.detach().reshape((-1, 3)).cpu().numpy()
            rays_mesh.m_vis.m_show_points=True
            #color based on sigma 
            sigma_a = radiance_field_flattened[:,:,:, self.siren_out_channels-1].detach().view(-1,1).repeat(1,3)
            rays_mesh.C=sigma_a.cpu().numpy()
            # Scene.show(rays_mesh, "rays_mesh_novel")
        if not novel:
            self.iter+=1


    

        radiance_field_flattened=radiance_field_flattened.view(-1,self.siren_out_channels)


        # "Unflatten" to obtain the radiance field.
        unflattened_shape = list(query_points.shape[:-1]) + [self.siren_out_channels]
        radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_predicted, depth_map, acc_map = render_volume_density(
        # rgb_predicted, depth_map, acc_map = render_volume_density_nerfplusplus(
        # rgb_predicted, depth_map, acc_map = render_volume_density2(
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




        # #NEW LOSSES
        new_loss=0
       




        return rgb_predicted,  depth_map, acc_map, new_loss
        # return img_directly_after_decoding






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