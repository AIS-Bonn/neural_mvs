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

#resize funcs 
import resize_right.resize_right as resize_right

#models from https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks
from segnet.model.SQNet import SQNet
from segnet.model.LinkNet import LinkNet
from segnet.model.LinkNet import LinkNetImprove
from segnet.model.SegNet import SegNet
import segnet.model.UNet as UNet_efficient
from segnet.model.ENet import ENet
from segnet.model.ERFNet import ERFNet
from segnet.model.CGNet import CGNet
from segnet.model.EDANet import EDANet
from segnet.model.ESNet import ESNet
from segnet.model.ESPNet import ESPNet
from segnet.model.LEDNet import LEDNet
# from segnet.model.ESPNet_v2.SegmentationModel import EESPNet_Seg
from segnet.model.ContextNet import ContextNet
from segnet.model.FastSCNN import FastSCNN
from segnet.model.DABNet import DABNet
from segnet.model.FSSNet import FSSNet
from segnet.model.FPENet import FPENet

#superres 
from super_res.edsr.edsr import EDSR


# from pytorch_memlab import LineProfiler
from pytorch_memlab import LineProfiler, profile, profile_every, set_target_gpu, clear_global_line_profiler

#latticenet 
# from latticenet_py.lattice.lattice_wrapper import LatticeWrapper
# from latticenet_py.lattice.diceloss import GeneralizedSoftDiceLoss
# from latticenet_py.lattice.lovasz_loss import LovaszSoftmax
# from latticenet_py.lattice.models import LNN
# from latticenet_py.callbacks.callback import *
# from latticenet_py.callbacks.viewer_callback import *
# from latticenet_py.callbacks.visdom_callback import *
# from latticenet_py.callbacks.state_callback import *
# from latticenet_py.callbacks.phase import *
# from latticenet_py.lattice.lattice_modules import *


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

class SpatialEncoder2D(torch.nn.Module):
    def __init__(self, nr_channels):
        super(SpatialEncoder2D, self).__init__()

        self.first_time=True

        self.learned_pe=LearnedPE(in_channels=11, num_encoding_functions=11, logsampling=True)

        self.start_nr_channels=nr_channels
        # self.first_conv = torch.nn.Conv2d(253, self.start_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() 
        self.first_conv = torch.nn.Conv2d(6, self.start_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() 
        cur_nr_channels=self.start_nr_channels
        

        #cnn for encoding
        self.resnet_list=torch.nn.ModuleList([])
        for i in range(1): #IF I use less layer it seems better, I think that using more layer gets more and more of a global feature and therefore is blurier. Optimally, I want the feature fron this module to change quite fast when going to one pixel to another one. So it should be quite high frequency, 
            #having no norm here is better than having a batchnorm. Maybe its because we use a batch of 1
            #also PAC works better than a conv
            #using norm with GroupNorm seems to work as good as no normalization but probably a bit more stable so we keep it with GN
            self.resnet_list.append( ResnetBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=False ) )
           

        self.relu=torch.nn.ReLU(inplace=False)
        # self.gelu=torch.nn.GELU()
        self.concat_coord=ConcatCoord() 


    def forward(self, x, frame):

        # x=self.concat_coord(x)

        initial_rgb=x

        height=x.shape[2]
        width=x.shape[3]
        channels=x.shape[1]


        #CONCATTING ALL OF THIS SHIT MAKES IT WORSE
        #concat also the ray direction and origin and coords
        fx=frame.K[0,0] ### 
        fy=frame.K[1,1] ### 
        cx=frame.K[0,2] ### 
        cy=frame.K[1,2] ### 
        tform_cam2world =torch.from_numpy( frame.tf_cam_world.inverse().matrix() ).to("cuda")
        ray_origins, ray_directions = get_ray_bundle(
            frame.height, frame.width, fx,fy,cx,cy, tform_cam2world, novel=False
        )
        # x=self.concat_coord(x)
        # ray_origins=ray_origins.view(1,height,width,-1).permute(0,3,1,2)
        ray_directions=ray_directions.view(1,height,width,-1).permute(0,3,1,2)
        # x=torch.cat([x,ray_origins, ray_directions],1)
        x=torch.cat([x, ray_directions],1)

        # channels=x.shape[1]
        # # print("nr channels is ", channels)
        # x=x.view(-1, channels)
        # x=self.learned_pe(x)
        # x=x.view(1,-1, height,width)


       
        #first conv
        x = self.first_conv(x)
        x=self.relu(x)

        after_first_conv=x

        #encode 
        # TIME_START("down_path")
        for i in range( len(self.resnet_list) ):
            x = self.resnet_list[i] (x, x) 
            # x = self.resnet_list[i] (x, initial_rgb) 

        # x=self.concat_coord(x)
        # x=torch.cat([x,initial_rgb],1)

        # x=x+after_first_conv
      

        return x

class SpatialEncoderDense2D(torch.nn.Module):
    def __init__(self, nr_channels):
        super(SpatialEncoderDense2D, self).__init__()

        self.first_time=True

        self.start_nr_channels=8
        self.first_conv = torch.nn.Conv2d(3, self.start_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() 
        cur_nr_channels=self.start_nr_channels
        

        #cnn for encoding 
        self.conv_list=torch.nn.ModuleList([])
        for i in range(12):
            self.conv_list.append( BlockPAC(in_channels=cur_nr_channels, out_channels=8, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True ) )
            cur_nr_channels+=8
           

        self.relu=torch.nn.ReLU(inplace=False)
        self.concat_coord=ConcatCoord() 


    def forward(self, x):

        # x=self.concat_coord(x)

        initial_rgb=x
       
        #first conv
        x = self.first_conv(x)
        x=self.relu(x)

        #encode 
        # TIME_START("down_path")
        stack=[x]
        for i in range( len(self.conv_list) ):
            input_conv=torch.cat(stack,1)
            x=self.conv_list[i](input_conv, input_conv)
            stack.append(x)

        # x=self.concat_coord(x)
        # x=torch.cat([x,initial_rgb],1)
      
        x=torch.cat(stack,1)

        # print("x final si ", x.shape)

        return x


class UNet(torch.nn.Module):
    def __init__(self, nr_channels_start, nr_channels_output, nr_stages, max_nr_channels=999990):
        super(UNet, self).__init__()


        # self.learned_pe=LearnedPE(in_channels=11, num_encoding_functions=11, logsampling=True)

        #params
        self.start_nr_channels=nr_channels_start
        self.nr_stages=nr_stages
        self.compression_factor=1.0



        #DELAYED creation 
        self.first_conv=None
        cur_nr_channels=self.start_nr_channels


        self.down_stages_list = torch.nn.ModuleList([])
        self.coarsen_list = torch.nn.ModuleList([])
        self.nr_layers_ending_stage=[]
        for i in range(self.nr_stages):
            print("cur nr_channels ", cur_nr_channels)
            self.down_stages_list.append( nn.Sequential(
                # ResnetBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[False, False], with_dropout=False, do_norm=True ),
                ResnetBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=True ),

                # GNReluConv(in_channels=cur_nr_channels, out_channels=cur_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.ReLU(), is_first_layer=False ),
                # GNReluConv(in_channels=cur_nr_channels, out_channels=cur_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.ReLU(), is_first_layer=False )
            ))
            self.nr_layers_ending_stage.append(cur_nr_channels)
            after_coarsening_nr_channels=int(cur_nr_channels*2*self.compression_factor)
            if after_coarsening_nr_channels> max_nr_channels:
                after_coarsening_nr_channels=max_nr_channels
            self.coarsen_list.append(  WNReluConv(in_channels=cur_nr_channels, out_channels=after_coarsening_nr_channels, kernel_size=2, stride=2, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.ReLU(), is_first_layer=False )  )
            cur_nr_channels= after_coarsening_nr_channels
            print("adding unet stage with output ", cur_nr_channels)


        self.bottleneck=nn.Sequential(
                # ResnetBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[False, False], with_dropout=False, do_norm=True ),
                ResnetBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=True ),

                # GNReluConv(in_channels=cur_nr_channels, out_channels=cur_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.ReLU(), is_first_layer=False ),
                # GNReluConv(in_channels=cur_nr_channels, out_channels=cur_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.ReLU(), is_first_layer=False )
            )

        self.up_stages_list = torch.nn.ModuleList([])
        self.squeeze_list = torch.nn.ModuleList([])
        for i in range(self.nr_stages):
            after_finefy_nr_channels=int(cur_nr_channels/2)
            self.squeeze_list.append(  WNReluConv(in_channels=cur_nr_channels, out_channels=after_finefy_nr_channels, kernel_size=2, stride=2, padding=0, dilation=1, bias=True, with_dropout=False, transposed=True, do_norm=True, activ=torch.nn.ReLU(), is_first_layer=False )  )
            #we now concat the features from the corresponding stage
            cur_nr_channels=after_finefy_nr_channels
            cur_nr_channels+= self.nr_layers_ending_stage.pop()
            self.up_stages_list.append( nn.Sequential(
                ##last conv should have a bias because it's not followed by a GN
                # ResnetBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[False, True], with_dropout=False, do_norm=True ),
                ResnetBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=True ),

                # GNReluConv(in_channels=cur_nr_channels, out_channels=cur_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.ReLU(), is_first_layer=False ),
                # GNReluConv(in_channels=cur_nr_channels, out_channels=cur_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.ReLU(), is_first_layer=False )
            ))
            print("up stage which outputs nr of layers ", cur_nr_channels)



        # #cnn for encoding
        # self.resnet_list=torch.nn.ModuleList([])
        # nr_layers=6
        # for i in range(nr_layers): 
        #     # print("creating curnnrchannels, ", cur_nr_channels)
        #     is_last_layer=i==nr_layers-1
        #     # self.resnet_list.append( ResnetBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[is_last_layer, is_last_layer], with_dropout=False, do_norm=True ) )
        #     self.resnet_list.append( ResnetBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=False ) )

        # print("last conv is ", cur_nr_channels)
        # self.last_conv1=DeformConv2d(cur_nr_channels, cur_nr_channels, kernel_size=3, stride=1, padding=1, bias=True).cuda()  
        # self.last_conv2=DeformConv2d(cur_nr_channels, cur_nr_channels, kernel_size=3, stride=1, padding=1, bias=True).cuda()  
        # self.last_conv1=PacConv2d(cur_nr_channels, cur_nr_channels, kernel_size=3, stride=1, padding=1, bias=True).cuda()  
        # self.last_conv2=PacConv2d(cur_nr_channels, cur_nr_channels, kernel_size=3, stride=1, padding=1, bias=True).cuda()  
        self.last_conv = torch.nn.Conv2d(cur_nr_channels, nr_channels_output, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() 
        # self.last_conv=DeformConv2d(cur_nr_channels, nr_channels_output, kernel_size=3, stride=1, padding=1, bias=True).cuda()  
        # self.last_conv =  GNReluConv(in_channels=cur_nr_channels, out_channels=nr_channels_output, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.relu, is_first_layer=False ).cuda()

        self.relu=torch.nn.ReLU(inplace=False)
        self.concat_coord=ConcatCoord() 

    # @profile
    # @profile_every(1)
    def forward(self, x):
        if self.first_conv==None:
            self.first_conv = torch.nn.Conv2d( x.shape[1], self.start_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() 

        
        #attempot 1
        # #run through the net 
        # x=self.first_conv(x)

        # for layer in self.resnet_list:
        #     x=layer(x,x)

        # x=self.concat_coord(x)


        #attempt 2 to make an actual unet
        initial_x=x
        x=self.first_conv(x)

        features_ending_stage=[]
        #down
        for i in range(self.nr_stages):
            # print("x stage ",i , "has shape ", x.shape)
            x=self.down_stages_list[i](x)
            features_ending_stage.append(x)
            # print("x before coarsening ", i , " has shape ", x.shape)
            x=self.coarsen_list[i](x)

        #bottleneck 
        x=self.bottleneck(x)
        # print("after bottlenck ", x.shape)

        #up
        for i in range(self.nr_stages):
            # print("before finefy ", i, " x is", x.shape)
            # x=self.finefy_list[i](x)
            # print("after finefy ", i, " x is", x.shape)
            vertical_feats= features_ending_stage.pop()
            x=self.squeeze_list[i](x) #upsample resolution and reduced the channels
            if x.shape[2]!=vertical_feats.shape[2] or x.shape[3]!=vertical_feats.shape[3]:
                # print("x has shape", x.shape, "vertical feat have shape ", vertical_feats.shape)
                x = torch.nn.functional.interpolate(x,size=(vertical_feats.shape[2], vertical_feats.shape[3]), mode='bilinear') #to make sure that the sized between the x and vertical feats match because the transposed conv may not neceserraly create the same size of the image as the one given as input
            # print("x is", x.shape , " verticalFeats ", vertical_feats.shape)
            x=torch.cat([x,vertical_feats],1)
            # print("x shape before seuqqeing is ", i, "  ", x.shape)
            # print("x shape after seuqqeing is ", i, "  ", x.shape)

            # print("vefore concating ", i, " x is ", x.shape, " vertical featus is ", vertical_feats.shape)
            x=self.up_stages_list[i](x)

        # print("x before the final conv has mean ", x.mean(), " and std ", x.std())
        # x=torch.cat([x,initial_x],1) #bad idea to concat the rgb here. It introduces way too much high frequency in the output which makes the ray marker get stuck in not knowing wheere to predict the depth so that the slicing slcies the correct features. IF we dont concat, then the unet features are kinda smooth and they act like a wise basin of converges that tell the lstm, to predict a depth close a certain position so that the features it slices will be better
        # x=self.last_conv1(x,x)
        # x=self.last_conv2(x,x)
        x=self.last_conv(x)
        
        return x


#Sine the unet can be quite big, I experiment here with just passing each level of the image pyramid through a one resnet block adn tehn concating the features from each level.
class FeaturePyramid(torch.nn.Module):
    def __init__(self, nr_channels_start, nr_channels_output, nr_stages):
        super(FeaturePyramid, self).__init__()


        # self.learned_pe=LearnedPE(in_channels=11, num_encoding_functions=11, logsampling=True)

        #params
        self.start_nr_channels=nr_channels_start
        self.nr_stages=nr_stages
        self.compression_factor=1.0



        #DELAYED creation 
        self.first_conv=None
        cur_nr_channels=self.start_nr_channels


        self.encoder_for_pyramid_lvl=[]
        for i in range(self.nr_stages):
            self.encoder_for_pyramid_lvl.append( nn.Sequential(
                # torch.nn.Conv2d( cur_nr_channels, cur_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda(),
                ResnetBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=True ),
            ))

        cur_nr_channels=cur_nr_channels*self.nr_stages

        # self.last_conv = nn.Sequential(
        #     # ResnetBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=True ),
        #     # torch.nn.Conv2d(cur_nr_channels, nr_channels_output, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda(),
        #     ResnetBlock2D(cur_nr_channels, kernel_size=1, stride=1, padding=0, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=True ),
        #     torch.nn.Conv2d(cur_nr_channels, nr_channels_output, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda(),
        # )
        self.last_conv = nn.Sequential(
            # ResnetBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=True ),
            # torch.nn.Conv2d(cur_nr_channels, nr_channels_output, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda(),
            # ResnetBlock2D(cur_nr_channels, kernel_size=1, stride=1, padding=0, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=True ),
            WNReluConv(in_channels=cur_nr_channels, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=None, is_first_layer=False ),
            torch.nn.GELU(),
            WNReluConv(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=None, is_first_layer=False ),
            torch.nn.GELU(),
            torch.nn.Conv2d(32, nr_channels_output, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda(),
        )
        self.relu=torch.nn.ReLU(inplace=False)
        self.concat_coord=ConcatCoord() 
        self.downscale_layer_per_lvl = None 
        
        # resize_right.ResizeLayer(in_shape, scale_factors=None, out_shape=None,
        #                                  interp_method=interp_methods.cubic, support_sz=None,
        #                                  antialiasing=True

    # @profile
    # @profile_every(1)
    def forward(self, x):

        if self.first_conv==None:
            self.first_conv = torch.nn.Conv2d( x.shape[1], self.start_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() 

        

        #attempt 2 to make an actual unet
        initial_x=x

        x=self.first_conv(x)

        #make som modules for resizing
        dummy_x=x
        scale_factor=1.0
        if(self.downscale_layer_per_lvl==None):
            self.downscale_layer_per_lvl=[]
            for i in range(self.nr_stages):
                if i!=0:
                    self.downscale_layer_per_lvl.append( resize_right.ResizeLayer(dummy_x.shape, scale_factors=0.5, out_shape=None,
                                         interp_method=resize_right.interp_methods.linear, support_sz=None,
                                         antialiasing=True).to("cuda")
                    )
                    dummy_x=self.downscale_layer_per_lvl[-1](dummy_x)

        # initial_per_lvl=[]
        # #make a subsample of the rgbs
        # for i in range(self.nr_stages):
        #     if i!=0:
        #         x=self.downscale_layer_per_lvl[i-1](initial_rgb)
        #         # x = torch.nn.functional.interpolate(x, size=( int(x.shape[2]/2), int(x.shape[3]/2) ), mode='bilinear')
        #     initial_per_lvl.append(x)


        
        features_per_lvl=[] 
        for i in range(self.nr_stages):
            #downsample if necessary
            # if i!=0:
                # x = torch.nn.functional.interpolate(initial_x, size=( int(x.shape[2]/2), int(x.shape[3]/2) ), mode='bilinear')
            # input_for_lvl= initial_per_lvl[i]
            if i!=0:
                x=self.downscale_layer_per_lvl[i-1](x)
            # print("encoding layer i", i, " ", x.shape)
            x=self.encoder_for_pyramid_lvl[i](x)
            features_per_lvl.append(x)

        #get all the features, upsample them and pass them through the last conv
        feat_upsampled_per_lvl=[]
        for i in range(self.nr_stages):
            feat= features_per_lvl[i]
            feat = torch.nn.functional.interpolate(feat, size=(initial_x.shape[2], initial_x.shape[3]), mode='bicubic')
            feat_upsampled_per_lvl.append(feat)

        x=torch.cat(feat_upsampled_per_lvl,1)

        # x=torch.cat([x,initial_x],1)
        x=self.last_conv(x)
       
        
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
        # self.model_params=model_params
        self.nr_downsamples=4
        self.nr_blocks_down_stage= [2,2,2,2,2]
        self.nr_blocks_bottleneck=1
        self.nr_blocks_up_stage= [1,1,1,1,1]
        self.nr_levels_down_with_normal_resnet=5
        self.nr_levels_up_with_normal_resnet=5
        # compression_factor=model_params.compression_factor()
        dropout_last_layer=False
        experiment="none"
        #check that the experiment has a valid string
        valid_experiment=["none", "slice_no_deform", "pointnet_no_elevate", "pointnet_no_local_mean", "pointnet_no_elevate_no_local_mean", "splat", "attention_pool"]
        if experiment not in valid_experiment:
            err = "Experiment " + experiment + " is not valid"
            sys.exit(err)





        self.distribute=DistributeLatticeModule(experiment) 
        # self.splat=SplatLatticeModule() 
        # self.pointnet_layers=model_params.pointnet_layers()
        # self.start_nr_filters=model_params.pointnet_start_nr_channels()
        # self.pointnet_layers=[16,32,64]
        self.pointnet_layers=[64,64] #when we use features like the pac features for each position, we don't really need to encode them that much
        self.start_nr_filters=64
        print("pointnet layers is ", self.pointnet_layers)
        self.point_net=PointNetModule( self.pointnet_layers, self.start_nr_filters, experiment)  
        self.conv_start=None




        #####################
        # Downsampling path #
        #####################
        self.resnet_blocks_per_down_lvl_list=torch.nn.ModuleList([])
        self.coarsens_list=torch.nn.ModuleList([])
        self.maxpool_list=torch.nn.ModuleList([])
        corsenings_channel_counts = []
        skip_connection_channel_counts = []
        cur_channels_count=self.start_nr_filters
        for i in range(self.nr_downsamples):
            
            #create the resnet blocks
            self.resnet_blocks_per_down_lvl_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_down_stage[i]):
                if i<self.nr_levels_down_with_normal_resnet:
                    print("adding down_resnet_block with nr of filters", cur_channels_count )
                    should_use_dropout=False
                    print("adding down_resnet_block with dropout", should_use_dropout )
                    self.resnet_blocks_per_down_lvl_list[i].append( ResnetBlock(cur_channels_count, [1,1], [False,False], should_use_dropout) )
                else:
                    print("adding down_bottleneck_block with nr of filters", cur_channels_count )
                    self.resnet_blocks_per_down_lvl_list[i].append( BottleneckBlock(cur_channels_count, [False,False,False]) )
            skip_connection_channel_counts.append(cur_channels_count)
            # nr_channels_after_coarsening=int(cur_channels_count*2*compression_factor)
            nr_channels_after_coarsening=int(cur_channels_count)
            print("adding bnReluCorsen which outputs nr of channels ", nr_channels_after_coarsening )
            self.coarsens_list.append( GnReluCoarsen(nr_channels_after_coarsening)) #is still the best one because it can easily learn the versions of Avg and Blur. and the Max version is the worse for some reason
            cur_channels_count=nr_channels_after_coarsening
            corsenings_channel_counts.append(cur_channels_count)

        #####################
        #     Bottleneck    #
        #####################
        self.resnet_blocks_bottleneck=torch.nn.ModuleList([])
        for j in range(self.nr_blocks_bottleneck):
                print("adding bottleneck_resnet_block with nr of filters", cur_channels_count )
                self.resnet_blocks_bottleneck.append( BottleneckBlock(cur_channels_count, [False,False,False]) )

        self.do_concat_for_vertical_connection=True
        #######################
        #   Upsampling path   #
        #######################
        self.finefy_list=torch.nn.ModuleList([])
        self.up_activation_list=torch.nn.ModuleList([])
        self.up_match_dim_list=torch.nn.ModuleList([])
        self.up_bn_match_dim_list=torch.nn.ModuleList([])
        self.resnet_blocks_per_up_lvl_list=torch.nn.ModuleList([])
        for i in range(self.nr_downsamples):
            nr_chanels_skip_connection=skip_connection_channel_counts.pop()

            # if the finefy is the deepest one int the network then it just divides by 2 the nr of channels because we know it didnt get as input two concatet tensors
            # nr_chanels_finefy=int(cur_channels_count/2)
            nr_chanels_finefy=int(nr_chanels_skip_connection)

            #do it with finefy
            print("adding bnReluFinefy which outputs nr of channels ", nr_chanels_finefy )
            self.finefy_list.append( GnReluFinefy(nr_chanels_finefy ))

            #after finefy we do a concat with the skip connection so the number of channels doubles
            if self.do_concat_for_vertical_connection:
                cur_channels_count=nr_chanels_skip_connection+nr_chanels_finefy
            else:
                cur_channels_count=nr_chanels_skip_connection

            self.resnet_blocks_per_up_lvl_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_up_stage[i]):
                is_last_conv=j==self.nr_blocks_up_stage[i]-1 and i==self.nr_downsamples-1 #the last conv of the last upsample is followed by a slice and not a bn, therefore we need a bias
                if i>=self.nr_downsamples-self.nr_levels_up_with_normal_resnet:
                    print("adding up_resnet_block with nr of filters", cur_channels_count ) 
                    self.resnet_blocks_per_up_lvl_list[i].append( ResnetBlock(cur_channels_count, [1,1], [False,is_last_conv], False) )
                else:
                    print("adding up_bottleneck_block with nr of filters", cur_channels_count ) 
                    self.resnet_blocks_per_up_lvl_list[i].append( BottleneckBlock(cur_channels_count, [False,False,is_last_conv] ) )



        self.expand=ExpandLatticeModule( 10, 0.05, True )



        # #a bit more control
        # # self.pointnet_layers=[16,32,64]
        # self.pointnet_layers=[64,64]
        # self.start_nr_filters=128
        # experiment="none"
        # #check that the experiment has a valid string
        # valid_experiment=["none", "slice_no_deform", "pointnet_no_elevate", "pointnet_no_local_mean", "pointnet_no_elevate_no_local_mean", "splat", "attention_pool"]
        # if experiment not in valid_experiment:
        #     err = "Experiment " + experiment + " is not valid"
        #     sys.exit(err)




        # self.distribute=DistributeLatticeModule(experiment) 
        # print("pointnet layers is ", self.pointnet_layers)
        # self.point_net=PointNetModule( self.pointnet_layers, self.start_nr_filters, experiment)  




        # #####################
        # # Downsampling path #
        # #####################
        # self.resnets=torch.nn.ModuleList([])
        # for i in range(8):
        #     self.resnets.append( ResnetBlock(self.start_nr_filters, [1,1], [False,False], False) )
         
      

    def forward(self, positions, sliced_local_features, sliced_global_features):

        # positions=torch.from_numpy(cloud.V.copy() ).float().to("cuda")
        # values=torch.from_numpy(cloud.C.copy() ).float().to("cuda")
        # values=torch.cat( [positions,values],1 )


        ls=Lattice.create("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/config/train.cfg", "splated_lattice")

        # distributed, indices, weights=self.distribute(ls, positions, values)

        # lv, ls=self.point_net(ls, distributed, indices)

        # lv, ls=self.expand(lv, ls, ls.positions() )

        
        # for i in range(len(self.resnets)):
        #     lv, ls = self.resnets[i]( lv, ls) 











        #ATTEMPT !
        # distributed, indices, weights=self.distribute(ls, positions, values)
        # lv, ls=self.point_net(ls, distributed, indices)

        #ATTEMP 2
        #get for each vertex just the mean values arount the area
        indices, weights=ls.just_create_verts(positions, True)
        ls.set_positions(positions)
        indices_long=indices.long()
        indices_long[indices_long<0]=0 #some indices may be -1 because they were not inserted into the hashmap, this will cause an error for scatter_max so we just set them to 0

        #average the weighted local features
        local_val_dim=sliced_local_features.shape[1]
        sliced_local_features=sliced_local_features.repeat(1,4)
        sliced_local_features=sliced_local_features.view(-1,local_val_dim)
        sliced_local_features=sliced_local_features*weights.view(-1,1) #weight the image features because they are local
        lv_local = torch_scatter.scatter_mean(sliced_local_features, indices_long, dim=0)

        #average the global features and then positionally encode them
        global_val_dim=sliced_global_features.shape[1]
        sliced_global_features=sliced_global_features.repeat(1,4)
        sliced_global_features=sliced_global_features.view(-1,global_val_dim)
        lv_global = torch_scatter.scatter_mean(sliced_global_features, indices_long, dim=0)
        #encode the average positions, ray directions, etc
        lv_global=positional_encoding(lv_global, num_encoding_functions=11)
        
        lv=torch.cat([lv_local, lv_global],1)
        ls.set_values(lv)
        #conv to get it to 64 or so
        if self.conv_start==None: 
            self.conv_start=ConvLatticeModule(nr_filters=64, neighbourhood_size=1, dilation=1, bias=True) #disable the bias becuse it is followed by a gn
        lv, ls = self.conv_start(lv,ls)

        

        # lv, ls, indices, weights = self.splat(ls, positions, values)

        # print("before lv", lv.shape)
        lv, ls=self.expand(lv, ls, ls.positions() )
        # print("after lv", lv.shape)


        
        fine_structures_list=[]
        fine_values_list=[]
        # TIME_START("down_path")
        for i in range(self.nr_downsamples):

            #resnet blocks
            for j in range(self.nr_blocks_down_stage[i]):
                # print("start downsample stage ", i , " resnet block ", j, "lv has shape", lv.shape, " ls has val dim", ls.val_dim() )
                lv, ls = self.resnet_blocks_per_down_lvl_list[i][j] ( lv, ls) 

            #saving them for when we do finefy so we can concat them there
            fine_structures_list.append(ls) 
            fine_values_list.append(lv)

            #now we do a downsample
            # print("start coarsen stage ", i, "lv has shape", lv.shape, "ls has val_dim", ls.val_dim() )
            lv, ls = self.coarsens_list[i] ( lv, ls)
            # print( "finished coarsen stage ", i, "lv has shape", lv.shape, "ls has val_dim", ls.val_dim() )

        # TIME_END("down_path")

        # #bottleneck
        for j in range(self.nr_blocks_bottleneck):
            # print("bottleneck stage", j,  "lv has shape", lv.shape, "ls has val_dim", ls.val_dim()  )
            lv, ls = self.resnet_blocks_bottleneck[j] ( lv, ls) 

        # multi_res_lattice=[] 

        #upsample (we start from the bottom of the U-net, so the upsampling that is closest to the blottlenck)
        # TIME_START("up_path")
        for i in range(self.nr_downsamples):

            fine_values=fine_values_list.pop()
            fine_structure=fine_structures_list.pop()


            #finefy
            # print("start finefy stage", i,  "lv has shape", lv.shape, "ls has val_dim ", ls.val_dim(),  "fine strcture has val dim ", fine_structure.val_dim() )
            lv, ls = self.finefy_list[i] ( lv, ls, fine_structure  )

            #concat or adding for the vertical connection
            if self.do_concat_for_vertical_connection: 
                lv=torch.cat((lv, fine_values ),1)
            else:
                lv+=fine_values

            #resnet blocks
            for j in range(self.nr_blocks_up_stage[i]):
                # print("start resnet block in upstage", i, "lv has shape", lv.shape, "ls has val dim" , ls.val_dim() )
                lv, ls = self.resnet_blocks_per_up_lvl_list[i][j] ( lv, ls) 
        # TIME_END("up_path")

        # print("lv has shape ", lv.shape)
            # multi_res_lattice.append( [lv,ls] )


        #for the last thing, concat the lv  with the intial lv so as to get in the values also the positiuon
        # lv=torch.cat([lv, positions],1)
        # ls.set_values(lv)



        # return multi_res_lattice
        return lv, ls




class SpatialLNNFixed(torch.nn.Module):
    def __init__(self ):
        super(SpatialLNNFixed, self).__init__()












        #a bit more control
        # self.nr_downsamples=4
        # self.nr_blocks_down_stage= [2,2,2,2,2]
        # self.nr_blocks_bottleneck=1
        # self.nr_blocks_up_stage= [2,2,2,2,2]
        # self.nr_levels_down_with_normal_resnet=5
        # self.nr_levels_up_with_normal_resnet=5

        #less
        self.nr_downsamples=0
        self.nr_blocks_down_stage= [2,2,2,2,2]
        self.nr_blocks_bottleneck=0
        self.nr_blocks_up_stage= [2,2,2,2,2]
        self.nr_levels_down_with_normal_resnet=5
        self.nr_levels_up_with_normal_resnet=5

        dropout_last_layer=False
        experiment="none"
        #check that the experiment has a valid string
        valid_experiment=["none", "slice_no_deform", "pointnet_no_elevate", "pointnet_no_local_mean", "pointnet_no_elevate_no_local_mean", "splat", "attention_pool"]
        if experiment not in valid_experiment:
            err = "Experiment " + experiment + " is not valid"
            sys.exit(err)





        self.distribute=DistributeLatticeModule(experiment) 
        # self.splat=SplatLatticeModule() 
        # self.pointnet_layers=model_params.pointnet_layers()
        # self.start_nr_filters=model_params.pointnet_start_nr_channels()
        # self.pointnet_layers=[16,32,64]
        self.pointnet_layers=[64,64] #when we use features like the pac features for each position, we don't really need to encode them that much
        self.start_nr_filters=64
        print("pointnet layers is ", self.pointnet_layers)
        # self.point_net=PointNetModule( self.pointnet_layers, self.start_nr_filters, experiment)  
        self.conv_start=None




        #####################
        # Downsampling path #
        #####################
        self.resnet_blocks_per_down_lvl_list=torch.nn.ModuleList([])
        self.coarsens_list=torch.nn.ModuleList([])
        self.maxpool_list=torch.nn.ModuleList([])
        corsenings_channel_counts = []
        skip_connection_channel_counts = []
        cur_channels_count=self.start_nr_filters
        for i in range(self.nr_downsamples):
            
            #create the resnet blocks
            self.resnet_blocks_per_down_lvl_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_down_stage[i]):
                print("adding down_resnet_block with nr of filters", cur_channels_count )
                should_use_dropout=False
                # self.resnet_blocks_per_down_lvl_list[i].append( ResnetBlock(cur_channels_count, [1,1], [False,False], should_use_dropout) )
                # self.resnet_blocks_per_down_lvl_list[i].append( ResnetBlockReluConvLattice(cur_channels_count, [1,1], [True,True], should_use_dropout) )
                # self.resnet_blocks_per_down_lvl_list[i].append( ResnetBlockConvReluLattice(cur_channels_count, [1,1], [True,True], should_use_dropout) )
                self.resnet_blocks_per_down_lvl_list[i].append( ConvRelu(cur_channels_count, dilation=1, bias=True, with_dropout=False) )
            skip_connection_channel_counts.append(cur_channels_count)
            # nr_channels_after_coarsening=int(cur_channels_count*2*compression_factor)
            nr_channels_after_coarsening=int(cur_channels_count)
            print("adding bnReluCorsen which outputs nr of channels ", nr_channels_after_coarsening )
            # self.coarsens_list.append( GnReluCoarsen(nr_channels_after_coarsening)) #is still the best one because it can easily learn the versions of Avg and Blur. and the Max version is the worse for some reason
            # self.coarsens_list.append( ReluCoarsen(nr_channels_after_coarsening, bias=True)) #is still the best one because it can easily learn the versions of Avg and Blur. and the Max version is the worse for some reason
            self.coarsens_list.append( CoarsenRelu(nr_channels_after_coarsening, bias=True)) #is still the best one because it can easily learn the versions of Avg and Blur. and the Max version is the worse for some reason
            cur_channels_count=nr_channels_after_coarsening
            corsenings_channel_counts.append(cur_channels_count)

        #####################
        #     Bottleneck    #
        #####################
        self.resnet_blocks_bottleneck=torch.nn.ModuleList([])
        for j in range(self.nr_blocks_bottleneck):
                print("adding bottleneck_resnet_block with nr of filters", cur_channels_count )
                # self.resnet_blocks_bottleneck.append( BottleneckBlock(cur_channels_count, [False,False,False]) )
                # self.resnet_blocks_bottleneck.append(  ResnetBlock(cur_channels_count, [1,1], [False,False], should_use_dropout) )
                # self.resnet_blocks_bottleneck.append(  ResnetBlockReluConvLattice(cur_channels_count, [1,1], [True,True], should_use_dropout) )
                # self.resnet_blocks_bottleneck.append(  ResnetBlockConvReluLattice(cur_channels_count, [1,1], [True,True], should_use_dropout) )
                self.resnet_blocks_bottleneck.append( ConvRelu(cur_channels_count, dilation=1, bias=True, with_dropout=False) )

        self.do_concat_for_vertical_connection=True
        #######################
        #   Upsampling path   #
        #######################
        self.finefy_list=torch.nn.ModuleList([])
        self.up_activation_list=torch.nn.ModuleList([])
        self.up_match_dim_list=torch.nn.ModuleList([])
        self.up_bn_match_dim_list=torch.nn.ModuleList([])
        self.resnet_blocks_per_up_lvl_list=torch.nn.ModuleList([])
        for i in range(self.nr_downsamples):
            nr_chanels_skip_connection=skip_connection_channel_counts.pop()

            # if the finefy is the deepest one int the network then it just divides by 2 the nr of channels because we know it didnt get as input two concatet tensors
            # nr_chanels_finefy=int(cur_channels_count/2)
            nr_chanels_finefy=int(nr_chanels_skip_connection)

            #do it with finefy
            print("adding bnReluFinefy which outputs nr of channels ", nr_chanels_finefy )
            # self.finefy_list.append( GnReluFinefy(nr_chanels_finefy ))
            # self.finefy_list.append( GnReluFinefyWTF(nr_chanels_finefy ))
            # self.finefy_list.append( ReluFinefy(nr_chanels_finefy, bias=True ))
            self.finefy_list.append( FinefyRelu(nr_chanels_finefy, bias=True ))

            #after finefy we do a concat with the skip connection so the number of channels doubles
            if self.do_concat_for_vertical_connection:
                cur_channels_count=nr_chanels_skip_connection+nr_chanels_finefy
            else:
                cur_channels_count=nr_chanels_skip_connection

            self.resnet_blocks_per_up_lvl_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_up_stage[i]):
                is_last_conv=j==self.nr_blocks_up_stage[i]-1 and i==self.nr_downsamples-1 #the last conv of the last upsample is followed by a slice and not a bn, therefore we need a bias
                print("adding up_resnet_block with nr of filters", cur_channels_count ) 
                # self.resnet_blocks_per_up_lvl_list[i].append( ResnetBlock(cur_channels_count, [1,1], [False,is_last_conv], False) )
                # self.resnet_blocks_per_up_lvl_list[i].append( ResnetBlockReluConvLattice(cur_channels_count, [1,1], [True,True], False) )
                # self.resnet_blocks_per_up_lvl_list[i].append( ResnetBlockConvReluLattice(cur_channels_count, [1,1], [True,True], False) )
                self.resnet_blocks_per_up_lvl_list[i].append( ConvRelu(cur_channels_count, dilation=1, bias=True, with_dropout=False) )



        self.expand=ExpandLatticeModule( 10, 0.05, True )



        # #a bit more control
        # # self.pointnet_layers=[16,32,64]
        # self.pointnet_layers=[64,64]
        # self.start_nr_filters=128
        # experiment="none"
        # #check that the experiment has a valid string
        # valid_experiment=["none", "slice_no_deform", "pointnet_no_elevate", "pointnet_no_local_mean", "pointnet_no_elevate_no_local_mean", "splat", "attention_pool"]
        # if experiment not in valid_experiment:
        #     err = "Experiment " + experiment + " is not valid"
        #     sys.exit(err)




        # self.distribute=DistributeLatticeModule(experiment) 
        # print("pointnet layers is ", self.pointnet_layers)
        # self.point_net=PointNetModule( self.pointnet_layers, self.start_nr_filters, experiment)  




        # #####################
        # # Downsampling path #
        # #####################
        # self.resnets=torch.nn.ModuleList([])
        # for i in range(8):
        #     self.resnets.append( ResnetBlock(self.start_nr_filters, [1,1], [False,False], False) )
         
      

    def forward(self, positions, sliced_local_features, sliced_global_features):

        # positions=torch.from_numpy(cloud.V.copy() ).float().to("cuda")
        # values=torch.from_numpy(cloud.C.copy() ).float().to("cuda")
        # values=torch.cat( [positions,values],1 )


        ls=Lattice.create("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/config/train.cfg", "splated_lattice")

        # distributed, indices, weights=self.distribute(ls, positions, values)

        # lv, ls=self.point_net(ls, distributed, indices)

        # lv, ls=self.expand(lv, ls, ls.positions() )

        
        # for i in range(len(self.resnets)):
        #     lv, ls = self.resnets[i]( lv, ls) 











        #ATTEMPT !
        # distributed, indices, weights=self.distribute(ls, positions, values)
        # lv, ls=self.point_net(ls, distributed, indices)

        #ATTEMP 2
        #get for each vertex just the mean values arount the area
        indices, weights=ls.just_create_verts(positions, True)
        ls.set_positions(positions)
        indices_long=indices.long()
        indices_long[indices_long<0]=0 #some indices may be -1 because they were not inserted into the hashmap, this will cause an error for scatter_max so we just set them to 0

        #average the weighted local features
        local_val_dim=sliced_local_features.shape[1]
        sliced_local_features=sliced_local_features.repeat(1,4)
        sliced_local_features=sliced_local_features.view(-1,local_val_dim)
        sliced_local_features=sliced_local_features*weights.view(-1,1) #weight the image features because they are local
        lv_local = torch_scatter.scatter_mean(sliced_local_features, indices_long, dim=0)

        #average the global features and then positionally encode them
        global_val_dim=sliced_global_features.shape[1]
        sliced_global_features=sliced_global_features.repeat(1,4)
        sliced_global_features=sliced_global_features.view(-1,global_val_dim)
        lv_global = torch_scatter.scatter_mean(sliced_global_features, indices_long, dim=0)
        #encode the average positions, ray directions, etc
        lv_global=positional_encoding(lv_global, num_encoding_functions=11)
        
        lv=torch.cat([lv_local, lv_global],1)
        ls.set_values(lv)
        #conv to get it to 64 or so
        if self.conv_start==None: 
            self.conv_start=ConvLatticeModule(nr_filters=128, neighbourhood_size=1, dilation=1, bias=True) #disable the bias becuse it is followed by a gn
        lv, ls = self.conv_start(lv,ls)

        

        # lv, ls, indices, weights = self.splat(ls, positions, values)

        # print("before lv", lv.shape)
        lv, ls=self.expand(lv, ls, ls.positions() )
        # print("after lv", lv.shape)


        
        fine_structures_list=[]
        fine_values_list=[]
        # TIME_START("down_path")
        for i in range(self.nr_downsamples):

            #resnet blocks
            for j in range(self.nr_blocks_down_stage[i]):
                # print("start downsample stage ", i , " resnet block ", j, "lv has shape", lv.shape, " ls has val dim", ls.val_dim() )
                lv, ls = self.resnet_blocks_per_down_lvl_list[i][j] ( lv, ls) 

            #saving them for when we do finefy so we can concat them there
            fine_structures_list.append(ls) 
            fine_values_list.append(lv)

            #now we do a downsample
            # print("start coarsen stage ", i, "lv has shape", lv.shape, "ls has val_dim", ls.val_dim() )
            lv, ls = self.coarsens_list[i] ( lv, ls)
            # print( "finished coarsen stage ", i, "lv has shape", lv.shape, "ls has val_dim", ls.val_dim() )

        # TIME_END("down_path")

        # #bottleneck
        for j in range(self.nr_blocks_bottleneck):
            # print("bottleneck stage", j,  "lv has shape", lv.shape, "ls has val_dim", ls.val_dim()  )
            lv, ls = self.resnet_blocks_bottleneck[j] ( lv, ls) 

        # multi_res_lattice=[] 

        #upsample (we start from the bottom of the U-net, so the upsampling that is closest to the blottlenck)
        # TIME_START("up_path")
        for i in range(self.nr_downsamples):

            fine_values=fine_values_list.pop()
            fine_structure=fine_structures_list.pop()


            #finefy
            # print("start finefy stage", i,  "lv has shape", lv.shape, "ls has val_dim ", ls.val_dim(),  "fine strcture has val dim ", fine_structure.val_dim() )
            lv, ls = self.finefy_list[i] ( lv, ls, fine_structure  )

            #concat or adding for the vertical connection
            if self.do_concat_for_vertical_connection: 
                lv=torch.cat((lv, fine_values ),1)
            else:
                lv+=fine_values

            #resnet blocks
            for j in range(self.nr_blocks_up_stage[i]):
                # print("start resnet block in upstage", i, "lv has shape", lv.shape, "ls has val dim" , ls.val_dim() )
                lv, ls = self.resnet_blocks_per_up_lvl_list[i][j] ( lv, ls) 
        # TIME_END("up_path")

        # print("lv has shape ", lv.shape)
            # multi_res_lattice.append( [lv,ls] )


        #for the last thing, concat the lv  with the intial lv so as to get in the values also the positiuon
        # lv=torch.cat([lv, positions],1)
        # ls.set_values(lv)



        # return multi_res_lattice
        return lv, ls


class LNN_2(torch.nn.Module):
    def __init__(self, nr_classes, model_params):
        super(LNN_2, self).__init__()
        self.nr_classes=nr_classes

        #a bit more control
        self.model_params=model_params
        self.nr_downsamples=model_params.nr_downsamples()
        self.nr_blocks_down_stage=model_params.nr_blocks_down_stage()
        self.nr_blocks_bottleneck=model_params.nr_blocks_bottleneck()
        self.nr_blocks_up_stage=model_params.nr_blocks_up_stage()
        self.nr_levels_down_with_normal_resnet=model_params.nr_levels_down_with_normal_resnet()
        self.nr_levels_up_with_normal_resnet=model_params.nr_levels_up_with_normal_resnet()
        compression_factor=model_params.compression_factor()
        dropout_last_layer=model_params.dropout_last_layer()
        experiment=model_params.experiment()
        #check that the experiment has a valid string
        valid_experiment=["none", "slice_no_deform", "pointnet_no_elevate", "pointnet_no_local_mean", "pointnet_no_elevate_no_local_mean", "splat", "attention_pool"]
        if experiment not in valid_experiment:
            err = "Experiment " + experiment + " is not valid"
            sys.exit(err)





        self.distribute=DistributeLatticeModule(experiment) 
        self.pointnet_layers=model_params.pointnet_layers()
        self.start_nr_filters=model_params.pointnet_start_nr_channels()
        print("pointnet layers is ", self.pointnet_layers)
        self.point_net=PointNetModule( self.pointnet_layers, self.start_nr_filters, experiment)  




        #####################
        # Downsampling path #
        #####################
        self.resnet_blocks_per_down_lvl_list=torch.nn.ModuleList([])
        self.coarsens_list=torch.nn.ModuleList([])
        self.maxpool_list=torch.nn.ModuleList([])
        corsenings_channel_counts = []
        skip_connection_channel_counts = []
        cur_channels_count=self.start_nr_filters
        for i in range(self.nr_downsamples):
            
            #create the resnet blocks
            self.resnet_blocks_per_down_lvl_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_down_stage[i]):
                if i<self.nr_levels_down_with_normal_resnet:
                    print("adding down_resnet_block with nr of filters", cur_channels_count )
                    should_use_dropout=False
                    print("adding down_resnet_block with dropout", should_use_dropout )
                    self.resnet_blocks_per_down_lvl_list[i].append( ResnetBlock(cur_channels_count, [1,1], [False,False], should_use_dropout) )
                else:
                    print("adding down_bottleneck_block with nr of filters", cur_channels_count )
                    self.resnet_blocks_per_down_lvl_list[i].append( BottleneckBlock(cur_channels_count, [False,False,False]) )
            skip_connection_channel_counts.append(cur_channels_count)
            nr_channels_after_coarsening=int(cur_channels_count*2*compression_factor)
            print("adding bnReluCorsen which outputs nr of channels ", nr_channels_after_coarsening )
            self.coarsens_list.append( GnReluCoarsen(nr_channels_after_coarsening)) #is still the best one because it can easily learn the versions of Avg and Blur. and the Max version is the worse for some reason
            cur_channels_count=nr_channels_after_coarsening
            corsenings_channel_counts.append(cur_channels_count)

        #####################
        #     Bottleneck    #
        #####################
        self.resnet_blocks_bottleneck=torch.nn.ModuleList([])
        for j in range(self.nr_blocks_bottleneck):
                print("adding bottleneck_resnet_block with nr of filters", cur_channels_count )
                self.resnet_blocks_bottleneck.append( BottleneckBlock(cur_channels_count, [False,False,False]) )

        self.do_concat_for_vertical_connection=True
        #######################
        #   Upsampling path   #
        #######################
        self.finefy_list=torch.nn.ModuleList([])
        self.up_activation_list=torch.nn.ModuleList([])
        self.up_match_dim_list=torch.nn.ModuleList([])
        self.up_bn_match_dim_list=torch.nn.ModuleList([])
        self.resnet_blocks_per_up_lvl_list=torch.nn.ModuleList([])
        for i in range(self.nr_downsamples):
            nr_chanels_skip_connection=skip_connection_channel_counts.pop()

            # if the finefy is the deepest one int the network then it just divides by 2 the nr of channels because we know it didnt get as input two concatet tensors
            nr_chanels_finefy=int(cur_channels_count/2)

            #do it with finefy
            print("adding bnReluFinefy which outputs nr of channels ", nr_chanels_finefy )
            self.finefy_list.append( GnReluFinefy(nr_chanels_finefy ))

            #after finefy we do a concat with the skip connection so the number of channels doubles
            if self.do_concat_for_vertical_connection:
                cur_channels_count=nr_chanels_skip_connection+nr_chanels_finefy
            else:
                cur_channels_count=nr_chanels_skip_connection

            self.resnet_blocks_per_up_lvl_list.append( torch.nn.ModuleList([]) )
            for j in range(self.nr_blocks_up_stage[i]):
                is_last_conv=j==self.nr_blocks_up_stage[i]-1 and i==self.nr_downsamples-1 #the last conv of the last upsample is followed by a slice and not a bn, therefore we need a bias
                if i>=self.nr_downsamples-self.nr_levels_up_with_normal_resnet:
                    print("adding up_resnet_block with nr of filters", cur_channels_count ) 
                    self.resnet_blocks_per_up_lvl_list[i].append( ResnetBlock(cur_channels_count, [1,1], [False,is_last_conv], False) )
                else:
                    print("adding up_bottleneck_block with nr of filters", cur_channels_count ) 
                    self.resnet_blocks_per_up_lvl_list[i].append( BottleneckBlock(cur_channels_count, [False,False,is_last_conv] ) )

        # self.slice_fast_cuda=SliceFastCUDALatticeModule(nr_classes=nr_classes, dropout_prob=dropout_last_layer, experiment=experiment)
        # self.slice=SliceLatticeModule()
        # self.classify=Conv1x1(out_channels=nr_classes, bias=True)
       
        self.logsoftmax=torch.nn.LogSoftmax(dim=1)


        if experiment!="none":
            warn="USING EXPERIMENT " + experiment
            print(colored("-------------------------------", 'yellow'))
            print(colored(warn, 'yellow'))
            print(colored("-------------------------------", 'yellow'))

    def forward(self, ls, positions, values):

        with torch.set_grad_enabled(False):
            ls, distributed, indices, weights=self.distribute(ls, positions, values)

        lv, ls=self.point_net(ls, distributed, indices)


        
        fine_structures_list=[]
        fine_values_list=[]
        # TIME_START("down_path")
        for i in range(self.nr_downsamples):

            #resnet blocks
            for j in range(self.nr_blocks_down_stage[i]):
                # print("start downsample stage ", i , " resnet block ", j, "lv has shape", lv.shape, " ls has val dim", ls.val_dim() )
                lv, ls = self.resnet_blocks_per_down_lvl_list[i][j] ( lv, ls) 

            #saving them for when we do finefy so we can concat them there
            fine_structures_list.append(ls) 
            fine_values_list.append(lv)

            #now we do a downsample
            # print("start coarsen stage ", i, "lv has shape", lv.shape, "ls has val_dim", ls.val_dim() )
            lv, ls = self.coarsens_list[i] ( lv, ls)
            # print( "finished coarsen stage ", i, "lv has shape", lv.shape, "ls has val_dim", ls.val_dim() )

        # TIME_END("down_path")

        # #bottleneck
        for j in range(self.nr_blocks_bottleneck):
            # print("bottleneck stage", j,  "lv has shape", lv.shape, "ls has val_dim", ls.val_dim()  )
            lv, ls = self.resnet_blocks_bottleneck[j] ( lv, ls) 


        #upsample (we start from the bottom of the U-net, so the upsampling that is closest to the blottlenck)
        # TIME_START("up_path")
        for i in range(self.nr_downsamples):

            fine_values=fine_values_list.pop()
            fine_structure=fine_structures_list.pop()


            #finefy
            # print("start finefy stage", i,  "lv has shape", lv.shape, "ls has val_dim ", ls.val_dim(),  "fine strcture has val dim ", fine_structure.val_dim() )
            lv, ls = self.finefy_list[i] ( lv, ls, fine_structure  )

            #concat or adding for the vertical connection
            if self.do_concat_for_vertical_connection: 
                lv=torch.cat((lv, fine_values ),1)
            else:
                lv+=fine_values

            #resnet blocks
            for j in range(self.nr_blocks_up_stage[i]):
                # print("start resnet block in upstage", i, "lv has shape", lv.shape, "ls has val dim" , ls.val_dim() )
                lv, ls = self.resnet_blocks_per_up_lvl_list[i][j] ( lv, ls) 
        # TIME_END("up_path")



        # sv =self.slice_fast_cuda(lv, ls, positions, indices, weights)
        # sv =self.slice(lv, ls, positions, indices, weights)
        # sv=self.classify(sv)


        # logsoftmax=self.logsoftmax(sv)


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
        cur_nr_channels+=128

        ### USE NO POSITIONS
        # cur_nr_channels=128
        # self.learned_pe_feat=LearnedPE(in_channels=cur_nr_channels, num_encoding_functions=4, logsampling=True)
        # cur_nr_channels=128 + 128*4*2    +256+3 #adding also the position encoded


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
        pos_enc=x
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



        ###DUMB
        # x=point_features
        # point_features_2d=point_features.view(-1,128)
        # feat_enc=self.learned_pe_feat( point_features_2d )
        # feat_enc=torch.cat([feat_enc,pos_enc],1)
        # x=feat_enc.view(height, width, nr_points, -1 )



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


#more similar to the pixelnerf https://arxiv.org/pdf/2012.02190v1.pdf
class SirenNetworkDirectPE_PixelNERF(MetaModule):
    def __init__(self, in_channels, out_channels):
        super(SirenNetworkDirectPE_PixelNERF, self).__init__()

        self.first_time=True

        self.nr_layers=4
        self.out_channels_per_layer=[128, 128, 128, 128, 128, 128]
     

        cur_nr_channels=in_channels
        self.net=torch.nn.ModuleList([])
        
        num_encodings=11
        self.learned_pe=LearnedPEGaussian(in_channels=in_channels, out_channels=256, std=5)
        cur_nr_channels=256+in_channels
        num_encoding_directions=4
        self.learned_pe_dirs=LearnedPE(in_channels=3, num_encoding_functions=num_encoding_directions, logsampling=True)
        dirs_channels=3+ 3*num_encoding_directions*2
        cur_nr_channels+=dirs_channels
        #now we concatenate the positions and directions and pass them through the initial conv
        self.first_conv=BlockSiren(activ=torch.relu, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[0], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, do_norm=False, with_dropout=False, transposed=False ).cuda()

        cur_nr_channels= self.out_channels_per_layer[0]

      

        for i in range(self.nr_layers):
            is_first_layer=i==0
            self.net.append( MetaSequential( ResnetBlockNerf(activ=torch.relu, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilations=[1,1], biases=[True,True], do_norm=False, with_dropout=False).cuda() ) )
           

        self.pred_sigma_and_rgb=MetaSequential(
            # BlockSiren(activ=torch.relu, in_channels=cur_nr_channels, out_channels=cur_nr_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            BlockSiren(activ=None, in_channels=cur_nr_channels, out_channels=4, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
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
        x=x.permute(2,3,0,1).contiguous() #from 71,107,30,3 to 30,3,71,107

        #also make the direcitons into image 
        ray_directions=ray_directions.view(-1,3)
        ray_directions=F.normalize(ray_directions, p=2, dim=1)
        ray_directions=self.learned_pe_dirs(ray_directions, params=get_subdict(params, 'learned_pe_dirs'))
        ray_directions=ray_directions.view(height, width, -1)
        ray_directions=ray_directions.permute(2,0,1).unsqueeze(0) #1,C,HW
        ray_directions=ray_directions.repeat(nr_points,1,1,1) #repeat as many times as samples that you have in a ray

        x=torch.cat([x,ray_directions],1) #nr_points, channels, height, width
        x=self.first_conv(x)


        #append to x the point_features
        point_features=point_features.view(height, width, nr_points, -1 )
        point_features=point_features.permute(2,3,0,1).contiguous() #N,C,H,W, where C is usually 128 or however big the feature vector is




        for i in range(len(self.net)):
            x=x+point_features
            x=self.net[i](x, params=get_subdict(params, 'net.'+str(i)  )  )

            # print("x has shape after resnet ", x.shape)

        sigma_and_rgb=self.pred_sigma_and_rgb(x)
        sigma_a=torch.relu( sigma_and_rgb[:,0:1, :, :] ) #first channel is the sigma
        rgb=torch.sigmoid(  sigma_and_rgb[:, 1:4, :, :]  )
        x=torch.cat([rgb, sigma_a],1)

        x=x.permute(2,3,0,1).contiguous() #from 30,nr_out_channels,71,107 to  71,107,30,4

       
        return  x



class SirenNetworkDirectPE_Simple(MetaModule):
    def __init__(self, in_channels, out_channels):
        super(SirenNetworkDirectPE_Simple, self).__init__()

        self.first_time=True

        self.nr_layers=5
        # self.out_channels_per_layer=[128, 128, 128, 128, 128, 128]
        # self.out_channels_per_layer=[96, 96, 96, 96, 96, 96]
        self.out_channels_per_layer=[32, 32, 32, 32, 32, 32]
        # self.out_channels_per_layer=[64, 64, 64, 64, 64, 64]
        # self.out_channels_per_layer=[32, 32, 32, 32, 32, 32]
     

        cur_nr_channels=in_channels
        self.net=torch.nn.ModuleList([])
        
        # gaussian encoding
        # self.learned_pe=LearnedPEGaussian(in_channels=in_channels, out_channels=256, std=5)
        # cur_nr_channels=256+in_channels
        #directional encoding runs way faster than the gaussian one and is free of thi std_dev hyperparameter which need to be finetuned depending on the scale of the scene
        num_encodings=11
        self.learned_pe=LearnedPE(in_channels=3, num_encoding_functions=num_encodings, logsampling=True)
        cur_nr_channels = in_channels + 3*num_encodings*2
        # #dir encoding
        # num_encoding_directions=4
        # self.learned_pe_dirs=LearnedPE(in_channels=3, num_encoding_functions=num_encoding_directions, logsampling=True)
        # dirs_channels=3+ 3*num_encoding_directions*2
        # cur_nr_channels+=dirs_channels

        #concat also the encoded features 
        # reduce_feat_channels=16
        # encoding_feat=6
        # self.conv_reduce_feat=BlockSiren(activ=torch.relu, in_channels=32, out_channels=reduce_feat_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, do_norm=False, with_dropout=False, transposed=False ).cuda()
        # self.learned_pe_features=LearnedPE(in_channels=reduce_feat_channels, num_encoding_functions=encoding_feat, logsampling=True)
        # feat_enc_channels=reduce_feat_channels+ reduce_feat_channels*encoding_feat*2
        # cur_nr_channels+=feat_enc_channels

        #now we concatenate the positions and directions and pass them through the initial conv
        self.first_conv=BlockSiren(activ=torch.relu, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[0], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, do_norm=False, with_dropout=False, transposed=False ).cuda()

        cur_nr_channels= self.out_channels_per_layer[0]

      

        for i in range(self.nr_layers):
            is_first_layer=i==0
            self.net.append( MetaSequential( BlockSiren(activ=torch.relu, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=is_first_layer).cuda() ) )
           

        # self.pred_sigma_and_rgb=MetaSequential(
        #     # BlockSiren(activ=torch.relu, in_channels=cur_nr_channels, out_channels=cur_nr_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     BlockSiren(activ=None, in_channels=cur_nr_channels, out_channels=4, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     )


        
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
        x=x.permute(2,3,0,1).contiguous() #from 71,107,30,3 to 30,3,71,107

        #also make the direcitons into image 
        ray_directions=ray_directions.view(-1,3)
        ray_directions=F.normalize(ray_directions, p=2, dim=1)
        ray_directions=self.learned_pe_dirs(ray_directions, params=get_subdict(params, 'learned_pe_dirs'))
        ray_directions=ray_directions.view(height, width, -1)
        ray_directions=ray_directions.permute(2,0,1).unsqueeze(0) #1,C,HW
        ray_directions=ray_directions.repeat(nr_points,1,1,1) #repeat as many times as samples that you have in a ray




        #skip to x the point_features
        if point_features!=None:
            point_features=point_features.view(height, width, nr_points, -1 )
            point_features=point_features.permute(2,3,0,1).contiguous() #N,C,H,W, where C is usually 128 or however big the feature vector is

            #concat also encoded features
            #THIS HELPS BUT ONLY IF WE USE LIKE 4-5 steps fo encoding, if we use the typical 11 as for the position then it gets unstable
            feat_reduce=self.conv_reduce_feat(point_features) #M x 8 x H xW
            feat_reduced_channels=feat_reduce.shape[1]
            feat_reduce=feat_reduce.permute(0,2,3,1).contiguous().view(-1,feat_reduced_channels)
            feat_enc=self.learned_pe_features(feat_reduce)
            feat_enc=feat_enc.view(nr_points, height, width, -1)
            feat_enc=feat_enc.permute(0,3,1,2).contiguous()
            x=torch.cat([x,feat_enc],1)


        # x=torch.cat([x,ray_directions],1) #nr_points, channels, height, width
        x=self.first_conv(x)

        for i in range(len(self.net)):
            if point_features!=None:
                x=x+point_features
            x=self.net[i](x, params=get_subdict(params, 'net.'+str(i)  )  )
            # x=x+point_features

            # print("x has shape after resnet ", x.shape)

        # sigma_and_rgb=self.pred_sigma_and_rgb(x)
        # sigma_a=torch.relu( sigma_and_rgb[:,0:1, :, :] ) #first channel is the sigma
        # rgb=torch.sigmoid(  sigma_and_rgb[:, 1:4, :, :]  )
        # x=torch.cat([rgb, sigma_a],1)

        # x=x.permute(2,3,0,1).contiguous() #from 30,nr_out_channels,71,107 to  71,107,30,4




        #predict the sigma and a feature vector for the rest of things
        sigma_and_feat=self.pred_sigma_and_feat(x,  params=get_subdict(params, 'pred_sigma_and_feat'))
        #get the feature vector for the rest of things and concat it with the direction
        sigma_a=torch.relu( sigma_and_feat[:,0:1, :, :] ) #first channel is the sigma
        feat=torch.relu( sigma_and_feat[:,1:sigma_and_feat.shape[1], :, : ] )
        # print("sigma and feat is ", sigma_and_feat.shape)
        # print(" feat is ", feat.shape)
        feat_and_dirs=torch.cat([feat, ray_directions], 1)
        #predict rgb
        # rgb=torch.sigmoid(  (self.pred_rgb(feat_and_dirs,  params=get_subdict(params, 'pred_rgb') ) +1.0)*0.5 )
        rgb=torch.sigmoid(  self.pred_rgb(feat_and_dirs,  params=get_subdict(params, 'pred_rgb') )  )
        #concat 
        # print("rgb is", rgb.shape)
        # print("sigma_a is", sigma_a.shape)
        x=torch.cat([rgb, sigma_a],1)

        x=x.permute(2,3,0,1).contiguous() #from 30,nr_out_channels,71,107 to  71,107,30,4


        
      



       
        return  x


class NERF_original(MetaModule):
    def __init__(self, in_channels, out_channels, use_ray_dirs):
        super(NERF_original, self).__init__()

        self.first_time=True

        self.nr_layers=4
        self.out_channels_per_layer=[128, 128, 128, 128, 128, 128]
        # self.out_channels_per_layer=[256, 256, 256, 256, 256, 256]
        # self.out_channels_per_layer=[512, 512, 512, 512, 512, 512]
        # self.out_channels_per_layer=[96, 96, 96, 96, 96, 96]
        # self.out_channels_per_layer=[32, 32, 32, 32, 32, 32]
        # self.out_channels_per_layer=[64, 64, 64, 64, 64, 64]
        # self.out_channels_per_layer=[32, 32, 32, 32, 32, 32]
        self.use_ray_dirs=use_ray_dirs
     

        cur_nr_channels=in_channels
        self.net=torch.nn.ModuleList([])
        
        # gaussian encoding
        self.learned_pe=LearnedPEGaussian(in_channels=in_channels, out_channels=256, std=8)
        cur_nr_channels=256+in_channels
        #directional encoding runs way faster than the gaussian one and is free of thi std_dev hyperparameter which need to be finetuned depending on the scale of the scene
        # num_encodings=8
        # self.learned_pe=LearnedPE(in_channels=3, num_encoding_functions=num_encodings, logsampling=True)
        # cur_nr_channels = in_channels + 3*num_encodings*2
        # #dir encoding
        # num_encoding_directions=4
        # self.learned_pe_dirs=LearnedPE(in_channels=3, num_encoding_functions=num_encoding_directions, logsampling=True)
        # dirs_channels=3+ 3*num_encoding_directions*2
        # cur_nr_channels+=dirs_channels

        #concat also the encoded features 
        # reduce_feat_channels=16
        # encoding_feat=6
        # self.conv_reduce_feat=BlockSiren(activ=torch.relu, in_channels=32, out_channels=reduce_feat_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, do_norm=False, with_dropout=False, transposed=False ).cuda()
        # self.learned_pe_features=LearnedPE(in_channels=reduce_feat_channels, num_encoding_functions=encoding_feat, logsampling=True)
        # feat_enc_channels=reduce_feat_channels+ reduce_feat_channels*encoding_feat*2
        # cur_nr_channels+=feat_enc_channels

        #now we concatenate the positions and directions and pass them through the initial conv
        # cur_nr_channels+=16
        self.first_conv=BlockNerf(activ=torch.relu, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[0],  bias=True).cuda()

        cur_nr_channels= self.out_channels_per_layer[0]

      

        for i in range(self.nr_layers):
            is_first_layer=i==0
            self.net.append( MetaSequential( BlockNerf(activ=torch.relu, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[i], bias=True).cuda(),
            # torch.nn.BatchNorm1d(self.out_channels_per_layer[i]), 
            ) )
           

        # self.pred_sigma_and_rgb=MetaSequential(
        #     # BlockSiren(activ=torch.relu, in_channels=cur_nr_channels, out_channels=cur_nr_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     BlockSiren(activ=None, in_channels=cur_nr_channels, out_channels=4, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
        #     )


        
        self.pred_sigma_and_feat=MetaSequential(
            # BlockSiren(activ=torch.relu, in_channels=cur_nr_channels, out_channels=cur_nr_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            BlockNerf(activ=None, in_channels=cur_nr_channels, out_channels=cur_nr_channels+1, bias=True).cuda(),
            )
        if use_ray_dirs:
            num_encoding_directions=4
            self.learned_pe_dirs=LearnedPE(in_channels=3, num_encoding_functions=num_encoding_directions, logsampling=True)
            dirs_channels=3+ 3*num_encoding_directions*2
            # new leaned pe with gaussian random weights as in  Fourier Features Let Networks Learn High Frequency 
            # self.learned_pe_dirs=LearnedPEGaussian(in_channels=in_channels, out_channels=64, std=10)
            # dirs_channels=64
            cur_nr_channels=cur_nr_channels+dirs_channels
        self.pred_rgb=MetaSequential( 
            BlockNerf(activ=torch.relu, in_channels=cur_nr_channels, out_channels=cur_nr_channels,  bias=True ).cuda(),
            BlockNerf(activ=None, in_channels=cur_nr_channels, out_channels=3,  bias=True ).cuda()    
            )






    def forward(self, x, ray_directions, point_features, nr_points_per_ray, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())


        if len(x.shape)!=2:
            print("Nerf_original forward: x should be a Nx3 matrix so 4 dimensions but it actually has ", x.shape, " so the lenght is ", len(x.shape))
            exit(1)

        # height=x.shape[0]
        # width=x.shape[1]
        # nr_points=x.shape[2]

        #from 71,107,30,3  to Nx3
        x=x.view(-1,3)
        x=self.learned_pe(x, params=get_subdict(params, 'learned_pe') )
        # x=x.view(height, width, nr_points, -1 )
        # x=x.permute(2,3,0,1).contiguous() #from 71,107,30,3 to 30,3,71,107

      



        # #skip to x the point_features
        # if point_features!=None:
        #     point_features=point_features.view(height, width, nr_points, -1 )
        #     point_features=point_features.permute(2,3,0,1).contiguous() #N,C,H,W, where C is usually 128 or however big the feature vector is

        #     #concat also encoded features
        #     #THIS HELPS BUT ONLY IF WE USE LIKE 4-5 steps fo encoding, if we use the typical 11 as for the position then it gets unstable
        #     feat_reduce=self.conv_reduce_feat(point_features) #M x 8 x H xW
        #     feat_reduced_channels=feat_reduce.shape[1]
        #     feat_reduce=feat_reduce.permute(0,2,3,1).contiguous().view(-1,feat_reduced_channels)
        #     feat_enc=self.learned_pe_features(feat_reduce)
        #     feat_enc=feat_enc.view(nr_points, height, width, -1)
        #     feat_enc=feat_enc.permute(0,3,1,2).contiguous()
        #     x=torch.cat([x,feat_enc],1)

        if point_features!=None:
            x=torch.cat([x, point_features],1)


        # x=torch.cat([x,ray_directions],1) #nr_points, channels, height, width
        x=self.first_conv(x)

        for i in range(len(self.net)):
            # if point_features!=None:
                # x=x+point_features
            # identity=x
            x=self.net[i](x, params=get_subdict(params, 'net.'+str(i)  )  )
            # x=x+point_features
            # x=x+identity

            # print("x has shape after resnet ", x.shape)

        # sigma_and_rgb=self.pred_sigma_and_rgb(x)
        # sigma_a=torch.relu( sigma_and_rgb[:,0:1, :, :] ) #first channel is the sigma
        # rgb=torch.sigmoid(  sigma_and_rgb[:, 1:4, :, :]  )
        # x=torch.cat([rgb, sigma_a],1)

        # x=x.permute(2,3,0,1).contiguous() #from 30,nr_out_channels,71,107 to  71,107,30,4




        #predict the sigma and a feature vector for the rest of things
        sigma_and_feat=self.pred_sigma_and_feat(x,  params=get_subdict(params, 'pred_sigma_and_feat'))
        #get the feature vector for the rest of things and concat it with the direction
        sigma_a=torch.relu( sigma_and_feat[:,0:1] ) #first channel is the sigma
        feat=torch.relu( sigma_and_feat[:,1:sigma_and_feat.shape[1] ] )
        # print("sigma and feat is ", sigma_and_feat.shape)
        # print(" feat is ", feat.shape)


        feat_and_dirs=feat
        if self.use_ray_dirs:
            #also make the direcitons into image 
            ray_directions=ray_directions.view(-1,3)
            ray_directions=F.normalize(ray_directions, p=2, dim=1)
            ray_directions=self.learned_pe_dirs(ray_directions, params=get_subdict(params, 'learned_pe_dirs'))
            # ray_directions=ray_directions.view(height, width, -1)
            # ray_directions=ray_directions.permute(2,0,1).unsqueeze(0) #1,C,HW
            # ray_directions=ray_directions.repeat(nr_points,1,1,1) #repeat as many times as samples that you have in a ray
            dim_ray_dir=ray_directions.shape[1]
            ray_directions=ray_directions.repeat(1, nr_points_per_ray) #repeat as many times as samples that you have in a ray
            ray_directions=ray_directions.view(-1,dim_ray_dir)
            feat_and_dirs=torch.cat([feat, ray_directions], 1)
        #predict rgb
        # rgb=torch.sigmoid(  (self.pred_rgb(feat_and_dirs,  params=get_subdict(params, 'pred_rgb') ) +1.0)*0.5 )
        rgb=torch.sigmoid(  self.pred_rgb(feat_and_dirs,  params=get_subdict(params, 'pred_rgb') )  )
        #concat 
        # print("rgb is", rgb.shape)
        # print("sigma_a is", sigma_a.shape)
        x=torch.cat([rgb, sigma_a],1)

        # x=x.permute(2,3,0,1).contiguous() #from 30,nr_out_channels,71,107 to  71,107,30,4


        
      



       
        return  x


class SIREN_original(MetaModule):
    def __init__(self, in_channels, out_channels, use_ray_dirs):
        super(SIREN_original, self).__init__()

        self.first_time=True

        self.nr_layers=4
        self.out_channels_per_layer=[128, 128, 128, 128, 128, 128]
        self.use_ray_dirs=use_ray_dirs
     

        cur_nr_channels=in_channels
        self.net=torch.nn.ModuleList([])
        

        #now we concatenate the positions and directions and pass them through the initial conv
        self.first_conv=BlockSiren(activ=torch.sin, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[0],  bias=True, is_first_layer=True).cuda()

        cur_nr_channels= self.out_channels_per_layer[0]

      

        for i in range(self.nr_layers):
            is_first_layer=i==0
            self.net.append( MetaSequential( BlockSiren(activ=torch.sin, in_channels=cur_nr_channels, out_channels=self.out_channels_per_layer[i], bias=True).cuda(),
            ) )
           

       
        
        self.pred_sigma_and_feat=MetaSequential(
            # BlockSiren(activ=torch.relu, in_channels=cur_nr_channels, out_channels=cur_nr_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=False, is_first_layer=False).cuda(),
            BlockSiren(activ=None, in_channels=cur_nr_channels, out_channels=cur_nr_channels+1, bias=True).cuda(),
            )
        if use_ray_dirs:
            num_encoding_directions=4
            self.learned_pe_dirs=LearnedPE(in_channels=3, num_encoding_functions=num_encoding_directions, logsampling=True)
            dirs_channels=3+ 3*num_encoding_directions*2
            # new leaned pe with gaussian random weights as in  Fourier Features Let Networks Learn High Frequency 
            # self.learned_pe_dirs=LearnedPEGaussian(in_channels=in_channels, out_channels=64, std=10)
            # dirs_channels=64
            cur_nr_channels=cur_nr_channels+dirs_channels
        cur_nr_channels=cur_nr_channels+32
        self.pred_rgb=MetaSequential( 
            BlockSiren(activ=torch.sin, in_channels=cur_nr_channels, out_channels=cur_nr_channels,  bias=True ).cuda(),
            BlockSiren(activ=None, in_channels=cur_nr_channels, out_channels=3,  bias=True ).cuda()    
            )






    def forward(self, x, ray_directions, point_features, nr_points_per_ray, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())


        if len(x.shape)!=2:
            print("Nerf_original forward: x should be a Nx3 matrix so 4 dimensions but it actually has ", x.shape, " so the lenght is ", len(x.shape))
            exit(1)

        # height=x.shape[0]
        # width=x.shape[1]
        # nr_points=x.shape[2]

        #from 71,107,30,3  to Nx3
        x=x.view(-1,3)
        # x=self.learned_pe(x, params=get_subdict(params, 'learned_pe') )
        # x=x.view(height, width, nr_points, -1 )
        # x=x.permute(2,3,0,1).contiguous() #from 71,107,30,3 to 30,3,71,107

      



        # #skip to x the point_features
        # if point_features!=None:
        #     point_features=point_features.view(height, width, nr_points, -1 )
        #     point_features=point_features.permute(2,3,0,1).contiguous() #N,C,H,W, where C is usually 128 or however big the feature vector is

        #     #concat also encoded features
        #     #THIS HELPS BUT ONLY IF WE USE LIKE 4-5 steps fo encoding, if we use the typical 11 as for the position then it gets unstable
        #     feat_reduce=self.conv_reduce_feat(point_features) #M x 8 x H xW
        #     feat_reduced_channels=feat_reduce.shape[1]
        #     feat_reduce=feat_reduce.permute(0,2,3,1).contiguous().view(-1,feat_reduced_channels)
        #     feat_enc=self.learned_pe_features(feat_reduce)
        #     feat_enc=feat_enc.view(nr_points, height, width, -1)
        #     feat_enc=feat_enc.permute(0,3,1,2).contiguous()
        #     x=torch.cat([x,feat_enc],1)

        # if point_features!=None:
        #     x=torch.cat([x, point_features],1)


        # x=torch.cat([x,ray_directions],1) #nr_points, channels, height, width
        x=self.first_conv(x)

        for i in range(len(self.net)):
            # if point_features!=None:
                # x=x+point_features
            x=self.net[i](x, params=get_subdict(params, 'net.'+str(i)  )  )



        #predict the sigma and a feature vector for the rest of things
        sigma_and_feat=self.pred_sigma_and_feat(x,  params=get_subdict(params, 'pred_sigma_and_feat'))
        #get the feature vector for the rest of things and concat it with the direction
        sigma_a=torch.relu( sigma_and_feat[:,0:1] ) #first channel is the sigma
        feat=torch.sin( sigma_and_feat[:,1:sigma_and_feat.shape[1] ] )
        # print("sigma and feat is ", sigma_and_feat.shape)
        # print(" feat is ", feat.shape)


        feat_and_dirs=feat
        if self.use_ray_dirs:
            #also make the direcitons into image 
            ray_directions=ray_directions.view(-1,3)
            ray_directions=F.normalize(ray_directions, p=2, dim=1)
            ray_directions=self.learned_pe_dirs(ray_directions, params=get_subdict(params, 'learned_pe_dirs'))
            # ray_directions=ray_directions.view(height, width, -1)
            # ray_directions=ray_directions.permute(2,0,1).unsqueeze(0) #1,C,HW
            # ray_directions=ray_directions.repeat(nr_points,1,1,1) #repeat as many times as samples that you have in a ray
            dim_ray_dir=ray_directions.shape[1]
            ray_directions=ray_directions.repeat(1, nr_points_per_ray) #repeat as many times as samples that you have in a ray
            ray_directions=ray_directions.view(-1,dim_ray_dir)
            feat_and_dirs=torch.cat([feat, ray_directions], 1)

        if point_features!=None:
            feat_and_dirs=torch.cat([feat_and_dirs, point_features],1)
        
        #predict rgb
        # rgb=torch.sigmoid(  (self.pred_rgb(feat_and_dirs,  params=get_subdict(params, 'pred_rgb') ) +1.0)*0.5 )
        rgb=torch.sigmoid(  self.pred_rgb(feat_and_dirs,  params=get_subdict(params, 'pred_rgb') )  )
        #concat 
        # print("rgb is", rgb.shape)
        # print("sigma_a is", sigma_a.shape)
        x=torch.cat([rgb, sigma_a],1)

        # x=x.permute(2,3,0,1).contiguous() #from 30,nr_out_channels,71,107 to  71,107,30,4


        
      



       
        return  x


class RGB_predictor_simple(MetaModule):
    def __init__(self, in_channels, out_channels, use_ray_dirs):
        super(RGB_predictor_simple, self).__init__()

        self.first_time=True

        self.use_ray_dirs = use_ray_dirs

        cur_nr_channels=in_channels
        self.learned_pe=LearnedPEGaussian(in_channels=in_channels, out_channels=256, std=9)
        cur_nr_channels=256+in_channels
        ###GAUSSIAN encoding works a tiny bit better but I prefer the normal positonal encoding because I don't have to deal witht he std paramter which is a bit sensitive
        # num_encodings=8
        # self.learned_pe=LearnedPE(in_channels=3, num_encoding_functions=num_encodings, logsampling=True)
        # cur_nr_channels = in_channels + 3*num_encodings*2


        # self.first_conv=BlockSiren(activ=torch.sin, in_channels=3, out_channels=128,  bias=True, is_first_layer=True).cuda()

        cur_nr_channels+=64 #for point features
        # cur_nr_channels+=3+ 3*4*2 #for the dirs of the neighbourin
        # cur_nr_channels+=3+ 3*4*2 #for the dirs of the neighbourin
        # cur_nr_channels+=32 #concating also the signed distnace
        # cur_nr_channels+=3 #concating also the normals
      
        if use_ray_dirs:
            num_encoding_directions=4
            self.learned_pe_dirs=LearnedPE(in_channels=3, num_encoding_functions=num_encoding_directions, logsampling=True)
            dirs_channels=3+ 3*num_encoding_directions*2
            # new leaned pe with gaussian random weights as in  Fourier Features Let Networks Learn High Frequency 
            # self.learned_pe_dirs=LearnedPEGaussian(in_channels=in_channels, out_channels=64, std=10)
            # dirs_channels=64
            # cur_nr_channels+=128 #point
            cur_nr_channels+=dirs_channels
        self.pred_feat=MetaSequential( 
            # torch.nn.GroupNorm( int(cur_nr_channels/2), cur_nr_channels).cuda(),
            BlockNerf(activ=torch.nn.GELU(), in_channels=cur_nr_channels, out_channels=64,  bias=True ).cuda(),
            # torch.nn.GroupNorm( int(cur_nr_channels/2), cur_nr_channels).cuda(),
            BlockNerf(activ=torch.nn.GELU(), in_channels=64, out_channels=64,  bias=True ).cuda()
            )
        self.pred_rgb=BlockNerf(activ=None, init="tanh", in_channels=64, out_channels=3,  bias=True ).cuda()    
        self.pred_mask=BlockNerf(activ=torch.sigmoid,  in_channels=64, out_channels=1,  bias=True ).cuda()    






    def forward(self, x, ray_directions, point_features, nr_points_per_ray, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())


        if len(x.shape)!=2:
            print("Nerf_original forward: x should be a Nx3 matrix so 4 dimensions but it actually has ", x.shape, " so the lenght is ", len(x.shape))
            exit(1)

        # height=x.shape[0]
        # width=x.shape[1]
        # nr_points=x.shape[2]

        #from 71,107,30,3  to Nx3
        x=x.view(-1,3)
        x=self.learned_pe(x, params=get_subdict(params, 'learned_pe') )
        # x=x.view(height, width, nr_points, -1 )
        # x=x.permute(2,3,0,1).contiguous() #from 71,107,30,3 to 30,3,71,107

      


        x=torch.cat([x,point_features],1)

  
        if self.use_ray_dirs:
            #also make the direcitons into image 
            ray_directions=ray_directions.view(-1,3)
            ray_directions=F.normalize(ray_directions, p=2, dim=1)
            ray_directions=self.learned_pe_dirs(ray_directions, params=get_subdict(params, 'learned_pe_dirs'))
            # ray_directions=ray_directions.view(height, width, -1)
            # ray_directions=ray_directions.permute(2,0,1).unsqueeze(0) #1,C,HW
            # ray_directions=ray_directions.repeat(nr_points,1,1,1) #repeat as many times as samples that you have in a ray
            dim_ray_dir=ray_directions.shape[1]
            ray_directions=ray_directions.repeat(1, nr_points_per_ray) #repeat as many times as samples that you have in a ray
            ray_directions=ray_directions.view(-1,dim_ray_dir)
            # print("x has mean and std ", x.mean(), " and std ", x.std(), " ray dirs has mean ", ray_directions.mean(), " and std ", ray_directions.std() )
            x=torch.cat([x, ray_directions], 1)

        # if point_features!=None:
            # x=torch.cat([x, point_features],1)
        
        #predict rgb
        # rgb=torch.sigmoid(  (self.pred_rgb(feat_and_dirs,  params=get_subdict(params, 'pred_rgb') ) +1.0)*0.5 )
        # x=torch.sigmoid(  self.pred_rgb(x,  params=get_subdict(params, 'pred_rgb') )  )
        x=  self.pred_feat(x,  params=get_subdict(params, 'pred_feat') )  
        last_features=x
        rgb=  self.pred_rgb(last_features,  params=get_subdict(params, 'pred_rgb') )  
        mask_pred=  self.pred_mask(last_features,  params=get_subdict(params, 'pred_mask') )  
        #concat 
        # print("rgb is", rgb.shape)
        # print("sigma_a is", sigma_a.shape)

        # x=x.permute(2,3,0,1).contiguous() #from 30,nr_out_channels,71,107 to  71,107,30,4


        # print("rgb is ", rgb.mean(), rgb.min(), rgb.max() ) 
      



       
        return  rgb, last_features, mask_pred




#Compute a feature vector given a 3D position. Similar to the function phi from scene representation network
class VolumetricFeature(MetaModule):
    def __init__(self, in_channels, out_channels, nr_layers, hidden_size, use_dirs):
        super(VolumetricFeature, self).__init__()

        self.first_time=True

        self.nr_layers=nr_layers
        self.hidden_size=hidden_size
        self.use_dirs=use_dirs
        

        cur_nr_channels=in_channels
        self.net=torch.nn.ModuleList([])
        
        # gaussian encoding
        self.learned_pe=LearnedPEGaussian(in_channels=in_channels, out_channels=256, std=8.0)
        cur_nr_channels=256+in_channels
        #directional encoding runs way faster than the gaussian one and is free of thi std_dev hyperparameter which need to be finetuned depending on the scale of the scene
        # num_encodings=8
        # self.learned_pe=LearnedPE(in_channels=3, num_encoding_functions=num_encodings, logsampling=True)
        # cur_nr_channels = in_channels + 3*num_encodings*2
   
        #now we concatenate the positions and directions and pass them through the initial conv
        self.first_conv=BlockNerf(activ=torch.relu, in_channels=cur_nr_channels, out_channels=hidden_size,  bias=True).cuda()
        cur_nr_channels= hidden_size

      

        for i in range(self.nr_layers):
            self.net.append( MetaSequential( BlockNerf(activ=torch.relu, in_channels=cur_nr_channels, out_channels=hidden_size, bias=True).cuda()
            ) )
           
        if use_dirs:
            num_encoding_directions=4
            self.learned_pe_dirs=LearnedPE(in_channels=3, num_encoding_functions=num_encoding_directions, logsampling=True)
            dirs_channels=3+ 3*num_encoding_directions*2
            cur_nr_channels+=dirs_channels

        #get the features until now, concat the directions if needed and computes the final feature 
        self.pred_feat=MetaSequential( 
            BlockNerf(activ=torch.relu, in_channels=cur_nr_channels, out_channels=hidden_size,  bias=True ).cuda(),
            BlockNerf(activ=None, in_channels=hidden_size, out_channels=out_channels,  bias=True ).cuda()    
            )



    def forward(self, x, ray_directions, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())


        if len(x.shape)!=2:
            print("Nerf_original forward: x should be a Nx3 matrix so 4 dimensions but it actually has ", x.shape, " so the lenght is ", len(x.shape))
            exit(1)


        #from 71,107,30,3  to Nx3
        x=x.view(-1,3)
        x=self.learned_pe(x, params=get_subdict(params, 'learned_pe') )

        x=self.first_conv(x)

        for i in range(len(self.net)):
            x=self.net[i](x, params=get_subdict(params, 'net.'+str(i)  )  )
            

        #if we use direction, we also concat those 
        if self.use_dirs:
            #also make the direcitons into image 
            ray_directions=ray_directions.view(-1,3)
            ray_directions=F.normalize(ray_directions, p=2, dim=1)
            ray_directions=self.learned_pe_dirs(ray_directions, params=get_subdict(params, 'learned_pe_dirs'))
            x=torch.cat([x, ray_directions])

        x=self.pred_feat(x)

        return  x


class VolumetricFeatureSiren(MetaModule):
    def __init__(self, in_channels, out_channels, nr_layers, hidden_size, use_dirs):
        super(VolumetricFeatureSiren, self).__init__()

        self.first_time=True

        self.nr_layers=nr_layers
        self.hidden_size=hidden_size
        self.use_dirs=use_dirs
        

        cur_nr_channels=in_channels
        self.net=torch.nn.ModuleList([])
        
        # gaussian encoding
       
   
        #now we concatenate the positions and directions and pass them through the initial conv
        self.first_conv=BlockSiren(activ=torch.sin, in_channels=cur_nr_channels, out_channels=hidden_size,  bias=True, is_first_layer=True).cuda()
        cur_nr_channels= hidden_size

      

        for i in range(self.nr_layers):
            self.net.append( MetaSequential( BlockSiren(activ=torch.sin, in_channels=cur_nr_channels, out_channels=hidden_size, bias=True).cuda()
            ) )
           
        if use_dirs:
            num_encoding_directions=4
            self.learned_pe_dirs=LearnedPE(in_channels=3, num_encoding_functions=num_encoding_directions, logsampling=True)
            dirs_channels=3+ 3*num_encoding_directions*2
            cur_nr_channels+=dirs_channels

        #get the features until now, concat the directions if needed and computes the final feature 
        self.pred_feat=MetaSequential( 
            BlockSiren(activ=torch.sin, in_channels=cur_nr_channels, out_channels=hidden_size,  bias=True ).cuda(),
            BlockSiren(activ=None, in_channels=hidden_size, out_channels=out_channels,  bias=True ).cuda()    
            )



    def forward(self, x, ray_directions, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())


        if len(x.shape)!=2:
            print("Nerf_original forward: x should be a Nx3 matrix so 4 dimensions but it actually has ", x.shape, " so the lenght is ", len(x.shape))
            exit(1)


        #from 71,107,30,3  to Nx3
        x=x.view(-1,3)
        # x=self.learned_pe(x, params=get_subdict(params, 'learned_pe') )

        x=self.first_conv(x)

        for i in range(len(self.net)):
            x=self.net[i](x, params=get_subdict(params, 'net.'+str(i)  )  )
            

        #if we use direction, we also concat those 
        if self.use_dirs:
            #also make the direcitons into image 
            ray_directions=ray_directions.view(-1,3)
            ray_directions=F.normalize(ray_directions, p=2, dim=1)
            ray_directions=self.learned_pe_dirs(ray_directions, params=get_subdict(params, 'learned_pe_dirs'))
            x=torch.cat([x, ray_directions])

        x=self.pred_feat(x)

        return  x





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


#inspired from the ray marcher from https://github.com/vsitzmann/scene-representation-networks/blob/master/custom_layers.py
class DifferentiableRayMarcher(torch.nn.Module):
    def __init__(self):
        super(DifferentiableRayMarcher, self).__init__()

        num_encodings=8
        self.learned_pe=LearnedPE(in_channels=3, num_encoding_functions=num_encodings, logsampling=True)
        # self.learned_pe=TorchScriptTraceWrapper( LearnedPE(in_channels=3, num_encoding_functions=num_encodings, logsampling=True) )
        # cur_nr_channels = in_channels + 3*num_encodings*2

        #model 
        self.lstm_hidden_size = 16
        self.lstm=None #Create this later, the volumentric feature can maybe change and therefore the features that get as input to the lstm will be different
        # self.out_layer = torch.nn.Linear(self.lstm_hidden_size, 1)
        self.out_layer = BlockNerf(activ=None, in_channels=self.lstm_hidden_size, out_channels=1,  bias=True ).cuda()
        # self.feature_computer= VolumetricFeature(in_channels=3, out_channels=64, nr_layers=2, hidden_size=64, use_dirs=False) 
        # self.feature_computer= VolumetricFeatureSiren(in_channels=3, out_channels=64, nr_layers=2, hidden_size=64, use_dirs=False) 
        # self.frame_weights_computer= FrameWeightComputer()
        # self.feature_aggregator=  FeatureAgregator() 
        # self.feature_aggregator=  FeatureAgregatorLinear() 
        # self.feature_aggregator=  FeatureAgregatorInvariant()  #loss is lower than FeatureAgregatorLinear but the normal map looks worse and more noisy
        self.feature_aggregator_traced=None
        self.slice_texture= SliceTextureModule()
        self.splat_texture= SplatTextureModule()
  
        self.feature_fuser = torch.nn.Sequential(
            BlockNerf(activ=torch.nn.GELU(), in_channels=3+3*num_encodings*2  +64, out_channels=32,  bias=True ).cuda(),
            # BlockNerf(activ=torch.nn.GELU(), in_channels=64, out_channels=64,  bias=True ).cuda(),
            # BlockNerf(activ=None, in_channels=64, out_channels=64,  bias=True ).cuda(),
        )

        self.concat_coord=ConcatCoord()

        #activ
        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()

        #params 
        self.nr_iters=10

        #starting depth per pixels 
        self.depth_per_pixel_train=None
        self.depth_per_pixel_test=None

      
    def forward(self, frame, ray_dirs, depth_min, frames_close, frames_features, weights, pixels_indices, novel=False):


        if novel:
            depth_per_pixel= torch.ones([frame.height*frame.width,1], dtype=torch.float32, device=torch.device("cuda")) 
            depth_per_pixel.fill_(depth_min/2.0)   #randomize the deptha  bith
        else:
            depth_per_pixel = torch.zeros((frame.height*frame.width, 1), device=torch.device("cuda") ).normal_(mean=depth_min, std=2e-2)

        #Select only certain pixels fro the image
        if pixels_indices is not None:
            depth_per_pixel = torch.index_select(depth_per_pixel, 0, pixels_indices)

        # #create the tensor if it's necesary #BUG FOR SOME REASON this makes the loss converge slower or not at all. so something is bugged here 
        # if self.depth_per_pixel_train is None and not novel:
        #     self.depth_per_pixel_train = torch.zeros((frame.height*frame.width, 1), device=torch.device("cuda") ).normal_(mean=depth_min, std=2e-2)
        # if self.depth_per_pixel_test is None and novel:
        #     self.depth_per_pixel_test= torch.ones([frame.height*frame.width,1], dtype=torch.float32, device=torch.device("cuda")) 
        #     self.depth_per_pixel_test.fill_(depth_min/2.0)   #randomize the deptha  bith
        # #set the depth per pixel to whichever tensor we need 
        # if novel:
        #     depth_per_pixel=self.depth_per_pixel_test.detach()
        # else:
        #     depth_per_pixel=self.depth_per_pixel_train.detach().normal_(mean=depth_min, std=2e-2)


        # depth_per_pixel.fill_(0.5)   #randomize the deptha  bith

        nr_nearby_frames=len(frames_close)
        R_list=[]
        t_list=[]
        K_list=[]
        for i in range(len(frames_close)):
            frame_selectd=frames_close[i]
            R_list.append( frame_selectd.R_tensor.view(1,3,3) )
            t_list.append( frame_selectd.t_tensor.view(1,1,3) )
            K_list.append( frame_selectd.K_tensor.view(1,3,3) )
        R_batched=torch.cat(R_list,0)
        t_batched=torch.cat(t_list,0)
        K_batched=torch.cat(K_list,0)
        #####when we project we assume all the frames have the same size
        height=frames_close[0].height
        width=frames_close[0].width


        #Ray direction in world coordinates
        # ray_dirs=torch.from_numpy(frame.ray_dirs).to("cuda").float()


        #attempt 2 unproject to 3D 
        camera_center=torch.from_numpy( frame.frame.pos_in_world() ).to("cuda")
        camera_center=camera_center.view(1,3)
        points3D = camera_center + depth_per_pixel*ray_dirs
        # show_3D_points(points3D, "points_3d_init")

        init_world_coords=points3D
        initial_depth=depth_per_pixel
        world_coords = [init_world_coords]
        depths = [initial_depth]
        signed_distances_for_marchlvl=[]
        states = [None]

        # weights=self.frame_weights_computer(frame, frames_close)

        #new loss that makes the std for the last iter and the other iter to be high
        new_loss=0.0

        # print("")
        for iter_nr in range(self.nr_iters):
            # print("iter is ", iter_nr)
            TIME_START("raymarch_iter")
        
            #compute the features at this position 
            # feat=self.feature_computer(points3D, ray_dirs) #a tensor of N x feat_size which contains for each position in 3D a feature representation around that point. Similar to phi from SRN
            # feat=self.feature_computer(world_coords[-1], ray_dirs) #a tensor of N x feat_size which contains for each position in 3D a feature representation around that point. Similar to phi from SRN
            TIME_START("raymarch_pe")
            feat=self.learned_pe(world_coords[-1])
            TIME_END("raymarch_pe")

            # TIME_START("raymarch_uv")
            TIME_START("rm_get_and_aggr")
            uv_tensor=compute_uv_batched(R_batched, t_batched, K_batched, height, width, world_coords[-1] )
            # TIME_END("raymarch_uv")


            # slice with grid_sample
            # TIME_START("raymarch_slice")
            uv_tensor=uv_tensor.view(nr_nearby_frames, -1, 1, 2) #Nr_framex x nr_pixels_cur_frame x 1 x 2
            sliced_feat_batched=torch.nn.functional.grid_sample( frames_features, uv_tensor, align_corners=False, mode="bilinear" ) #sliced features is N,C,H,W
            feat_dim=sliced_feat_batched.shape[1]
            sliced_feat_batched=sliced_feat_batched.permute(0,2,3,1) # from N,C,H,W to N,H,W,C
            sliced_feat_batched=sliced_feat_batched.view(len(frames_close), -1, feat_dim) #make it nr_frames x nr_pixels x FEATDIM
            # TIME_END("raymarch_slice")
           
            
            #attempt 2 
            # TIME_START("raymarch_aggr")
            # weights_one= torch.ones([3,1], dtype=torch.float32, device=torch.device("cuda"))  #the features shount not be weighted here because we want to match completely between the 3 images
            # img_features_aggregated= self.feature_aggregator(sliced_feat_batched, weights, novel)
            mean=sliced_feat_batched.mean(dim=0)
            std=sliced_feat_batched.std(dim=0)
            img_features_aggregated=torch.cat([mean,std],1)
            # TIME_END("raymarch_aggr")
            TIME_END("rm_get_and_aggr")

            # #get std for each lvl 
            # std=sliced_feat_batched.std(dim=0).mean()
            # # print("std for lvl ", iter_nr, " std is ", std.item())
            # # new loss that makes the std for the last iter and the other iter to be high
            # if iter_nr == self.nr_iters-1: #last iter
            #     new_loss+=std
            # if iter_nr == self.nr_iters-2:
            #     new_loss-=torch.clamp(std, 0.0, 1.0)
            
            

            
            TIME_START("raymarch_fuse")
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            feat=torch.cat([feat,img_features_aggregated],1)
            feat=self.feature_fuser(feat)
            # print(prof)
            TIME_END("raymarch_fuse")


            #create the lstm if not created 
            TIME_START("raymarch_lstm")
            if self.lstm==None:
                self.lstm = torch.nn.LSTMCell(input_size=feat.shape[1], hidden_size=self.lstm_hidden_size ).to("cuda")
                self.lstm.apply(init_recurrent_weights)
                lstm_forget_gate_init(self.lstm)
                # self.lstm = torch.nn.Sequential(
                #     BlockNerf(activ=torch.nn.GELU(), in_channels=feat.shape[1], out_channels=32,  bias=True ).cuda(),
                #     BlockNerf(activ=torch.nn.GELU(), in_channels=32, out_channels=32,  bias=True ).cuda(),
                #     BlockNerf(activ=None, in_channels=32, out_channels=16,  bias=True ).cuda(),
                # )

            #run through the lstm
            state = self.lstm(feat, states[-1])
            # state = self.lstm(feat)
            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-10, max=10))
            # if state.requires_grad:
                # state.register_hook(lambda x: x.clamp(min=-10, max=10))

            signed_distance= self.out_layer(state[0])
            # signed_distance= self.out_layer(state)
            TIME_END("raymarch_lstm")
            # print("signed_distance iter", iter_nr, " is ", signed_distance.mean())
            # signed_distance=torch.abs(signed_distance) #the distance only increases
            #the output of the lstm after abs will probably be on average around 0.5 (because before the abs it was zero meaned and kinda spread around [-1,1])
            # however, doing nr_steps*0.5 will likely put the depth above the scene scale which is normally 1.0
            # therefore we expect each step to be 1.0/nr_steps so for 10 steps each steps should to 0.1
            # print("sined dist min for iter i", iter_nr, " ", signed_distance.min().item(), " max ", signed_distance.max().item() )
            depth_scaling=1.0/(1.0*self.nr_iters) #1.0 is the scene scale and we expect on average that every step will do a movement of 0.5, maybe the average movement is more like 0.25 idunno
            signed_distance=signed_distance*depth_scaling
            # print("signed_distance iter", iter_nr, " is ", signed_distance.mean())
            

            new_world_coords = world_coords[-1] + ray_dirs * signed_distance
            states.append(state)
            world_coords.append(new_world_coords)
            signed_distances_for_marchlvl.append(signed_distance)

            # if iter_nr==self.nr_iters-1:
                # show_3D_points(new_world_coords, "points_3d_"+str(iter_nr))
            TIME_END("raymarch_iter")

        #get the depth at this final 3d position
        depth= (new_world_coords-camera_center).norm(dim=1, keepdim=True)

        #return also the world coords at every march
        world_coords.pop(0)

        return new_world_coords, depth, world_coords, signed_distances_for_marchlvl, new_loss


class DifferentiableRayMarcherHierarchical(torch.nn.Module):
    def __init__(self):
        super(DifferentiableRayMarcherHierarchical, self).__init__()

        num_encodings=8
        self.learned_pe=LearnedPE(in_channels=3, num_encoding_functions=num_encodings, logsampling=True)
        # cur_nr_channels = in_channels + 3*num_encodings*2

        #model 
        self.lstm_hidden_size = 16
        self.lstm=None #Create this later, the volumentric feature can maybe change and therefore the features that get as input to the lstm will be different
        # self.out_layer = torch.nn.Linear(self.lstm_hidden_size, 1)
        self.out_layer = BlockNerf(activ=None, in_channels=self.lstm_hidden_size, out_channels=1,  bias=True ).cuda()
        self.frame_weights_computer= FrameWeightComputer()
        self.feature_aggregator= FeatureAgregator()
        self.slice_texture= SliceTextureModule()
      
        self.feature_fuser = torch.nn.Sequential(
            BlockNerf(activ=torch.nn.GELU(), in_channels=3+3*num_encodings*2  +32, out_channels=64,  bias=True ).cuda(),
            BlockNerf(activ=torch.nn.GELU(), in_channels=64, out_channels=64,  bias=True ).cuda(),
            BlockNerf(activ=None, in_channels=64, out_channels=64,  bias=True ).cuda(),
        )

        # self.feature_fuser = torch.nn.Sequential(
        #     torch.nn.Conv2d(3+3*num_encodings*2  +32, 64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda(),
        #     torch.nn.GELU(),
        #     torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda(),
        # )
        

        #activ
        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()

        #params 
        self.nr_iters=2
        self.nr_resolutions=3

      
    def forward(self, frame, depth_min, frames_close, frames_features, novel=False):

        
        #make a series of frames subsaqmpled
        frames_subsampled=[]
        frames_subsampled.append(frame)
        for i in range(self.nr_resolutions):
            frames_subsampled.append(frame.subsampled_frames[i])
        frames_subsampled.reverse()

        #debug the frames
        # for i in range(len(frames_subsampled)):
            # frame_subsampled=frames_subsampled[i]
            # print("frames is ", frame_subsampled.height)


        #go from each level of the hierarchy and ray march from there 
        for res_iter in range(len(frames_subsampled)):
            frame_subsampled=frames_subsampled[res_iter]

            # print("res iter", res_iter)
            # print("height and width is ", frame_subsampled.height, " ", frame_subsampled.width)

            if res_iter==0:
                if novel:
                    depth_per_pixel= torch.ones([frame_subsampled.height*frame_subsampled.width,1], dtype=torch.float32, device=torch.device("cuda")) 
                    depth_per_pixel.fill_(depth_min/2.0)   #randomize the deptha  bith
                else:
                    depth_per_pixel = torch.zeros((frame_subsampled.height*frame_subsampled.width, 1), device=torch.device("cuda") ).normal_(mean=depth_min, std=2e-2)
            else: 
                ## if any other level above the coarsest one, then we upsample the depth using nerest neighbour
                depth_per_pixel =depth.view( 1,1, frames_subsampled[res_iter-1].height, frames_subsampled[res_iter-1].width ) #make it into N,C,H,W
                # depth_per_pixel = torch.nn.functional.interpolate(depth_per_pixel ,size=(frame_subsampled.height, frame_subsampled.width ), mode='nearest')
                depth_per_pixel = torch.nn.functional.interpolate(depth_per_pixel ,size=(frame_subsampled.height, frame_subsampled.width ), mode='bilinear')
                depth_per_pixel=depth_per_pixel.view(frame_subsampled.height*frame_subsampled.width, 1) #make it into Nx1

            ray_dirs=frame_subsampled.ray_dirs

            # print("ray_dirs is ", ray_dirs.shape)

            #attempt 2 unproject to 3D 
            camera_center=torch.from_numpy( frame_subsampled.frame.pos_in_world() ).to("cuda")
            camera_center=camera_center.view(1,3)
            points3D = camera_center + depth_per_pixel*ray_dirs

            #start runnign marches 
            init_world_coords=points3D
            initial_depth=depth_per_pixel
            world_coords = [init_world_coords]
            depths = [initial_depth]
            signed_distances_for_marchlvl=[]
            states = [None]


            weights=self.frame_weights_computer(frame, frames_close)

            for iter_nr in range(self.nr_iters):
                TIME_START("raymarch_iter")
            
                #compute the features at this position 
                feat=self.learned_pe(world_coords[-1])

                uv_tensor=compute_uv_batched(frames_close, world_coords[-1] )


                # slice with grid_sample
                uv_tensor=uv_tensor.view(-1, frame_subsampled.height, frame_subsampled.width, 2)
                sliced_feat_batched=torch.nn.functional.grid_sample( frames_features, uv_tensor, align_corners=False ) #sliced features is N,C,H,W
                feat_dim=sliced_feat_batched.shape[1]
                sliced_feat_batched=sliced_feat_batched.permute(0,2,3,1) # from N,C,H,W to N,H,W,C
                sliced_feat_batched=sliced_feat_batched.view(len(frames_close), -1, feat_dim) #make it 1 x N x FEATDIM
            
                
                #attempt 2 
                TIME_START("raymarch_aggr")
                img_features_aggregated= self.feature_aggregator(frame, frames_close, sliced_feat_batched, weights)

                feat=torch.cat([feat,img_features_aggregated],1)

                feat=self.feature_fuser(feat)
                TIME_END("raymarch_aggr")


                #create the lstm if not created 
                TIME_START("raymarch_lstm")
                if self.lstm==None:
                    self.lstm = torch.nn.LSTMCell(input_size=feat.shape[1], hidden_size=self.lstm_hidden_size ).to("cuda")
                    self.lstm.apply(init_recurrent_weights)
                    lstm_forget_gate_init(self.lstm)

                #run through the lstm
                state = self.lstm(feat, states[-1])
                if state[0].requires_grad:
                    state[0].register_hook(lambda x: x.clamp(min=-10, max=10))

                signed_distance= self.out_layer(state[0])
                TIME_END("raymarch_lstm")
                # print("signed_distance iter", iter_nr, " is ", signed_distance.mean())
                signed_distance=torch.abs(signed_distance) #the distance only increases
                #the output of the lstm after abs will probably be on average around 0.5 (because before the abs it was zero meaned and kinda spread around [-1,1])
                # however, doing nr_steps*0.5 will likely put the depth above the scene scale which is normally 1.0
                # therefore we expect each step to be 1.0/nr_steps so for 10 steps each steps should to 0.1
                depth_scaling=1.0/(1.0*self.nr_iters*self.nr_resolutions) #1.0 is the scene scale and we expect on average that every step will do a movement of 0.5, maybe the average movement is more like 0.25 idunno
                signed_distance=signed_distance*depth_scaling
                # print("signed_distance iter", iter_nr, " is ", signed_distance.mean())
                


                new_world_coords = world_coords[-1] + ray_dirs * signed_distance
                states.append(state)
                world_coords.append(new_world_coords)
                signed_distances_for_marchlvl.append(signed_distance)

                # if iter_nr==self.nr_iters-1:
                    # show_3D_points(new_world_coords, "points_3d_"+str(res_iter))



            #get the depth at this final 3d position
            depth= (new_world_coords-camera_center).norm(dim=1, keepdim=True)


        return new_world_coords, depth, None, signed_distances_for_marchlvl

#it masks off the pixels that are already converged
class DifferentiableRayMarcherMasked(torch.nn.Module):
    def __init__(self):
        super(DifferentiableRayMarcherMasked, self).__init__()

        num_encodings=8
        self.learned_pe=LearnedPE(in_channels=3, num_encoding_functions=num_encodings, logsampling=True)
        # cur_nr_channels = in_channels + 3*num_encodings*2

        #model 
        self.lstm_hidden_size = 16
        # self.lstm = torch.nn.LSTMCell(input_size=self.n_feature_channels, hidden_size=hidden_size)
        self.lstm=None #Create this later, the volumentric feature can maybe change and therefore the features that get as input to the lstm will be different
        # self.out_layer = torch.nn.Linear(self.lstm_hidden_size, 1)
        self.out_layer = BlockNerf(activ=None, in_channels=self.lstm_hidden_size, out_channels=1,  bias=True ).cuda()
        # self.feature_computer= VolumetricFeature(in_channels=3, out_channels=64, nr_layers=2, hidden_size=64, use_dirs=False) 
        # self.feature_computer= VolumetricFeatureSiren(in_channels=3, out_channels=64, nr_layers=2, hidden_size=64, use_dirs=False) 
        self.frame_weights_computer= FrameWeightComputer()
        self.feature_aggregator= FeatureAgregator()
        self.slice_texture= SliceTextureModule()
        self.splat_texture= SplatTextureModule()
        # self.feature_fuser = torch.nn.Sequential(
        #     # torch.nn.Linear(64+32, 64),
        #     # torch.nn.Linear(3+3*num_encodings*2  +32, 64),
        #     BlockNerf(activ=torch.relu, in_channels=cur_nr_channels, out_channels=cur_nr_channels,  bias=True ).cuda(),
        #     # torch.nn.ReLU(),
        #     torch.nn.Linear(64, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 64)
        # )
        self.feature_fuser = torch.nn.Sequential(
            BlockNerf(activ=torch.nn.GELU(), in_channels=3+3*num_encodings*2  +32, out_channels=64,  bias=True ).cuda(),
            BlockNerf(activ=torch.nn.GELU(), in_channels=64, out_channels=64,  bias=True ).cuda(),
            BlockNerf(activ=None, in_channels=64, out_channels=64,  bias=True ).cuda(),
        )

        #activ
        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()

        #params 
        self.nr_iters=10

      
    def forward(self, frame, depth_min, frames_close, frames_features, novel=False):

        if novel:
            depth_per_pixel= torch.ones([frame.height*frame.width,1], dtype=torch.float32, device=torch.device("cuda")) 
            depth_per_pixel.fill_(depth_min/2.0)   #randomize the deptha  bith
        else:
            depth_per_pixel = torch.zeros((frame.height*frame.width, 1), device=torch.device("cuda") ).normal_(mean=depth_min, std=2e-2)

        # depth_per_pixel.fill_(0.5)   #randomize the deptha  bith

        #Ray direction in world coordinates
        # ray_dirs_mesh=frame.pixels2dirs_mesh()
        # ray_dirs=torch.from_numpy(ray_dirs_mesh.V.copy()).to("cuda").float() #Nx3
        ray_dirs=frame.ray_dirs

        #compute points_2d 
        # points2d_screen_mesh=frame.pixels2coords()
        # points2d_screen=torch.from_numpy(points2d_screen_mesh.V.copy()).to("cuda").float()

        #unproject to 3D
        # K_inv= torch.from_numpy(  np.linalg.inv(frame.K.copy())    ).to("cuda")
        # tf_world_cam =torch.from_numpy( frame.tf_cam_world.inverse().matrix() ).to("cuda")
        # point_3D_camera_coord= K_inv * points2d_screen; 
        # point_3D_camera_coord= point_3D_camera_coord*depth_per_pixel
        # Eigen::Vector3d point_3D_world_coord=tf_world_cam*point_3D_camera_coord;

        #attempt 2 unproject to 3D 
        camera_center=torch.from_numpy( frame.frame.pos_in_world() ).to("cuda")
        camera_center=camera_center.view(1,3)
        points3D = camera_center + depth_per_pixel*ray_dirs
        # show_3D_points(points3D, "points_3d_init")

        init_world_coords=points3D
        initial_depth=depth_per_pixel
        world_coords = [init_world_coords]
        depths = [initial_depth]
        signed_distances_for_marchlvl=[]
        states = [None]

        weights=self.frame_weights_computer(frame, frames_close)

        for iter_nr in range(self.nr_iters):
            # print("iter is ", iter_nr)
            TIME_START("raymarch_iter")
        
            #compute the features at this position 
            # feat=self.feature_computer(points3D, ray_dirs) #a tensor of N x feat_size which contains for each position in 3D a feature representation around that point. Similar to phi from SRN
            # feat=self.feature_computer(world_coords[-1], ray_dirs) #a tensor of N x feat_size which contains for each position in 3D a feature representation around that point. Similar to phi from SRN
            feat=self.learned_pe(world_coords[-1])

            #concat also the features from images 
            feat_sliced_per_frame=[]
            TIME_START("raymarch_allslice")
            for i in range(len(frames_close)):
                TIME_START("raymarch_sliceiter")
                frame_close=frames_close[i].frame
                frame_features=frames_features[i]
                uv=compute_uv(frame_close, world_coords[-1])
                frame_features_for_slicing= frame_features.permute(0,2,3,1).squeeze().contiguous() # from N,C,H,W to H,W,C
                dummy, dummy, sliced_local_features= self.slice_texture(frame_features_for_slicing, uv)
                feat_sliced_per_frame.append(sliced_local_features.unsqueeze(0))  #make it 1 x N x FEATDIM
                TIME_END("raymarch_sliceiter")
            TIME_END("raymarch_allslice")
            # img_features_aggregated=torch.cat(feat_sliced_per_frame,1)
            # img_features_aggregated= feat_sliced_per_frame[0] - feat_sliced_per_frame[1]
            #attempt 1
            # img_features_concat=torch.cat(feat_sliced_per_frame,0) #we concat and compute mean and std similar to https://ibrnet.github.io/static/paper.pdf
            # img_features_mean=img_features_concat.mean(dim=0)
            # img_features_std=img_features_concat.std(dim=0)
            # img_features_aggregated=torch.cat([img_features_mean,img_features_std],1)
            
            #attempt 2 
            TIME_START("raymarch_aggr")
            img_features_aggregated= self.feature_aggregator(frame, frames_close, feat_sliced_per_frame, weights)

            feat=torch.cat([feat,img_features_aggregated],1)

            feat=self.feature_fuser(feat)
            TIME_END("raymarch_aggr")


            #create the lstm if not created 
            TIME_START("raymarch_lstm")
            if self.lstm==None:
                self.lstm = torch.nn.LSTMCell(input_size=feat.shape[1], hidden_size=self.lstm_hidden_size ).to("cuda")
                self.lstm.apply(init_recurrent_weights)
                lstm_forget_gate_init(self.lstm)

            #run through the lstm
            state = self.lstm(feat, states[-1])
            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-10, max=10))

            signed_distance= self.out_layer(state[0])
            TIME_END("raymarch_lstm")
            # print("signed_distance iter", iter_nr, " is ", signed_distance.mean())
            signed_distance=torch.abs(signed_distance) #the distance only increases
            #the output of the lstm after abs will probably be on average around 0.5 (because before the abs it was zero meaned and kinda spread around [-1,1])
            # however, doing nr_steps*0.5 will likely put the depth above the scene scale which is normally 1.0
            # therefore we expect each step to be 1.0/nr_steps so for 10 steps each steps should to 0.1
            depth_scaling=1.0/(1.0*self.nr_iters) #1.0 is the scene scale and we expect on average that every step will do a movement of 0.5, maybe the average movement is more like 0.25 idunno
            signed_distance=signed_distance*depth_scaling
            # print("signed_distance iter", iter_nr, " is ", signed_distance.mean())
            
            #debug signed_distance 
            signed_distance_vis=signed_distance.view(1,1, frame.height, frame.width)
            signed_distance_vis=signed_distance_vis/depth_scaling
            signed_distance_vis_mat=tensor2mat(signed_distance_vis) 
            Gui.show(signed_distance_vis_mat, "signed_distance_"+str(iter_nr))


            # print("ray_dirs is ", ray_dirs.shape)
            # print("signed distance is ", signed_distance.shape)

            new_world_coords = world_coords[-1] + ray_dirs * signed_distance
            states.append(state)
            world_coords.append(new_world_coords)
            signed_distances_for_marchlvl.append(signed_distance)

            # if iter_nr==self.nr_iters-1:
                # show_3D_points(new_world_coords, "points_3d_"+str(iter_nr))
            # show_3D_points(new_world_coords, "points_3d_"+str(iter_nr))
            TIME_END("raymarch_iter")

        #get the depth at this final 3d position
        depth= (new_world_coords-camera_center).norm(dim=1, keepdim=True)

        # points3D = camera_center + depth_per_pixel*ray_dirs

        #return also the world coords at every march
        world_coords.pop(0)


        return new_world_coords, depth, world_coords, signed_distances_for_marchlvl
        # return points3D, depth_per_pixel








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
        # self.encoder=EncoderLNN(self.z_size)
        # self.decoder=DecoderTo2D(self.z_size, 6)

        # self.spatial_lnn=SpatialLNN()
        self.spatial_lnn=SpatialLNNFixed()
        self.spatial_2d=SpatialEncoder2D(nr_channels=64)
        # self.spatial_2d=SpatialEncoderDense2D(nr_channels=32)
        self.slice=SliceLatticeModule()
        self.slice_texture= SliceTextureModule()
        self.splat_texture= SplatTextureModule()
        # self.learned_pe=LearnedPEGaussian(in_channels=3, out_channels=256, std=5)

        # self.z_size+=3 #because we add the direcitons
        # self.siren_net = SirenNetwork(in_channels=3, out_channels=4)
        # self.siren_net = SirenNetworkDirect(in_channels=3, out_channels=4)
        # self.siren_net = SirenNetworkDirect(in_channels=3, out_channels=3)
        # self.siren_net = SirenNetworkDirect(in_channels=3+3*self.num_encodings*2, out_channels=self.siren_out_channels)
        # self.siren_net = SirenNetworkDirectPE(in_channels=3, out_channels=self.siren_out_channels)
        self.siren_net = SirenNetworkDirectPE_Simple(in_channels=3, out_channels=self.siren_out_channels)
        # self.siren_net = SirenNetworkDirectPE_PixelNERF(in_channels=3, out_channels=self.siren_out_channels)
        
        # self.siren_net = SirenNetworkDirectPETrim(in_channels=3, out_channels=self.siren_out_channels)
        # self.siren_net = SirenNetworkDense(in_channels=3+3*self.num_encodings*2, out_channels=4)
        # self.nerf_net = NerfDirect(in_channels=3+3*self.num_encodings*2, out_channels=4)
        # self.hyper_net = HyperNetwork(hyper_in_features=self.nr_points_z*3*2, hyper_hidden_layers=1, hyper_hidden_features=512, hypo_module=self.siren_net)
        # self.hyper_net = HyperNetworkPrincipledInitialization(hyper_in_features=self.nr_points_z*3*2, hyper_hidden_layers=1, hyper_hidden_features=512, hypo_module=self.siren_net)
        # self.hyper_net = HyperNetworkPrincipledInitialization(hyper_in_features=self.z_size, hyper_hidden_layers=1, hyper_hidden_features=512, hypo_module=self.siren_net)
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

        # cam_params=9 + 3 + 3+ 1
        # self.cam_embedder = torch.nn.Sequential(
        #     BlockLinear(  in_channels=cam_params, out_channels=64,  bias=True,  activ=torch.relu ),
        #     # BlockLinear(  in_channels=64, out_channels=64,  bias=True,  activ=torch.relu ),
        #     BlockLinear(  in_channels=64, out_channels=64,  bias=True,  activ=None )
        # )
        # self.z_with_cam_embedder = torch.nn.Sequential(
        #     BlockLinear(  in_channels=self.z_size+64, out_channels=self.z_size,  bias=True,  activ=torch.relu ),
        #     # BlockLinear(  in_channels=self.z_size, out_channels=self.z_size,  bias=True,  activ=torch.relu ),
        #     BlockLinear(  in_channels=self.z_size, out_channels=self.z_size,  bias=True,  activ=None )
        # )
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
        sliced_local_features_list=[]
        sliced_global_features_list=[]
        for i in range(len(frames_for_encoding)):
            cur_cloud=frames_for_encoding[i].cloud.clone()
            cur_cloud.random_subsample(0.8)
            mesh.add( cur_cloud )
            img_features=self.spatial_2d(frames_for_encoding[i].rgb_tensor, frames_for_encoding[i] )
            #DO PCA
            if i==0:
                #show the features 
                height=img_features.shape[2]
                width=img_features.shape[3]
                img_features_for_pca=img_features.squeeze(0).permute(1,2,0).contiguous()
                img_features_for_pca=img_features_for_pca.view(height*width, -1)
                pca=PCA.apply(img_features_for_pca)
                pca=pca.view(height, width, 3)
                pca=pca.permute(2,0,1).unsqueeze(0)
                pca_mat=tensor2mat(pca)
                Gui.show(pca_mat, "pca_mat")



            uv1=frames_for_encoding[i].compute_uv(cur_cloud) #uv for projecting this cloud into this frame
            # uv2=frames_for_encoding[i].compute_uv_with_assign_color(cur_cloud) #uv for projecting this cloud into this frame
            uv=uv1

            ####DEBUG 
            # print("uv1", uv1)
            # print("uv2", uv2)
            # diff = ((uv1-uv2)**2).mean()
            # print("diff in uv is ", diff)

            uv_tensor=torch.from_numpy(uv).float().to("cuda")
            uv_tensor= uv_tensor*2 -1

            #flip
            uv_tensor[:,1]=-uv_tensor[:,1]

            # print("uv tensor min max is ", uv_tensor.min(), " " , uv_tensor.max())

            # print("uv tensor is ", uv_tensor)


            #concat to the ray origin and ray dir
            fx=frames_for_encoding[i].K[0,0] ### 
            fy=frames_for_encoding[i].K[1,1] ### 
            cx=frames_for_encoding[i].K[0,2] ### 
            cy=frames_for_encoding[i].K[1,2] ### 
            tform_cam2world =torch.from_numpy( frames_for_encoding[i].tf_cam_world.inverse().matrix() ).to("cuda")
            ray_origins, ray_directions = get_ray_bundle(
                frames_for_encoding[i].height, frames_for_encoding[i].width, fx,fy,cx,cy, tform_cam2world, novel=False
            )
            # print("ray origins ", ray_origins.shape)
            # print("ray_directions ", ray_directions.shape)


            # print("img_features ", img_features.shape)
            # img_features=img_features.squeeze(0)
            img_features=img_features.squeeze(0)
            # img_features=frames_for_encoding[i].rgb_tensor.squeeze(0)
            img_features=img_features.permute(1,2,0)
            # print("img_featuresis ", img_features.shape)

            # img_features=torch.cat([img_features, ray_origins, ray_directions], 2)
            #we distinguish between local features and global features
            local_features=img_features
            global_features=torch.cat([ray_origins, ray_directions],2)

            dummy, dummy, sliced_local_features= self.slice_texture(local_features, uv_tensor)
            dummy, dummy, sliced_global_features= self.slice_texture(global_features, uv_tensor)
            # print("current_sliced_local_features is ", sliced_local_features.shape)
            # print("current_sliced_global_features is ", sliced_global_features.shape)
            sliced_local_features_list.append(sliced_local_features)
            sliced_global_features_list.append(sliced_global_features)

            #####debug by splatting again-----------------------------
            # color_val=torch.from_numpy(cur_cloud.C.copy() ).float().to("cuda")
            # texture= self.splat_texture(sliced_features, uv_tensor, 40)
            # # texture= self.splat_texture(color_val, uv_tensor, 40)
            # texture=texture.permute(2,0,1).unsqueeze(0)
            # texture=texture[:,0:3, : , :]
            # print("texture is ", texture.shape)
            # tex_mat=tensor2mat(texture)
            # Gui.show(tex_mat,"tex_splatted")



        #compute psitions and values
        positions=torch.from_numpy(mesh.V.copy() ).float().to("cuda")
        sliced_local_features=torch.cat(sliced_local_features_list,0)
        sliced_global_features=torch.cat(sliced_global_features_list,0)
        sliced_global_features=torch.cat([sliced_global_features, positions],1)
        # print("final sliced_local_features is ", sliced_local_features.shape)
        # print("final sliced_global_features is ", sliced_global_features.shape)
        # color_values=torch.from_numpy(mesh.C.copy() ).float().to("cuda")
        # values=color_values
        # values=sliced_features
        # values=torch.cat( [positions,values],1 )
        # print("values is ", values.shape)

        #concat also the encoded positions 
        # pos_encod=self.siren_net.learned_pe(positions )
        # values=torch.cat( [values,pos_encod],1 )

        #check diff DEBUG------------------------------------------------------------
        # diff=((sliced_features-color_values)**2).mean()
        # print("diff is ", diff)
        # print("color values is ", color_values)
        # print("sliced color is ", sliced_features)


        
        mesh.m_vis.m_show_points=True
        mesh.m_vis.set_color_pervertcolor()
        Scene.show(mesh, "cloud")


        #pass the mesh through the lattice 
        TIME_START("spatial_lnn")
        # lv, ls = self.spatial_lnn(positions, values)
        lv, ls = self.spatial_lnn(positions, sliced_local_features, sliced_global_features)
        TIME_END("spatial_lnn")
       





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
        depth_samples_per_ray=100
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
        TIME_START("slice")
        # multires_sliced=[]
        # for i in range(len(multi_res_lattice)):
        #     lv=multi_res_lattice[i][0]
        #     ls=multi_res_lattice[i][1]
        #     point_features=self.slice(lv, ls, flattened_query_points_for_slicing)
        #     # print("sliced at res i", i, " is ", point_features.shape)
        #     multires_sliced.append(point_features.unsqueeze(2) )
        point_features=self.slice(lv, ls, flattened_query_points_for_slicing)
        TIME_END("slice")
        #aggregate all the features 
        # aggregated_point_features=multires_sliced[0]
        # for i in range(len(multi_res_lattice)-1):
            # aggregated_point_features=aggregated_point_features+ multires_sliced[i+1]
        # aggregated_point_features=aggregated_point_features/ len(multi_res_lattice)
        #attemopt 2 aggegare with maxpool
        # aggregated_point_features=torch.cat(multires_sliced,2)
        # aggregated_point_features, _=aggregated_point_features.max(dim=2)
        #attemopt 2 aggegare with ,ean
        # aggregated_point_features=torch.cat(multires_sliced,2)
        # aggregated_point_features =aggregated_point_features.mean(dim=2)
        #attempt 3 start at the coarser one and then replace parts of it with the finer ones
        # aggregated_point_features=multires_sliced[0] #coarsers sliced features
        # for i in range(len(multi_res_lattice)-1):
        #     cur_sliced_features= multires_sliced[i+1]
        #     cur_valid_features_mask= cur_sliced_features!=0.0
        #     # aggregated_point_features[cur_valid_features_mask] =  cur_sliced_features[cur_valid_features_mask]
        #     aggregated_point_features.masked_scatter(cur_valid_features_mask, cur_sliced_features)
        #attempt 4, just get the finest one
        # aggregated_point_features=multires_sliced[ len(multi_res_lattice)-1  ]
        aggregated_point_features=point_features

        #having a good feature for a point in the ray should somehow conver information to the other points in the ray so we need to pass some information between all of them
        #for each ray we get the maximum feature
        # aggregated_point_features_img=aggregated_point_features.view(height,width,depth_samples_per_ray,-1 )
        # aggregated_point_features_img_max, _=aggregated_point_features_img.max(dim=2, keepdim=True)
        # aggregated_point_features_img=aggregated_point_features_img+ aggregated_point_features_img_max
        # aggregated_point_features= aggregated_point_features_img.view( height*width*depth_samples_per_ray, -1)

        ##aggregate by summing the max over all resolutions
        # aggregated_point_features_all=torch.cat(multires_sliced,2)
        # aggregated_point_features_max, _=aggregated_point_features_all.max(dim=2)
        # aggregated_point_features=aggregated_point_features.squeeze(2)+ aggregated_point_features_max






        # print("self, iter is ",self.iter, )

        # radiance_field_flattened = self.siren_net(query_points.to("cuda") )-3.0 
        # radiance_field_flattened = self.siren_net(query_points.to("cuda") )
        # flattened_query_points/=2.43
        TIME_START("just_siren")
        radiance_field_flattened = self.siren_net(flattened_query_points.to("cuda"), ray_directions, aggregated_point_features ) #radiance field has shape height,width, nr_samples,4
        TIME_END("just_siren")
        # radiance_field_flattened = self.siren_net(flattened_query_points.to("cuda"), ray_directions, params=siren_params )
        # radiance_field_flattened = self.siren_net(query_points.to("cuda"), params=siren_params )
        # radiance_field_flattened = self.nerf_net(flattened_query_points.to("cuda") ) 
        # radiance_field_flattened = self.nerf_net(flattened_query_points.to("cuda"), params=siren_params ) 

        #debug
        if not novel and self.iter%100==0:
            rays_mesh=Mesh()
            # rays_mesh.V=query_points.detach().reshape((-1, 3)).cpu().numpy()
            # rays_mesh.m_vis.m_show_points=True
            # #color based on sigma 
            # sigma_a = radiance_field_flattened[:,:,:, self.siren_out_channels-1].detach().view(-1,1).repeat(1,3)
            # rays_mesh.C=sigma_a.cpu().numpy()
            # Scene.show(rays_mesh, "rays_mesh_novel")
        if not novel:
            self.iter+=1


    
        radiance_field_flattened=radiance_field_flattened.view(-1,self.siren_out_channels)


        # "Unflatten" to obtain the radiance field.
        unflattened_shape = list(query_points.shape[:-1]) + [self.siren_out_channels]
        radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        TIME_START("render_volume")
        rgb_predicted, depth_map, acc_map = render_volume_density(
        # rgb_predicted, depth_map, acc_map = render_volume_density_nerfplusplus(
        # rgb_predicted, depth_map, acc_map = render_volume_density2(
            # radiance_field, ray_origins.to("cuda"), depth_values.to("cuda"), self.siren_out_channels
            radiance_field, ray_origins, depth_values, self.siren_out_channels
        )
        TIME_END("render_volume")

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



class DepthPredictor(torch.nn.Module):
    def __init__(self, model_params):
        super(DepthPredictor, self).__init__()

        self.lnn=LNN(128, model_params)
        self.lattice=Lattice.create("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/config/train.cfg", "splated_lattice")
        self.slice_texture= SliceTextureModule()
        self.splat_texture= SplatTextureModule()
        self.cnn_2d=UNet(32,1)
        self.concat_coord=ConcatCoord() 

    def forward(self, frame, mesh):

        # #pass the mesh_full through a lnn to get features OR maybse pass only the mesh_sparse which correpsonds to this frame_query
        # positions=torch.from_numpy(mesh.V.copy() ).float().to("cuda")
        # values=torch.from_numpy(mesh.C.copy() ).float().to("cuda")
        # values=torch.cat([positions,values],1)
        # logsoftmax, sv=self.lnn(self.lattice, positions, values)

        # #splat the features of the point onto the frame_query tensor
        # uv=frame.compute_uv(mesh) #uv for projecting this cloud into this frame
        # uv_tensor=torch.from_numpy(uv).float().to("cuda")
        # uv_tensor= uv_tensor*2 -1
        # uv_tensor[:,1]=-uv_tensor[:,1] #flip
        # texture= self.splat_texture(sv, uv_tensor, frame.height) #we assume that the height is the same as the weight
        # #divide by the homogeneous coords
        # val_dim=texture.shape[2]-1
        # texture=texture[:,:,0:val_dim] / (texture[:,:,val_dim:val_dim+1] +0.0001)
        # if frame.height!=frame.width:
        #     print("The splat texture only can create square textures but the frames doesnt have a square size, so in order to create a depth map, we need to resize the image or improve the splat_texture function")
        #     exit(1)


        #Run a CNN to produce a depth map of this frame_query
        # texture=texture.unsqueeze(0) #N,H,W,C
        # texture=texture.permute(0,3,1,2) #converts to N,C,H,W
        # depth=self.cnn_2d(texture)

        #DEBUG put the rgb in there 
        rgb_query=mat2tensor(frame.rgb_32f, False).to("cuda")
        cnn_input=self.concat_coord(rgb_query)
        coords=cnn_input[:,0:2, :, :] 
        coords_encoded=positional_encoding(coords, num_encoding_functions=6)
        cnn_input=torch.cat([coords_encoded, rgb_query],1)
        # cnn_input=coords_encoded
        depth=self.cnn_2d(cnn_input)

        #if we predict depth, we know it has to be be positive
        depth=torch.relu(depth) + 0.01 #added a tiny epsilon because depth of 0 gets an invalid uv tensor afterwards and it just get a black color

        #####----Another option is to return the Z coordinate, but the one relative to the original of the world instead of the depth which is relative to the camera

        return depth

class Net2(torch.nn.Module):
    def __init__(self, model_params):
        super(Net2, self).__init__()

        self.first_time=True

        #models
        # self.lnn=LNN_2(128, model_params)
        self.lattice=Lattice.create("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/config/train.cfg", "splated_lattice")

        #activ
        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()

        #params 
        self.siren_out_channels=4


        
        self.slice=SliceLatticeModule()
        self.slice_texture= SliceTextureModule()
        self.splat_texture= SplatTextureModule()

       
        # self.siren_net = SirenNetworkDirectPE_Simple(in_channels=3, out_channels=self.siren_out_channels)
        self.siren_net = NERF_original(in_channels=3, out_channels=self.siren_out_channels)

      
    def forward(self, frame, mesh, depth_min, depth_max, pixel_indices):

        use_lattice_features=False


        if use_lattice_features:
            #compute psitions and values
            positions=torch.from_numpy(mesh.V.copy() ).float().to("cuda")
            values=torch.from_numpy(mesh.C.copy() ).float().to("cuda")
        
            #pass the mesh through the lattice 
            TIME_START("spatial_lnn")
            lv, ls = self.lnn(self.lattice, positions, values)
            TIME_END("spatial_lnn")
       





        TIME_START("full_siren")
        #siren has to receive some 3d points as query, The 3d points are located along rays so we can volume render and compare with the image itself
        #most of it is from https://github.com/krrish94/nerf-pytorch/blob/master/tiny_nerf.py
        height=frame.height
        width=frame.width
        K= torch.from_numpy( frame.K.copy() )
        fx=K[0,0] ### 
        fy=K[1,1] ### 
        cx=K[0,2] ### 
        cy=K[1,2] ### 
        tf_world_cam =torch.from_numpy( frame.tf_cam_world.inverse().matrix() ).to("cuda")
        near_thresh=depth_min
        far_thresh=depth_max
        depth_samples_per_ray=80
        # chunksize=400*400

        # Get the "bundle" of rays through all image pixels.
        TIME_START("ray_bundle")
        ray_origins, ray_directions = get_ray_bundle(
            height, width, fx,fy,cx,cy, tf_world_cam, novel=False
        )
        ray_origins=ray_origins.view(-1,3)
        ray_directions=ray_directions.view(-1,3)
        TIME_END("ray_bundle")

        ###TODO get only a subsample of the ray origin and ray_direction, maybe like in here https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/14
        # idxs=None
        # ray_origins_original=ray_origins.clone() 
        # if use_chunking:
        #     TIME_START("random_pixels")
        #     chunck_size= min(30*30, height*width)
        #     weights = torch.ones([height*width], dtype=torch.float32, device=torch.device("cuda"))  #equal probability to choose each pixel
        #     idxs=torch.multinomial(weights, chunck_size, replacement=False)
        #     # idxs=torch.arange(height*width).to("cuda")
        #     # print("idx is ", idxs.min(), idxs.max() )
        #     # print("ray_origins ", ray_origins.shape)
        #     # ray_origins=ray_origins[idxs]
        #     ray_origins=torch.index_select(ray_origins, 0, idxs.long())
        #     # print("ray_origins after selecting ", ray_origins.shape)
        #     # ray_directions=ray_directions[idxs]
        #     ray_directions=torch.index_select(ray_directions, 0, idxs.long())
        #     TIME_END("random_pixels")
        ray_origins=torch.index_select(ray_origins, 0, pixel_indices)
        ray_directions=torch.index_select(ray_directions, 0, pixel_indices)
        # print("ray_origins_original", ray_origins_original)
        # print("ray_origins_sliced", ray_origins)
        # diff = ((ray_origins_original -ray_origins)**2).mean()
        # print("diff is ", diff)
        # exit(1)

        TIME_START("sample")
        # #just set the two tensors to the min and max 
        # near_thresh_tensor= torch.ones([1,1,height,width], dtype=torch.float32, device=torch.device("cuda")) 
        # far_thresh_tensor= torch.ones([1,1,height,width], dtype=torch.float32, device=torch.device("cuda")) 
        # near_thresh_tensor.fill_(depth_min)
        # far_thresh_tensor.fill_(depth_max)


        # query_points, depth_values = compute_query_points_from_rays2(
            # ray_origins, ray_directions, near_thresh_tensor, far_thresh_tensor, depth_samples_per_ray, randomize=True
        # )
        query_points, depth_values = compute_query_points_from_rays(
            ray_origins, ray_directions, depth_min, depth_max, depth_samples_per_ray, randomize=True
        )
        query_points=query_points.view(-1,3)
        depth_values=depth_values.view(-1)
        # print("query_points", query_points.shape)
        # print("depth_values", depth_values.shape)
        TIME_END("sample")

        # "Flatten" the query points.
        # print("query points is ", query_points.shape)
        flattened_query_points = query_points.reshape((-1, 3))
        # print("flattened_query_points is ", flattened_query_points.shape)

        # TIME_START("pos_encode")
        # flattened_query_points = positional_encoding(flattened_query_points, num_encoding_functions=self.num_encodings, log_sampling=True)
        # flattened_query_points=self.leaned_pe(flattened_query_points.to("cuda") )
        # print("flattened_query_points after pe", flattened_query_points.shape)
        # flattened_query_points=flattened_query_points.view(height,width,depth_samples_per_ray,-1 )
        # print("flatened_query_pointss is ", flatened_query_pointss.shape)
        # TIME_END("pos_encode")


        #slice from lattice 
        TIME_START("slice")
        # multires_sliced=[]
        # for i in range(len(multi_res_lattice)):
        #     lv=multi_res_lattice[i][0]
        #     ls=multi_res_lattice[i][1]
        #     point_features=self.slice(lv, ls, flattened_query_points_for_slicing)
        #     # print("sliced at res i", i, " is ", point_features.shape)
        #     multires_sliced.append(point_features.unsqueeze(2) )
        if use_lattice_features:
            flattened_query_points_for_slicing= flattened_query_points.view(-1,3)
            point_features=self.slice(lv, ls, flattened_query_points_for_slicing)
        TIME_END("slice")
        #aggregate all the features 
        # aggregated_point_features=multires_sliced[0]
        # for i in range(len(multi_res_lattice)-1):
            # aggregated_point_features=aggregated_point_features+ multires_sliced[i+1]
        # aggregated_point_features=aggregated_point_features/ len(multi_res_lattice)
        #attemopt 2 aggegare with maxpool
        # aggregated_point_features=torch.cat(multires_sliced,2)
        # aggregated_point_features, _=aggregated_point_features.max(dim=2)
        #attemopt 2 aggegare with ,ean
        # aggregated_point_features=torch.cat(multires_sliced,2)
        # aggregated_point_features =aggregated_point_features.mean(dim=2)
        #attempt 3 start at the coarser one and then replace parts of it with the finer ones
        # aggregated_point_features=multires_sliced[0] #coarsers sliced features
        # for i in range(len(multi_res_lattice)-1):
        #     cur_sliced_features= multires_sliced[i+1]
        #     cur_valid_features_mask= cur_sliced_features!=0.0
        #     # aggregated_point_features[cur_valid_features_mask] =  cur_sliced_features[cur_valid_features_mask]
        #     aggregated_point_features.masked_scatter(cur_valid_features_mask, cur_sliced_features)
        #attempt 4, just get the finest one
        # aggregated_point_features=multires_sliced[ len(multi_res_lattice)-1  ]
        aggregated_point_features=None
        if use_lattice_features:
            aggregated_point_features=point_features.contiguous()

        #having a good feature for a point in the ray should somehow conver information to the other points in the ray so we need to pass some information between all of them
        #for each ray we get the maximum feature
        # aggregated_point_features_img=aggregated_point_features.view(height,width,depth_samples_per_ray,-1 )
        # aggregated_point_features_img_max, _=aggregated_point_features_img.max(dim=2, keepdim=True)
        # aggregated_point_features_img=aggregated_point_features_img+ aggregated_point_features_img_max
        # aggregated_point_features= aggregated_point_features_img.view( height*width*depth_samples_per_ray, -1)

        ##aggregate by summing the max over all resolutions
        # aggregated_point_features_all=torch.cat(multires_sliced,2)
        # aggregated_point_features_max, _=aggregated_point_features_all.max(dim=2)
        # aggregated_point_features=aggregated_point_features.squeeze(2)+ aggregated_point_features_max






        # print("self, iter is ",self.iter, )

        # radiance_field_flattened = self.siren_net(query_points.to("cuda") )-3.0 
        # radiance_field_flattened = self.siren_net(query_points.to("cuda") )
        # flattened_query_points/=2.43
        # print("flattened_query_points ", flattened_query_points.shape)
        TIME_START("just_siren")
        # flattened_query_points=flattened_query_points.half()
        # ray_directions=ray_directions.half()
        # radiance_field_flattened = self.siren_net(flattened_query_points.to("cuda"), ray_directions, aggregated_point_features ) #radiance field has shape height,width, nr_samples,4
        radiance_field_flattened = self.siren_net(flattened_query_points.to("cuda"), ray_directions, aggregated_point_features, depth_samples_per_ray ) #radiance field has shape height,width, nr_samples,4
        # radiance_field_flattened=radiance_field_flattened.float()
        TIME_END("just_siren")
        # radiance_field_flattened = self.siren_net(flattened_query_points.to("cuda"), ray_directions, params=siren_params )
        # radiance_field_flattened = self.siren_net(query_points.to("cuda"), params=siren_params )
        # radiance_field_flattened = self.nerf_net(flattened_query_points.to("cuda") ) 
        # radiance_field_flattened = self.nerf_net(flattened_query_points.to("cuda"), params=siren_params ) 

        # #debug
        # if not novel and self.iter%100==0:
        #     rays_mesh=Mesh()
        #     # rays_mesh.V=query_points.detach().reshape((-1, 3)).cpu().numpy()
        #     # rays_mesh.m_vis.m_show_points=True
        #     # #color based on sigma 
        #     # sigma_a = radiance_field_flattened[:,:,:, self.siren_out_channels-1].detach().view(-1,1).repeat(1,3)
        #     # rays_mesh.C=sigma_a.cpu().numpy()
        #     # Scene.show(rays_mesh, "rays_mesh_novel")
        # if not novel:
        #     self.iter+=1

        # #DEBUG 
        # rays_mesh=Mesh()
        # rays_mesh.V=query_points.detach().reshape((-1, 3)).cpu().numpy()
        # rays_mesh.m_vis.m_show_points=True
        # #color based on sigma 
        # sigma_a = radiance_field_flattened[:, self.siren_out_channels-1].detach().view(-1,1).repeat(1,3)
        # rays_mesh.C=sigma_a.cpu().numpy()
        # Scene.show(rays_mesh, "rays_mesh_novel")


    
        radiance_field_flattened=radiance_field_flattened.view(-1,self.siren_out_channels)


        # "Unflatten" to obtain the radiance field.
        unflattened_shape = list(query_points.shape[:-1]) + [self.siren_out_channels]
        radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        TIME_START("render_volume")
        radiance_field=radiance_field.view(-1, depth_samples_per_ray, 4)
        depth_values=depth_values.view(-1,depth_samples_per_ray)
        # print("radiance_field shapoe si ", radiance_field.shape)
        # print("ray_origins shapoe si ", ray_origins.shape)
        # print("depth_values shapoe si ", depth_values.shape)
        rgb_predicted, depth_map, acc_map = render_volume_density(
        # rgb_predicted, depth_map, acc_map = render_volume_density_nerfplusplus(
        # rgb_predicted, depth_map, acc_map = render_volume_density2(
            # radiance_field, ray_origins.to("cuda"), depth_values.to("cuda"), self.siren_out_channels
            radiance_field, ray_origins, depth_values, self.siren_out_channels
        )
        TIME_END("render_volume")

        # print("rgb predicted has shpae ", rgb_predicted.shape)
        # rgb_predicted=rgb_predicted.view(1,3,height,width)
        # rgb_predicted=rgb_predicted.permute(2,0,1).unsqueeze(0).contiguous()
        # print("depth map size is ", depth_map.shape)
        # depth_map=depth_map.unsqueeze(0).unsqueeze(0).contiguous()
        # depth_map_mat=tensor2mat(depth_map)
        TIME_END("full_siren")

        # print("rgb_predicted is ", rgb_predicted.shape)

        # if not use_chunking:
            # rgb_predicted=rgb_predicted.view(height,width,3)
            # rgb_predicted=rgb_predicted.permute(2,0,1).unsqueeze(0).contiguous()
        # else:
            # rgb_predicted=rgb_predicted.view(-1,3).contiguous()

        # rgb_predicted=rgb_predicted.view(height,width,3)
        # rgb_predicted=rgb_predicted.permute(2,0,1).unsqueeze(0).contiguous()




        # #NEW LOSSES
        new_loss=0
       




        return rgb_predicted,  depth_map, acc_map, new_loss
        # return img_directly_after_decoding






        return x

#Instead of doing ray marching like in NERF we use a LSTM to update the ray step similar to Scene representation network
class Net3_SRN(torch.nn.Module):
    def __init__(self, model_params, do_superres):
        super(Net3_SRN, self).__init__()

        self.first_time=True

        #models
        self.unet=UNet( nr_channels_start=16, nr_channels_output=32, nr_stages=4, max_nr_channels=64)
        # self.unet_rgb=UNet( nr_channels_start=16, nr_channels_output=32, nr_stages=1, max_nr_channels=32)
        # self.unet=FeaturePyramid( nr_channels_start=16, nr_channels_output=32, nr_stages=5)

        # self.unet= SQNet(classes=32)
        # self.unet= LinkNet(classes=32) #converges
        # self.unet= LinkNetImprove(classes=32)  #looks ok
        # self.unet= SegNet(classes=32) #looks ok
        # self.unet= UNet_efficient.UNet(classes=32) #converges
        # self.unet= ENet(classes=16) #eror
        # self.unet= ERFNet(classes=16) #eror
        # self.unet= CGNet(classes=32)
        # self.unet= EDANet(classes=16) #converges
        # self.unet= ESNet(classes=32)
        # self.unet= ESPNet(classes=16) #error
        # self.unet= LEDNet(classes=16) #converges
        # self.unet= ContextNet(classes=16) #converges
        # self.unet= FastSCNN(classes=32)
        # self.unet= DABNet(classes=32)
        # self.unet= FSSNet(classes=16) #error
        # self.unet= FPENet(classes=32)

        self.do_superres=do_superres
        if do_superres:
            edsr_args=EDSR_args()
            edsr_args.n_in_channels=67
            edsr_args.n_resblocks=4
            edsr_args.n_feats=64
            edsr_args.scale=1
            # self.super_res=EDSR(edsr_args)
            self.super_res=UNet( nr_channels_start=67, nr_channels_output=3, nr_stages=3, max_nr_channels=64)

        self.ray_marcher=DifferentiableRayMarcher()
        # self.ray_marcher=DifferentiableRayMarcherHierarchical()
        # self.ray_marcher=DifferentiableRayMarcherMasked()
        # self.rgb_predictor = NERF_original(in_channels=3, out_channels=4, use_ray_dirs=True)
        # self.rgb_predictor = SIREN_original(in_channels=3, out_channels=4, use_ray_dirs=True)
        # self.embedd_sd=BlockNerf(activ=None, init="sigmoid", in_channels=1, out_channels=32,  bias=True ).cuda()    
        self.rgb_predictor = RGB_predictor_simple(in_channels=3, out_channels=4, use_ray_dirs=True)
        # self.rgb_refiner=UNet( nr_channels_start=64, nr_channels_output=3, nr_stages=3)
        # self.s_weight = torch.nn.Parameter(torch.randn(1))  #from equaiton 3 here https://arxiv.org/pdf/2010.08888.pdf
        # with torch.set_grad_enabled(False):
            # self.s_weight.fill_(0.5)
            # self.s_weight.fill_(10.0)
        # self.frame_weights_computer= FrameWeightComputer()
        self.feature_aggregator= FeatureAgregator()
        # self.feature_aggregator= FeatureAgregatorLinear()


        #activ
        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()

        #params


        
        self.slice_texture= SliceTextureModule()
        self.splat_texture= SplatTextureModule()
        self.concat_coord=ConcatCoord() 


      
    def forward(self, frame, ray_dirs, rgb_close_batch, depth_min, depth_max, frames_close, weights, pixels_indices, novel=False):

        TIME_START("unet_everything")
        # frames_features=[]
        # for frame_close in frames_close:
        #     rgb_gt=frame_close.rgb_tensor
        #     frame_features=self.unet(rgb_gt)
        #     frames_features.append(frame_features)

        # rgb_batch_list=[]
        # for frame_close in frames_close:
        #     rgb_gt=mat2tensor(frame_close.rgb_32f, False).to("cuda")
        #     rgb_batch_list.append(rgb_gt)
        # rgb_batch=torch.cat(rgb_batch_list,0)
        #pass through unet 
        TIME_START("unet")
        # with  torch.autograd.profiler.profile(profile_memory=True, record_shapes=True, use_cuda=True, with_stack=True,) as prof:
        # exit(1)
        # unet_input=torch.cat([rgb_close_batch, ray_dirs_close_batch],1)
        frames_features=self.unet( rgb_close_batch )
        # frames_features=self.unet( unet_input )
        # exit(1)
        # print(prof.table(sort_by="cuda_memory_usage", row_limit=20) )
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))

        TIME_END("unet")
        frames_features_list=[]
        for i in range(len(frames_close)):
            frames_features_list.append(frames_features[i:i+1, :,:,:])
        TIME_END("unet_everything")

        #select only the ray dirst for the selected pixels 
        if pixels_indices is not None:
            ray_dirs= torch.index_select(ray_dirs, 0, pixels_indices)

        TIME_START("ray_march")
        point3d, depth, points3d_for_marchlvl, signed_distances_for_marchlvl, raymarcher_loss = self.ray_marcher(frame, ray_dirs, depth_min, frames_close, frames_features, weights, pixels_indices, novel)
        TIME_END("ray_march")

        # print("len points3d_for_marchlvl", len(points3d_for_marchlvl))
        # print("len signed_distances_for_marchlvl", len(signed_distances_for_marchlvl))


        # ray_dirs_mesh=frame.pixels2dirs_mesh()
        # ray_dirs=torch.from_numpy(ray_dirs_mesh.V.copy()).to("cuda").float() #Nx3
        # ray_dirs=torch.from_numpy(frame.ray_dirs).to("cuda").float()


        # ###predict RGB for every march lvl of the lstm 
        # rgb_pred_for_marchlvl=[]
        # TIME_START("rgb_pred_allmarch")
        # for i in range(len(points3d_for_marchlvl)):
        #     points3d_for_lvl= points3d_for_marchlvl[i]
        #     #concat also the features from images 
        #     feat_sliced_per_frame=[]
        #     for i in range(len(frames_close)):
        #         frame_close=frames_close[i].frame
        #         frame_features=frames_features[i]
        #         uv=compute_uv(frame_close, points3d_for_lvl )
        #         frame_features_for_slicing= frame_features.permute(0,2,3,1).squeeze().contiguous() # from N,C,H,W to H,W,C
        #         dummy, dummy, sliced_local_features= self.slice_texture(frame_features_for_slicing, uv)
        #         feat_sliced_per_frame.append(sliced_local_features.unsqueeze(0)) #make it 1 x N x FEATDIM
            
        #     ##attempt 4 
        #     weights=self.frame_weights_computer(frame, frames_close)
        #     img_features_aggregated= self.feature_aggregator(frame, frames_close, feat_sliced_per_frame, weights)
            
        #     radiance_field_flattened = self.rgb_predictor(points3d_for_lvl, ray_dirs, point_features=img_features_aggregated, nr_points_per_ray=1, params=None  ) #radiance field has shape height,width, nr_samples,4

        #     rgb_pred=radiance_field_flattened[:, 0:3]
        #     rgb_pred=rgb_pred.view(frame.height, frame.width,3)
        #     rgb_pred=rgb_pred.permute(2,0,1).unsqueeze(0)
        #     rgb_pred_for_marchlvl.append(rgb_pred) 
        # TIME_END("rgb_pred_allmarch")









        #concat also the features from images 
        # feat_sliced_per_frame=[]
        # for i in range(len(frames_close)):
        #     frame_close=frames_close[i]
        #     frame_features=frames_features_list[i]
        #     uv=compute_uv(frame_close, point3d )
        #     frame_features_for_slicing= frame_features.permute(0,2,3,1).squeeze().contiguous() # from N,C,H,W to H,W,C
        #     dummy, dummy, sliced_local_features= self.slice_texture(frame_features_for_slicing, uv)
        #     feat_sliced_per_frame.append(sliced_local_features.unsqueeze(0)) #make it 1 x N x FEATDIM
        # feat_sliced_per_frame=torch.cat(feat_sliced_per_frame,0)

        #rgb gets other features than the ray marcher 
        # frames_features_rgb=self.unet_rgb(rgb_close_batch)
        frames_features_rgb=frames_features

        nr_nearby_frames=len(frames_close)
        R_list=[]
        t_list=[]
        K_list=[]
        for i in range(len(frames_close)):
            frame_selected=frames_close[i]
            R_list.append( frame_selected.R_tensor.view(1,3,3) )
            t_list.append( frame_selected.t_tensor.view(1,1,3) )
            K_list.append( frame_selected.K_tensor.view(1,3,3) )
        R_batched=torch.cat(R_list,0)
        t_batched=torch.cat(t_list,0)
        K_batched=torch.cat(K_list,0)
        ######when we project we assume all the frames have the same size
        height=frames_close[0].height
        width=frames_close[0].width


        # uv_tensor=compute_uv_batched_original(frames_close, point3d )
        uv_tensor=compute_uv_batched(R_batched, t_batched, K_batched, height, width,  point3d )
        # slice with grid_sample
        uv_tensor=uv_tensor.view(nr_nearby_frames, -1, 1,  2) #nrnearby_frames x nr_pixels x 1 x 2
        sliced_feat_batched=torch.nn.functional.grid_sample( frames_features_rgb, uv_tensor, align_corners=False, mode="bilinear" ) #sliced features is N,C,H,W
        feat_dim=sliced_feat_batched.shape[1]
        sliced_feat_batched=sliced_feat_batched.permute(0,2,3,1) # from N,C,H,W to N,H,W,C
        sliced_feat_batched=sliced_feat_batched.view(len(frames_close), -1, feat_dim) #make it 1 x N x FEATDIM
          

        ##attempt 4 
        # weights=self.frame_weights_computer(frame, frames_close)
        # img_features_aggregated= self.feature_aggregator(frame, frames_close, sliced_feat_batched, weights)
        img_features_aggregated= self.feature_aggregator( sliced_feat_batched, weights)
        std= img_features_aggregated[:, -16]

        # show the frames and with a line weight depending on the weight
        # if novel:
            # print("weights is ", weights)
        # if novel:
        for i in range(len(frames_close)):
            frustum_mesh=frames_close[i].frame.create_frustum_mesh(0.02)
            frustum_mesh.m_vis.m_line_width= (weights[i])*15
            frustum_mesh.m_vis.m_line_color=[0.0, 1.0, 0.0] #green
            frustum_mesh.m_force_vis_update=True
            Scene.show(frustum_mesh, "frustum_neighb_"+str(i) ) 
        

        #concat also the signed distnace at the last iteration of the lstm 
        #BAD IDEA, concatting the signed distance allows the network to predict weird wobbly depth that still gets mapped to a correct color. The distance left should not be a feature that hte network can use since this is a feature that actually comes from the lstm which is kinda like an aggregation of all the features along the ray. However the RGB predictor should only be allowed to use the feature at the current position because this forces the position to be correct so that the sliced features from the images are as good as possible for predicting color
        # signed_dist=signed_distances_for_marchlvl[ -1 ]
        # signed_dist_embedded=self.embedd_sd(signed_dist)
        # img_features_aggregated=torch.cat([img_features_aggregated, signed_dist_embedded],1)

        # #concat also the nromals  THEY ACTUALLY hur the perfomance because the normals are way to noisy, espetially at the beggining
        # points3D_img=point3d.view(1, frame.height, frame.width, 3)
        # points3D_img=points3D_img.permute(0,3,1,2) #from N,H,W,C to N,C,H,W
        # normal_img=compute_normal(points3D_img)
        # normal=normal_img.permute(0,2,3,1) # from N,C,H,W to N,H,W,C
        # normal=normal.view(-1,3)
        # img_features_aggregated=torch.cat([img_features_aggregated, normal],1)





        TIME_START("rgb_predict")
        radiance_field_flattened, last_features, mask_pred = self.rgb_predictor(point3d, ray_dirs, point_features=img_features_aggregated, nr_points_per_ray=1, params=None  ) #radiance field has shape height,width, nr_samples,4
        TIME_END("rgb_predict")

        if pixels_indices==None:
            rgb_pred=radiance_field_flattened[:, 0:3]
            rgb_pred=rgb_pred.view(frame.height, frame.width,3)
            rgb_pred=rgb_pred.permute(2,0,1).unsqueeze(0)
            rgb_pred=torch.tanh(rgb_pred) #similar to the differentiable neural rendering

            mask_pred=mask_pred.view(frame.height, frame.width,1)
            mask_pred=mask_pred.permute(2,0,1).unsqueeze(0)

            # #refine the prediction with some Unet so we have also some spatial context
            if self.do_superres:
                TIME_START("superres")
                last_features=last_features.view(frame.height, frame.width,-1)
                last_features=last_features.permute(2,0,1).unsqueeze(0)
                rgb_low_res=torch.cat([rgb_pred,last_features],1)
                # print("rgb_low_res", rgb_low_res.shape)
                rgb_refined=self.super_res(rgb_low_res )
                # print("rgb_refined",rgb_refined.shape)
                # # rgb_refined=self.rgb_refiner(rgb_pred)
                TIME_END("superres")
            else:
                rgb_refined=None


            depth=depth.view(frame.height, frame.width,1)
            depth=depth.permute(2,0,1).unsqueeze(0)
        else:
            rgb_pred=radiance_field_flattened[:, 0:3]
            rgb_pred=rgb_pred.permute(1,0) #3 x nr_pixels
            mask_pred=None
            rgb_refined=None
            depth=None



        # # #DEBUG 
        # if novel:
        #     #show the PCAd features of the closest frame
        #     img_features=frames_features_list[0]
        #     height=img_features.shape[2]
        #     width=img_features.shape[3]
        #     img_features_for_pca=img_features.squeeze(0).permute(1,2,0).contiguous()
        #     img_features_for_pca=img_features_for_pca.view(height*width, -1)
        #     pca=PCA.apply(img_features_for_pca)
        #     pca=pca.view(height, width, 3)
        #     pca=pca.permute(2,0,1).unsqueeze(0)
        #     pca_mat=tensor2mat(pca)
        #     Gui.show(pca_mat, "pca_mat")

        return rgb_pred, rgb_refined, depth, mask_pred, signed_distances_for_marchlvl, std, raymarcher_loss, point3d


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



class DeferredNeuralRenderer(torch.nn.Module):
    def __init__(self):
        super(DeferredNeuralRenderer, self).__init__()

        self.first_time=True

        #models
        num_encoding_directions=4
        self.learned_pe_dirs=LearnedPE(in_channels=3, num_encoding_functions=num_encoding_directions, logsampling=True)
        self.unet=UNet( nr_channels_start=32, nr_channels_output=3, nr_stages=3, max_nr_channels=64)
        # self.unet=torch.nn.Sequential(
        #     BlockNerf(activ=torch.nn.GELU(), in_channels=32, out_channels=64,  bias=True ).cuda(),
        #     BlockNerf(activ=torch.nn.GELU(), in_channels=64, out_channels=128,  bias=True ).cuda(),
        #     BlockNerf(activ=torch.nn.GELU(), in_channels=128, out_channels=64,  bias=True ).cuda(),
        #     BlockNerf(activ=None, in_channels=64, out_channels=3,  bias=True ).cuda(),
        # )
        # self.unet=torch.nn.Sequential(
        #     BlockSiren(activ=torch.sin, in_channels=32, out_channels=64,  bias=True, is_first_layer=True, scale_init=10).cuda(),
        #     BlockSiren(activ=torch.sin, in_channels=64, out_channels=128,  bias=True, is_first_layer=False).cuda(),
        #     BlockSiren(activ=torch.sin, in_channels=128, out_channels=64,  bias=True, is_first_layer=False).cuda(),
        #     BlockNerf(activ=torch.tanh, in_channels=64, out_channels=3,  bias=True).cuda(),
        # )
        # self.unet=torch.nn.Sequential(
        #     torch.nn.Conv2d( 59, 64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda(),
        #     ResnetBlock2D(64, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=True ),
        #     ResnetBlock2D(64, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=True ),
        #     torch.nn.Conv2d( 64, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True).cuda() 
        # )
      

        #too high resolution will lead to flickering because we will optimize only some texels that are sampled during training but during testing we will sample other texels
        max_texture_size=512
        self.texture1= torch.zeros((1,32, max_texture_size, max_texture_size )).to("cuda") 
        self.texture2= torch.zeros((1,32, max_texture_size//2, max_texture_size//2 )).to("cuda") 
        self.texture3= torch.zeros((1,32, max_texture_size//4, max_texture_size//4 )).to("cuda") 
        self.texture4= torch.zeros((1,32, max_texture_size//8, max_texture_size//8 )).to("cuda") 
        self.texture1.requires_grad=True
        self.texture2.requires_grad=True
        self.texture3.requires_grad=True
        self.texture4.requires_grad=True
       


        #activ
        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()

        self.i=0
      

    #from https://github.com/SSRSGJYD/NeuralTexture/blob/master/model/pipeline.py
    def _spherical_harmonics_basis(self, extrinsics):
        '''
        extrinsics: a tensor shaped (N, 3)
        output: a tensor shaped (N, 9)
        '''
        batch = extrinsics.shape[0]
        sh_bands = torch.ones((batch, 9), dtype=torch.float)
        coff_0 = 1 / (2.0*math.sqrt(np.pi))
        coff_1 = math.sqrt(3.0) * coff_0
        coff_2 = math.sqrt(15.0) * coff_0
        coff_3 = math.sqrt(1.25) * coff_0
        # l=0
        sh_bands[:, 0] = coff_0
        # l=1
        sh_bands[:, 1] = extrinsics[:, 1] * coff_1
        sh_bands[:, 2] = extrinsics[:, 2] * coff_1
        sh_bands[:, 3] = extrinsics[:, 0] * coff_1
        # l=2
        sh_bands[:, 4] = extrinsics[:, 0] * extrinsics[:, 1] * coff_2
        sh_bands[:, 5] = extrinsics[:, 1] * extrinsics[:, 2] * coff_2
        sh_bands[:, 6] = (3.0 * extrinsics[:, 2] * extrinsics[:, 2] - 1.0) * coff_3
        sh_bands[:, 7] = extrinsics[:, 2] * extrinsics[:, 0] * coff_2
        sh_bands[:, 8] = (extrinsics[:, 0] * extrinsics[:, 0] - extrinsics[:, 2] * extrinsics[:, 2]) * coff_2
        return sh_bands


      
    def forward(self, frame, uv_tensor, ray_directions):
        # ray_directions=ray_directions.view(-1,3)
        # ray_directions=F.normalize(ray_directions, p=2, dim=1)
        # ray_directions=self.learned_pe_dirs(ray_directions, params=None)
        # ray_directions=ray_directions.view(1, frame.height, frame.width, -1).permute(0,3,1,2) #from N,H,W,C to N,C,H,W
        # # print("ray_directions", ray_directions.shape)

        # feat_input=torch.cat([texture_features,ray_directions],1)
        # # print("feat_input", feat_input.shape)

        # rgb_pred=self.unet( feat_input )
        # # print("rgb_pred",rgb_pred.shape)


        #sample features 
        texture_features1=torch.nn.functional.grid_sample( self.texture1, uv_tensor, align_corners=False, mode="bilinear" ) 
        texture_features2=torch.nn.functional.grid_sample( self.texture2, uv_tensor, align_corners=False, mode="bilinear" ) 
        texture_features3=torch.nn.functional.grid_sample( self.texture3, uv_tensor, align_corners=False, mode="bilinear" ) 
        texture_features4=torch.nn.functional.grid_sample( self.texture4, uv_tensor, align_corners=False, mode="bilinear" ) 
        texture_features= (texture_features1 + texture_features2 + texture_features3 + texture_features4) / 4




        ##atttempt 2 using spherical harmonics
        ray_directions=ray_directions.view(-1,3)
        ray_directions=F.normalize(ray_directions, p=2, dim=1)
        # basis = self._spherical_harmonics_basis(ray_directions).cuda()
        # basis=basis.view(1, frame.height, frame.width, 9).permute(0,3,1,2) #from N,H,W,C to N,C,H,W
        # print("basis is ", basis.shape)

        # texture_features[:, 3:12, :, :] = texture_features[:, 3:12, :, :] * basis
        # feat_input=texture_features
        # feat_input=torch.cat([texture_features,basis],1)
        #concating the dirs
        ray_directions=self.learned_pe_dirs(ray_directions, params=None)
        feat_dirs=ray_directions.shape[1]
        ray_directions=ray_directions.view(1,uv_tensor.shape[1], uv_tensor.shape[2], feat_dirs).permute(0,3,1,2) #from N,H,W,C to N,C,H,W
        feat_input=torch.cat([texture_features,ray_directions],1)

        # feat_nr=feat_input.shape[1]
        # feat_input=feat_input.permute(0,2,3,1).view(-1,feat_nr)# from N,C,H,W to N,H,W,C

        rgb_pred=self.unet( feat_input )
        rgb_pred=torch.tanh(rgb_pred)

        # rgb_pred=rgb_pred.view(1,frame.height, frame.width, 3).permute(0,3,1,2) #from N,H,W,C to N,C,H,W
        self.i+=1

        # #show pca texture 
        # if(self.i%100*10)==0:
        #     img_features=self.texture3
        #     height=img_features.shape[2]
        #     width=img_features.shape[3]
        #     img_features_for_pca=img_features.squeeze(0).permute(1,2,0).contiguous()
        #     img_features_for_pca=img_features_for_pca.view(height*width, -1)
        #     pca=PCA.apply(img_features_for_pca)
        #     pca=pca.view(height, width, 3)
        #     pca=pca.permute(2,0,1).unsqueeze(0)
        #     pca_mat=tensor2mat(pca)
        #     Gui.show(pca_mat, "pca_mat")
       
        return rgb_pred 

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

class PCA(Function):
    # @staticmethod
    def forward(ctx, sv): #sv corresponds to the slices values, it has dimensions nr_positions x val_full_dim

        # http://agnesmustar.com/2017/11/01/principal-component-analysis-pca-implemented-pytorch/


        X=sv.detach().cpu()#we switch to cpu of memory issues when doing svd on really big imaes
        k=3
        # print("x is ", X.shape)
        X_mean = torch.mean(X,0)
        # print("x_mean is ", X_mean.shape)
        X = X - X_mean.expand_as(X)

        U,S,V = torch.svd(torch.t(X)) 
        C = torch.mm(X,U[:,:k])
        # print("C has shape ", C.shape)
        # print("C min and max is ", C.min(), " ", C.max() )
        C-=C.min()
        C/=C.max()
        # print("after normalization C min and max is ", C.min(), " ", C.max() )

        return C

