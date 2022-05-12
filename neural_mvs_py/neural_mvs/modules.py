import torch
import torch.nn as nn
import torch.nn.functional as F
from easypbr  import *
import sys
import math


from neural_mvs_py.neural_mvs.utils import *


#wraps module, and changes them to become a torchscrip version of them during inference
class TorchScriptTraceWrapper(torch.nn.Module):
    def __init__(self, module):
        super(TorchScriptTraceWrapper, self).__init__()

        self.module=module
        self.module_traced=None

    def forward(self, *args):
        args_list=[]
        for arg in args:
            args_list.append(arg)
        if self.module_traced==None:
                self.module_traced = torch.jit.trace(self.module, args_list )
        return self.module_traced(*args)

class FrameWeightComputer(torch.nn.Module):

    def __init__(self ):
        super(FrameWeightComputer, self).__init__()

        # self.s_weight = torch.nn.Parameter(torch.randn(1))  #from equaiton 3 here https://arxiv.org/pdf/2010.08888.pdf
        # with torch.set_grad_enabled(False):
        #     # self.s_weight.fill_(0.5)
        #     self.s_weight.fill_(10.0)

        ######CREATING A NEW parameter for the s_weight for some reason destroys the rest of the network and it doesnt optimize anymore. Either way, it barely changes so we just set it to 10
        # self.s_weight=10
        self.s_weight=1

    def forward(self, frame, frames_close):
        cur_dir=frame.look_dir
        exponential_weight_towards_neighbour=[]
        for i in range(len(frames_close)):
            dir_neighbour=frames_close[i].look_dir
            dot= torch.dot( cur_dir.view(-1), dir_neighbour.view(-1) )
            s_dot= self.s_weight*(dot-1)
            exp=torch.exp(s_dot)
            exponential_weight_towards_neighbour.append(exp.view(1))
        all_exp=torch.cat(exponential_weight_towards_neighbour)
        exp_minimum= all_exp.min()
        unnormalized_weights=[]
        for i in range(len(frames_close)):
            cur_exp= exponential_weight_towards_neighbour[i]
            exp_sub_min= cur_exp-exp_minimum
            unnormalized_weight= torch.relu(exp_sub_min)
            unnormalized_weights.append(unnormalized_weight)
            # print("unnormalized_weight", unnormalized_weight)
        all_unormalized_weights=torch.cat(unnormalized_weights)
        weight_sum=all_unormalized_weights.sum()
        weights=[]
        for i in range(len(frames_close)):
            unnormalized_weight= unnormalized_weights[i]
            weight= unnormalized_weight/weight_sum
            weights.append(weight)
        weights=torch.cat(weights)

        # ##attempt 2 by just using barycentric coords
        # frames_close_list=[]
        # for framepy in frames_close:
        #     frames_close_list.append(framepy.frame)
        # weights_vec=SFM.compute_frame_weights(frame.frame, frames_close_list)
        # # print("weigrs vec is ", weights_vec)
        # weights=torch.from_numpy( np.array(weights_vec) ).float().to("cuda")
        # #clamp them
        # weights=torch.clamp(weights,0.0, 1.0)


        return weights






class WNConvActiv(torch.nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.Mish(), init=None, do_norm=False, is_first_layer=False ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.GELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.sin, init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.SELU(inplace=False), init=None ):
        super(WNConvActiv, self).__init__()
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.bias=bias 
        self.conv= None
        self.norm= None
        # self.relu=torch.nn.ReLU(inplace=False)
        self.activ=activ
        self.with_dropout=with_dropout
        self.transposed=transposed
        # self.cc=ConcatCoord()
        self.init=init
        self.do_norm=do_norm
        self.is_first_layer=is_first_layer

        if with_dropout:
            self.drop=torch.nn.Dropout2d(0.2)

        # if do_norm:
        #     nr_groups=32
        #     #if the groups is not diivsalbe so for example if we have 80 params
        #     if in_channels%nr_groups!=0:
        #         nr_groups= int(in_channels/4)
        #     if in_channels==32:
        #         nr_groups= int(in_channels/4)
        #     # print("nr groups is ", nr_groups, " in channels ", in_channels)
        #     self.norm = torch.nn.GroupNorm(nr_groups, in_channels).cuda()
            # self.norm = torch.nn.GroupNorm(in_channels, in_channels).cuda()
            # self.norm = torch.nn.GroupNorm( int(in_channels/4), in_channels).cuda()
        # self.norm = torch.nn.GroupNorm(1, in_channels).cuda()
        if do_norm:
            if self.transposed:
                self.conv= ConvTranspose2dWN(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda() 
            else:
                self.conv=  Conv2dWN(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  
                # self.conv=  DeformConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias).cuda()  
        else: 
            if self.transposed:
                self.conv=torch.nn.ConvTranspose2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()
            else:
                self.conv= torch.nn.Conv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()
        
        
        # self.conv=  torch.nn.Conv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  
        

       
        # print("initializing with kaiming uniform")
        # torch.nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
        # if self.bias is not False:
        #     fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     torch.nn.init.uniform_(self.conv.bias, -bound, bound)
    

    def forward(self, x):
        # if params is None:
            # params = OrderedDict(self.named_parameters())
        # print("params is", params)

        # x=self.cc(x)

        in_channels=x.shape[1]

        # if self.do_norm:
            # x=self.norm(x)
        x = self.conv(x )
        if self.activ !=None: 
            x=self.activ(x)
       

        return x



class TwoBlock2D(torch.nn.Module):

    def __init__(self, out_channels, kernel_size, stride, padding, dilations, biases, with_dropout, do_norm=False, activ=torch.nn.Mish(), is_first_layer=False, block_type=WNConvActiv ):
    # def __init__(self, out_channels, kernel_size, stride, padding, dilations, biases, with_dropout, do_norm=False, activ=torch.nn.GELU(), is_first_layer=False ):
    # def __init__(self, out_channels, kernel_size, stride, padding, dilations, biases, with_dropout, do_norm=False, activ=torch.nn.GELU(), is_first_layer=False ):
        super(TwoBlock2D, self).__init__()

        #again with bn-relu-conv
        # self.conv1=GnReluConv(out_channels, kernel_size=3, stride=1, padding=1, dilation=dilations[0], bias=biases[0], with_dropout=False, transposed=False)
        # self.conv2=GnReluConv(out_channels, kernel_size=3, stride=1, padding=1, dilation=dilations[0], bias=biases[0], with_dropout=with_dropout, transposed=False)

        # self.conv1=BlockForResnet(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=False, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=is_first_layer )
        # self.conv2=BlockForResnet(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=with_dropout, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=False )

        # self.conv1=BlockPAC(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=False, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=is_first_layer )
        # self.conv2=BlockPAC(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=with_dropout, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=False )

        self.conv1=block_type(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=False, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=is_first_layer )
        self.conv2=block_type(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=with_dropout, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=False )

    def forward(self, x):
        identity=x
        # x=self.conv1(x, x)
        # x=self.conv2(x, x)
        x=self.conv1(x)
        x=self.conv2(x)
        return x




class ConcatCoord(torch.nn.Module):
    def __init__(self):
        super(ConcatCoord, self).__init__()

    def forward(self, x):

        #concat the coordinates in x an y as in coordconv https://github.com/Wizaron/coord-conv-pytorch/blob/master/coord_conv.py
        image_height=x.shape[2]
        image_width=x.shape[3]
        x_coords = 2.0 * torch.arange(image_width).unsqueeze(
            0).expand(image_height, image_width) / (image_width - 1.0) - 1.0
        y_coords = 2.0 * torch.arange(image_height).unsqueeze(
            1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        coords = torch.stack((x_coords, y_coords), dim=0).float()
        coords=coords.unsqueeze(0)
        coords=coords.repeat(x.shape[0],1,1,1)
        # print("coords have size ", coords.size())
        x_coord = torch.cat((coords.to("cuda"), x), dim=1)

        return x_coord

class PositionalEncoding(torch.nn.Module):
    def __init__(self, in_channels, num_encoding_functions):
        super(PositionalEncoding, self).__init__()
        self.in_channels=in_channels
        self.num_encoding_functions=num_encoding_functions

        out_channels=in_channels*self.num_encoding_functions*2

        self.conv= torch.nn.Linear(in_channels, int(out_channels/2), bias=False).cuda()  #in the case we set the weight ourselves
        self.init_weights()


        #we dont train because that causes it to overfit to the input views and not generalize the specular effects to novel views
        self.conv.weight.requires_grad = False

    def init_weights(self):
        with torch.no_grad():
            num_input = self.in_channels
            self.conv.weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
            # print("weight is ", self.conv.weight.shape) #60x3

            #we make the same as the positonal encoding, which is mutiplying each coordinate with this linespaced frequencies
            lin=2.0 ** torch.linspace(
                0.0,
                self.num_encoding_functions - 1,
                self.num_encoding_functions,
                dtype=torch.float32,
                device=torch.device("cuda"),
            )
            lin_size=lin.shape[0]
            weight=torch.zeros([self.in_channels, self.num_encoding_functions*self.in_channels], dtype=torch.float32, device=torch.device("cuda") )
            for i in range(self.in_channels):
                weight[i:i+1,   i*lin_size:i*lin_size+lin_size ] = lin

            weight=weight.t().contiguous()

            #set the new weights =
            self.conv.weight=torch.nn.Parameter(weight)
            # self.conv.weight.requires_grad=False
            # print("weight is", weight.shape)
            # print("bias is", self.conv.bias.shape)
            # print("weight is", weight)

            self.weights_initialized=True


    def forward(self, x):

        with torch.no_grad():

            x_proj = self.conv(x)

            # if self.only_sin:
                # return torch.cat([x, torch.sin(x_proj) ], -1)
            # else:
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj), x], -1)

