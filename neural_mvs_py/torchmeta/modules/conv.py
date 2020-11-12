import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

from collections import OrderedDict
from torch.nn.modules.utils import _single, _pair, _triple
from torchmeta.modules.module import MetaModule

class MetaConv1d(nn.Conv1d, MetaModule):
    __doc__ = nn.Conv1d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv1d(F.pad(input, expanded_padding, mode='circular'),
                            params['weight'], bias, self.stride,
                            _single(0), self.dilation, self.groups)

        return F.conv1d(input, params['weight'], bias, self.stride,
                        self.padding, self.dilation, self.groups)

class MetaConv2d(nn.Conv2d, MetaModule):
    # __doc__ = nn.Conv2d.__doc__

    # def __init__(self, *args):
        # super().__init__(*args)

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,  bias=True):
        super(MetaConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    # def __init__(self, *args, **kwargs):
    #     super(MetaConv2d, self).__init__(*args, **kwargs)

    #     # super(MetaConv2d, self).__init__()
        # self.alpha_weights = torch.nn.Parameter(torch.randn(1))
        # torch.nn.init.constant_(self.alpha_weights, 0.01)
        # self.alpha_bias = torch.nn.Parameter(torch.randn(1))
        # torch.nn.init.constant_(self.alpha_bias, 0.01)

    def forward(self, input, params=None, incremental=False):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            params['weight'], bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        # return F.conv2d(input, params['weight'], bias, self.stride,
        # print("am in metaconv2 and input has shape ", input.shape)
        # print("am in metaconv2 and weight has shape ", params['weight'].shape)

        weights=params['weight']

        # #if it's the first layer of siren initialize accordingly 
        # weight_init=params['weight'].clone().detach()
        # print("weight has shape ", params['weight'].shape)
        # num_input=params['weight'].shape[1]
        # num_output=params['weight'].shape[0]
        # if(params['weight'].shape==torch.Size([128, 2, 1, 1]) ):
        #     print("we are in the first layer of siren")
        #     weight_init.uniform_(-1 / num_input, 1 / num_input)
        #     # print("weight init is ", weight_init)
        #     # print("weight computed is ", weights)
        #     weights=weights+weight_init
        #     # weights=weight_init
        # elif(params['weight'].shape==torch.Size([128, 128, 1, 1]) ):
        #     print("any other layer of siren")
        #     weight_init.uniform_(-np.sqrt(6 / num_input)/30 , np.sqrt(6 / num_input)/30 )
        #     weights=weights+weight_init
        #     # weights=weight_init



        #if we didn't pass any new parameters into params argument then params[weight] and self.weight are actually the same tensor
        if incremental:
            have_new_weights=False
            if params['weight'].data_ptr() != self.weight.data_ptr():
                have_new_weights=True
            #instead of predicting the whole weights we predict delta weight of the initial ones
            if have_new_weights:
                initial_weights=self.weight.clone().detach() # the initial weight do not get optimized
                # print("initial_weights", initial_weights.shape)
                # initial_weights.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
                # print("initial_weights", initial_weights)
                # print("self.alpha_weights", self.alpha_weights)
                # new_weights=initial_weights+weights * self.alpha_weights #we add the new weights with a learnable alpha that starts very low at the beggining to ensure that we mostly use the initial weights and then move more towards the new ones
                new_weights=initial_weights+weights * 0.1 #we add the new weights with a learnable alpha that starts very low at the beggining to ensure that we mostly use the initial weights and then move more towards the new ones
                weights=new_weights
                #do the same for bias
                if bias is not None:
                    initial_bias=self.bias.clone().detach() # the initial weight do not get optimized
                    # print("initial_bias", initial_bias)
                    # new_bias=initial_bias+bias * self.alpha_bias
                    new_bias=initial_bias+bias * 0.1
                    bias=new_bias
            # else: 
                # print("we are just using old weights")

        

        return F.conv2d(input, weights, bias, self.stride,
                        self.padding, self.dilation, self.groups)

class MetaConv3d(nn.Conv3d, MetaModule):
    __doc__ = nn.Conv3d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2,
                                (self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv3d(F.pad(input, expanded_padding, mode='circular'),
                            params['weight'], bias, self.stride,
                            _triple(0), self.dilation, self.groups)

        return F.conv3d(input, params['weight'], bias, self.stride,
                        self.padding, self.dilation, self.groups)
