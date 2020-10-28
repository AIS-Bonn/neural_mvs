import torch.nn as nn
import torch.nn.functional as F
import torch


from collections import OrderedDict
from torchmeta.modules.module import MetaModule

class MetaLinear(nn.Linear, MetaModule):
    # __doc__ = nn.Linear.__doc__

    def __init__(self, in_channels, out_channels,  bias=True):
        super(MetaLinear, self).__init__(in_channels, out_channels, bias=bias)

        self.alpha_weights = torch.nn.Parameter(torch.randn(1))
        torch.nn.init.constant_(self.alpha_weights, 0.01)
        self.alpha_bias = torch.nn.Parameter(torch.randn(1))
        torch.nn.init.constant_(self.alpha_bias, 0.01)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        weights=params['weight']

        #if we didn't pass any new parameters into params argument then params[weight] and self.weight are actually the same tensor
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
            new_weights=initial_weights+weights * self.alpha_weights #we add the new weights with a learnable alpha that starts very low at the beggining to ensure that we mostly use the initial weights and then move more towards the new ones
            weights=new_weights
            #do the same for bias
            if bias is not None:
                initial_bias=self.bias.clone().detach() # the initial weight do not get optimized
                # print("initial_bias", initial_bias)
                new_bias=initial_bias+bias * self.alpha_bias
                bias=new_bias
        # else: 
            # print("we are just using old weights") 


        return F.linear(input, weights, bias)

class MetaBilinear(nn.Bilinear, MetaModule):
    __doc__ = nn.Bilinear.__doc__

    def forward(self, input1, input2, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.bilinear(input1, input2, params['weight'], bias)
