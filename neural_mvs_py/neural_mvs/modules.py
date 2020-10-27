import torch
import torch.nn as nn
import torch.nn.functional as F
from easypbr  import *
import sys
import math

from neural_mvs_py.neural_mvs.funcs import *


from torchmeta.modules.conv import MetaConv2d
from torchmeta.modules.module import *
from torchmeta.modules.utils import *
from torchmeta.modules import (MetaModule, MetaSequential)

from neural_mvs_py.neural_mvs.pac import *



def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))



class Block(MetaModule):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.relu, init=None, do_norm=False, is_first_layer=False ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.GELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.sin, init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.SELU(inplace=False), init=None ):
        super(Block, self).__init__()
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

        self.relu=torch.nn.ReLU(inplace=True)

        # self.conv=None

        # self.norm = torch.nn.BatchNorm2d(in_channels, momentum=0.01).cuda()
        # self.norm = torch.nn.BatchNorm2d(self.out_channels, momentum=0.01).cuda()
        # self.norm = torch.nn.GroupNorm(1, in_channels).cuda()

        # self.sine_scale=torch.nn.Parameter(torch.Tensor(1)).cuda()
        # self.sine_scale=torch.nn.Parameter(torch.randn(3,6)).cuda()
        # torch.nn.init.constant_(self.sine_scale, 30)
        # with torch.set_grad_enabled(False):
            # self.sine_scale=30
        # self.sine_scale.requires_grad = True
        # self.wtf=torch.nn.Linear(10,10)
        # self.weight = torch.nn.Parameter(torch.Tensor(10, 10))
        # torch.nn.init.uniform_(self.sine_scale, -1, 1)
        # self.W = torch.nn.Parameter(torch.randn(3,4,5))
        self.sine_scale = torch.nn.Parameter(torch.randn(1))
        torch.nn.init.constant_(self.sine_scale, 30)
        # self.W.requires_grad = True

        if not self.transposed:
            self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
        else:
            self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )

        if self.init=="zero":
                torch.nn.init.zeros_(self.conv[-1].weight) 
        if self.activ==torch.sin:
            with torch.no_grad():
                # print(":we are usign sin")
                # print("in channels is ", in_channels, " and conv weight size 1 is ", self.conv.weight.size(-1) )
                # num_input = self.conv.weight.size(-1)
                num_input = in_channels
                # num_input = self.out_channels
                # See supplement Sec. 1.5 for discussion of factor 30
                if self.is_first_layer:
                    # self.conv[-1].weight.uniform_(-1 / num_input, 1 / num_input)
                    # self.conv[-1].weight.uniform_(-1 / num_input*2, 1 / num_input*2)
                    self.conv[-1].weight.uniform_(-1 / num_input, 1 / num_input)
                    # print("conv 1 is ", self.conv[-1].weight )
                else:
                    self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input)/30 , np.sqrt(6 / num_input)/30 )
                    # self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
                    # self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input)/7 , np.sqrt(6 / num_input)/7 )
                    # print("conv any other is ", self.conv[-1].weight )
        if self.activ==torch.relu:
            print("initializing with kaiming uniform")
            torch.nn.init.kaiming_uniform_(self.conv[-1].weight, a=math.sqrt(5), mode='fan_out', nonlinearity='relu')
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv[-1].weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.conv[-1].bias, -bound, bound)
        

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        # print("params is", params)

        # x=self.cc(x)

        in_channels=x.shape[1]


        #create modules if they are not created
        # if self.norm is None:
        #     # self.norm = torch.nn.GroupNorm(in_channels, in_channels).cuda()
        #     nr_groups=32
        #     nr_params=in_channels
        #     if nr_params<=32:
        #         nr_groups=int(nr_params/2)
            # self.norm = torch.nn.GroupNorm(nr_groups, nr_params).cuda()
            # self.norm = torch.nn.GroupNorm(16, self.out_channels).cuda()
            # self.norm = torch.nn.GroupNorm(1, in_channels).cuda()
            # self.norm = torch.nn.GroupNorm(in_channels, in_channels).cuda()
            # self.norm = torch.nn.GroupNorm(self.out_channels, self.out_channels).cuda()
            # self.norm = torch.nn.BatchNorm2d(in_channels, momentum=0.01).cuda()
            # self.norm = torch.nn.BatchNorm2d(self.out_channels, momentum=0.01).cuda()
        # if self.conv is None:
        #     # self.net=[]
        #     if not self.transposed:
        #         self.conv= MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  
        #     else:
        #         self.conv= MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda() 

        #     if self.init=="zero":
        #         torch.nn.init.zeros_(self.conv.weight) 
        #     if self.activ==torch.sin:
        #         with torch.no_grad():
        #             # print(":we are usign sin")
        #             # print("in channels is ", in_channels, " and conv weight size 1 is ", self.conv.weight.size(-1) )
        #             # num_input = self.conv.weight.size(-1)
        #             num_input = in_channels
        #             # num_input = self.out_channels
        #             # See supplement Sec. 1.5 for discussion of factor 30
        #             if self.is_first_layer:
        #                 self.conv.weight.uniform_(-1 / num_input, 1 / num_input)
        #             else:
        #                 self.conv.weight.uniform_(-np.sqrt(6 / num_input)/30 , np.sqrt(6 / num_input)/30 )

      


        #pass the tensor through the modules
        # if self.do_norm:
            # x=self.norm(x) # TODO The vae seems to work a lot better without any normalization but more testing might be needed
        # if self.do_norm:
            # x=self.norm(x) # TODO The vae seems to work a lot better without any normalization but more testing might be needed
        # x=gelu(x)
        # x=self.activ(x)
        # if self.with_dropout:
            # x = self.drop(x)
        # if self.activ==torch.sin:
            # print("am in a sin acitvation, x before conv is ", x.shape)
            # print("am in a sin acitvation, conv has params with shape ", self.conv[-1].weight.shape)
        # x=self.norm(x) # TODO The vae seems to work a lot better without any normalization but more testing might be needed

        # if self.is_first_layer:
        #     x=30*x
        

        # print("before conv, x has mean and std " , x.mean() , " std ", x.std() )
        x = self.conv(x, params=get_subdict(params, 'conv') )
        # if self.do_norm:
            # print("norm")
            # if(x.shape[1]%16==0):
                # x=self.norm(x) # TODO The vae seems to work a lot better without any normalization but more testing might be needed
            # x=self.norm(x) # TODO The vae seems to work a lot better without any normalization but more testing might be needed
        # x=self.relu(x)
        # x=self.norm(x) # TODO The vae seems to work a lot better without any normalization but more testing might be needed
        if self.activ==torch.sin:
            # x=30*x
            x=30*x
            # x=self.sine_scale*x
            # print("self.sine_scale", self.sine_scale)
            # print("before activ, x has mean and std " , x.mean() , " std ", x.std() )
            # print("before activ, x*30 has mean and std " , (x*30).mean() , " std ", (x*30).std() )
            # x=self.activ(30*x)
            x=self.activ(x)
            # print("after activ, x has mean and std " , x.mean() , " std ", x.std() )
        elif self.activ is not None:
            # x=self.activ(x)
            x=self.relu(x)
            # print("after activ, x has mean and std " , x.mean() , " std ", x.std() )
        # x=gelu(x)
        # x=torch.sin(x)
        # x=torch.sigmoid(x)

        return x

class BNReluConv(MetaModule):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.relu, init=None, do_norm=False, is_first_layer=False ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.GELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.sin, init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.SELU(inplace=False), init=None ):
        super(BNReluConv, self).__init__()
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

       
        self.norm = torch.nn.BatchNorm2d(in_channels).cuda()
        # self.norm = torch.nn.GroupNorm(1, in_channels).cuda()
        self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
        

       
        print("initializing with kaiming uniform")
        torch.nn.init.kaiming_uniform_(self.conv[-1].weight, a=math.sqrt(5), mode='fan_out', nonlinearity='tanh')
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv[-1].weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.conv[-1].bias, -bound, bound)
    

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        # print("params is", params)

        # x=self.cc(x)

        in_channels=x.shape[1]


        if self.activ !=None: 
            x=self.norm(x)
            x=self.activ(x)
        x = self.conv(x, params=get_subdict(params, 'conv') )
       

        return x


class BlockPAC(torch.nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.relu, init=None, do_norm=False, is_first_layer=False ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.GELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.sin, init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.SELU(inplace=False), init=None ):
        super(BlockPAC, self).__init__()
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

       
        self.norm = torch.nn.BatchNorm2d(in_channels).cuda()
        self.conv=PacConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias).cuda() 
        

       
        # print("initializing with kaiming uniform")
        # torch.nn.init.kaiming_uniform_(self.conv[-1].weight, a=math.sqrt(5), mode='fan_out', nonlinearity='tanh')
        # if self.bias is not None:
        #     fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv[-1].weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     torch.nn.init.uniform_(self.conv[-1].bias, -bound, bound)
    

    def forward(self, x, guide):
        # if params is None:
            # params = OrderedDict(self.named_parameters())
        # print("params is", params)

        # x=self.cc(x)

        in_channels=x.shape[1]


        x = self.conv(x, guide )

        if self.activ !=None: 
            x=self.activ(x)
       

        return x

class BlockSiren(MetaModule):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ReLU(inplace=False), init=None, do_norm=False, is_first_layer=False ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.GELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.sin, init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.SELU(inplace=False), init=None ):
        super(BlockSiren, self).__init__()
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

        self.relu=torch.nn.ReLU()
        self.leaky_relu=torch.nn.LeakyReLU(negative_slope=0.1)

        # self.conv=None

        # self.norm = torch.nn.BatchNorm2d(in_channels, momentum=0.01).cuda()
        # self.norm = torch.nn.BatchNorm2d(self.out_channels, momentum=0.01).cuda()
        # self.norm = torch.nn.GroupNorm(1, in_channels).cuda()

        # self.sine_scale=torch.nn.Parameter(torch.Tensor(1)).cuda()
        # self.sine_scale=torch.nn.Parameter(torch.randn(3,6)).cuda()
        # torch.nn.init.constant_(self.sine_scale, 30)
        # with torch.set_grad_enabled(False):
            # self.sine_scale=30
        # self.sine_scale.requires_grad = True
        # self.wtf=torch.nn.Linear(10,10)
        # self.weight = torch.nn.Parameter(torch.Tensor(10, 10))
        # torch.nn.init.uniform_(self.sine_scale, -1, 1)
        # self.W = torch.nn.Parameter(torch.randn(3,4,5))
        self.sine_scale = torch.nn.Parameter(torch.randn(1))
        torch.nn.init.constant_(self.sine_scale, 30)
        # self.W.requires_grad = True

        if not self.transposed:
            self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
            # self.conv_alt= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
        else:
            self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )

        if self.init=="zero":
                torch.nn.init.zeros_(self.conv[-1].weight) 
        if self.activ==torch.sin or self.activ==None:
            with torch.no_grad():
                # print(":we are usign sin")
                # print("in channels is ", in_channels, " and conv weight size 1 is ", self.conv.weight.size(-1) )
                # num_input = self.conv.weight.size(-1)
                num_input = in_channels
                # num_input = self.out_channels
                # See supplement Sec. 1.5 for discussion of factor 30
                if self.is_first_layer:
                    # self.conv[-1].weight.uniform_(-1 / num_input, 1 / num_input)
                    self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
                    # print("conv 1 is ", self.conv[-1].weight )
                else:
                    # self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input)/30 , np.sqrt(6 / num_input)/30 )
                    self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
                    # print("conv any other is ", self.conv[-1].weight )
                # self.conv[-1].bias.zero_()

        self.iter=1

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
       

        in_channels=x.shape[1]

        x_input=x


        # print("before conv, x has mean and std " , x.mean() , " std ", x.std() )
        x = self.conv(x, params=get_subdict(params, 'conv') )
        # x_relu=self.conv_alt(x_input,  params=get_subdict(params, 'conv_alt') )
        # x_relu=self.leaky_relu(x_relu)
        if self.activ==torch.sin:
            # print("before 30x, x has mean and std " , x.mean().item() , " std ", x.std().item(), " min: ", x.min().item(),  "max ", x.max().item() )
            if self.is_first_layer: 
                # x_conv_scaled=30*x_conv
                # x=x*(5+self.iter*0.01)
                x=90*x
            # x_conv_scaled=x_conv
            else: 
                x=x*1
            # print("before activ, x has mean and std " , x.mean().item() , " std ", x.std().item(), " min: ", x.min().item(),  "max ", x.max().item() )
            x=self.activ(x)
            # x_relu=self.relu(x_conv)
            #each x will map into a certain period of the sine depending on their value, the network has to be aware of which sine it will activate
            # x_pos = x/30
            # x=torch.cat( [x_sine, x_relu],1)
            # x=torch.cat( [x_sine, x_pos],1)
            # x=x_sine+x_conv%(3.14)
            # x=x_sine + x_relu
            # x=x_sine
            # print("after activ, x has mean and std " , x.mean().item() , " std ", x.std().item(), " min: ", x.min().item(),  "max ", x.max().item() )


            # # if self.is_first_layer:
            # #check the layer
            # print("x.shape ", x.shape)
            # nr_layers=x.shape[1]
            # for i in range(10):
            #     layer=x[:,i:i+1, :, :]
            #     layer=(layer+1.0)*0.5
            #     layer_mat=tensor2mat(layer)
            #     Gui.show(layer_mat, "layer_"+str(i))


        elif self.activ is not None:
            x=self.activ(x)
        # elif self.activ is None:
            # x=x_conv

        # print("x has shape ", x.shape)

        # x=x+x_relu

        return x

class BlockForResnet(MetaModule):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ReLU(inplace=False), init=None, do_norm=False, is_first_layer=False ):
        super(BlockForResnet, self).__init__()
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

        # self.conv=None

        # self.norm = torch.nn.BatchNorm2d(in_channels, momentum=0.01).cuda()
        self.norm = torch.nn.BatchNorm2d(in_channels).cuda()
        # self.norm = torch.nn.BatchNorm2d(self.out_channels, momentum=0.01).cuda()
        # self.norm = torch.nn.GroupNorm(1, in_channels).cuda()


        if not self.transposed:
            self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
        else:
            self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )

        if self.init=="zero":
                torch.nn.init.zeros_(self.conv[-1].weight) 
        if self.activ==torch.sin:
            with torch.no_grad():
                # print(":we are usign sin")
                # print("in channels is ", in_channels, " and conv weight size 1 is ", self.conv.weight.size(-1) )
                # num_input = self.conv.weight.size(-1)
                num_input = in_channels
                # num_input = self.out_channels
                # See supplement Sec. 1.5 for discussion of factor 30
                if self.is_first_layer:
                    # self.conv[-1].weight.uniform_(-1 / num_input, 1 / num_input)
                    # self.conv[-1].weight.uniform_(-1 / num_input*2, 1 / num_input*2)
                    self.conv[-1].weight.uniform_(-1 / num_input, 1 / num_input)
                    # print("conv 1 is ", self.conv[-1].weight )
                else:
                    self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input)/30 , np.sqrt(6 / num_input)/30 )
                    # self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
                    # self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input)/7 , np.sqrt(6 / num_input)/7 )
                    # print("conv any other is ", self.conv[-1].weight )

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        # print("params is", params)

        # x=self.cc(x)

        in_channels=x.shape[1]


        x=self.norm(x)
        x=self.activ(x)
        x = self.conv(x, params=get_subdict(params, 'conv') )

        return x




class ResnetBlock(torch.nn.Module):

    def __init__(self, out_channels, kernel_size, stride, padding, dilations, biases, with_dropout, do_norm=False, activ=torch.nn.ReLU(inplace=False), is_first_layer=False ):
    # def __init__(self, out_channels, kernel_size, stride, padding, dilations, biases, with_dropout, do_norm=False, activ=torch.nn.GELU(), is_first_layer=False ):
        super(ResnetBlock, self).__init__()

        #again with bn-relu-conv
        # self.conv1=GnReluConv(out_channels, kernel_size=3, stride=1, padding=1, dilation=dilations[0], bias=biases[0], with_dropout=False, transposed=False)
        # self.conv2=GnReluConv(out_channels, kernel_size=3, stride=1, padding=1, dilation=dilations[0], bias=biases[0], with_dropout=with_dropout, transposed=False)

        self.conv1=BlockForResnet(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=False, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=is_first_layer )
        self.conv2=BlockForResnet(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=with_dropout, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=False )

        # self.conv1=ConvRelu(out_channels, kernel_size=3, stride=1, padding=1, dilation=dilations[0], bias=biases[0], with_dropout=False, transposed=False)
        # self.conv2=ConvRelu(out_channels, kernel_size=3, stride=1, padding=1, dilation=dilations[0], bias=biases[0], with_dropout=with_dropout, transposed=False)

    def forward(self, x):
        identity=x
        x=self.conv1(x)
        x=self.conv2(x)
        x+=identity
        return x

class ConcatCoord(torch.nn.Module):
    def __init__(self):
        super(ConcatCoord, self).__init__()

    def forward(self, x):

        #concat the coordinates in x an y as in coordconv https://github.com/Wizaron/coord-conv-pytorch/blob/master/coord_conv.py
        image_height=x.shape[2]
        image_width=x.shape[3]
        y_coords = 2.0 * torch.arange(image_height).unsqueeze(
            1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(image_width).unsqueeze(
            0).expand(image_height, image_width) / (image_width - 1.0) - 1.0
        coords = torch.stack((y_coords, x_coords), dim=0).float()
        coords=coords.unsqueeze(0)
        coords=coords.repeat(x.shape[0],1,1,1)
        # print("coords have size ", coords.size())
        x_coord = torch.cat((coords.to("cuda"), x), dim=1)

        return x_coord

class LearnedPE(torch.nn.Module):
    def __init__(self, in_channels, num_encoding_functions, logsampling ):
        super(LearnedPE, self).__init__()
        self.num_encoding_functions=num_encoding_functions
        self.logsampling=logsampling

        out_channels=3*self.num_encoding_functions*2
       
        # self.conv= torch.nn.Linear(in_channels, out_channels, bias=True).cuda()  
        self.conv= torch.nn.Linear(in_channels, int(out_channels/2), bias=True).cuda()  #in the case we set the weight ourselves

        with torch.no_grad():
            num_input = in_channels
            self.conv.weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
            print("weight is ", self.conv.weight.shape) #60x3
            
            #we make the same as the positonal encoding, which is mutiplying each coordinate with this linespaced frequencies
            lin=2.0 ** torch.linspace(
                0.0,
                num_encoding_functions - 1,
                num_encoding_functions,
                dtype=torch.float32,
                device=torch.device("cuda"),
            )
            lin_size=lin.shape[0]
            weight=torch.zeros([in_channels, num_encoding_functions*in_channels], dtype=torch.float32, device=torch.device("cuda") )
            for i in range(in_channels):
                weight[i:i+1,   i*lin_size:i*lin_size+lin_size ] = lin

            weight=weight.t().contiguous()

            #set the new weights = 
            self.conv.weight=torch.nn.Parameter(weight)
            # self.conv.weight.requires_grad=False
            print("weight is", weight.shape)
            print("weight is", weight)


    def forward(self, x):
        
        x_input=x

        # print("self.conv.weight", self.conv.weight)

        # print("x ", x.shape)
        x = self.conv(x)
        # print("after conv", x.shape)
        # x=90*x
        x_sin=torch.sin(x)
        x_cos=torch.cos(x)
        x=torch.cat([x_sin, x_cos],1)

        x=torch.cat([x,x_input],1)

        # print("after sin")
        # x=x.contiguous()
          

        return x
