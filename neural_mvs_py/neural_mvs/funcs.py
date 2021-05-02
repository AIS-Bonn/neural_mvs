import torch
from torch.autograd import Function
from torch import Tensor

import sys
from easypbr  import Profiler
import numpy as np
import time
import math
from neuralmvs import NeuralMVS
from neuralmvs import SFM
from easypbr import * 

#Just to have something close to the macros we have in c++
def profiler_start(name):
    if(Profiler.is_profiling_gpu()):
        torch.cuda.synchronize()
    Profiler.start(name)
def profiler_end(name):
    if(Profiler.is_profiling_gpu()):
        torch.cuda.synchronize()
    Profiler.end(name)
TIME_START = lambda name: profiler_start(name)
TIME_END = lambda name: profiler_end(name)



# print("creating neural mbs")
neural_mvs=NeuralMVS.create()
# print("created neural mvs")

#inits from SRN paper  https://github.com/vsitzmann/scene-representation-networks/blob/8165b500816bb1699f5a34782455f2c4b6d4f35a/custom_layers.py
def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name: continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.)




def show_3D_points(points_3d_tensor, color=None):
    mesh=Mesh()
    mesh.V=points_3d_tensor.detach().double().reshape((-1, 3)).cpu().numpy()

    if color is not None:
        color_channels_last=color.permute(0,2,3,1).detach() # from n,c,h,w to N,H,W,C
        color_channels_last=color_channels_last.view(-1,3).contiguous()
        # color_channels_last=color_channels_last.permute() #from bgr to rgb
        color_channels_last=torch.index_select(color_channels_last, 1, torch.LongTensor([2,1,0]).cuda() ) #switch the columns so that we grom from bgr to rgb
        mesh.C=color_channels_last.detach().double().reshape((-1, 3)).cpu().numpy()
        mesh.m_vis.set_color_pervertcolor()

    mesh.m_vis.m_show_points=True
    # Scene.show(mesh, name)

    return mesh


class SplatTexture(Function):
    @staticmethod
    def forward(ctx, values_tensor, uv_tensor, texture_height, texture_width):

        ctx.save_for_backward(values_tensor, uv_tensor)
        # ctx.neural_mvs=neural_mvs

        # texture = neural_mvs.splat_texture(values_tensor, uv_tensor, texture_size)
        texture = NeuralMVS.splat_texture(values_tensor, uv_tensor, texture_height, texture_width)

        return texture

    @staticmethod
    def backward(ctx, grad_texture):

        #I only want the gradient with respect to the UV tensor 
        values_tensor, uv_tensor =ctx.saved_tensors
        # neural_mvs=ctx.neural_mvs

        # grad_val = neural_mesh_obj.slice_texture(grad_texture, uv_tensor)
        # grad_values, grad_uv =neural_mvs.splat_texture_backward( grad_texture, values_tensor, uv_tensor )
        grad_values, grad_uv =NeuralMVS.splat_texture_backward( grad_texture, values_tensor, uv_tensor )
        # TODO we might need also the grad for values
        
        # return None, None, None
        # return None, grad_uv, None
        return grad_values, grad_uv, None

class SliceTexture(Function):
    @staticmethod
    def forward(ctx, texture, uv_tensor):

        ctx.save_for_backward(texture, uv_tensor)
        # ctx.texture_size=texture.shape[1]
        # ctx.neural_mvs=neural_mvs

        # values_not_normalized = neural_mvs.slice_texture(texture, uv_tensor)
        values_not_normalized = NeuralMVS.slice_texture(texture, uv_tensor)

        return values_not_normalized

    @staticmethod
    def backward(ctx, grad_values_not_normalized):

        #I only want the gradient with respect to the UV tensor 
        texture, uv_tensor =ctx.saved_tensors
        # texture_size=ctx.texture_size
        # neural_mvs=ctx.neural_mvs

        # grad_texture = neural_mesh_obj.splat_texture(grad_values, uv_tensor, texture_size)
        # grad_texture, grad_uv = neural_mvs.slice_texture_backward(grad_values_not_normalized, texture, uv_tensor)
        grad_texture, grad_uv = NeuralMVS.slice_texture_backward(grad_values_not_normalized, texture, uv_tensor)
        # TODO we need also the grad for uv
        
        
        # return None, None
        # return grad_texture, None
        return grad_texture, grad_uv



def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=1)