import torch
from torch.autograd import Function
from torch import Tensor

import sys
from easypbr  import Profiler
import numpy as np
import time
import math
from neuralmvs import NeuralMVS

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




# class SplatTexture(Function):
#     @staticmethod
#     def forward(ctx, values_tensor, uv_tensor, texture_size):

#         ctx.save_for_backward(values_tensor, uv_tensor)

#         texture = neural_mesh_obj.splat_texture(values_tensor, uv_tensor, texture_size)

#         return texture

#     @staticmethod
#     def backward(ctx, grad_texture):

#         #I only want the gradient with respect to the UV tensor 
#         values_tensor, uv_tensor =ctx.saved_tensors

#         # grad_val = neural_mesh_obj.slice_texture(grad_texture, uv_tensor)
#         grad_values, grad_uv =neural_mesh_obj.splat_texture_backward( grad_texture, values_tensor, uv_tensor )
#         # TODO we might need also the grad for values
        
#         # return None, None, None
#         # return None, grad_uv, None
#         return grad_values, grad_uv, None

# class SliceTexture(Function):
#     @staticmethod
#     def forward(ctx, texture, uv_tensor):

#         ctx.save_for_backward(texture, uv_tensor)
#         # ctx.texture_size=texture.shape[1]

#         values_not_normalized = neural_mesh_obj.slice_texture(texture, uv_tensor)

#         return values_not_normalized

#     @staticmethod
#     def backward(ctx, grad_values_not_normalized):

#         #I only want the gradient with respect to the UV tensor 
#         texture, uv_tensor =ctx.saved_tensors
#         # texture_size=ctx.texture_size

#         # grad_texture = neural_mesh_obj.splat_texture(grad_values, uv_tensor, texture_size)
#         grad_texture, grad_uv = neural_mesh_obj.slice_texture_backward(grad_values_not_normalized, texture, uv_tensor)
#         # TODO we need also the grad for uv
        
        
#         # return None, None
#         # return grad_texture, None
#         return grad_texture, grad_uv



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