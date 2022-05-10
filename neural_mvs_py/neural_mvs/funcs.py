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
    torch.cuda.synchronize()
    Profiler.start(name)
def profiler_end(name):
    if(Profiler.is_profiling_gpu()):
        torch.cuda.synchronize()
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


