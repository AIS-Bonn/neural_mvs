#!/usr/bin/env python3.6

import torch
from torch.autograd import Function
# from torch.autograd import gradcheck
from torch import Tensor

import sys
import os
import numpy as np
# from matplotlib import pyplot as plt 
# http://wiki.ros.org/Packages#Client_Library_Support
# import rospkg
# rospack = rospkg.RosPack()
# sf_src_path=rospack.get_path('surfel_renderer')
# sf_build_path=os.path.abspath(sf_src_path + "/../../build/surfel_renderer")
# sys.path.append(sf_build_path) #contains the modules of pycom

# from DataLoaderTest  import *
# from lattice_py import LatticePy
# import visdom
# import torchnet
# from lr_finder import LRFinder
# from scores import Scores
# from model_ctx import ModelCtx
# from lattice_funcs import *
# from lattice_modules import *
# from models import *
from neural_mesh_py.neural_mesh.funcs import *
from gradcheck_custom import *

# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(threshold=50000)
# torch.set_printoptions(profile="full")
torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)

# config_file="lnn_grad_check.cfg"
# with_viewer=True


def check_splat(positions_tensor, uv_tensor, texture_size):
    gradcheck(SplatTexture.apply, (positions_tensor, uv_tensor, texture_size), eps=1e-5) 

def check_slice(texture, uv_tensor):
    gradcheck(SliceTexture.apply, (texture, uv_tensor), eps=1e-5) 



def run():

    while True:

        #get positions and values 
        texture_size= 32
        positions_tensor = torch.rand(16,3).to("cuda")
        uv_tensor = torch.rand(16,2).to("cuda")
        texture = torch.rand(texture_size, texture_size, 4).to("cuda")
        uv_tensor=(uv_tensor*2)-1.0 #get it in range -1, 1
        uv_tensor*=0.1 #make a it a bit smaller to avoid border effects when accesign the texture
        #print
        print("positions_tensor is ", positions_tensor)
        print("uv_tensor is ", uv_tensor)
        #set the reguired grads
        # positions_tensor.requires_grad=True
        # texture.requires_grad=True
        uv_tensor.requires_grad=True


        
        check_splat(positions_tensor, uv_tensor, texture_size)
        # check_slice(texture, uv_tensor)

        print("FINISHED GRAD CHECK")

        return


def main():
    run()



if __name__ == "__main__":
    main()  # This is what you would have, but the following is useful:

    # # These are temporary, for debugging, so meh for programming style.
    # import sys, trace

    # # If there are segfaults, it's a good idea to always use stderr as it
    # # always prints to the screen, so you should get as much output as
    # # possible.
    # sys.stdout = sys.stderr

    # # Now trace execution:
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.run('main()')