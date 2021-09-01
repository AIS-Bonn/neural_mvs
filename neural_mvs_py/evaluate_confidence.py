#!/usr/bin/env python3.6

import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler

import sys
import os
import numpy as np
from tqdm import tqdm
import time
import random

from easypbr  import *
from dataloaders import *

import cv2

config_file="evaluate_confidence.cfg"

torch.manual_seed(0)
random.seed(0)
# torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(edgeitems=3)

# #initialize the parameters used for training
# train_params=TrainParams.create(config_file)    
# model_params=ModelParams.create(config_file)    


def map_range( input_val, input_start, input_end,  output_start,  output_end):
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    # input_clamped=max(input_start, min(input_end, input_val))
    input_clamped=torch.clamp(input_val, input_start, input_end)
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start)



def run():
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
    # if train_params.with_viewer():
        # view=Viewer.create(config_path)


    first_time=True

    confidence_mat = cv2.imread("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/recordings/from_other_comps/llff_flower_with_confidence/confidence/1.png" )
    confidence_colored_mat = cv2.imread("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/recordings/from_other_comps/llff_flower_with_confidence/confidence_colored/1.png" )
    pred_mat = cv2.imread("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/recordings/from_other_comps/llff_flower_overfit/0/rgb/1.png" )
    gt_mat = cv2.imread("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/recordings/from_other_comps/llff_flower_overfit/0/gt/1.png" )

    # diff=gt_mat-pred_mat
    #diff
    gt_tensor=torch.from_numpy(gt_mat).float()/255
    pred_tensor=torch.from_numpy(pred_mat).float()/255
    print("pred_tensor", pred_tensor.dtype)
    diff_tensor=(gt_tensor-pred_tensor) #H,W,3
    diff_tensor=diff_tensor.norm(dim=2, keepdim=True)
    print("diff tensor has shape", diff_tensor.shape)
    print("diff_tensor min max is", diff_tensor.min(), " ", diff_tensor.max() )
    # diff_tensor_max=diff_tensor.max()
    # diff_tensor=diff_tensor_max-diff_tensor
    diff_tensor=map_range(diff_tensor,0, 1.4, 0.0, 1.0)
    diff_tensor=diff_tensor.max()-diff_tensor
    diff_tensor=diff_tensor**12
    diff_tensor=diff_tensor.max()-diff_tensor
    # diff_tensor=map_range(diff_tensor,0, 0.3, 0.0, 1.0)
    diff_tensor=diff_tensor*255
    diff_tensor=diff_tensor.to(torch.uint8)
    print("diff_tensor", diff_tensor.dtype)
    print("diff_tensor min max is", diff_tensor.min(), " ", diff_tensor.max() )
    # diff_tensor=map_range(diff_tensor,0, 1.0, 1.0, 0.0)

    #color
    diff_mat=diff_tensor.numpy()
    # diff_color = cv2.applyColorMap(diff_mat, cv2.COLORMAP_JET)
    diff_color = cv2.applyColorMap(diff_mat, cv2.COLORMAP_COOL)
    # diff_color = cv2.applyColorMap(diff_mat, cv2.COLORMAP_SUMMER)



    cv2.imwrite("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/recordings/confidence_evaluation/confidence.png", confidence_mat)
    cv2.imwrite("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/recordings/confidence_evaluation/gt.png", gt_mat)
    cv2.imwrite("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/recordings/confidence_evaluation/pred.png", pred_mat)
    cv2.imwrite("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/recordings/confidence_evaluation/confidence_colored.png", confidence_colored_mat)
    cv2.imwrite("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/recordings/confidence_evaluation/diff_color.png", diff_color)



    # while True:

        # view.update()


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