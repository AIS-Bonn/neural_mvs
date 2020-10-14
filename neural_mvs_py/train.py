#!/usr/bin/env python3.6

import torch
import torch.nn.functional as F

import sys
import os
import numpy as np
from tqdm import tqdm
import time

from easypbr  import *
from dataloaders import *
from neuralmvs import *
from neural_mvs.models import *

from callbacks.callback import *
from callbacks.viewer_callback import *
from callbacks.visdom_callback import *
from callbacks.state_callback import *
from callbacks.phase import *

from optimizers.over9000.radam import *
from optimizers.over9000.lookahead import *
from optimizers.over9000.novograd import *

#debug 
from easypbr import Gui
from easypbr import Scene
# from neural_mvs.modules import *

#lnet 
# from deps.lnets.lnets.utils.math.autodiff import *


config_file="train.cfg"

torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.autograd.set_detect_anomaly(True)

# #initialize the parameters used for training
train_params=TrainParams.create(config_file)    
model_params=ModelParams.create(config_file)    



def run():
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
    if train_params.with_viewer():
        view=Viewer.create(config_path)


    first_time=True

    experiment_name="default"



    cb_list = []
    if(train_params.with_visdom()):
        cb_list.append(VisdomCallback(experiment_name))
    if(train_params.with_viewer()):
        cb_list.append(ViewerCallback())
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)
    #create loaders
    # loader=DataLoaderImg(config_path)
    # loader.start()
    loader=TinyLoader.create(config_file)
    #create phases
    phases= [
        Phase('train', loader, grad=True),
        # Phase('test', loader_test, grad=False)
    ]
    #model 
    # model=Net().to("cuda")
    # model.train()

    loss_fn=torch.nn.MSELoss()

    show_every=19

    while True:

        for phase in phases:
            cb.epoch_started(phase=phase)
            cb.phase_started(phase=phase)
            # model.train(phase.grad)


            # pbar = tqdm(total=phase.loader.nr_samples())
            while ( phase.samples_processed_this_epoch < phase.loader.nr_frames()):

                frame=loader.get_frame(0)
                is_training = phase.grad

                Gui.show(frame.rgb_32f, "img"+str(0))

                # torch.manual_seed(0)

                # if(phase.iter_nr%show_every==0):
                #     Gui.show(frame.rgb_32f, "rgb")

                # #forward
                # with torch.set_grad_enabled(is_training):

                #     rgb_tensor=mat2tensor(frame.rgb_32f, False).to("cuda")

                #     # params=rgb_tensor.clone()

                #     TIME_START("forward")
                #     out_tensor=model(rgb_tensor)
                #     TIME_END("forward")

                #     loss=((out_tensor-rgb_tensor)**2).mean()



                #     if(phase.iter_nr%show_every==0):
                #         out_mat=tensor2mat(out_tensor)
                #         Gui.show(out_mat, "output")
        
                #     #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                #     if first_time:
                #         first_time=False
                #         optimizer=RAdam( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                #         # optimizer=torch.optim.AdamW( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=True, factor=0.1)

                #     cb.after_forward_pass(loss=loss, phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 
                #     # pbar.update(1)

                # #backward
                # if is_training:
                #     # if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                #         # scheduler.step(phase.epoch_nr + float(phase.samples_processed_this_epoch) / phase.loader.nr_samples() )
                #     optimizer.zero_grad()
                #     cb.before_backward_pass()
                #     TIME_START("backward")
                #     loss.backward()
                #     TIME_END("backward")
                #     cb.after_backward_pass()
                #     # grad_clip=0.01
                #     # torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                #     # torch.nn.utils.clip_grad_norm_(uv_regressor.parameters(), grad_clip)
                #     # summary(model)
                #     # exit()

                #     # print("fcmu grad norm", model.fc_mu.weight.grad.norm())
                #     # print("first_conv norm", model.first_conv.weight.grad.norm())

                #     optimizer.step()



                # if phase.loader.is_finished():
                #     # pbar.close()
                #     # if is_training: #we reduce the learning rate when the test iou plateus
                #     # print("what")
                #     # if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                #         # scheduler.step(phase.loss_acum_per_epoch) #for ReduceLROnPlateau
                #     cb.epoch_ended(phase=phase, model=model, save_checkpoint=train_params.save_checkpoint(), checkpoint_path=train_params.checkpoint_path() ) 
                #     cb.phase_ended(phase=phase) 
                #     # if not phase.grad:


                if train_params.with_viewer():
                    view.update()


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
