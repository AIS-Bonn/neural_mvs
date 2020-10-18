#!/usr/bin/env python3.6

import torch
import torch.nn.functional as F

import sys
import os
import numpy as np
from tqdm import tqdm
import time
import random

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

    # experiment_name="default"
    # experiment_name="n4"
    experiment_name="s"



    cb_list = []
    if(train_params.with_visdom()):
        cb_list.append(VisdomCallback(experiment_name))
    if(train_params.with_viewer()):
        cb_list.append(ViewerCallback())
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)

    #create loaders
    # loader=TinyLoader.create(config_file)
    loader=DataLoaderVolRef(config_path)
    loader.start()
    #load all the images on cuda already so it's faster
    # imgs=[]
    # for i in range(loader.nr_frames()):
    #     img_cpu=loader.get_frame(i).rgb_32f
    #     print("img has size ", loader.get_frame(i).width, " ", loader.get_frame(i).height)
    #     img_tensor=mat2tensor(img_cpu, False)
    #     imgs.append( img_tensor.to("cuda") )
    

    #create phases
    phases= [
        Phase('train', loader, grad=True),
        # Phase('test', loader_test, grad=False)
    ]
    #model 
    model=Net().to("cuda")
    # model=SirenNetwork().to("cuda")
    # model = VAE(nc=3, ngf=128, ndf=128, latent_variable_size=500).to("cuda")
    model.train()

    loss_fn=torch.nn.MSELoss()

    show_every=40
    # show_every=1

    while True:

        for phase in phases:
            cb.epoch_started(phase=phase)
            cb.phase_started(phase=phase)
            model.train(phase.grad)
            is_training=phase.grad


            # pbar = tqdm(total=phase.loader.nr_samples())
            for i in range(phase.loader.nr_samples()):

                if phase.loader.has_data():

                    torch.manual_seed(0)

                    # #get a reference frame
                    # ref_idx=random.randint(0, phase.loader.nr_frames()-1 )
                    # # ref_idx=0
                    # if(phase.iter_nr%show_every==0):
                    #     img=tensor2mat(imgs[ref_idx])
                    #     Gui.show(img, "ref")
                    #     frustum=phase.loader.get_frame(ref_idx).create_frustum_mesh(0.1)
                    #     Scene.show(frustum, "frustum"+str(ref_idx))
                    # #get a ground truth frame
                    # gt_idx=random.randint(0, phase.loader.nr_frames()-1 )
                    # # gt_idx=0
                    # if(phase.iter_nr%show_every==0):
                    #     img=tensor2mat(imgs[gt_idx])
                    #     Gui.show(img, "gt")

                    ref_frame=phase.loader.get_color_frame()
                    # gt_frame=phase.loader.closest_color_frame(ref_frame)
                    gt_frame=ref_frame
                    #get depth frames
                    ref_depth_frame=phase.loader.get_depth_frame()
                    # gt_depth_frame=phase.loader.closest_depth_frame(ref_frame)
                    gt_depth_frame=ref_depth_frame

                    #debug
                    # cloud=ref_depth_frame.depth2world_xyz_mesh()
                    # frustum=ref_depth_frame.create_frustum_mesh(0.1)
                    # Scene.show(cloud, "cloud")
                    # Scene.show(frustum, "frustum")


                    


                    #forward
                    with torch.set_grad_enabled(is_training):

                        # ref_rgb_tensor=imgs[ref_idx]
                        # gt_rgb_tensor=imgs[gt_idx]


                        #get only valid pixels
                        # ref_frame.rgb_32f=ref_frame.rgb_with_valid_depth(ref_depth_frame) 
                        # gt_frame.rgb_32f=gt_frame.rgb_with_valid_depth(gt_depth_frame) 
                        gt_frame.rgb_32f=ref_frame.rgb_32f

                        ref_rgb_tensor=mat2tensor(ref_frame.rgb_32f, False).to("cuda")
                        gt_rgb_tensor=mat2tensor(gt_frame.rgb_32f, False).to("cuda")
                        ref_rgb_tensor=ref_rgb_tensor.contiguous()

                        if(phase.iter_nr%show_every==0):
                            # print("width and height ", ref_frame.width)
                            Gui.show(ref_frame.rgb_32f, "ref")
                            Gui.show(gt_frame.rgb_32f, "gt")

                        # #try another view
                        # with torch.set_grad_enabled(False):
                        #     render_tf=gt_frame.tf_cam_world
                        #     render_tf.rotate_axis_angle([0,1,0], random.randint(-60,60) )
                        #     out_tensor=model(ref_rgb_tensor, ref_frame.tf_cam_world, render_tf )
                        #     # out_tensor=model(ref_rgb_tensor, render_tf, render_tf )
                        #     if(phase.iter_nr%show_every==0):
                        #         out_mat=tensor2mat(out_tensor)
                        #         Gui.show(out_mat, "novel")



                        TIME_START("forward")
                        # out_tensor=model(ref_rgb_tensor, ref_frame.tf_cam_world, gt_frame.tf_cam_world )
                        out_tensor=model(ref_rgb_tensor, ref_frame.tf_cam_world, gt_frame.tf_cam_world )
                        # out_tensor, mu, logvar = model(ref_rgb_tensor)
                        TIME_END("forward")


                        # print("out tensor  ", out_tensor.min(), " ", out_tensor.max())
                        # print("out tensor  ", gt_rgb_tensor.min(), " ", gt_rgb_tensor.max())
                        loss=((out_tensor-gt_rgb_tensor)**2).mean()
                        # loss=(((out_tensor-gt_rgb_tensor)**2)*gt_rgb_tensor) .mean()
                        # loss=loss_fn(out_tensor, gt_rgb_tensor)
                        # print("loss is ", loss)



                        if(phase.iter_nr%show_every==0):
                            out_mat=tensor2mat(out_tensor)
                            Gui.show(out_mat, "output")
            
                        #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                        if first_time:
                            first_time=False
                            optimizer=RAdam( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                            # optimizer=torch.optim.AdamW( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.1)

                        cb.after_forward_pass(loss=loss, phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 
                    #     # pbar.update(1)

                    #backward
                    if is_training:
                        # if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                            # scheduler.step(phase.epoch_nr + float(phase.samples_processed_this_epoch) / phase.loader.nr_samples() )
                        optimizer.zero_grad()
                        cb.before_backward_pass()
                        TIME_START("backward")
                        loss.backward()
                        TIME_END("backward")
                        cb.after_backward_pass()
                        # grad_clip=0.01
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        # torch.nn.utils.clip_grad_norm_(uv_regressor.parameters(), grad_clip)
                        # summary(model)
                        # exit()

                        # print("fcmu grad norm", model.fc_mu.weight.grad.norm())
                        # print("first_conv norm", model.first_conv.weight.grad.norm())

                        optimizer.step()

                if train_params.with_viewer():
                    view.update()

            # finished all the images 
            # pbar.close()
            if is_training: #we reduce the learning rate when the test iou plateus
                # if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # scheduler.step(phase.loss_acum_per_epoch) #for ReduceLROnPlateau
                cb.epoch_ended(phase=phase, model=model, save_checkpoint=train_params.save_checkpoint(), checkpoint_path=train_params.checkpoint_path() ) 
                cb.phase_ended(phase=phase) 
                # phase.epoch_nr+=1


                # if train_params.with_viewer():
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
