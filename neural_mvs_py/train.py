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

from neural_mvs.smooth_loss import *
from neural_mvs.ssim import * #https://github.com/VainF/pytorch-msssim

#debug 
from easypbr import Gui
from easypbr import Scene
# from neural_mvs.modules import *


#lnet 
# from deps.lnets.lnets.utils.math.autodiff import *


config_file="train.cfg"

torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
# torch.set_printoptions(edgeitems=5)

# #initialize the parameters used for training
train_params=TrainParams.create(config_file)    
model_params=ModelParams.create(config_file)    


class FramePY():
    def __init__(self, frame, znear, zfar):
        #get mask 
        self.mask_tensor=mat2tensor(frame.mask, False).to("cuda").repeat(1,3,1,1)
        self.mask=frame.mask
        #get rgb with mask applied 
        self.rgb_tensor=mat2tensor(frame.rgb_32f, False).to("cuda")
        # self.rgb_tensor=self.rgb_tensor*self.mask_tensor
        self.rgb_32f=tensor2mat(self.rgb_tensor)
        #get tf and K
        self.tf_cam_world=frame.tf_cam_world
        self.K=frame.K
        #weight and hegiht
        self.height=self.rgb_tensor.shape[2]
        self.width=self.rgb_tensor.shape[3]
        #create tensor to store the bound in z near and zfar for every pixel of this image
        self.znear_zfar = torch.nn.Parameter(  torch.ones([1,2,self.height,self.width], dtype=torch.float32, device=torch.device("cuda"))  )
        with torch.no_grad():
            self.znear_zfar[:,0,:,:]=znear
            self.znear_zfar[:,1,:,:]=zfar
        # self.znear_zfar.requires_grad=True
        self.cloud=frame.depth2world_xyz_mesh()
        self.cloud=frame.assign_color(self.cloud)
        self.cloud.remove_vertices_at_zero()
    def create_frustum_mesh(self, scale):
        frame=Frame()
        frame.K=self.K
        frame.tf_cam_world=self.tf_cam_world
        frame.width=self.width
        frame.height=self.height
        cloud=frame.create_frustum_mesh(scale)
        return cloud
    def compute_uv(self, cloud):
        frame=Frame()
        frame.rgb_32f=self.rgb_32f
        frame.K=self.K
        frame.tf_cam_world=self.tf_cam_world
        frame.width=self.width
        frame.height=self.height
        uv=frame.compute_uv(cloud)
        return uv
    def compute_uv_with_assign_color(self, cloud):
        frame=Frame()
        frame.rgb_32f=self.rgb_32f
        frame.K=self.K
        frame.tf_cam_world=self.tf_cam_world
        frame.width=self.width
        frame.height=self.height
        cloud=frame.assign_color(cloud)
        return cloud.UV.copy()
        
        

     



def run():
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
    if train_params.with_viewer():
        view=Viewer.create(config_path)


    first_time=True

    # experiment_name="default"
    # experiment_name="n4"
    experiment_name="s_3_mean"

    use_ray_compression=False





    cb_list = []
    if(train_params.with_visdom()):
        cb_list.append(VisdomCallback(experiment_name))
    if(train_params.with_viewer()):
        cb_list.append(ViewerCallback())
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)

    #create loaders
    # loader=TinyLoader.create(config_file)
    loader=DataLoaderShapeNetImg(config_path)
    # loader=DataLoaderNerf(config_path)
    # loader=DataLoaderVolRef(config_path)
    # loader.load_only_from_idxs( [0,1,2,3,4,5,6,7] )
    # loader.set_shuffle(False)
    # loader.set_overfit(False)
    # loader.load_only_from_idxs( [0,2,4,6] )
    # loader.start()
    loader_test=DataLoaderShapeNetImg(config_path)
    # loader_test=DataLoaderNerf(config_path)
    # loader_test=DataLoaderVolRef(config_path)
    # loader_test.load_only_from_idxs( [0,2,4,6] )
    # loader_test.load_only_from_idxs( [9,10,11,12,13,14,15,16] ) #one full row at the same height
    # loader_test.load_only_from_idxs( [10,12,14,16] )
    # loader_test.load_only_from_idxs( [8,10,12,14,16,18,20,22,24,26,28] )
    # loader_test.load_only_from_idxs( [10] )
    # loader_test.set_shuffle(True)
    # loader_test.set_overfit(True) #so we don't reload the image after every reset but we just keep on training on it
    # loader_test.start()
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
    # model=SirenNetwork(in_channels=2, out_channels=3).to("cuda")
    # model = VAE(nc=3, ngf=128, ndf=128, latent_variable_size=500).to("cuda")
    model.train()

    loss_fn=torch.nn.MSELoss()

    # show_every=39
    show_every=10


    
    #for vase
    depth_min=0.6
    depth_max=2.0
    #for nerf
    # depth_min=2.0
    # depth_max=5.0



    initial_render_frame=None
        
    novel_cam=Camera()
    novel_cam.set_position([0, 0.0001, 0.0001])
    # novel_cam.set_lookat([-0.02, 0.1, -1.3]) #for the figure
    # novel_cam.set_lookat([-0.02, 0.1, -1.0]) #for vase
    # novel_cam.set_lookat([-0.02, 0.2, -1.0]) #for socrates
    # novel_cam.set_lookat([0.0, 0.0, 0.0]) #for car
    novel_cam.m_fov=40
    novel_cam.m_near=0.01
    novel_cam.m_far=3

  
   

    while True:

        for phase in phases:
            cb.epoch_started(phase=phase)
            cb.phase_started(phase=phase)
            model.train(phase.grad)
            is_training=phase.grad

            # if loader_test.finished_reading_scene():
            #     print("is finished")
            # else: 
            #     print("is NOT")


            # pbar = tqdm(total=phase.loader.nr_samples())
            # for i in range(loader_test.nr_samples()):
            # for i in range( len(frames_for_training) ):
            if loader_test.finished_reading_scene():

                # if phase.loader.has_data() and loader_test.has_data():
                # if loader_test.has_data():
                if True: #Shapenet IMg always had ata at this point 
                  
                    #preload all frames_for_encoding 
                    frames_for_encoding=[]
                    all_imgs_poses_cam_world_list=[]
                    for i in range(6): #####IF YOU CHANGE THIS change also the nr_imgs_to_read in the train.cfg
                        frame=loader_test.get_random_frame()
                        frame_py=FramePY(frame, depth_min, depth_max) 
                        frames_for_encoding.append(frame_py)
                        all_imgs_poses_cam_world_list.append(frame.tf_cam_world)

                    #load a random frame for gt 
                    gt_frame=frames_for_encoding[ random.randint(0, len(frames_for_encoding)-1 )]
                    gt_rgb_tensor=mat2tensor(gt_frame.rgb_32f, False).to("cuda")
                    mask=mat2tensor(gt_frame.mask, False).to("cuda")

                
                    #start reading new scene
                    if(phase.iter_nr%1==0):
                        loader_test.start_reading_next_scene()


                  


                    # #debug
                    if(phase.iter_nr%show_every==0):
                        # cloud=gt_depth_frame.depth2world_xyz_mesh()
                        frustum=gt_frame.create_frustum_mesh(0.1)
                        # Scene.show(cloud, "cloud")
                        Scene.show(frustum, "frustum")
                        #show the first frame of the ones used for encoding
                        Gui.show(frames_for_encoding[0].rgb_32f, "rgb_for_enc")
                        # Scene.show(gt_frame.cloud, "cloud")





                    #forward
                    with torch.set_grad_enabled(is_training):


                        if(phase.iter_nr%show_every==0):
                            Gui.show(gt_frame.rgb_32f, "gt")

                        # #try another view
                        with torch.set_grad_enabled(False):
                            if initial_render_frame==None:
                                initial_render_frame=gt_frame.tf_cam_world
                                # novel_cam.set_position(gt_frame.tf_cam_world.inverse().translation())
                                novel_cam.transform_model_matrix( gt_frame.tf_cam_world.inverse() )
                                novel_cam.set_lookat([0.0, 0.0, 0.0]) #for car
                                # novel_cam.set_lookat([-0.02, 0.1, -0.9]) #for vase
                            # print("novel cam tf is ", novel_cam.view_matrix_affine().to_float().matrix() )
                            render_tf=initial_render_frame
                            novel_cam.orbit_y(10)
                            render_tf=novel_cam.view_matrix_affine().to_float()

                          
                            out_tensor,  depth_map, acc_map, new_loss =model(gt_frame, frames_for_encoding, all_imgs_poses_cam_world_list, render_tf, gt_frame.K, depth_min, depth_max, use_ray_compression, novel=True )
                            # out_tensor=model(ref_rgb_tensor, renrgb_siren,der_tf, render_tf )
                            if(phase.iter_nr%1==0):
                                out_mat=tensor2mat(out_tensor)
                                Gui.show(out_mat, "novel")
                                frustum=novel_cam.create_frustum_mesh(0.1, [100,70])
                                Scene.show(frustum, "frustum_novel")

                               



                     



                        TIME_START("forward")
                        out_tensor,  depth_map, acc_map, new_loss=model(gt_frame, frames_for_encoding, all_imgs_poses_cam_world_list, gt_frame.tf_cam_world, gt_frame.K, depth_min, depth_max, use_ray_compression )
                        TIME_END("forward")


                      
                       

                        rgb_loss=((out_tensor-gt_rgb_tensor)**2).mean()
                        # rgb_loss=( torch.abs(out_tensor-gt_rgb_tensor) ).mean()
                       

                      
                        #SSIM LOSS does not mae things better, it may even make then worse
                        # ssim_loss= 1 - ms_ssim( gt_rgb_tensor, out_tensor, win_size=3, data_range=1.0, size_average=True )
                        # loss=rgb_loss*0.5 + ssim_loss*0.5
                        loss=rgb_loss
                      

                      


                     

                        gt_rgb_tensor_mat=tensor2mat(gt_rgb_tensor)
                        Gui.show(gt_rgb_tensor_mat, "gt_rgb_tensor")


                        rgb_loss=loss



                        if(phase.iter_nr%show_every==0):
                            out_mat=tensor2mat(out_tensor)
                            Gui.show(out_mat, "output")
            
                        #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                        if first_time:
                            first_time=False
                            optimizer=RAdam( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )

                            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.1)
                            # scheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=300)
                            # scheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)
                            # lambda1 = lambda epoch: 0.9999 ** phase.iter_nr
                            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
                            optimizer.zero_grad()

                        cb.after_forward_pass(loss=rgb_loss, phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 
                        # pbar.update(1)

                    #backward
                    if is_training:
                        # if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                            # scheduler.step(phase.epoch_nr + float(phase.samples_processed_this_epoch) / phase.loader.nr_samples() )
                            # scheduler.step(phase.epoch_nr + float(phase.samples_processed_this_epoch) / loader_test.nr_samples() )
                            # scheduler.step(phase.iter_nr /10000  ) #go to zero every 10k iters
                        # if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                            # scheduler.step()
                        optimizer.zero_grad()
                        cb.before_backward_pass()
                        TIME_START("backward")
                        loss.backward()
                        TIME_END("backward")
                        cb.after_backward_pass()
                        grad_clip=0.01
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        # torch.nn.utils.clip_grad_norm_(uv_regressor.parameters(), grad_clip)
                        # summary(model)
                        # exit()

                      
                        optimizer.step()

                        # if (phase.iter_nr%10==0):
                        #     optimizer.step() # DO it only once after getting gradients for all images
                        #     optimizer.zero_grad()


                if train_params.with_viewer():
                    view.update()

            # finished all the images 
            # pbar.close()
            if is_training and loader_test.is_finished(): #we reduce the learning rate when the test iou plateus
            # if is_training: #we reduce the learning rate when the test iou plateus
                # optimizer.step() # DO it only once after getting gradients for all images
                # optimizer.zero_grad()
                # if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # scheduler.step(phase.loss_acum_per_epoch) #for ReduceLROnPlateau
                cb.epoch_ended(phase=phase, model=model, save_checkpoint=train_params.save_checkpoint(), checkpoint_path=train_params.checkpoint_path() ) 
                cb.phase_ended(phase=phase) 
                # phase.epoch_nr+=1
                loader_test.reset()
                # random.shuffle(frames_for_encoding)
                # random.shuffle(frames_for_training)
                # time.sleep(0.1) #give the loaders a bit of time to load


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
