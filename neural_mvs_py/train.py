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
    def create_frustum_mesh(self, scale):
        frame=Frame()
        frame.K=self.K
        frame.tf_cam_world=self.tf_cam_world
        frame.width=self.width
        frame.height=self.height
        cloud=frame.create_frustum_mesh(scale)
        return cloud

     



def run():
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
    if train_params.with_viewer():
        view=Viewer.create(config_path)


    first_time=True

    # experiment_name="default"
    # experiment_name="n4"
    experiment_name="s_60trim"

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


    #prepare all images and concat them batchwise
    # all_imgs_list=[]
    # all_imgs_poses_cam_world_list=[]
    # all_imgs_Ks_list=[]
    #load some images for encoding
    #for nerf and shapentimg
    # for i in range(4):
    #     ref_frame=loader.get_random_frame()
    #     ref_rgb_tensor=mat2tensor(ref_frame.rgb_32f, False).to("cuda")
    #     all_imgs_list.append(ref_rgb_tensor)
    #     all_imgs_poses_cam_world_list.append(ref_frame.tf_cam_world)
    #     all_imgs_Ks_list.append(ref_frame.K)
    #for volref 
    # while True:
    #     for i in range(loader.nr_samples()):
    #         if loader.has_data():
    #             ref_frame=loader.get_color_frame()
    #             depth_frame=loader.get_depth_frame()
    #             # ref_frame.rgb_32f=ref_frame.rgb_with_valid_depth(depth_frame) 
    #             ref_rgb_tensor=mat2tensor(ref_frame.rgb_32f, False).to("cuda")
    #             all_imgs_list.append(ref_rgb_tensor)
    #             all_imgs_poses_cam_world_list.append(ref_frame.tf_cam_world)
    #             all_imgs_Ks_list.append(ref_frame.K)
    #             print("appending")
    #     if loader.is_finished():
    #         loader.reset()
    #         break
    # all_imgs=torch.cat(all_imgs_list,0).contiguous().to("cuda")
    # print("all imgs have shape ", all_imgs.shape)

    #for vase
    depth_min=0.5
    depth_max=1.6
    #for nerf
    # depth_min=2.0
    # depth_max=5.0

    # #preload all frames_for_encoding 
    # frames_for_encoding=[]
    # while True:
    #     for i in range(loader.nr_samples()):
    #         if loader.has_data():
    #             frame=loader.get_color_frame()
    #             frame_py=FramePY(frame, depth_min, depth_max) 
    #             frames_for_encoding.append(frame_py)
    #     if loader.is_finished():
    #         loader.reset()
    #         break

    # #preload all frames for training
    # frames_for_training=[]
    # while True:
    #     for i in range(loader_test.nr_samples()):
    #         if loader_test.has_data():
    #             frame=loader_test.get_color_frame()
    #             frame_py=FramePY(frame, depth_min, depth_max) 
    #             frames_for_training.append(frame_py)
    #     if loader_test.is_finished():
    #         loader_test.reset()
    #         break




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


            # pbar = tqdm(total=phase.loader.nr_samples())
            # for i in range(loader_test.nr_samples()):
            # for i in range( len(frames_for_training) ):
            if loader_test.finished_reading_scene():

                # if phase.loader.has_data() and loader_test.has_data():
                # if loader_test.has_data():
                if True: #Shapenet IMg always had ata at this point 
                # if loader_test.finished_reading_scene():
                    # loader_test.start_reading_next_scene()

                    # torch.manual_seed(0)


                    # gt_frame=loader_test.get_random_frame() #load from the gt loader
                    # gt_frame=loader_test.get_next_frame() #load from the gt loader
                    # gt_frame=loader_test.get_color_frame() #fro shapenet vol ref
                    # gt_depth_frame=loader_test.get_depth_frame() #load from the gt loader
                    # i=0

                    # gt_frame=frames_for_training[i]
                    # gt_rgb_tensor=frames_for_training[i].rgb_tensor
                    # mask=frames_for_training[i].mask_tensor

                                    
                    #preload all frames_for_encoding 
                    frames_for_encoding=[]
                    all_imgs_poses_cam_world_list=[]
                    for i in range(6):
                        frame=loader_test.get_random_frame()
                        frame_py=FramePY(frame, depth_min, depth_max) 
                        frames_for_encoding.append(frame_py)
                        all_imgs_poses_cam_world_list.append(frame.tf_cam_world)

                    #load a random frame for gt 
                    # gt_frame=loader_test.get_random_frame()
                    gt_frame=frames_for_encoding[0]
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



                    # #show frustums 
                    # frustum_ref=ref_frame.create_frustum_mesh(0.1)
                    # Scene.show(frustum_ref, "frustum_ref"+str(phase.samples_processed_this_epoch))
                    # frustum_gt=gt_frame.create_frustum_mesh(0.1)
                    # frustum_gt.m_vis.m_line_color=[0, 1.0, 0.0]
                    # Scene.show(frustum_gt, "frustum_gt"+str(phase.samples_processed_this_epoch))
                    


                    #forward
                    with torch.set_grad_enabled(is_training):

                        # ref_rgb_tensor=imgs[ref_idx]
                        # gt_rgb_tensor=imgs[gt_idx]


                        #get only valid pixels
                        # ref_frame.rgb_32f=ref_frame.rgb_with_valid_depth(ref_depth_frame) 
                        # gt_frame.rgb_32f=gt_frame.rgb_with_valid_depth(gt_depth_frame) 
                        # gt_frame.rgb_32f=ref_frame.rgb_32f


                        # ref_rgb_tensor=mat2tensor(ref_frame.rgb_32f, False).to("cuda")
                        # gt_rgb_tensor=mat2tensor(gt_frame.rgb_32f, False).to("cuda")
                        # mask=mat2tensor(gt_frame.mask, False).to("cuda").repeat(1,3,1,1)
                        # gt_rgb_tensor=gt_rgb_tensor*mask
                        # gt_frame.rgb_32f=tensor2mat(gt_rgb_tensor)
                        # mask=gt_rgb_tensor>0.0
                        # mask=mat2tensor(gt_frame.mask, False).to("cuda")
                        # ref_rgb_tensor=ref_rgb_tensor.contiguous()

                        #EXPERIMENT make the gt tensor just black and white  SO we predict just black for background and white for the objects
                        # gt_rgb_tensor=mask*1.0
                        # gt_frame.rgb_32f=tensor2mat(gt_rgb_tensor)

                        # Gui.show(gt_frame.mask, "mask")

                        if(phase.iter_nr%show_every==0):
                            # print("width and height ", ref_frame.width)
                            # Gui.show(ref_frame.rgb_32f, "ref")
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
                            # print("render tf is ", render_tf.matrix())

                            # render_K=novel_cam.intrinsics(100, 70)
                            # print("novel cam translation is ", render_tf.translation())
                            # print("gt fra,e translation is ", gt_frame.tf_cam_world.translation())
                            # exit(1)
                            # out_tensor=model(ref_rgb_tensor, ref_frame.tf_cam_world, render_tf )
                            out_tensor,  depth_map, acc_map, new_loss=model(gt_frame, frames_for_encoding, all_imgs_poses_cam_world_list, render_tf, gt_frame.K, depth_min, depth_max, use_ray_compression, novel=True )
                            # out_tensor=model(ref_rgb_tensor, renrgb_siren,der_tf, render_tf )
                            if(phase.iter_nr%10==0):
                                out_mat=tensor2mat(out_tensor)
                                Gui.show(out_mat, "novel")
                                # rgb_siren_mat=tensor2mat(rgb_siren)
                                # Gui.show(rgb_siren_mat, "novel_siren")
                                #show the frustum of the novel view
                                # frame=Frame()
                                # frame.K=gt_frame.K
                                # frame.width=gt_frame.width
                                # frame.height=gt_frame.height
                                # frame.tf_cam_world=render_tf
                                frustum=novel_cam.create_frustum_mesh(0.1, [100,70])
                                Scene.show(frustum, "frustum_novel")



                            



                        TIME_START("forward")
                        # out_tensor=model(ref_rgb_tensor, ref_frame.tf_cam_world, gt_frame.tf_cam_world )
                        out_tensor, depth_map, acc_map, new_loss=model(gt_frame, frames_for_encoding, all_imgs_poses_cam_world_list, gt_frame.tf_cam_world, gt_frame.K, depth_min, depth_max, use_ray_compression )
                        # out_tensor=model(gt_rgb_tensor)
                        # out_tensor, mu, logvar = model(ref_rgb_tensor)
                        TIME_END("forward")

                        #calculate smoothness loss 
                        smooth_loss=inverse_depth_smoothness_loss(depth_map*mask, gt_rgb_tensor)
                        # print("smooth_loss", smooth_loss)

                        

                        with torch.set_grad_enabled(False):
                            if(phase.iter_nr%show_every==0):
                                # print("depth map has shape ", depth_map.shape)
                                # print("mask has shape ", mask.shape)
                                depth_map=depth_map*mask
                                # depth_map=depth_map-1.5 #it's in range 1 to 2 meters so now we set it to range 0 to 1
                                # depth_map_nonzero=depth_map!=0.0
                                # print("min max", depth_map.min(), " ", depth_map.max(), " mean ", depth_map.mean() )
                                depth_map_ranged=map_range(depth_map, depth_min, depth_max, 0.0, 1.0).repeat(1,3,1,1)
                                depth_map_mat=tensor2mat(depth_map_ranged)
                                Gui.show(depth_map_mat, "depth")
                                # #gt depth
                                # depth_gt=mat2tensor(gt_depth_frame.depth, False)
                                # depth_gt=depth_gt.repeat(1,3,1,1)
                                # depth_gt=map_range(depth_gt, 0.9, 1.7, 0.0, 1.0)
                                # depth_gt_mat=tensor2mat(depth_gt)
                                # Gui.show(depth_gt_mat, "depth_gt")

                            # #show the znear zfar
                            # if(phase.iter_nr%show_every==0):
                            #     znear=gt_frame.znear_zfar[:,0:1,:,:].repeat(1,3,1,1)
                            #     zfar=gt_frame.znear_zfar[:,1:2,:,:].repeat(1,3,1,1)
                            #     # print("znear has hsape", znear.shape)
                            #     znear_ranged=map_range(znear, depth_min, depth_max, 0.0, 1.0)
                            #     zfar_ranged=map_range(zfar, depth_min, depth_max, 0.0, 1.0)
                            #     Gui.show(tensor2mat(znear_ranged), "znear_ranged")
                            #     Gui.show(tensor2mat(zfar_ranged), "zfar_ranged")


                        # #render the depth map
                        # if(phase.iter_nr%show_every==0):
                        #     depth_map_1_ch=depth_map[:,0:1, :, :].contiguous().clone()
                        #     depth_map_1_ch=depth_map_1_ch*mask[:,0:1, :, :]
                        #     depth_mat_cv=tensor2mat(depth_map_1_ch)
                        #     gt_depth_frame.depth=depth_mat_cv
                        #     # print("what")
                        #     cloud_pred=gt_depth_frame.depth2world_xyz_mesh()
                        #     # print("what2")
                        #     cloud_pred.m_vis.m_point_color=[0.0, 1.0, 0.0]
                        #     Scene.show(cloud_pred, "cloud_pred")

                       

                        # print("out tensor  ", out_tensor.min(), " ", out_tensor.max())
                        # print("out tensor  ", gt_rgb_tensor.min(), " ", gt_rgb_tensor.max())
                        rgb_loss=((out_tensor-gt_rgb_tensor)**2).mean()
                        # loss+=((rgb_siren-gt_rgb_tensor)**2).mean()
                        # loss=(((out_tensor-gt_rgb_tensor)**2)).mean()  / loader_test.nr_samples()
                        # loss=(((out_tensor-gt_rgb_tensor)**2)).mean()  / 10
                        # loss=loss_fn(out_tensor, gt_rgb_tensor)
                        # print("loss is ", loss.item())

                        # loss+=smooth_loss*0.00001*phase.iter_nr
                        # loss+=smooth_loss*0.01
                        ##PUT also the new losses
                        # loss+=new_loss*0.001*phase.iter_nr

                        loss=rgb_loss + smooth_loss*0.001

                        #make a loss to bring znear anzfar close 
                        if use_ray_compression:
                            znear=gt_frame.znear_zfar[:,0,:,:]
                            zfar=gt_frame.znear_zfar[:,1,:,:]
                            ray_shortness_loss=((zfar-znear)**2).mean()
                            loss+=ray_shortness_loss*0.01


                        #debug the diff map 
                        diff=(((out_tensor-gt_rgb_tensor)**2))
                        diff_mat=tensor2mat(diff)
                        Gui.show(diff_mat, "diff_mat")



                        if(phase.iter_nr%show_every==0):
                            # out_mat=tensor2mat(out_tensor)
                            # out_mat=tensor2mat(out_tensor*mask)
                            out_mat=tensor2mat(out_tensor)
                            Gui.show(out_mat, "output")
                            # rgb_siren_mat=tensor2mat(rgb_siren)
                            # Gui.show(rgb_siren_mat, "output_siren")
            
                        #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                        if first_time:
                            first_time=False
                            # optimizer=RAdam( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                            # optimizer = RAdam([
                            #     {'params': model.siren_net.net[0][0].sine_scale, 'lr': 0.1 },
                            #     {'params': model.siren_net.net[1][0].sine_scale, 'lr': 0.1 },
                            #     {'params': model.siren_net.net[2][0].sine_scale, 'lr': 0.1 },
                            # ], lr=train_params.lr(), weight_decay=0.0)

                            # #make a parameter group for the sirens and another for the rest
                            # param_rest=[]
                            # param_sine_scale=[]
                            # # for p in model.parameters():
                            # for name, param in model.named_parameters():
                            #     # print(param)
                            #     if "sine_scale" in name:
                            #         print (name)
                            #         param_sine_scale.append(param)
                            #     else: 
                            #         param_rest.append(param)
                            # # param_rest=model.parameters() 
                            # # print(param_rest)
                            # # optimizer=RAdam( param_rest, lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                            # optimizer = RAdam([
                            #     {'params': param_sine_scale, 'lr': 0.5 },
                            #     {'params': param_rest, 'lr': train_params.lr() },
                            # ], lr=train_params.lr(), weight_decay=0.0)



                            # optimizer=torch.optim.AdamW( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )

                            #optimizer which contains all the znear and zfar also https://pytorch.org/docs/stable/optim.html#per-parameter-options
                            if use_ray_compression:
                                param_znear_zfar=[]
                                for f in frames_for_training:
                                    param_znear_zfar.append(f.znear_zfar)
                            # optimizer=torch.optim.AdamW( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )

                            if use_ray_compression:
                                optimizer=torch.optim.AdamW ([
                                    {'params': model.parameters()},
                                    {'params': param_znear_zfar, 'lr': train_params.lr() }
                                ], lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                            else:
                                optimizer=torch.optim.AdamW( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )






                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.1)
                            optimizer.zero_grad()

                        cb.after_forward_pass(loss=rgb_loss, phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 
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

                        # ##aply more lr to the sine scale
                        # with torch.set_grad_enabled(False):
                        #     # model.siren_net.net[0][0].sine_scale
                        #     nr_siren_layers=model.siren_net.nr_layers
                        #     for i in range( nr_siren_layers ):
                        #         grad=model.siren_net.net[i][0].sine_scale.grad
                        #         print("grad norm is ", grad.norm() )
                        #         model.siren_net.net[i][0].sine_scale.data += grad.detach()*-10000

                        # print("fcmu grad norm", model.fc_mu.weight.grad.norm())
                        # print("first_conv norm", model.first_conv.weight.grad.norm())

                        #check the grad for the znear zfar
                        # print("rma grad is ", gt_frame.znear_zfar.grad.norm() )

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
