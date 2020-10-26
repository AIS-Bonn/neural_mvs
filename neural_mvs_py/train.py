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


def map_range( input_val, input_start, input_end,  output_start,  output_end):
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    # input_clamped=max(input_start, min(input_end, input_val))
    input_clamped=torch.clamp(input_val, input_start, input_end)
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start)




def run():
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
    if train_params.with_viewer():
        view=Viewer.create(config_path)


    first_time=True

    # experiment_name="default"
    # experiment_name="n4"
    experiment_name="s_1"




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
    # loader.load_only_from_idxs( [0,1,2,3,4,5,6,7] )
    loader.set_shuffle(False)
    loader.set_overfit(False)
    loader.load_only_from_idxs( [0,2,4,6] )
    loader.start()
    loader_test=DataLoaderVolRef(config_path)
    # loader_test.load_only_from_idxs( [0,2,4,6] )
    # loader_test.load_only_from_idxs( [9,10,11,12,13,14,15,16] ) #one full row at the same height
    # loader_test.load_only_from_idxs( [10,12,14,16] )
    # loader_test.load_only_from_idxs( [10] )
    loader_test.set_shuffle(True)
    # loader_test.set_overfit(True) #so we don't reload the image after every reset but we just keep on training on it
    loader_test.start()
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
    all_imgs_list=[]
    all_imgs_poses_cam_world_list=[]
    all_imgs_Ks_list=[]
    while True:
        for i in range(loader.nr_samples()):
            if loader.has_data():
                ref_frame=loader.get_color_frame()
                depth_frame=loader.get_depth_frame()
                # ref_frame.rgb_32f=ref_frame.rgb_with_valid_depth(depth_frame) 
                ref_rgb_tensor=mat2tensor(ref_frame.rgb_32f, False).to("cuda")
                all_imgs_list.append(ref_rgb_tensor)
                all_imgs_poses_cam_world_list.append(ref_frame.tf_cam_world)
                all_imgs_Ks_list.append(ref_frame.K)
                print("appending")
        if loader.is_finished():
            loader.reset()
            break
    all_imgs=torch.cat(all_imgs_list,0).contiguous().to("cuda")
    print("all imgs have shape ", all_imgs.shape)

    initial_render_frame=None
        
    novel_cam=Camera()
    novel_cam.set_position([0, 0.3, 0.2])
    # novel_cam.set_lookat([-0.02, 0.1, -1.3]) #for the figure
    novel_cam.set_lookat([-0.02, 0.1, -1.0]) #for vase
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
            for i in range(loader_test.nr_samples()):

                # if phase.loader.has_data() and loader_test.has_data():
                if loader_test.has_data():

                    # torch.manual_seed(0)


                    gt_frame=loader_test.get_color_frame() #load from the gt loader
                    gt_depth_frame=loader_test.get_depth_frame() #load from the gt loader

                    # #debug
                    cloud=gt_depth_frame.depth2world_xyz_mesh()
                    frustum=gt_depth_frame.create_frustum_mesh(0.1)
                    Scene.show(cloud, "cloud")
                    Scene.show(frustum, "frustum")


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
                        gt_frame.rgb_32f=gt_frame.rgb_with_valid_depth(gt_depth_frame) 
                        # gt_frame.rgb_32f=ref_frame.rgb_32f

                        # ref_rgb_tensor=mat2tensor(ref_frame.rgb_32f, False).to("cuda")
                        gt_rgb_tensor=mat2tensor(gt_frame.rgb_32f, False).to("cuda")
                        mask=gt_rgb_tensor>0.0
                        # ref_rgb_tensor=ref_rgb_tensor.contiguous()

                        #EXPERIMENT make the gt tensor just black and white  SO we predict just black for background and white for the objects
                        # gt_rgb_tensor=mask*1.0
                        # gt_frame.rgb_32f=tensor2mat(gt_rgb_tensor)

                        if(phase.iter_nr%show_every==0):
                            # print("width and height ", ref_frame.width)
                            # Gui.show(ref_frame.rgb_32f, "ref")
                            Gui.show(gt_frame.rgb_32f, "gt")

                        # #try another view
                        with torch.set_grad_enabled(False):
                            if initial_render_frame==None:
                                initial_render_frame=gt_frame.tf_cam_world
                                novel_cam.set_position(gt_frame.tf_cam_world.inverse().translation())
                            render_tf=initial_render_frame
                            novel_cam.orbit_y(10)
                            render_tf=novel_cam.view_matrix_affine()
                            render_K=novel_cam.intrinsics(100, 70)
                            # print("novel cam translation is ", render_tf.translation())
                            # print("gt fra,e translation is ", gt_frame.tf_cam_world.translation())
                            # exit(1)
                            # out_tensor=model(ref_rgb_tensor, ref_frame.tf_cam_world, render_tf )
                            out_tensor, depth_map, acc_map, new_loss=model(gt_rgb_tensor, all_imgs, all_imgs_poses_cam_world_list, render_tf, gt_frame.K, novel=True )
                            # out_tensor=model(ref_rgb_tensor, render_tf, render_tf )
                            if(phase.iter_nr%1==0):
                                out_mat=tensor2mat(out_tensor)
                                Gui.show(out_mat, "novel")
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
                        out_tensor, depth_map, acc_map, new_loss=model(gt_rgb_tensor, all_imgs, all_imgs_poses_cam_world_list, gt_frame.tf_cam_world, gt_frame.K )
                        # out_tensor=model(gt_rgb_tensor)
                        # out_tensor, mu, logvar = model(ref_rgb_tensor)
                        TIME_END("forward")

                        #calculate smoothness loss 
                        # smooth_loss=inverse_depth_smoothness_loss(depth_map, gt_rgb_tensor)
                        # print("smooth_loss", smooth_loss)

                        

                        with torch.set_grad_enabled(False):
                            if(phase.iter_nr%show_every==0):
                                # print("depth map has shape ", depth_map.shape)
                                # print("mask has shape ", mask.shape)
                                depth_map=depth_map*mask
                                # depth_map=depth_map-1.5 #it's in range 1 to 2 meters so now we set it to range 0 to 1
                                # depth_map_nonzero=depth_map!=0.0
                                # print("min max", depth_map.min(), " ", depth_map.max())
                                depth_map=map_range(depth_map, 0.7, 1.2, 0.0, 1.0)
                                depth_map_mat=tensor2mat(depth_map)
                                Gui.show(depth_map_mat, "depth")

                        # print("out tensor  ", out_tensor.min(), " ", out_tensor.max())
                        # print("out tensor  ", gt_rgb_tensor.min(), " ", gt_rgb_tensor.max())
                        loss=((out_tensor-gt_rgb_tensor)**2).mean()
                        # loss=(((out_tensor-gt_rgb_tensor)**2)).mean()  / loader_test.nr_samples()
                        # loss=(((out_tensor-gt_rgb_tensor)**2)).mean()  / 10
                        # loss=loss_fn(out_tensor, gt_rgb_tensor)
                        # print("loss is ", loss)

                        # loss+=smooth_loss*0.00001*phase.iter_nr
                        ##PUT also the new losses
                        # loss+=new_loss*0.001*phase.iter_nr

                        #debug the diff map 
                        diff=(((out_tensor-gt_rgb_tensor)**2))
                        diff_mat=tensor2mat(diff)
                        Gui.show(diff_mat, "diff_mat")



                        if(phase.iter_nr%show_every==0):
                            # out_mat=tensor2mat(out_tensor)
                            # out_mat=tensor2mat(out_tensor*mask)
                            out_mat=tensor2mat(out_tensor)
                            Gui.show(out_mat, "output")
            
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



                            optimizer=torch.optim.AdamW( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.1)
                            optimizer.zero_grad()

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

                        optimizer.step()

                        # if (phase.iter_nr%10==0):
                        #     optimizer.step() # DO it only once after getting gradients for all images
                        #     optimizer.zero_grad()


                if train_params.with_viewer():
                    view.update()

            # finished all the images 
            # pbar.close()
            if is_training and loader_test.is_finished(): #we reduce the learning rate when the test iou plateus
                # optimizer.step() # DO it only once after getting gradients for all images
                # optimizer.zero_grad()
                # if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # scheduler.step(phase.loss_acum_per_epoch) #for ReduceLROnPlateau
                cb.epoch_ended(phase=phase, model=model, save_checkpoint=train_params.save_checkpoint(), checkpoint_path=train_params.checkpoint_path() ) 
                cb.phase_ended(phase=phase) 
                # phase.epoch_nr+=1
                loader_test.reset()
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
