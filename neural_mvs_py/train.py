#!/usr/bin/env python3.6

import torch

import numpy as np
import random

from easypbr  import *
from dataloaders import *
from neuralmvs import *
from neural_mvs.models import *
from neural_mvs.modules import *
from neural_mvs.utils import *

from callbacks.callback import *
from callbacks.viewer_callback import *
from callbacks.visdom_callback import *
from callbacks.tensorboard_callback import *
from callbacks.state_callback import *
from callbacks.phase import *

from optimizers.over9000.radam import *


import piq


#debug 
from easypbr import Gui
from easypbr import Scene




config_file="train.cfg"

torch.manual_seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = True

# #initialize the parameters used for training
train_params=TrainParams.create(config_file)    



def run():
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
    if train_params.with_viewer():
        view=Viewer.create(config_path)


    first_time=True
    experiment_name="s36"


    predict_confidence_map=True
    multi_res_loss=True



    cb_list = []
    if(train_params.with_visdom()):
        cb_list.append(VisdomCallback(experiment_name))
    if(train_params.with_viewer()):
        cb_list.append(ViewerCallback())
    if(train_params.with_tensorboard()):
        cb_list.append(TensorboardCallback(experiment_name))
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)

    #create loaders
    loader_train, loader_test=create_loader(train_params.dataset_name(), config_path)
    frames_train = get_frames(loader_train)
    frames_test = get_frames(loader_test)
    dataset_params = compute_dataset_params(loader_train, frames_train)
   

  
    #create phases
    phases= [
        Phase('train', loader_train, grad=True),
        Phase('test', loader_test, grad=False)
    ]
    phases[0].frames=frames_train 
    phases[1].frames=frames_test
    #model 
    model=None
    model=Net(predict_confidence_map, multi_res_loss).to("cuda")
    model.train()

    scheduler=None
    frame_weights_computer= FrameWeightComputer()

    show_every=10
    use_novel_orbit_frame=False #for testing we can either use the frames from the loader or create new ones that orbit around the object
    eval_every_x_epoch=30


    max_test_psnr=0.0



    while True:

        for phase in phases:
            cb.epoch_started(phase=phase)
            cb.phase_started(phase=phase)
            model.train(phase.grad)
            is_training=phase.grad

            if phase.loader.has_data(): # the nerf loader will always return true because it preloads all data
             
                nr_scenes=1 
                nr_frames=phase.loader.nr_samples()
                if use_novel_orbit_frame and not is_training:
                    nr_frames=360
                else:
                    if isinstance(loader_train, DataLoaderShapeNetImg) or isinstance(loader_train, DataLoaderSRN) or isinstance(loader_train, DataLoaderDTU):
                        if is_training:
                            nr_scenes=phase.loader.nr_scenes()
                            nr_frames=1 #train on only a random frame from each scene
                        else: #when we evaluate we evalaute over everything
                            nr_scenes= phase.loader.nr_scenes()

                #if we are evalauting, we evaluate every once in a while so most of the time we just skip this
                if not is_training and phase.epoch_nr%eval_every_x_epoch!=0:
                    nr_scenes=0


                for scene_idx in range(nr_scenes):
                    for i in range( nr_frames ):
                        if phase.grad:
                            frame=random.choice(phase.frames)
                        else:
                            if not use_novel_orbit_frame:
                                frame=phase.frames[i]
                            else:
                                #get novel frame that is an orbit around the origin 
                                frame=phase.frames[0]
                                model_matrix = frame.frame.tf_cam_world.inverse()
                                model_matrix=model_matrix.orbit_y_around_point([0,0,0], 360/nr_frames)
                                frame.frame.tf_cam_world = model_matrix.inverse()
                                frame=FramePY(frame.frame)
                        TIME_START("all")

                        ##PREPARE data 
                        with torch.set_grad_enabled(False):


                            frame.load_images()

                           

                            discard_same_idx=is_training # if we are training we don't select the frame with the same idx, if we are testing even if they have the same idx there are from different sets ( test set and train set)
                            if isinstance(loader_train, DataLoaderShapeNetImg) or isinstance(loader_train, DataLoaderSRN) or isinstance(loader_train, DataLoaderDTU):
                                discard_same_idx=True
                            frames_to_consider_for_neighbourhood=frames_train
                            if isinstance(loader_train, DataLoaderShapeNetImg) or isinstance(loader_train, DataLoaderSRN) or isinstance(loader_train, DataLoaderDTU): #if it's these loader we cannot take the train frames for testing because they dont correspond to the same object
                                frames_to_consider_for_neighbourhood=phase.frames
                            do_close_computation_with_delaunay=True
                            if not do_close_computation_with_delaunay:
                                frames_close=get_close_frames(loader_train, frame, frames_to_consider_for_neighbourhood, 8, discard_same_idx) 
                                weights= frame_weights_computer(frame, frames_close)
                            else:
                                frames_close, weights=get_close_frames_barycentric(frame, frames_to_consider_for_neighbourhood, discard_same_idx, dataset_params.sphere_center, dataset_params.sphere_radius, dataset_params.triangulation_type)
                                weights= torch.from_numpy(weights.copy()).to("cuda").float() 

                            #load the image data for this frames that we selected
                            for i in range(len(frames_close)):
                                frames_close[i].load_images()

                            rgb_close_fulres_batch_list=[]
                            for frame_close in frames_close:
                                rgb_close_frame=mat2tensor(frame_close.frame.rgb_32f, True).to("cuda")
                                rgb_close_fulres_batch_list.append(rgb_close_frame)
                            rgb_close_fullres_batch=torch.cat(rgb_close_fulres_batch_list,0)


                            #load the image data for this frames that we selected
                            # for i in range(len(frames_close)):
                                # Gui.show(frames_close[i].frame.rgb_32f,"close_"+phase.name+"_"+str(i) )


                            #prepare rgb data and rest of things
                            rgb_gt, ray_dirs, rgb_close_batch, ray_dirs_close_batch, ray_diff = prepare_data(frame, frames_close)



                        #view current active frame
                        if train_params.with_viewer(): 
                            frustum_mesh=frame.frame.create_frustum_mesh(dataset_params.frustum_size)
                            frustum_mesh.m_vis.m_line_width=3
                            frustum_mesh.m_vis.m_line_color=[1.0, 0.0, 1.0] #purple
                            Scene.show(frustum_mesh, "frustum_activ" )
                            

                            #show the curstums of the close frames
                            for i in range(len(frames_close)):
                                frustum_mesh=frames_close[i].frame.create_frustum_mesh(dataset_params.frustum_size)
                                frustum_mesh.m_vis.m_line_width= (weights[i])*15
                                frustum_mesh.m_vis.m_line_color=[0.0, 1.0, 0.0] #green
                                frustum_mesh.m_force_vis_update=True
                                Scene.show(frustum_mesh, "frustum_neighb_"+str(i) ) 






                        #forward 
                        with torch.set_grad_enabled(is_training):


                            TIME_START("forward")
                            rgb_pred, depth_pred, point3d, new_loss, depth_for_each_res, confidence_map, depth_for_each_step=model(dataset_params, frame, ray_dirs, rgb_close_batch, rgb_close_fullres_batch, ray_dirs_close_batch, ray_diff, frame, frames_close, weights, novel=not phase.grad)
                            TIME_END("forward")



                            if predict_confidence_map:
                                rgb_pred_with_confidence_blending=confidence_map*rgb_pred + (1-confidence_map)*rgb_gt

                         
                        
                            #loss
                            loss=0
                            if predict_confidence_map:
                                rgb_loss_l1= ((rgb_gt- rgb_pred_with_confidence_blending).abs()).mean()
                                #loss for the conidence to be close to 1.0
                                loss_mask=((1.0-confidence_map)**2).mean()
                                loss+=loss_mask*0.1
                            else:
                                rgb_loss_l1= ((rgb_gt- rgb_pred).abs()).mean()
                            psnr_index = piq.psnr(rgb_gt, torch.clamp(rgb_pred,0.0,1.0), data_range=1.0 )
                            loss+=rgb_loss_l1
                            loss+=new_loss

                          
                            #at the beggining we just optimize so that the lstm predicts the center of the sphere 
                            weight=map_range( torch.tensor(phase.iter_nr), 0, 1000, 0.0, 1.0)
                            loss*=weight


                            #constant loss that says that the depth should be have values above the  dataset_params.raymarch_depth_min, keeps the depht from flipping to the other side of the camera
                            diff= depth_pred-  dataset_params.raymarch_depth_min # ideally this is only positive values but if it has negative values then we apply the loss
                            diff=torch.clamp(diff, max=0.0) #make it run from the negative to the 0 so if the depth is above the minimum then the loss is zero
                            loss+=-diff.mean()*100 #the more negtive the depth goes in the other direction, the more the loss increses
                

                            #loss that pushes the points to be in the middle of the space 
                            if phase.iter_nr<1000:
                                loss+= (torch.abs( point3d.norm(dim=1,keepdim=True) -  dataset_params.estimated_scene_dist_from_origin )).mean() *0.2


                        
                        
                            #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                            if first_time:
                                first_time=False
                              
                                optimizer=RAdam( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, mode='max', patience=500) #for LLFF when we overfit
                                optimizer.zero_grad()

                            cb.after_forward_pass(loss=loss.item(), psnr=psnr_index.item(), loss_rgb=rgb_loss_l1.item(), phase=phase, lr=optimizer.param_groups[0]["lr"], rgb_pred=rgb_pred.clamp(0,1), rgb_gt=rgb_gt.clamp(0,1), confidence_map=confidence_map.clamp(0,1), point3d=point3d  ) #visualizes the prediction 


                        #backward
                        if is_training:
                            optimizer.zero_grad()
                            cb.before_backward_pass()
                            TIME_START("backward")
                            loss.backward()
                            TIME_END("backward")
                            cb.after_backward_pass()

                            # summary(model)
                          
                         

                            optimizer.step()

                 

                        TIME_END("all")



                        if train_params.with_viewer(): 
                            with torch.set_grad_enabled(False):
                                #VIEW pred
                                if phase.iter_nr%show_every==0:
                                    #view diff 
                                    diff=( rgb_gt-rgb_pred)**2*10
                                    Gui.show(tensor2mat(diff),"diff_"+phase.name)
                                    Gui.show( tensor2mat(rgb_pred).rgb2bgr() ,"rgb_pred_"+phase.name)
                                    if predict_confidence_map:
                                        Gui.show( tensor2mat(confidence_map) ,"confidence_"+phase.name)
                                 
                                    #view gt
                                    Gui.show(tensor2mat(rgb_gt).rgb2bgr(),"rgb_gt_"+phase.name)
                              
                                    Gui.show(tensor2mat(rgb_close_batch[0:1, :,:,:] ).rgb2bgr(),"rgbclose" )

                                
                                #VIEW 3d points   at the end of the ray march
                                camera_center=torch.from_numpy( frame.frame.pos_in_world() ).to("cuda")
                                camera_center=camera_center.view(1,3)
                                
                                #view normal
                                normal_img=compute_normal(point3d)
                                normal_vis=(normal_img+1.0)*0.5
                                Gui.show(tensor2mat(normal_vis), "normal")

                                normal=normal_img.permute(0,2,3,1) # from n,c,h,w to N,H,W,C
                                normal=normal.view(-1,3)
                               

                                #show things
                                points3d_mesh=show_3D_points( nchw2lin(point3d), color=rgb_pred)
                                points3d_mesh.NV= normal.detach().cpu().numpy()
                                Scene.show(points3d_mesh, "points3d_mesh")

                        if train_params.with_viewer():
                            view.update()



                        #load the next scene 
                    TIME_START("load")
                    if isinstance(loader_train, DataLoaderShapeNetImg) or isinstance(loader_train, DataLoaderSRN) or isinstance(loader_train, DataLoaderDTU):
                        TIME_START("justload")
                       
                        phase.loader.start_reading_next_scene()
                        #wait until they are read
                        while True:
                            if( phase.loader.finished_reading_scene() ): 
                                break
                        TIME_END("justload")
                        frames_list=[]
                        for i in range(phase.loader.nr_samples()):
                            frame_cur=phase.loader.get_frame_at_idx(i)
                            frames_list.append(FramePY(frame_cur, create_subsamples=True))
                        phase.frames=frames_list
                    TIME_END("load")














            # finished all the images 
            if not is_training and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(phase.scores.avg_psnr())
            cb.epoch_ended(phase=phase, model=model, save_checkpoint=train_params.save_checkpoint(), checkpoint_path=train_params.checkpoint_path(), save_every_x_epoch=train_params.save_every_x_epoch() ) 
            cb.phase_ended(phase=phase) 


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
