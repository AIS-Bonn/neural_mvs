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
from neuralmvs import *
from neural_mvs.models import *
from neural_mvs.modules import *
from neural_mvs.utils import *
from neural_mvs.MS_SSIM_L1_loss import *

from callbacks.callback import *
from callbacks.viewer_callback import *
from callbacks.visdom_callback import *
from callbacks.state_callback import *
from callbacks.phase import *

# from optimizers.over9000.radam import *
from optimizers.over9000.lookahead import *
from optimizers.over9000.novograd import *
from optimizers.over9000.ranger import *
from optimizers.over9000.rangerlars import *
from optimizers.over9000.apollo import *
from optimizers.over9000.lamb import *
from optimizers.adahessian import *
import optimizers.gradient_centralization.ranger2020 as GC_Ranger #incorporated also gradient centralization but it seems to converge slower than the Ranger from over9000
import optimizers.gradient_centralization.Adam as GC_Adam
import optimizers.gradient_centralization.RAdam as GC_RAdam

import piq

from neural_mvs.smooth_loss import *
from neural_mvs.ssim import * #https://github.com/VainF/pytorch-msssim
import neural_mvs.warmup_scheduler as warmup  #https://github.com/Tony-Y/pytorch_warmup
from torchsummary.torchsummary import *
import torchvision.transforms.functional as TF

#debug 
from easypbr import Gui
from easypbr import Scene
# from neural_mvs.modules import *


#lnet 
# from deps.lnets.lnets.utils.math.autodiff import *


config_file="train.cfg"

torch.manual_seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(edgeitems=3)

# #initialize the parameters used for training
train_params=TrainParams.create(config_file)    
model_params=ModelParams.create(config_file)    



def run():
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
    if train_params.with_viewer():
        view=Viewer.create(config_path)


    first_time=True
    # experiment_name="13lhighlr"
    experiment_name="s_"


    # use_ray_compression=False
    # do_superres=True
    # predict_occlusion_map=False



    cb_list = []
    if(train_params.with_visdom()):
        cb_list.append(VisdomCallback(experiment_name))
    if(train_params.with_viewer()):
        cb_list.append(ViewerCallback())
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
    #model 
    model=None
    model=Net3_SRN(model_params).to("cuda")
    model.train()

    scheduler=None
    smooth = InverseDepthSmoothnessLoss()
    ssim_l1_criterion = MS_SSIM_L1_LOSS(compensation=1.0)
    frame_weights_computer= FrameWeightComputer()

    show_every=1
    factor_subsample_close_frames=2 #0 means that we use the full resoslution fot he image, anything above 0 means that we will subsample the RGB_closeframes from which we compute the features
    factor_subsample_depth_pred=2
    use_novel_orbit_frame=False #for testing we can either use the frames from the loader or create new ones that orbit aorund the object
    eval_every_x_epoch=30


    #get all the frames train in am array, becuase it's faster to have everything already on the gpu
    phases[0].frames=frames_train 
    phases[1].frames=frames_test
    #Show only the visdom for the testin
    phases[0].show_visdom=False
    phases[1].show_visdom=True
  

    grad_history = []
    max_test_psnr=0.0


    while True:

        for phase in phases:
            cb.epoch_started(phase=phase)
            cb.phase_started(phase=phase)
            model.train(phase.grad)
            is_training=phase.grad

            if phase.loader.has_data(): # the nerf will always return true because it preloads all data, the shapenetimg dataset will return true when the scene it actually loaded
             
                nr_frames=0
                nr_scenes=1 #if we use something like dtu we just train on on1 scene and one sample at a time but when trainign we iterate throguh all the scens and all the frames
                nr_frames=phase.loader.nr_samples()
                if use_novel_orbit_frame and not is_training:
                    nr_frames=360
                else:
                    if isinstance(loader_train, DataLoaderShapeNetImg) or isinstance(loader_train, DataLoaderSRN) or isinstance(loader_train, DataLoaderDTU):
                        if is_training:
                            nr_scenes=phase.loader.nr_scenes()
                            nr_frames=1
                        else: #when we evaluate we evalaute over everything
                            nr_scenes= phase.loader.nr_scenes()
                            nr_frames=phase.loader.nr_samples()
                    else: 
                        nr_frames=phase.loader.nr_samples()
                # for i in range(phase.loader.nr_samples()):

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

                            # print("frame rgb path is ", frame.frame.rgb_path)

                            frame.load_images()
                            #get a subsampled frame if necessary
                            frame_full_res=frame
                            if factor_subsample_depth_pred!=0:
                                frame=frame.subsampled_frames[factor_subsample_depth_pred-1]

                            discard_same_idx=is_training # if we are training we don't select the frame with the same idx, if we are testing, even if they have the same idx there are from different sets ( test set and train set)
                            if isinstance(loader_train, DataLoaderShapeNetImg) or isinstance(loader_train, DataLoaderSRN) or isinstance(loader_train, DataLoaderDTU):
                                discard_same_idx=True
                            frames_to_consider_for_neighbourhood=frames_train
                            if isinstance(loader_train, DataLoaderShapeNetImg) or isinstance(loader_train, DataLoaderSRN) or isinstance(loader_train, DataLoaderDTU): #if it's these loader we cannot take the train frames for testing because they dont correspond to the same object
                                frames_to_consider_for_neighbourhood=phase.frames
                            do_close_computation_with_delaunay=True
                            if not do_close_computation_with_delaunay:
                                frames_close=get_close_frames(loader_train, frame, frames_to_consider_for_neighbourhood, 3, discard_same_idx) #the neighbour are only from the training set
                                weights= frame_weights_computer(frame, frames_close)
                            else:
                                frames_close, weights=get_close_frames_barycentric(frame, frames_to_consider_for_neighbourhood, discard_same_idx, dataset_params.sphere_center, dataset_params.sphere_radius, dataset_params.triangulation_type)
                                weights= torch.from_numpy(weights.copy()).to("cuda").float() 

                            #load the image data for this frames that we selected
                            for i in range(len(frames_close)):
                                frames_close[i].load_images()

                            rgb_close_fulres_batch_list=[]
                            for frame_close in frames_close:
                                rgb_close_frame=mat2tensor(frame_close.frame.rgb_32f, False).to("cuda")
                                rgb_close_fulres_batch_list.append(rgb_close_frame)
                            rgb_close_fullres_batch=torch.cat(rgb_close_fulres_batch_list,0)

                            #the frames close may need to be subsampled
                            if factor_subsample_close_frames!=0:
                                frames_close_subsampled=[]
                                for frame_close in frames_close:
                                    frame_subsampled= frame_close.subsampled_frames[factor_subsample_close_frames-1]
                                    frames_close_subsampled.append(frame_subsampled)
                                frames_close= frames_close_subsampled

                            #load the image data for this frames that we selected
                            for i in range(len(frames_close)):
                                Gui.show(frames_close[i].frame.rgb_32f,"close_"+phase.name+"_"+str(i) )


                            #prepare rgb data and rest of things
                            rgb_gt_fullres, rgb_gt, ray_dirs, rgb_close_batch, ray_dirs_close_batch = prepare_data(frame_full_res, frame, frames_close)

                        #random crop of  uv_tensor, ray_dirs and rgb_gt_selected https://discuss.pytorch.org/t/cropping-batches-at-the-same-position/24550/5
                        # TIME_START("crop")
                        # if rand_true(0.5) and is_training:
                        # # if False:
                        #     max_size= min(frame.height, frame.width )
                        #     max_size=random.randint(max_size//4, max_size)
                        #     crop_indices = torchvision.transforms.RandomCrop.get_params( uv_tensor, output_size=(max_size, max_size))
                        #     i, j, h, w = crop_indices
                        #     # uv_tensor = TF.crop(uv_tensor, i, j, h, w)
                        #     rgb_gt_selected= TF.crop(rgb_gt_selected, i, j, h, w)
                        #     ray_dirs=ray_dirs.view(1,frame.height, frame.width, 3).permute(0,3,1,2) #from N,H,W,C to N,C,H,W
                        #     ray_dirs= TF.crop(ray_dirs, i, j, h, w)
                        #     #in order to make the unet access pixels that are further apart or closer apart, we upscale the cropped image to the full size
                        #     uv_tensor = torch.nn.functional.interpolate(uv_tensor,size=(frame.height, frame.width), mode='nearest')
                        #     rgb_gt_selected = torch.nn.functional.interpolate(rgb_gt_selected,size=(frame.height, frame.width), mode='bilinear')
                        #     ray_dirs = torch.nn.functional.interpolate(ray_dirs,size=(frame.height, frame.width), mode='bilinear')
                        #     #back to Nx3 rays
                        #     ray_dirs=ray_dirs.permute(0,2,3,1).reshape(-1,3) #from NCHW to NHWC
                        # TIME_END("crop")



                        #VIEW gt 
                      
                        #view current active frame
                        frustum_mesh=frame.frame.create_frustum_mesh(0.02)
                        frustum_mesh.m_vis.m_line_width=3
                        frustum_mesh.m_vis.m_line_color=[1.0, 0.0, 1.0] #purple
                        Scene.show(frustum_mesh, "frustum_activ" )




                        #forward attempt 2 using a network with differetnaible ray march
                        with torch.set_grad_enabled(is_training):


                            TIME_START("forward")
                            rgb_pred, depth_pred, point3d=model(dataset_params, frame, ray_dirs, rgb_close_batch, rgb_close_fullres_batch, ray_dirs_close_batch, frames_close, weights, novel=not phase.grad)
                            TIME_END("forward")
                          
                            #sometimes the refined one doesnt upsample nicely to the full res 
                            # if rgb_refined.shape!=rgb_gt_fullres:
                            ###TODO do not upsample here but rather upsample the feature maps in the unet of rgb_refiner
                                # rgb_refined=torch.nn.functional.interpolate(rgb_refined,size=(rgb_gt_fullres.shape[2], rgb_gt_fullres.shape[3]), mode='bilinear')
                            rgb_lowres=torch.nn.functional.interpolate(rgb_pred,size=(frame.height, frame.width), mode='bilinear')

                         
                        
                            #loss
                            loss=0
                            rgb_loss_l1= ((rgb_gt_fullres- rgb_pred).abs()).mean()
                            psnr_index = piq.psnr(rgb_gt_fullres, torch.clamp(rgb_pred,0.0,1.0), data_range=1.0 )
                            loss+=rgb_loss_l1
                            if not is_training and psnr_index.item()>max_test_psnr:
                                max_test_psnr=psnr_index.detach().item()
                                
                          
                            #at the beggining we just optimize so that the lstm predicts the center of the sphere 
                            weight=map_range( torch.tensor(phase.iter_nr), 0, 1000, 0.0, 1.0)
                            loss*=weight
                

                            #loss that pushes the points to be in the middle of the space 
                            if phase.iter_nr<1000:
                                loss+=( ( torch.from_numpy(dataset_params.estimated_scene_center).view(1,3).cuda()-point3d).norm(dim=1)).mean()*0.2


                        
                        
                            #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                            if first_time:
                                first_time=False
                              
                                optimizer=GC_Adam.AdamW( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                                # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)
                                # scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10000)
                                # scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, mode='max', patience=10000) 
                                # warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=3000)
                                optimizer.zero_grad()

                            cb.after_forward_pass(loss=psnr_index.item(), phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 


                        #backward
                        if is_training:
                            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                                scheduler.step(phase.iter_nr /10000  ) #go to zero every 10k iters
                            # warmup_scheduler.dampen()
                            optimizer.zero_grad()
                            cb.before_backward_pass()
                            TIME_START("backward")
                            loss.backward()
                            # loss.backward(create_graph=True) #IS NEEDED BY ADAHESIAN but it doesnt work becasue grid sampler doesnt have a second derrivative
                            TIME_END("backward")
                            cb.after_backward_pass()
                          

                            #try something autoclip https://github.com/pseeth/autoclip/blob/master/autoclip.py 
                            clip_percentile=10
                            obs_grad_norm = get_grad_norm(model)
                            # print("grad norm", obs_grad_norm)
                            grad_history.append(obs_grad_norm)
                            clip_value = np.percentile(grad_history, clip_percentile)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                            # print("clip_value", clip_value)

                            # model.summary()
                            # exit()

                            optimizer.step()

                     
                        if not is_training and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(max_test_psnr)

                        TIME_END("all")



                        if True: 
                            with torch.set_grad_enabled(False):
                                #VIEW pred
                                if phase.iter_nr%show_every==0:
                                    #view diff 
                                    diff=( rgb_gt_fullres-rgb_pred)**2*10
                                    Gui.show(tensor2mat(diff),"diff_"+phase.name)
                                    Gui.show( tensor2mat(rgb_pred) ,"rgb_pred_"+phase.name)
                                 
                                    #view gt
                                    Gui.show(tensor2mat(rgb_gt),"rgb_gt_"+phase.name)
                              
                                    Gui.show(tensor2mat(rgb_close_batch[0:1, :,:,:] ),"rgbclose" )

                                
                                #VIEW 3d points   at the end of the ray march
                                camera_center=torch.from_numpy( frame.frame.pos_in_world() ).to("cuda")
                                camera_center=camera_center.view(1,3)
                                points3D = camera_center + depth_pred.view(-1,1)*ray_dirs
                                
                                #view normal
                                points3D_img=points3D.view(1, frame.height, frame.width, 3)
                                points3D_img=points3D_img.permute(0,3,1,2) #from N,H,W,C to N,C,H,W
                                normal_img=compute_normal(points3D_img)
                                normal_vis=(normal_img+1.0)*0.5
                                Gui.show(tensor2mat(normal_vis), "normal")

                                #mask based on grazing angle between normal and view angle
                                normal=normal_img.permute(0,2,3,1) # from n,c,h,w to N,H,W,C
                                normal=normal.view(-1,3)
                                # dot_view_normal= (ray_dirs * normal).sum(dim=1,keepdim=True)
                                # dot_view_normal_mask= dot_view_normal>-0.1 #ideally the dot will be -1, if it goes to 0.0 it's bad and 1.0 is even worse
                                # dot_view_normal_mask=dot_view_normal_mask.repeat(1,3) #repeat 3 times for rgb
                                # points3D[dot_view_normal_mask]=0.0

                                #show things
                                points3d_mesh=show_3D_points(points3D, color=rgb_lowres)
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
            # pbar.close()
            if True: #if we reached this point we already read all the images so there is no need to check if the loader is finished 
                print("epoch finished", phase.epoch_nr, " phase rag is", phase.grad)
                cb.epoch_ended(phase=phase, model=model, save_checkpoint=train_params.save_checkpoint(), checkpoint_path=train_params.checkpoint_path(), save_every_x_epoch=train_params.save_every_x_epoch() ) 
                cb.phase_ended(phase=phase) 
                # phase.epoch_nr+=1
                # loader_test.reset()
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
