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
from neural_mvs.modules import *
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
from optimizers.over9000.apollo import *
from optimizers.adahessian import *
import optimizers.gradient_centralization.ranger2020 as GC_Ranger #incorporated also gradient centralization but it seems to converge slower than the Ranger from over9000
import optimizers.gradient_centralization.Adam as GC_Adam
import optimizers.gradient_centralization.RAdam as GC_RAdam

from neural_mvs.smooth_loss import *
from neural_mvs.ssim import * #https://github.com/VainF/pytorch-msssim
import neural_mvs.warmup_scheduler as warmup  #https://github.com/Tony-Y/pytorch_warmup

#debug 
from easypbr import Gui
from easypbr import Scene
# from neural_mvs.modules import *


#lnet 
# from deps.lnets.lnets.utils.math.autodiff import *


config_file="eval.cfg"

torch.manual_seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
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

    # experiment_name="s13_rg_ac_0.003"
    experiment_name="s1ad_0.001"

    use_ray_compression=False
    do_superres=True
    predict_occlusion_map=False





    cb_list = []
    if(train_params.with_visdom()):
        cb_list.append(VisdomCallback(experiment_name))
    if(train_params.with_viewer()):
        cb_list.append(ViewerCallback())
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)

    #create loaders
    # loader_train=DataLoaderNerf(config_path)
    # loader_test=DataLoaderNerf(config_path)
    # loader_train=DataLoaderColmap(config_path)
    # loader_test=DataLoaderColmap(config_path)
    # loader_train.set_mode_all()
    # loader_test.set_mode_all()
    # loader_train.start()
    # loader_test.start()
    loader_train, loader_test=create_loader(train_params.dataset_name(), config_path)

    #create phases
    phases= [
        Phase('train', loader_train, grad=True),
        Phase('test', loader_test, grad=False)
    ]
    #model 
    model=None
    model=Net3_SRN(model_params, do_superres).to("cuda")
    model.eval()

    scheduler=None
    concat_coord=ConcatCoord() 
    smooth = InverseDepthSmoothnessLoss()
    ssim_l1_criterion = MS_SSIM_L1_LOSS()

    show_every=1



    #get all the frames train in am array, becuase it's faster to have everything already on the gpu
    frames_train=[]
    frames_test=[]
    for i in range(loader_train.nr_samples()):
        frame=loader_train.get_frame_at_idx(i)
        frames_train.append(FramePY(frame, create_subsamples=True))
    for i in range(loader_test.nr_samples()):
        frame=loader_test.get_frame_at_idx(i)
        frames_test.append(FramePY(frame, create_subsamples=True))
    phases[0].frames=frames_train 
    phases[1].frames=frames_test
    #Show only the visdom for the testin
    phases[0].show_visdom=False
    phases[1].show_visdom=True




    #get the triangulation of the frames 
    frame_centers, frame_idxs = frames_to_points(frames_train)
    sphere_center, sphere_radius=SFM.fit_sphere(frame_centers)
    if isinstance(loader_train, DataLoaderShapeNetImg) or isinstance(loader_train, DataLoaderDTU):
        sphere_center= np.array([0,0,0])
        sphere_radius= np.amax(np.linalg.norm(frame_centers- sphere_center, axis=1))
    print("sphere center and raidys ", sphere_center, " radius ", sphere_radius)
    frame_weights_computer= FrameWeightComputer()

    # triangulated_mesh, sphere_center, sphere_radius=SFM.compute_triangulation(loader_train.get_all_frames())

    #depth min max for nerf 
    depth_min=2
    depth_max=5
    #depth min max for home photos
    depth_min=3.5
    depth_max=11.5
    #depth min max for home photos after scaling the scenne
    depth_min=0.15
    depth_max=1.0

    new_frame=None

    grad_history = []

    torch.cuda.empty_cache()
    print( torch.cuda.memory_summary() )

    neural_mvs_gui=NeuralMVSGUI.create(view)


    frame=FramePY(frames_test[0].frame)
    tf_world_cam=frame.tf_cam_world.inverse()
    print("initial pos", tf_world_cam.translation() )
    print("initial quat", tf_world_cam.quat() )

    #set the camera to be in the same position as the first frame
    # view.m_camera.set_position(  tf_world_cam.translation() )
    # view.m_camera.set_quat(  tf_world_cam.quat() )
    view.m_camera.set_model_matrix(tf_world_cam)
    view.m_camera.set_dist_to_lookat(0.3)
    # view.m_camera.set_lookat( frame.frame.look_dir()*0.5  )

    #check that the quat is correct 
    pos= view.m_camera.model_matrix_affine().translation()
    print("pos of the cam is ", pos)
    quat=view.m_camera.model_matrix_affine().quat()
    print("quat of the cam is ", quat)

    #usa_subsampled_frames
    factor_subsample_close_frames=2 #0 means that we use the full resoslution fot he image, anything above 0 means that we will subsample the RGB_closeframes from which we compute the features
    factor_subsample_depth_pred=2

    while True:
        with torch.set_grad_enabled(False):

        
            #get the model matrix of the view and set it to the frame
            cam_tf_world_cam= view.m_camera.model_matrix_affine()
            frame.frame.tf_cam_world=cam_tf_world_cam.inverse()
            frame=FramePY(frame.frame, create_subsamples=True)


            # discard_same_idx=False
            # do_close_computation_with_delaunay=True
            # if not do_close_computation_with_delaunay:
            #     frames_close=get_close_frames(loader_train, frame, frames_train, 5, discard_same_idx) #the neighbour are only from the training set
            #     weights= frame_weights_computer(frame, frames_close)
            # else:
            #     frames_close, weights=get_close_frames_barycentric(frame, frames_train, discard_same_idx, sphere_center, sphere_radius)
            #     weights= torch.from_numpy(weights.copy()).to("cuda").float() 

            discard_same_idx= False
            if isinstance(loader_train, DataLoaderShapeNetImg) or isinstance(loader_train, DataLoaderSRN) or isinstance(loader_train, DataLoaderDTU):
                discard_same_idx=True
            frames_to_consider_for_neighbourhood=frames_train
            if isinstance(loader_train, DataLoaderShapeNetImg) or isinstance(loader_train, DataLoaderSRN) or isinstance(loader_train, DataLoaderDTU): #if it's these loader we cannot take the train frames for testing because they dont correspond to the same object
                frames_to_consider_for_neighbourhood=frames_test
            do_close_computation_with_delaunay=True
            if not do_close_computation_with_delaunay:
                frames_close=get_close_frames(loader_train, frame, frames_to_consider_for_neighbourhood, 5, discard_same_idx) #the neighbour are only from the training set
                weights= frame_weights_computer(frame, frames_close)
            else:
                frames_close, weights=get_close_frames_barycentric(frame, frames_to_consider_for_neighbourhood, discard_same_idx, sphere_center, sphere_radius)
                weights= torch.from_numpy(weights.copy()).to("cuda").float() 

            #load frames
            frame.load_images()

            frame_full_res=frame
            if factor_subsample_depth_pred!=0 and first_time:
                frame=frame.subsampled_frames[factor_subsample_depth_pred-1]

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

            #prepare rgb data and rest of things
            rgb_gt=mat2tensor(frame.frame.rgb_32f, False).to("cuda")
            mask_tensor=mat2tensor(frame.frame.mask, False).to("cuda")
            ray_dirs=torch.from_numpy(frame.ray_dirs).to("cuda").float()
            rgb_close_batch_list=[]
            for frame_close in frames_close:
                rgb_close_frame=mat2tensor(frame_close.frame.rgb_32f, False).to("cuda")
                rgb_close_batch_list.append(rgb_close_frame)
            rgb_close_batch=torch.cat(rgb_close_batch_list,0)
            #make also a batch fo directions
            raydirs_close_batch_list=[]
            for frame_close in frames_close:
                ray_dirs_close=torch.from_numpy(frame_close.ray_dirs).to("cuda").float()
                ray_dirs_close=ray_dirs_close.view(1, frame_close.height, frame_close.width, 3)
                ray_dirs_close=ray_dirs_close.permute(0,3,1,2) #from N,H,W,C to N,C,H,W
                raydirs_close_batch_list.append(ray_dirs_close)
            ray_dirs_close_batch=torch.cat(raydirs_close_batch_list,0)


            pixels_indices=None

            rgb_pred, rgb_refined, depth_pred, mask_pred, signed_distances_for_marchlvl, std, raymarcher_loss, point3d=model(frame, ray_dirs, rgb_close_batch, rgb_close_fullres_batch, ray_dirs_close_batch, depth_min, depth_max, frames_close, weights, pixels_indices, novel=True)
            # print("depth_pred", depth_pred.mean())

            if first_time:
                first_time=False
                #TODO load checkpoint
                # now that all the parameters are created we can fill them with a model from a file
                # model.load_state_dict(torch.load( "/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/saved_models/fine_leaves_home_plant/model_e_900.pt" ))
                # model.load_state_dict(torch.load( "/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/saved_models/dtu_sub2_sr_v6/model_e_2500.pt" ))
                model.load_state_dict(torch.load( "/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/saved_models/dtu_sub2_sr_v9_nopos_HR/model_e_650.pt" ))


            camera_center=torch.from_numpy( frame.frame.pos_in_world() ).to("cuda")
            camera_center=camera_center.view(1,3)
            points3D = camera_center + depth_pred.view(-1,1)*ray_dirs
            if neural_mvs_gui.m_show_rgb:
                pred_mat=tensor2mat(rgb_pred)
            if neural_mvs_gui.m_show_depth:
                depth_vis=depth_pred.view(1,1,frame.height,frame.width)
                # depth_vis=map_range(depth_vis, 0.2, 0.6, 0.0, 1.0) #for the colamp fine leaves
                depth_vis=map_range(depth_vis, neural_mvs_gui.m_min_depth, neural_mvs_gui.m_max_depth, 0.0, 1.0) #for the colamp fine leaves
                depth_vis=depth_vis.repeat(1,3,1,1)
                pred_mat=tensor2mat(depth_vis)
            if neural_mvs_gui.m_show_normal:
                points3D_img=points3D.view(1, frame.height, frame.width, 3)
                points3D_img=points3D_img.permute(0,3,1,2) #from N,H,W,C to N,C,H,W
                normal_img=compute_normal(points3D_img)
                normal_vis=(normal_img+1.0)*0.5
                pred_mat=tensor2mat(normal_vis)
            Gui.show(pred_mat,"pred")
            Gui.show(tensor2mat(rgb_refined),"pred_refined")






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
