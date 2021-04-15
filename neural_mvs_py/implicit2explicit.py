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

from neural_mvs.smooth_loss import *
from neural_mvs.ssim import * #https://github.com/VainF/pytorch-msssim
import neural_mvs.warmup_scheduler as warmup  #https://github.com/Tony-Y/pytorch_warmup
from torchsummary.torchsummary import *

#debug 
from easypbr import Gui
from easypbr import Scene
# from neural_mvs.modules import *

import subprocess


#lnet 
# from deps.lnets.lnets.utils.math.autodiff import *


config_file="train.cfg"

torch.manual_seed(0)
random.seed(0)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(edgeitems=3)

# #initialize the parameters used for training
train_params=TrainParams.create(config_file)    
model_params=ModelParams.create(config_file)    

# d

     



def run():
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
    if train_params.with_viewer():
        view=Viewer.create(config_path)


    first_time=True
    experiment_name="s_"


    use_ray_compression=False
    do_superres=False
    predict_occlusion_map=True




    cb_list = []
    if(train_params.with_visdom()):
        cb_list.append(VisdomCallback(experiment_name))
    if(train_params.with_viewer()):
        cb_list.append(ViewerCallback())
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)

    #create loaders
    loader_train, loader_test=create_loader(train_params.dataset_name(), config_path)

   

    #create phases
    phases= [
        Phase('train', loader_train, grad=True),
        # Phase('test', loader_test, grad=False)
    ]
    #model 
    model=None
    model=Net3_SRN(model_params, do_superres).to("cuda")
    model.eval()

    scheduler=None
    concat_coord=ConcatCoord() 
    smooth = InverseDepthSmoothnessLoss()
    ssim_l1_criterion = MS_SSIM_L1_LOSS(compensation=1.0)

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
    # phases[1].frames=frames_test
    #Show only the visdom for the testin
    phases[0].show_visdom=False
    # phases[1].show_visdom=True

    #get the triangulation of the frames 
    frame_centers, frame_idxs = frames_to_points(frames_train)
    sphere_center, sphere_radius=SFM.fit_sphere(frame_centers)
    #if ithe shapentimg we put the center to zero because we know where it is
    if isinstance(loader_train, DataLoaderShapeNetImg):
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
    #usa_subsampled_frames
    factor_subsample_close_frames=0 #0 means that we use the full resoslution fot he image, anything above 0 means that we will subsample the RGB_closeframes from which we compute the features
    factor_subsample_depth_pred=0
    use_novel_orbit_frame=False #for testing we can either use the frames from the loader or create new ones that orbit aorund the object

    new_frame=None

    grad_history = []

    torch.cuda.empty_cache()
    print( torch.cuda.memory_summary() )

    while True:

        points3d_meshes=[]
        if first_time: #we get all the 3D points from all the train frame just once
            for phase in phases:
                cb.epoch_started(phase=phase)
                cb.phase_started(phase=phase)
                # model.train(phase.grad)
                is_training=phase.grad

                # if phase.loader.finished_reading_scene(): #For shapenet
                if phase.loader.has_data(): # the nerf will always return true because it preloads all data, the shapenetimg dataset will return true when the scene it actually loaded

                
                    for i in range(phase.loader.nr_samples()):
                    # for i in range(2):
                        frame=phase.frames[i]
                        TIME_START("all")
                        #get a subsampled frame if necessary
                        frame_full_res=frame
                        if factor_subsample_depth_pred!=0:
                            frame=frame.subsampled_frames[factor_subsample_depth_pred-1]
                        

                        # if frame.frame_idx!=83 or is_training:
                        #     continue

                        ##PREPARE data 
                        with torch.set_grad_enabled(False):
                            discard_same_idx=is_training # if we are training we don't select the frame with the same idx, if we are testing, even if they have the same idx there are from different sets ( test set and train set)
                            do_close_computation_with_delaunay=True
                            if not do_close_computation_with_delaunay:
                                frames_close=get_close_frames(loader_train, frame, frames_train, 5, discard_same_idx) #the neighbour are only from the training set
                                weights= frame_weights_computer(frame, frames_close)
                            else:
                                frames_close, weights=get_close_frames_barycentric(frame, frames_train, discard_same_idx, sphere_center, sphere_radius)
                                weights= torch.from_numpy(weights.copy()).to("cuda").float() 

                            #the frames close may need to be subsampled
                            if factor_subsample_close_frames!=0:
                                frames_close_subsampled=[]
                                for frame_close in frames_close:
                                    frame_subsampled= frame_close.subsampled_frames[factor_subsample_close_frames-1]
                                    frames_close_subsampled.append(frame_subsampled)
                                frames_close= frames_close_subsampled

                            #load the image data for this frames that we selected
                            frame.load_images()
                            for i in range(len(frames_close)):
                                frames_close[i].load_images()


                            #prepare rgb data and rest of things
                            rgb_gt=mat2tensor(frame.frame.rgb_32f, False).to("cuda")
                            rgb_gt_fullres=mat2tensor(frame_full_res.frame.rgb_32f, False).to("cuda")
                            mask_tensor=mat2tensor(frame.frame.mask, False).to("cuda")
                            ray_dirs=torch.from_numpy(frame.ray_dirs).to("cuda").float()
                            rgb_close_batch_list=[]
                            for frame_close in frames_close:
                                rgb_close_frame=mat2tensor(frame_close.frame.rgb_32f, False).to("cuda")
                                rgb_close_batch_list.append(rgb_close_frame)
                            rgb_close_batch=torch.cat(rgb_close_batch_list,0)
                        # print("frame is height widht", frame.height, " ", frame.width) #colmap has 189x252
                        # print("frame has shape ", rgb_gt.shape)
                        # print("rgb close frame ", rgb_close_frame.shape)

                        # print( torch.cuda.memory_summary() )

                        #select certian pixels 
                        pixels_indices=None
                        use_pixel_indices=False
                        rgb_gt_selected=rgb_gt
                        
                        #view current active frame
                        frustum_mesh=frame.frame.create_frustum_mesh(0.02)
                        frustum_mesh.m_vis.m_line_width=3
                        frustum_mesh.m_vis.m_line_color=[1.0, 0.0, 1.0] #purple
                        Scene.show(frustum_mesh, "frustum_activ" )




                        #forward attempt 2 using a network with differetnaible ray march
                        with torch.set_grad_enabled(False):


                            TIME_START("forward")
                            # print( torch.cuda.memory_summary() )
                            # with profiler.profile(profile_memory=True, record_shapes=True, use_cuda=True) as prof:
                            rgb_pred, rgb_refined, depth_pred, mask_pred, signed_distances_for_marchlvl, std, raymarcher_loss, point3d=model(frame, ray_dirs, rgb_close_batch, depth_min, depth_max, frames_close, weights, pixels_indices, novel=not phase.grad)
                            TIME_END("forward")

                            if first_time:
                                first_time=False
                                # now that all the parameters are created we can fill them with a model from a file
                                # model.load_state_dict(torch.load( "/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/saved_models/nerf_lego_sub4/model_e_100.pt" ))
                                model.load_state_dict(torch.load( "/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/saved_models/head_s16/model_e_100.pt" ))
                                rgb_pred, rgb_refined, depth_pred, mask_pred, signed_distances_for_marchlvl, std, raymarcher_loss, point3d=model(frame, ray_dirs, rgb_close_batch, depth_min, depth_max, frames_close, weights, pixels_indices, novel=not phase.grad)
                            # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))


                            #sometimes the refined one doesnt upsample nicely to the full res 
                            if do_superres and rgb_refined.shape!=rgb_gt_fullres:
                                rgb_refined=torch.nn.functional.interpolate(rgb_refined,size=(rgb_gt_fullres.shape[2], rgb_gt_fullres.shape[3]), mode='bilinear')

                        TIME_END("all")

                        if not ( use_pixel_indices and is_training): 
                            with torch.set_grad_enabled(False):
                                #VIEW pred
                                #make masks 
                                mask_pred_thresh=mask_pred<0.3
                                rgb_pred_channels_last=rgb_pred.permute(0,2,3,1) # from n,c,h,w to N,H,W,C
                                rgb_pred_zeros=rgb_pred_channels_last.view(-1,3).norm(dim=1, keepdim=True)
                                rgb_pred_zeros_mask= rgb_pred_zeros<0.05
                                rgb_pred_zeros_mask_img= rgb_pred_zeros_mask.view(1,1,frame.height,frame.width)
                                rgb_pred_zeros_mask=rgb_pred_zeros_mask.repeat(1,3) #repeat 3 times for rgb
                                if phase.iter_nr%show_every==0:
                                    #view diff 
                                    diff=( rgb_gt-rgb_pred)**2*10
                                    Gui.show(tensor2mat(diff),"diff_"+phase.name)
                                    #mask
                                    if predict_occlusion_map:
                                        rgb_pred.masked_fill_(mask_pred_thresh, 0.0)
                                    rgb_pred_mat=tensor2mat(rgb_pred)
                                    Gui.show(rgb_pred_mat,"rgb_pred_"+phase.name)
                                    if do_superres:
                                        rgb_refined_mat=tensor2mat(rgb_refined)
                                        Gui.show(rgb_refined_mat,"rgb_refined_"+phase.name)
                                    #view gt
                                    Gui.show(tensor2mat(rgb_gt),"rgb_gt_"+phase.name)
                                    Gui.show(tensor2mat(mask_pred),"mask_pred_"+phase.name)
                                    Gui.show(tensor2mat(mask_pred_thresh*1.0),"mask_pred_t_"+phase.name)
                                    # print("depth_pred min max ", depth_pred.min(), depth_pred.max())
                                    depth_vis=depth_pred.view(1,1,frame.height,frame.width)
                                    # depth_vis=map_range(depth_vis, 0.35, 0.6, 0.0, 1.0) #for the lego shape
                                    # depth_vis=map_range(depth_vis, 0.2, 0.6, 0.0, 1.0) #for the colamp fine leaves
                                    depth_vis=map_range(depth_vis, 0.9, 1.5, 0.0, 1.0) #for the shapenetimgs
                                    depth_vis=depth_vis.repeat(1,3,1,1)
                                    depth_vis.masked_fill_(rgb_pred_zeros_mask_img, 0.0)
                                    Gui.show(tensor2mat(depth_vis),"depth_"+phase.name)
                                    #show rgb for frame close 
                                    Gui.show(tensor2mat(rgb_close_batch_list[0]),"rgbclose" )

                                
                                #VIEW 3d points   at the end of the ray march
                                camera_center=torch.from_numpy( frame.frame.pos_in_world() ).to("cuda")
                                camera_center=camera_center.view(1,3)
                                points3D = camera_center + depth_pred.view(-1,1)*ray_dirs
                                #get the point that have a color of black (correspond to background) and put them to zero
                                rgb_pred_channels_last=rgb_pred.permute(0,2,3,1) # from n,c,h,w to N,H,W,C
                                rgb_pred_zeros=rgb_pred_channels_last.view(-1,3).norm(dim=1, keepdim=True)
                                rgb_pred_zeros_mask= rgb_pred_zeros<0.05
                                rgb_pred_zeros_mask=rgb_pred_zeros_mask.repeat(1,3) #repeat 3 times for rgb
                                points3D[rgb_pred_zeros_mask]=0.0 #MASK the point in the background
                                points3D.masked_fill_(mask_pred_thresh.view(-1,1), 0.0) # mask point occlueded
                                #mask also the points that still have a signed distance 
                                signed_dist=signed_distances_for_marchlvl[ -1 ]
                                signed_dist_mask= signed_dist.abs()>0.03
                                signed_dist_mask=signed_dist_mask.repeat(1,3) #repeat 3 times for rgb
                                # points3D[signed_dist_mask]=0.0

                                #view normal
                                points3D_img=points3D.view(1, frame.height, frame.width, 3)
                                points3D_img=points3D_img.permute(0,3,1,2) #from N,H,W,C to N,C,H,W
                                normal_img=compute_normal(points3D_img)
                                normal_vis=(normal_img+1.0)*0.5
                                # normal_vis=normal_img
                                rgb_pred_zeros_mask_img=rgb_pred_zeros_mask.view(1, frame.height, frame.width, 3)
                                signed_dist_mask_img=signed_dist_mask.view(1, frame.height, frame.width, 3)
                                rgb_pred_zeros_mask_img=rgb_pred_zeros_mask_img.permute(0,3,1,2) #from N,H,W,C to N,C,H,W
                                signed_dist_mask_img=signed_dist_mask_img.permute(0,3,1,2) #from N,H,W,C to N,C,H,W
                                normal_vis[rgb_pred_zeros_mask_img]=0.0
                                # normal_vis[signed_dist_mask_img]=0.0
                                normal_mat=tensor2mat(normal_vis)
                                Gui.show(normal_mat, "normal")

                                #mask based on grazing angle between normal and view angle
                                normal=normal_img.permute(0,2,3,1) # from n,c,h,w to N,H,W,C
                                normal=normal.view(-1,3)
                                dot_view_normal= (ray_dirs * normal).sum(dim=1,keepdim=True)
                                dot_view_normal_mask= dot_view_normal>-0.1 #ideally the dot will be -1, if it goes to 0.0 it's bad and 1.0 is even worse
                                dot_view_normal_mask=dot_view_normal_mask.repeat(1,3) #repeat 3 times for rgb
                                # points3D[dot_view_normal_mask]=0.0

                                #show things
                                # if is_training:
                                    # show_3D_points(points3D, "points_3d_"+str(frame.frame_idx), color=rgb_pred)
                                points3d_mesh=show_3D_points(points3D, color=rgb_pred)
                                points3d_mesh.NV= normal.detach().cpu().numpy()
                                Scene.show(points3d_mesh, "points3d_mesh")
                                points3d_meshes.append(points3d_mesh)




                        if train_params.with_viewer():
                            view.update()


        #finished getting all the points so now we just merge all of them
        if points3d_meshes:
            points3d_mesh=Mesh()
            for i in range(len(points3d_meshes)):
                print("merging  mesh ash shape ", points3d_mesh.V.shape)
                points3d_mesh.add(points3d_meshes[i])
            print("final mesh ash shape ", points3d_mesh.V.shape)
            file_path_root= "/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/meshes/"
            file_name="head_s16"
            file_path= file_path_root+file_name+"_pc.ply"
            points3d_mesh.remove_vertices_at_zero()
            points3d_mesh.save_to_file(file_path)
            # Scene.show(points3d_mesh, "points3d_mesh")

            #run poisson
            poisson_path="/media/rosu/Data/phd/ws/PoissonRecon/Bin/Linux/"
            args= "--in "+file_path + " --out "+ file_path_root+file_name+"_mesh.ply" + " --bType 2"  + " --depth 10  --samplesPerNode 10 --density --pointWeight 0.1 --verbose"
            full_cmd=  poisson_path+"PoissonRecon " + args
            subprocess.run( full_cmd , shell=True)  # doesn't capture output
            #trim 
            args= "--in "+file_path_root+file_name+"_mesh.ply" + " --out "+ file_path_root+file_name+"_mesh_trimmed.ply" + " --trim 7" 
            full_cmd=  poisson_path+"SurfaceTrimmer " + args
            subprocess.run( full_cmd , shell=True)  # doesn't capture output
            #run the unwrapping from blender
            args= " -- --in_path "+file_path_root+file_name+"_mesh_trimmed.ply" + " --out_path "+ file_path_root+file_name+"_mesh_trimmed_uv.ply"
            blender_unwrap_script_path="/media/rosu/Data/phd/ws/misc_scripts/blender_unwrap_one_mesh.py"
            full_cmd=  "blender --python "+ blender_unwrap_script_path + args
            subprocess.run( full_cmd , shell=True)  # doesn't capture output

            #load mesh 
            mesh_trimmed=Mesh(file_path_root+file_name+"_mesh_trimmed_uv.ply")
            Scene.show(mesh_trimmed, "mesh_trimmed")

            

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
