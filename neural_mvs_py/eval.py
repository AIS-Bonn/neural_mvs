#!/usr/bin/env python3.6

import torch
import torch.nn.functional as F

import sys
import os
import numpy as np
from tqdm import tqdm
import time
import random
import os.path
from os import path

import piq


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


    predict_confidence_map=False
    multi_res_loss=True






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
        # Phase('train', loader_train, grad=True),
        Phase('test', loader_test, grad=False)
    ]
    #model 
    model=None
    model=Net3_SRN(model_params, predict_confidence_map, multi_res_loss).to("cuda")
    model.eval()

    show_every=1

    #get all the frames train in am array, becuase it's faster to have everything already on the gpu
    phases[0].frames=frames_test 
    #Show only the visdom for the testin
    phases[0].show_visdom=True




    #get the triangulation of the frames 
    frame_centers, frame_idxs = frames_to_points(frames_train)
    sphere_center, sphere_radius=SFM.fit_sphere(frame_centers)
    if isinstance(loader_train, DataLoaderShapeNetImg) or isinstance(loader_train, DataLoaderDTU):
        sphere_center= np.array([0,0,0])
        sphere_radius= np.amax(np.linalg.norm(frame_centers- sphere_center, axis=1))
    print("sphere center and raidys ", sphere_center, " radius ", sphere_radius)
    frame_weights_computer= FrameWeightComputer()

    # triangulated_mesh, sphere_center, sphere_radius=SFM.compute_triangulation(loader_train.get_all_frames())


    new_frame=None
    color_mngr=ColorMngr()

    if train_params.with_viewer():
        neural_mvs_gui=NeuralMVSGUI.create(view)


    frame=FramePY(frames_test[0].frame)
    tf_world_cam=frame.tf_cam_world.inverse()
    K=frame.frame.K.copy()
    print("initial pos", tf_world_cam.translation() )
    print("initial quat", tf_world_cam.quat() )

    #set the camera to be in the same position as the first frame
    # view.m_camera.set_position(  tf_world_cam.translation() )
    # view.m_camera.set_quat(  tf_world_cam.quat() )
    # view.m_camera.set_model_matrix(tf_world_cam)
    # view.m_camera.set_dist_to_lookat(0.3)


    #attempt 2 to get two cameras working 
    cam_for_pred=Camera()
    cam_for_pred.set_model_matrix(tf_world_cam)
    # cam_for_pred.set_dist_to_lookat(0.3) #for dtu
    cam_for_pred.set_dist_to_lookat(0.5) #for nerf

    #check that the quat is correct 
    if train_params.with_viewer():
        pos= view.m_camera.model_matrix_affine().translation()
        print("pos of the cam is ", pos)
        quat=view.m_camera.model_matrix_affine().quat()
        print("quat of the cam is ", quat)

    #usa_subsampled_frames
    factor_subsample_close_frames=0 #0 means that we use the full resoslution fot he image, anything above 0 means that we will subsample the RGB_closeframes from which we compute the features
    factor_subsample_depth_pred=0


    use_spiral=False
    if use_spiral:
        poses_on_spiral= make_list_of_poses_on_spiral(frames_train, path_zflat=False )

    img_nr=0
    psnr_acum=0
    ssim_acum=0
    lpips_acum=0


    # while True: #Do it infinitely if we want to just visualize things
    if True: #to just do it once
        img_nr=0
        for phase in phases:
                cb.epoch_started(phase=phase)
                cb.phase_started(phase=phase)
                model.train(phase.grad)
                is_training=phase.grad

                if phase.loader.has_data(): # the nerf will always return true because it preloads all data, the shapenetimg dataset will return true when the scene it actually loaded
                
                    nr_frames=0
                    nr_scenes=1 #if we use something like dtu we just train on on1 scene and one sample at a time but when trainign we iterate throguh all the scens and all the frames
                    nr_frames=phase.loader.nr_samples()
                    if isinstance(loader_train, DataLoaderShapeNetImg) or isinstance(loader_train, DataLoaderSRN) or isinstance(loader_train, DataLoaderDTU):
                        nr_scenes= phase.loader.nr_scenes()
                        nr_frames=phase.loader.nr_samples()
                    else: 
                        nr_frames=phase.loader.nr_samples()


                    for scene_idx in range(nr_scenes):
                        for i in range( nr_frames ):
                            frame=phase.frames[i]
                            TIME_START("all")

                            with torch.set_grad_enabled(False):

                                # print("frame rgb path is ", frame.frame.rgb_path)

                                frame.load_images()
                                #get a subsampled frame if necessary
                                frame_full_res=frame
                                # print("frame_full_res has size ", frame_full_res.height, " ", frame_full_res.width)
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
                                    frames_close=get_close_frames(loader_train, frame, frames_to_consider_for_neighbourhood, 8, discard_same_idx) #the neighbour are only from the training set
                                    weights= frame_weights_computer(frame, frames_close)
                                else:
                                    frames_close, weights=get_close_frames_barycentric(frame, frames_to_consider_for_neighbourhood, discard_same_idx, dataset_params.sphere_center, dataset_params.sphere_radius, dataset_params.triangulation_type)
                                    weights= torch.from_numpy(weights.copy()).to("cuda").float() 
                                frames_close_full_res = frames_close
                                # print("weights",weights)

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


                                #prepare rgb data and rest of things
                                rgb_gt_fullres, rgb_gt, ray_dirs, rgb_close_batch, ray_dirs_close_batch, ray_diff = prepare_data(frame_full_res, frames_close_full_res, frame, frames_close)

                                rgb_pred, depth_pred, point3d, new_loss, depth_for_each_res, confidence_map=model(dataset_params, frame, ray_dirs, rgb_close_batch, rgb_close_fullres_batch, ray_dirs_close_batch, ray_diff, frame_full_res, frames_close, weights, novel=True)


                                if first_time:
                                    first_time=False
                                    #TODO load checkpoint
                                    # now that all the parameters are created we can fill them with a model from a file
                                    # model.load_state_dict(torch.load( "/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/saved_models/lego/model_e_31_score_25.798268527984618.pt" ))
                                    model.load_state_dict(torch.load( "/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/saved_models/dtu/model_e_38_score_0.pt" ))
                                    #rerun 
                                    rgb_pred, depth_pred, point3d, new_loss, depth_for_each_res, confidence_map=model(dataset_params, frame, ray_dirs, rgb_close_batch, rgb_close_fullres_batch, ray_dirs_close_batch, ray_diff, frame_full_res, frames_close, weights, novel=True)



                                # path="/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/recordings/test4"
                                path="/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/recordings/test4_dtu_eval"

                                #compute psnr, ssim an lpips
                                psnr = piq.psnr(rgb_gt_fullres, torch.clamp(rgb_pred,0.0,1.0), data_range=1.0 )
                                ssim = piq.ssim(rgb_gt_fullres, torch.clamp(rgb_pred,0.0,1.0) )
                                lpips: torch.Tensor = piq.LPIPS()(  rgb_gt_fullres, torch.clamp(rgb_pred,0.0,1.0)  )
                                print("psnr", psnr.item())
                                # print("ssim", ssim.item())
                                # print("lpips", lpips.item())
                                psnr_acum+=psnr.item()
                                ssim_acum+=ssim.item()
                                lpips_acum+=lpips.item()







                                #use the mask
                                mask_list=[]
                                use_mask=True #whne we use the mask we set the masked ares to white
                                if use_mask:
                                    mask_list.append( mat2tensor(frame.frame.mask, False).cuda().repeat(1,1,1,1) )
                                    mask_list.append( mat2tensor(frame.subsampled_frames[0].frame.mask, False).cuda().repeat(1,1,1,1) )
                                    mask_list.append( mat2tensor(frame.subsampled_frames[1].frame.mask, False).cuda().repeat(1,1,1,1) )
                                    mask_list.append( mat2tensor(frame.subsampled_frames[2].frame.mask, False).cuda().repeat(1,1,1,1) )






                                #compute all visualizable thing
                                #RGB
                                rgb_mat =  tensor2mat(rgb_pred)
                                #GT
                                gt_mat =  tensor2mat(rgb_gt_fullres)
                                #depth
                                depth_mats=[]
                                depth_mats_colored=[]
                                for i in range(len(depth_for_each_res)):
                                    depth_for_lvl=depth_for_each_res[i].repeat(1,3,1,1) #make it have 3 channels

                                    #range it
                                    if  isinstance(loader_test, DataLoaderNerf):
                                        depth_for_lvl=map_range(depth_for_lvl, 0.35, 0.7, 1.0, 0.0) ######Is dataset specific FOR NERF synthetic
                                    if  isinstance(loader_test, DataLoaderLLFF):
                                        depth_for_lvl=map_range(depth_for_lvl, 0.0, 1.0, 1.0, 0.0) ######Is dataset specific 
                                    if  isinstance(loader_test, DataLoaderDTU):
                                        depth_for_lvl=map_range(depth_for_lvl, 0.15, 1.0, 1.0, 0.0) ######Is dataset specific FOR NERF synthetic

                                    if use_mask: #concat a alpha channel
                                        # depth_for_lvl[1-mask_list[i]]=[54/255, 15/255, 107/255]
                                        depth_for_lvl=torch.cat([depth_for_lvl, mask_list[i] ], 1)
                                    depth_mat=tensor2mat(depth_for_lvl)
                                    depth_mats.append(depth_mat)
                                    #color it
                                    depth_mat_colored= color_mngr.mat2color(depth_mat, "magma")
                                    # if use_mask: 
                                        # depth_mat_colored_tensor=mat2tensor(depth_mat_colored,False)
                                        # depth_mat_colored_tensor[1-mask_list[i]]=1.0
                                        # depth_mat_colored=tensor2mat(depth_mat_colored_tensor)
                                    depth_mats_colored.append(depth_mat_colored)
                                #normal 
                                points3D_img=point3d
                                normal_img=compute_normal(points3D_img)
                                normal_vis=(normal_img+1.0)*0.5
                                if use_mask:
                                    # normal_vis[1-mask_list[0]]=1.0
                                    normal_vis=torch.cat([normal_vis, mask_list[0] ], 1)
                                normal_mat=tensor2mat(normal_vis)
                                #confidence
                                if confidence_map!=None:
                                    confidence_mat =  tensor2mat(confidence_map)



                                #write to disk
                                if(not os.path.exists(path)):
                                    print("path does not exist, are you sure you are on the correct machine", path)
                                    exit(1)
                                path=os.path.join(path,str(scene_idx))
                                rgb_path=os.path.join(path,"rgb")
                                gt_path=os.path.join(path,"gt")
                                depth_paths=[]
                                depth_colored_paths=[]
                                for i in range(len(depth_for_each_res)):
                                    depth_paths.append( os.path.join(path,"depth/depth_"+str(i)) )
                                    depth_colored_paths.append( os.path.join(path,"depth/depth_colored_"+str(i)) )
                                normal_path=os.path.join(path,"normal")
                                confidence_path=os.path.join(path,"confidence")
                                #make the paths
                                os.makedirs(rgb_path, exist_ok=True)
                                os.makedirs(gt_path, exist_ok=True)
                                for i in range(len(depth_for_each_res)):
                                    os.makedirs(depth_paths[i], exist_ok=True)
                                    os.makedirs(depth_colored_paths[i], exist_ok=True)
                                os.makedirs(normal_path, exist_ok=True)
                                os.makedirs(confidence_path, exist_ok=True)
                                #write
                                rgb_mat.to_cv8u().to_file(rgb_path+"/"+str(img_nr)+".png")
                                gt_mat.to_cv8u().to_file(gt_path+"/"+str(img_nr)+".png")
                                for i in range(len(depth_mats)):
                                    depth_mats[i].to_cv8u().to_file(depth_paths[i]+"/"+str(img_nr)+".png")
                                for i in range(len(depth_mats_colored)):
                                    depth_mats_colored[i].to_cv8u().to_file(depth_colored_paths[i]+"/"+str(img_nr)+".png")
                                normal_mat.to_cv8u().to_file(normal_path+"/"+str(img_nr)+".png")
                                if confidence_map!=None:
                                    confidence_mat.to_cv8u().to_file(confidence_path+"/"+str(img_nr)+".png")


                                img_nr+=1



                                #SHOW
                                if(train_params.with_viewer()):
                                    #RGB
                                    Gui.show(rgb_mat, "rgb_mat")
                                    #depth
                                    for i in range(len(depth_mats)):
                                        Gui.show(depth_mats[i], "depth_mat_"+str(i))
                                    for i in range(len(depth_mats_colored)):
                                        Gui.show(depth_mats_colored[i], "depth_mat_colored_"+str(i))
                                    #normal
                                    Gui.show(normal_mat, "normal_mat")
                                    #confidence 
                                    if confidence_map!=None:
                                        Gui.show(confidence_mat, "confidence_mat")
                                    



                                    view.update()


                        TIME_START("load")
                        if isinstance(loader_train, DataLoaderShapeNetImg) or isinstance(loader_train, DataLoaderSRN) or isinstance(loader_train, DataLoaderDTU):
                            TIME_START("justload")

                            img_nr=0
                        
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


    #print avg values
    psnr_avg=psnr_acum/img_nr
    ssim_avg=ssim_acum/img_nr
    lpips_avg=lpips_acum/img_nr
    print("EVALUATION METRICs, averaged over all images")
    print("psnr_avg", psnr_avg)
    print("ssim_avg", ssim_avg)
    print("lpips_avg", lpips_avg)

                















































    ################################################################## OLD CODE START HERE 

    # while True:
    #     with torch.set_grad_enabled(False):

        
    #         #get the model matrix of the view and set it to the frame
    #         # cam_tf_world_cam= view.m_camera.model_matrix_affine()
    #         if use_spiral:
    #             tf_world_cam_hwf = poses_on_spiral[ img_nr% len(poses_on_spiral) ]
    #             # print("pose_on spiral ", tf_world_cam_hwf)
    #             tf_world_cam = tf_world_cam_hwf[:, 0:4]
    #             hwf=tf_world_cam_hwf[:, 4:5]
    #             # print("hwf", hwf)
    #             # print("tf_world_cam", tf_world_cam)
    #             row = np.array([ 0,0,0,1  ])  #the last row in the matrix
    #             row = row.reshape((1, 4))
    #             tf_world_cam = np.concatenate([ tf_world_cam, row ], 0)
    #             # print("tf_world_cam with lasr tow", tf_world_cam)
    #             tf_cam_world = np.linalg.inv(tf_world_cam)
    #             # print("tf_cam_world with lasr tow", tf_cam_world)
    #             # tf_cam_world[:,0:1] = -tf_cam_world[:,0:1]

    #             tf_cam_world_eigen= Affine3f()
    #             tf_cam_world_eigen.from_matrix(tf_cam_world)
    #             tf_cam_world_eigen.flip_z()
    #             frame.frame.tf_cam_world= tf_cam_world_eigen
    #             #restrict also the focal a bit
    #             # new_K=K.copy()
    #             # new_K=new_K*0.25
    #             # new_K[0,0]=new_K[0,0]*1.2
    #             # new_K[1,1]=new_K[1,1]*1.2
    #             # new_K[2,2]=1.0
    #             # frame.frame.K=new_K
    #             # print("normal K is ", frame.frame.K)
    #             # print("new K is ", new_K)
    #         else:
    #             view.m_camera=cam_for_pred
    #             cam_tf_world_cam= cam_for_pred.model_matrix_affine()
    #             frame.frame.tf_cam_world=cam_tf_world_cam.inverse()
    #         frame=FramePY(frame.frame, create_subsamples=True)
    #         #recalculate the dirs because the frame changes in space 
    #         ray_dirs_mesh=frame.frame.pixels2dirs_mesh()
    #         frame.ray_dirs=ray_dirs_mesh.V.copy() #Nx3



    #         #show the current frame 
    #         frustum_mesh=frame.frame.create_frustum_mesh(dataset_params.frustum_size)
    #         frustum_mesh.m_vis.m_line_color=[0.0, 1.0, 1.0] #green
    #         frustum_mesh.m_force_vis_update=True
    #         # print("frame.frameK", frame.frame.K)
    #         Scene.show(frustum_mesh, "frustum_cur" ) 



    #         discard_same_idx= False
    #         if isinstance(loader_train, DataLoaderShapeNetImg) or isinstance(loader_train, DataLoaderSRN) or isinstance(loader_train, DataLoaderDTU):
    #             discard_same_idx=True
    #         frames_to_consider_for_neighbourhood=frames_train
    #         if isinstance(loader_train, DataLoaderShapeNetImg) or isinstance(loader_train, DataLoaderSRN) or isinstance(loader_train, DataLoaderDTU): #if it's these loader we cannot take the train frames for testing because they dont correspond to the same object
    #             frames_to_consider_for_neighbourhood=frames_test
    #         do_close_computation_with_delaunay=True
    #         if not do_close_computation_with_delaunay:
    #             frames_close=get_close_frames(loader_train, frame, frames_to_consider_for_neighbourhood, 7, discard_same_idx) #the neighbour are only from the training set
    #             weights= frame_weights_computer(frame, frames_close)
    #         else:
    #             triangulation_type="sphere"
    #             if  isinstance(loader_train, DataLoaderLLFF):
    #                 triangulation_type="plane"
    #             frames_close, weights=get_close_frames_barycentric(frame, frames_to_consider_for_neighbourhood, discard_same_idx, sphere_center, sphere_radius, triangulation_type)
    #             weights= torch.from_numpy(weights.copy()).to("cuda").float() 
    #         frames_close_full_res = frames_close


    #         #double check why are the tf_matrices weird


    #         #load frames
    #         frame.load_images()

    #         if factor_subsample_depth_pred!=0 and first_time:
    #             frame=frame.subsampled_frames[factor_subsample_depth_pred-1]
    #         if first_time:
    #             frame_full_res=frame
    #         # print("frame_full_res", frame_full_res.height, " ", frame_full_res.width)

    #         # print("K after subsample is ", frame.frame.K)

    #         for i in range(len(frames_close)):
    #             frames_close[i].load_images()

    #         rgb_close_fulres_batch_list=[]
    #         for frame_close in frames_close:
    #             rgb_close_frame=mat2tensor(frame_close.frame.rgb_32f, False).to("cuda")
    #             rgb_close_fulres_batch_list.append(rgb_close_frame)
    #         rgb_close_fullres_batch=torch.cat(rgb_close_fulres_batch_list,0)

    #         #the frames close may need to be subsampled
    #         if factor_subsample_close_frames!=0:
    #             frames_close_subsampled=[]
    #             for frame_close in frames_close:
    #                 frame_subsampled= frame_close.subsampled_frames[factor_subsample_close_frames-1]
    #                 frames_close_subsampled.append(frame_subsampled)
    #             frames_close= frames_close_subsampled


    #         # rgb_gt_fullres, rgb_gt, ray_dirs, rgb_close_batch, ray_dirs_close_batch = prepare_data(frame_full_res, frame, frames_close)
    #         #recalcualte the directions because the frame moved
    #         frame_full_res.frame.tf_cam_world= tf_cam_world_eigen
    #         ray_dirs_mesh=frame_full_res.frame.pixels2dirs_mesh()
    #         frame_full_res.ray_dirs=ray_dirs_mesh.V.copy() #Nx3
    #         rgb_gt_fullres, rgb_gt, ray_dirs, rgb_close_batch, ray_dirs_close_batch, ray_diff = prepare_data(frame_full_res, frames_close_full_res, frame, frames_close)


    #         rgb_pred, depth_pred, point3d, new_loss, depth_for_each_res, confidence_map=model(dataset_params, frame, ray_dirs, rgb_close_batch, rgb_close_fullres_batch, ray_dirs_close_batch, ray_diff, frame_full_res, frames_close, weights, novel=True)
    #         # print("depth_pred", depth_pred.mean())

    #         if first_time:
    #             first_time=False
    #             #TODO load checkpoint
    #             # now that all the parameters are created we can fill them with a model from a file
    #             model.load_state_dict(torch.load( "/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/saved_models/leaves/model_e_100_score_21.298389434814453.pt" ))


    #         #normal
    #         points3D_img=point3d
    #         normal_img=compute_normal(points3D_img)
    #         normal_vis=(normal_img+1.0)*0.5
    #         #masks
    #         # rgb_refined_downsized= torch.nn.functional.interpolate(rgb_refined, size=(rgb_pred.shape[2], rgb_pred.shape[3]), mode='bilinear')
    #         # rgb_pred_channels_last=rgb_refined_downsized.permute(0,2,3,1) # from n,c,h,w to N,H,W,C
    #         # rgb_pred_zeros=rgb_pred_channels_last.view(-1,3).norm(dim=1, keepdim=True)
    #         # rgb_pred_zeros_mask= rgb_pred_zeros<0.01
    #         # rgb_pred_zeros_mask=rgb_pred_zeros_mask.repeat(1,3) #repeat 3 times for rgb
    #         # rgb_pred_zeros_mask_img=rgb_pred_zeros_mask.view(1,frame.height,frame.width,3)
    #         # rgb_pred_zeros_mask_img=rgb_pred_zeros_mask_img.permute(0,3,1,2)
    #         if(train_params.with_viewer()):
    #             depth_vis=depth_pred.view(1,1,frame.height,frame.width)
    #             depth_vis=map_range(depth_vis, neural_mvs_gui.m_min_depth, neural_mvs_gui.m_max_depth, 0.0, 1.0) #for the colamp fine leaves
    #             depth_vis=depth_vis.repeat(1,3,1,1)
    #             if neural_mvs_gui.m_show_rgb:
    #                     pred_mat=tensor2mat(rgb_pred)
    #                 # depth_vis[rgb_pred_zeros_mask_img]=1.0 #MASK the point in the background
    #             if neural_mvs_gui.m_show_depth:
    #                 pred_mat=tensor2mat(depth_vis)
    #             if neural_mvs_gui.m_show_normal:
    #                 # print("normal_vis has min max", normal_vis.min(), normal_vis.max())
    #                 pred_mat=tensor2mat(normal_vis)
    #             Gui.show(pred_mat,"Depth")
    #         # Gui.show(tensor2mat(rgb_refined),"RGB")
    #         # #show 3d points 
    #         # normal=normal_img.permute(0,2,3,1) # from n,c,h,w to N,H,W,C
    #         # normal=normal.view(-1,3)
    #         # rgb_refined_downsized= torch.nn.functional.interpolate(rgb_refined, size=(rgb_pred.shape[2], rgb_pred.shape[3]), mode='bilinear')
    #         # rgb_pred_channels_last=rgb_refined_downsized.permute(0,2,3,1) # from n,c,h,w to N,H,W,C
    #         # rgb_pred_zeros=rgb_pred_channels_last.view(-1,3).norm(dim=1, keepdim=True)
    #         # rgb_pred_zeros_mask= rgb_pred_zeros<0.05
    #         # rgb_pred_zeros_mask=rgb_pred_zeros_mask.repeat(1,3) #repeat 3 times for rgb
    #         # points3D[rgb_pred_zeros_mask]=0.0 #MASK the point in the background
    #         # points3d_mesh=show_3D_points(points3D, color=rgb_refined_downsized)
    #         # points3d_mesh.NV= normal.detach().cpu().numpy()
    #         # Scene.show(points3d_mesh, "points3d_mesh")
    #         # for i in range(len(frames_close)):
    #         #     Gui.show(frames_close[i].frame.rgb_32f,"close_"+str(i) )


            
    #         depth_vis=depth_pred.view(1,1,frame.height,frame.width)
    #         depth_vis=map_range(depth_vis, 0.0, 1.0, 0.0, 1.0) #for the colamp fine leaves
    #         depth_vis=depth_vis.repeat(1,3,1,1)
    #         #tensor2mat(rgb_pred).to_cv8u().to_file("/home/user/rosu/c_ws/src/phenorob/neural_mvs/recordings/leaves_eval/rgb/rgb"+str(img_nr)+".png")
    #         #tensor2mat(depth_vis).to_cv8u().to_file("/home/user/rosu/c_ws/src/phenorob/neural_mvs/recordings/leaves_eval/depth/depth"+str(img_nr)+".png")
    #         # tensor2mat(rgb_pred).to_cv8u().to_file("/home/user/rosu/c_ws/src/phenorob/neural_mvs/recordings/flower_eval/rgb/rgb"+str(img_nr)+".png")
    #         # tensor2mat(depth_vis).to_cv8u().to_file("/home/user/rosu/c_ws/src/phenorob/neural_mvs/recordings/flower_eval/depth/depth"+str(img_nr)+".png")
    #         tensor2mat(rgb_pred).to_cv8u().to_file("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/recordings/leaves_eval/rgb/rgb"+str(img_nr)+".png")
    #         tensor2mat(depth_vis).to_cv8u().to_file("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/recordings/leaves_eval/depth/depth"+str(img_nr)+".png")
    #         #also each res 
    #         for i  in  range(len(depth_for_each_res)):
    #             depth_LR=depth_for_each_res[i]
    #             depth_vis_LR=map_range(depth_LR, 0.0, 1.0, 0.0, 1.0) #for the colamp fine leaves
    #             depth_vis_LR=depth_vis_LR.repeat(1,3,1,1)
    #             tensor2mat(depth_vis_LR).to_cv8u().to_file("/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/recordings/leaves_eval/depth_"+str(i)+"/depth"+str(img_nr)+".png")


    #         img_nr+=1



    #     # view.update()
    #     # view.m_camera=view.m_default_camera
    #     # if neural_mvs_gui.m_control_secondary_cam: 
    #     #     #if we control the secondary cam we set the secondary cam in the viewer and then do an update which will do a glfwpoolevents so that moves the camera
    #     #     view.m_camera=cam_for_pred
    #     #     view.m_swap_buffers=False #we don't need to swap buffers as we only needed to do the update because we updated the movement of this camera
    #     #     view.update()
    #     # #render the real 3D scene
    #     # view.m_camera=view.m_default_camera
    #     # view.m_swap_buffers=True
    #     if(train_params.with_viewer()):
    #         view.update()


#################################################################################### old code finished here  


         

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
