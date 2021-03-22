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
from optimizers.over9000.apollo import *
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

def get_close_frames(loader, frame_py, all_frames_py_list, nr_frames_close, discard_same_idx):
    frames_close=loader.get_close_frames(frame_py.frame, nr_frames_close, discard_same_idx)
    # print("frames_close in py is ", len(frames_close))
    # print("all_frames_py_list", len(all_frames_py_list))
    #fromt his frame close get the frames from frame_py_list with the same indexes
    frames_selected=[]
    for frame in frames_close:
        frame_idx=frame.frame_idx
        # print("looking for a frame with frame_idx", frame_idx)
        #find in all_frames_py_list the one with this frame idx
        for frame_py in all_frames_py_list:
            # print("cur frame has frmaeidx", frame_py.frame_idx)
            if frame_py.frame_idx==frame_idx:
                frames_selected.append(frame_py)

    return frames_selected


def get_close_frames_barycentric(frame_py, all_frames_py_list, discard_same_idx, sphere_center, sphere_radius):

    if discard_same_idx:
        frame_centers, frame_idxs = frames_to_points(all_frames_py_list, discard_frame_with_idx=frame_py.frame_idx)
    else:
        frame_centers, frame_idxs = frames_to_points(all_frames_py_list )

    triangulated_mesh=SFM.compute_triangulation_stegreographic( frame_centers, sphere_center, sphere_radius )

    face, weights= SFM.compute_closest_triangle( frame_py.frame.pos_in_world(), triangulated_mesh )

    #from the face get the vertices that we triangulated
    # print("frame idx ", frame_idx_0, frame_idx_1, frame_idx_2 )
    frame_idx_0= frame_idxs[face[0]]
    frame_idx_1= frame_idxs[face[1]]
    frame_idx_2= frame_idxs[face[2]]

    selected_frames=[]
    frame_0=None
    frame_1=None
    frame_2=None
    #We have to add them in the same order, so first frame0 andthen 1 and then 2
    for frame in all_frames_py_list:
        if frame.frame_idx==frame_idx_0:
            frame_0=frame
        if frame.frame_idx==frame_idx_1:
            frame_1=frame
        if frame.frame_idx==frame_idx_2:
            frame_2=frame
    selected_frames.append(frame_0)
    selected_frames.append(frame_1)
    selected_frames.append(frame_2)
    

    return selected_frames, weights


class FramePY():
    def __init__(self, frame, create_subsamples=False):
        #get mask 
        self.frame=frame
        # #We do NOT store the tensors on the gpu and rather load them whenever is endessary. This is because we can have many frames and can easily run out of memory
        # if not frame.mask.empty():
        #     mask_tensor=mat2tensor(frame.mask, False).to("cuda").repeat(1,3,1,1)
        #     self.frame.mask=frame.mask
        # else:
        #     mask_tensor= torch.ones((1,1,frame.height,frame.width), device=torch.device("cuda") )
        #     self.frame.mask=tensor2mat(mask_tensor)
        # #get rgb with mask applied 
        # rgb_tensor=mat2tensor(frame.rgb_32f, False).to("cuda")
        # rgb_tensor=rgb_tensor*mask_tensor

        if frame.mask.empty():
            mask_tensor= torch.ones((1,1,frame.height,frame.width))
            self.frame.mask=tensor2mat(mask_tensor)

        # self.frame.rgb_32f=tensor2mat(rgb_tensor)
        #get tf and K
        self.tf_cam_world=frame.tf_cam_world
        self.K=frame.K
        self.R_tensor=torch.from_numpy( frame.tf_cam_world.linear() ).to("cuda")
        self.t_tensor=torch.from_numpy( frame.tf_cam_world.translation() ).to("cuda")
        self.K_tensor = torch.from_numpy( frame.K ).to("cuda")
        #weight and hegiht
        # self.height=self.rgb_tensor.shape[2]
        # self.width=self.rgb_tensor.shape[3]
        self.height=frame.height
        self.width=frame.width
        #misc
        self.frame_idx=frame.frame_idx
        # self.loader=loader
        #Ray direction in world coordinates
        ray_dirs_mesh=frame.pixels2dirs_mesh()
        # self.ray_dirs=torch.from_numpy(ray_dirs_mesh.V.copy()).to("cuda").float() #Nx3
        self.ray_dirs=ray_dirs_mesh.V.copy() #Nx3
        #lookdir and cam center
        self.camera_center=torch.from_numpy( frame.pos_in_world() ).to("cuda")
        self.camera_center=self.camera_center.view(1,3)
        self.look_dir=torch.from_numpy( frame.look_dir() ).to("cuda")
        self.look_dir=self.look_dir.view(1,3)
        #create tensor to store the bound in z near and zfar for every pixel of this image
        # self.znear_zfar = torch.nn.Parameter(  torch.ones([1,2,self.height,self.width], dtype=torch.float32, device=torch.device("cuda"))  )
        # with torch.no_grad():
        #     self.znear_zfar[:,0,:,:]=znear
        #     self.znear_zfar[:,1,:,:]=zfar
        # self.znear_zfar.requires_grad=True
        # self.cloud=frame.depth2world_xyz_mesh()
        # self.cloud=frame.assign_color(self.cloud)
        # self.cloud.remove_vertices_at_zero()

        #make a list of subsampled frames
        if create_subsamples:
            self.subsampled_frames=[]
            for i in range(3):
                if i==0:
                    frame_subsampled=frame.subsample(2)
                else:
                    frame_subsampled=frame_subsampled.subsample(2)
                self.subsampled_frames.append(FramePY(frame_subsampled, create_subsamples=False))

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

    # experiment_name="s13_rg_ac_0.003"
    experiment_name="s2_aggrmeanstd"

    use_ray_compression=False





    cb_list = []
    if(train_params.with_visdom()):
        cb_list.append(VisdomCallback(experiment_name))
    if(train_params.with_viewer()):
        cb_list.append(ViewerCallback())
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)

    #create loaders
    loader_train=DataLoaderNerf(config_path)
    loader_test=DataLoaderNerf(config_path)
    # loader_train=DataLoaderColmap(config_path)
    # loader_test=DataLoaderColmap(config_path)
    loader_train.set_mode_train()
    loader_test.set_mode_test()
    loader_train.start()
    loader_test.start()

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
    concat_coord=ConcatCoord() 
    smooth = InverseDepthSmoothnessLoss()
    ssim_l1_criterion = MS_SSIM_L1_LOSS()

    show_every=1



    #get all the frames train in am array, becuase it's faster to have everything already on the gpu
    frames_train=[]
    frames_test=[]
    for i in range(loader_train.nr_samples()):
        frame=loader_train.get_frame_at_idx(i)
        frames_train.append(FramePY(frame))
    for i in range(loader_test.nr_samples()):
        frame=loader_test.get_frame_at_idx(i)
        frames_test.append(FramePY(frame))
    phases[0].frames=frames_train 
    phases[1].frames=frames_test
    #Show only the visdom for the testin
    phases[0].show_visdom=False
    phases[1].show_visdom=True

    #show all the train and test frames 
    # for i in range(loader_train.nr_samples()):
    #     frame=loader_train.get_frame_at_idx(i)
    #     frustum_mesh=frame.create_frustum_mesh(0.02)
    #     frustum_mesh.m_vis.m_line_width=1
    #     Scene.show(frustum_mesh, "frustum_train_"+str(frame.frame_idx) )
    # for i in range(loader_test.nr_samples()):
    #     frame=loader_test.get_frame_at_idx(i)
    #     frustum_mesh=frame.create_frustum_mesh(0.02)
    #     frustum_mesh.m_vis.m_line_width=1
    #     frustum_mesh.m_vis.m_line_color=[0.0, 0.0, 1.0] #blue
    #     Scene.show(frustum_mesh, "frustum_test_"+str(frame.frame_idx) )

 
    
    # #compute 3D 
    # sfm=SFM.create()
    # selected_frame_idx=np.arange(30) #For colmap
    # # selected_frame_idx=[10]
    # frames_query_selected=[]
    # frames_target_selected=[]
    # frames_all_selected=[]
    # meshes_for_query_frames=[]
    # for i in range(loader_train.nr_samples()):
    # # for i in range(1 ):
    #     # frame_0=loader_train.get_frame_at_idx(i+3) 
    #     if i in selected_frame_idx:
    #         frame_query=loader_train.get_frame_at_idx(i) 
    #         # frame_target=loader_train.get_closest_frame(frame_query)
    #         frame_target=loader_train.get_close_frames(frame_query, 1, True)[0]
    #         frames_query_selected.append(frame_query)
    #         frames_target_selected.append(frame_target)
    #         frames_all_selected.append(frame_query)
    #         frames_all_selected.append(frame_target)
    #         mesh_sparse, keypoints_distances_eigen, keypoints_indices_eigen=sfm.compute_3D_keypoints_from_frames(frame_query, frame_target  )
    #         meshes_for_query_frames.append(mesh_sparse)
         


    # #fuse all the meshes into one
    # mesh_full=Mesh()
    # for mesh in meshes_for_query_frames:
    #     mesh_full.add(mesh)
    # mesh_full.m_vis.m_show_points=True
    # mesh_full.m_vis.set_color_pervertcolor()
    # Scene.show(mesh_full, "mesh_full" )
    # print("scene scale is ", Scene.get_scale())


    # #get for each frame_query the distances of the keypoints
    # frame_idx2keypoint_data={}
    # for i in range(loader_train.nr_samples()):
    #     frame_query=loader_train.get_frame_at_idx(i) 
    #     frame_target=loader_train.get_closest_frame(frame_query)
    #     mesh_sparse, keypoints_distances_eigen, keypoints_indices_eigen=sfm.compute_3D_keypoints_from_frames(frame_query, frame_target  )
    #     keypoints_distances=torch.from_numpy(keypoints_distances_eigen.copy()).to("cuda")
    #     keypoints_indices=torch.from_numpy(keypoints_indices_eigen.copy()).to("cuda")
    #     keypoints_3d =torch.from_numpy(mesh_sparse.V.copy()).float().to("cuda")
    #     keypoint_data=[keypoints_distances, keypoints_indices, keypoints_3d]
    #     frame_idx2keypoint_data[frame_query.frame_idx] = keypoint_data


    #get the triangulation of the frames 
    frame_centers, frame_idxs = frames_to_points(frames_train)
    sphere_center, sphere_radius=SFM.fit_sphere(frame_centers)
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

    while True:

        for phase in phases:
            cb.epoch_started(phase=phase)
            cb.phase_started(phase=phase)
            model.train(phase.grad)
            is_training=phase.grad

            # if loader_test.finished_reading_scene(): #For shapenet
            if True: #for nerf

                # if phase.loader.has_data() and loader_test.has_data():
                # if phase.loader.has_data(): #for nerf
                # if True: #Shapenet IMg always had ata at this point 
                # for frame_idx, frame in enumerate(frames_all_selected):
                for i in range(phase.loader.nr_samples()):
                    if phase.grad:
                        frame=random.choice(phase.frames)
                    else:
                        frame=phase.frames[i]
                    TIME_START("all")

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
                        #prepare rgb data and rest of things
                        rgb_gt=mat2tensor(frame.frame.rgb_32f, False).to("cuda")
                        mask_tensor=mat2tensor(frame.frame.mask, False).to("cuda")
                        ray_dirs=torch.from_numpy(frame.ray_dirs).to("cuda").float()
                        rgb_close_batch_list=[]
                        for frame_close in frames_close:
                            rgb_close_frame=mat2tensor(frame_close.frame.rgb_32f, False).to("cuda")
                            rgb_close_batch_list.append(rgb_close_frame)
                        rgb_close_batch=torch.cat(rgb_close_batch_list,0)
                    # print("frame is height widht", frame.height, " ", frame.width) #colmap has 189x252

                    # print( torch.cuda.memory_summary() )




                    #VIEW gt 
                    # if phase.iter_nr%show_every==0:
                        # rgb_mat=tensor2mat(rgb_gt)
                        # Gui.show(rgb_mat,"rgb_gt")

                    #view current active frame
                    frustum_mesh=frame.frame.create_frustum_mesh(0.02)
                    frustum_mesh.m_vis.m_line_width=3
                    frustum_mesh.m_vis.m_line_color=[1.0, 0.0, 1.0] #purple
                    Scene.show(frustum_mesh, "frustum_activ" )


                    #DEBUG run only one iter of the training 
                    # if is_training:
                        # break






                    #forward attempt 2 using a network with differetnaible ray march
                    with torch.set_grad_enabled(is_training):


                        TIME_START("forward")
                        # print( torch.cuda.memory_summary() )
                        # with profiler.profile(profile_memory=True, record_shapes=True, use_cuda=True) as prof:
                        rgb_pred, rgb_refined, depth_pred, mask_pred, signed_distances_for_marchlvl, std=model(frame, ray_dirs, rgb_close_batch, depth_min, depth_max, frames_close, weights, novel=not phase.grad)
                        TIME_END("forward")
                        # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

                        # print("rgb_close_batch",rgb_close_batch.shape)
                        # summary(model, [ (1, 1), (100*100,3), (3, 3, 100, 100), (1,1), (1,1),(1,1), weights.shape, (1,1)   ]  )
                        # model.summary()
                        # exit(1)

                        #mask the prediction so we copy the values from the rgb_gt into the parts of rgb_pred so that the loss for those pixels is zero
                        #since the copy cannot be done with a mask because that is just 0 and 1 which mean it cannot propagate gradient, we do it with a blend
                        # rgb_pred=mask_pred*rgb_pred + (1-mask_pred)*rgb_gt
                     
                        #loss
                        loss=0
                        rgb_loss=(( rgb_gt-rgb_pred)**2).mean()
                        rgb_loss_l1=(( rgb_gt-rgb_pred).abs()).mean()
                        rgb_loss_ssim_l1 = ssim_l1_criterion(rgb_gt, rgb_pred)
                        # loss+=rgb_loss_ssim_l1
                        loss+=rgb_loss
                        # loss+=rgb_loss_l1
                        #loss on the rgb_refiend
                        # rgb_refined_loss=(( rgb_gt-rgb_pred)**2).mean()
                        # rgb_refined_loss_ssim_l1 = ssim_l1_criterion(rgb_gt, rgb_refined)
                        # loss+=rgb_refined_loss_ssim_l1*0.5

                        #make the mask to be mostly white
                        # loss_mask=((1.0-mask_pred)**2).mean()
                        # loss+=loss_mask*100
            

                        # #loss on depth 
                        # if is_training: #when testing we don;t compute the loss towards the keypoint depth because we have no keypoints for those frames
                        #     keypoint_data=frame_idx2keypoint_data[frame.frame_idx]
                        #     keypoint_distances=keypoint_data[0]
                        #     keypoint_instances=keypoint_data[1]
                        #     keypoints_3d=keypoint_data[2]
                        #     depth_pred=depth_pred.view(-1,1)
                        #     depth_pred_keypoints= torch.index_select(depth_pred, 0, keypoint_instances.long())
                        #     mask_keypoints= torch.index_select(mask_tensor.view(-1,1), 0, keypoint_instances.long()) #the parts that are in the background need no loss on the depth
                        #     loss_depth= (( keypoint_distances- depth_pred_keypoints)**2)
                        #     loss_depth=loss_depth*mask_keypoints
                        #     loss_depth= loss_depth.mean()
                        #     if phase.iter_nr<1000:
                        #         loss+=loss_depth*100

                            #smoothness loss
                            # depth_pred=depth_pred.view(1, frame.height, frame.width, 1).permute(0,3,1,2) #from N,H,W,C to N,C,H,W
                            # smooth_loss = smooth(depth_pred*mask_tensor, rgb_gt)
                            # loss+=smooth_loss* min(phase.iter_nr*0.1,1000)
                            # print("smooth_loss",smooth_loss)

                        #loss on the signed distance, making it be zero as soon as possible for all levels of the mark
                        # if is_training: 
                        #     for i in range(len(signed_distances_for_marchlvl)):
                        #         signed_dist=signed_distances_for_marchlvl[i]
                        #         # loss+= (signed_dist**2).mean()*100*i #first distance is allowed to move, and the more we move the more we expect it to stop moving
                        #         weight=2**i
                        #         if weight>1000:
                        #             weight=1000
                        #         loss+= (signed_dist**2).mean()*weight*0.1 #first distance is allowed to move, and the more we move the more we expect it to stop moving

                        #loss on the signed distance, making it be zero but only for the last level
                        # if is_training: 
                        #     signed_dist=signed_distances_for_marchlvl[ -1 ]
                        #     signed_dist=signed_dist.view(1,1,frame.height, frame.width)
                        #     signed_dist=signed_dist*frame.mask_tensor
                        #     loss+= (signed_dist**2).mean()*100 #first distance is allowed to move, and the more we move the more we expect it to stop moving

                        #loss on the stdm at the finaly of the ray tracing the std of the features shoulb be zeo
                        # if is_training: 
                            # loss+=(std**2).mean()
                            # print("st loss ", (std**2).mean())





                 





                        # debug the keypoints 
                        # ray_dirs_mesh=frame.pixels2dirs_mesh()
                        # ray_dirs=torch.from_numpy(ray_dirs_mesh.V.copy()).to("cuda").float() #Nx3
                        # camera_center=torch.from_numpy( frame.pos_in_world() ).to("cuda")
                        # camera_center=camera_center.view(1,3)
                        # rays_dirs_keypoints = torch.index_select(ray_dirs, 0, keypoint_instances.long())
                        # keypoints3d_reprojected = camera_center + keypoint_distances.view(-1,1)*rays_dirs_keypoints
                        # show_3D_points(keypoints3d_reprojected, " keypoints_debug_"+str(frame.frame_idx))
                        # show_3D_points(keypoints_3d, " keypoints_CORR_debug_"+str(frame.frame_idx))
                        # error= (keypoints_3d- keypoints3d_reprojected).norm(dim=1)
                        # print("-----pytorch stuf-------------")
                        # print("keypoints_3d ", keypoints_3d)
                        # print("keypoints3d_reprojected ", keypoints3d_reprojected[0,:])
                        # print("rays_dirs_keypoints ", rays_dirs_keypoints[0,:])
                        # print("keypoint_distances", keypoint_distances[0])
                        # print("error is ", error[0])
                        # print("camera_center", camera_center)
                        # exit(1)


                        # print("keypoint_distances ", keypoint_distances)
                        # print("depth_pred_keypoints ", depth_pred_keypoints)
                        # print("keypoint_instances", keypoint_instances)






                        # #attempt 2 multiscale loss
                        # loss=0
                        # rgb_loss_scale_fine=None
                        # for i in range(1):
                        #     if i!=0:
                        #         frame=frame.subsample(4)

                        #     rgb_pred=model(frame, mesh_full, depth_min, depth_max)

                        #     #view pred
                        #     rgb_pred_mat=tensor2mat(rgb_pred)
                        #     Gui.show(rgb_pred_mat,"rgb_pred_"+str(i) )

                        #     #loss
                        #     rgb_gt=mat2tensor(frame.rgb_32f, False).to("cuda")
                        #     # print("rgb_gt is ", rgb_gt.shape)
                        #     # print("rgb_pred is ", rgb_pred.shape)
                        #     rgb_loss=(( rgb_gt-rgb_pred)**2).mean()
                        #     # rgb_loss_l1=(torch.abs(rgb_gt-rgb_pred)).mean()
                        #     loss+=rgb_loss

                        #     if i==0:
                        #         rgb_loss_scale_fine=rgb_loss

                      
                      
                        #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                        if first_time:
                            first_time=False
                            # optimizer=RAdam( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                            # optimizer=GC_RAdam.RAdam( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                            # optimizer=Apollo( model.parameters(), lr=train_params.lr(), warmup=500 )
                            # optimizer=Ranger( model.parameters(), lr=train_params.lr() )
                            # optimizer=GC_Ranger.Ranger( model.parameters(), lr=train_params.lr() )
                            # optimizer=Adahessian( model.parameters(), lr=train_params.lr() ) #DO NOT USE, it requires loss.backward(create_graph=True) to compute second derivatives but that doesnt work because the grid sampler doenst have second deiv
                            # optimizer=Novograd( model.parameters(), lr=train_params.lr() )
                            # optimizer=torch.optim.AdamW( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                            optimizer=GC_Adam.AdamW( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                            # optimizer=torch.optim.SGD( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay(), momentum=0.9, nesterov=True )
                            # optimizer=Lookahead(optimizer, alpha=0.5, k=6)
                            # optimizer=torch.optim.AdamW( 
                            #     [
                            #         {'params': model.ray_marcher.parameters()},
                            #         {'params': model.rgb_predictor.parameters(), 'lr': train_params.lr()*0.1 }
                            #     ], lr=train_params.lr(), weight_decay=train_params.weight_decay()

                            #  )
                            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)
                            # warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=200)
                            optimizer.zero_grad()

                        cb.after_forward_pass(loss=rgb_loss.item(), phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 
                        # cb.after_forward_pass(loss=0, phase=phase, lr=0) #visualizes the predictio



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
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

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

                    # if is_training and phase.iter_nr%2==0: #we reduce the learning rate when the test iou plateus
                    #     optimizer.step() # DO it only once after getting gradients for all images
                    #     optimizer.zero_grad()

                    TIME_END("all")


                    with torch.set_grad_enabled(False):
                        #VIEW pred
                        #make masks 
                        mask_pred_thresh=mask_pred<0.9
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
                            # rgb_pred.masked_fill_(mask_pred_thresh, 0.0)
                            rgb_pred_mat=tensor2mat(rgb_pred)
                            Gui.show(rgb_pred_mat,"rgb_pred_"+phase.name)
                            # rgb_refined_mat=tensor2mat(rgb_refined)
                            # Gui.show(rgb_refined_mat,"rgb_refined_"+phase.name)
                            #view gt
                            Gui.show(tensor2mat(rgb_gt),"rgb_gt_"+phase.name)
                            Gui.show(tensor2mat(mask_pred),"mask_pred_"+phase.name)
                            Gui.show(tensor2mat(mask_pred_thresh*1.0),"mask_pred_t_"+phase.name)
                            # print("depth_pred min max ", depth_pred.min(), depth_pred.max())
                            depth_vis=depth_pred.view(1,1,frame.height,frame.width)
                            depth_vis=map_range(depth_vis, 0.35, 0.6, 0.0, 1.0) #for the lego shape
                            # depth_vis=map_range(depth_vis, 0.2, 0.6, 0.0, 1.0) #for the colamp fine leaves
                            depth_vis=depth_vis.repeat(1,3,1,1)
                            depth_vis.masked_fill_(rgb_pred_zeros_mask_img, 0.0)
                            Gui.show(tensor2mat(depth_vis),"depth_"+phase.name)

                        
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
                        # points3D.masked_fill_(mask_pred_thresh.view(-1,1), 0.0) # mask point occlueded
                        #mask also the points that still have a signed distance 
                        signed_dist=signed_distances_for_marchlvl[ -1 ]
                        signed_dist_mask= signed_dist.abs()>0.03
                        signed_dist_mask=signed_dist_mask.repeat(1,3) #repeat 3 times for rgb
                        points3D[signed_dist_mask]=0.0

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
                        points3D[dot_view_normal_mask]=0.0

                        #show things
                        # if is_training:
                            # show_3D_points(points3D, "points_3d_"+str(frame.frame_idx), color=rgb_pred)
                        points3d_mesh=show_3D_points(points3D, color=rgb_pred)
                        points3d_mesh.NV= normal.detach().cpu().numpy()
                        Scene.show(points3d_mesh, "points3d_mesh")




























                    # #novel view
                    # #show a novel view 
                    # if phase.iter_nr%show_every==0:
                    #     with torch.set_grad_enabled(False):
                    #         model.eval()
                    #         #create novel view
                    #         if new_frame==None:
                    #             new_frame=FramePY()
                    #             frame_to_start=frames_train[0]
                    #             new_frame.tf_cam_world=frame_to_start.tf_cam_world
                    #             new_frame.K=frame_to_start.K.copy()
                    #             new_frame.height=frame_to_start.height
                    #             new_frame.width=frame_to_start.width
                    #             new_frame.frame=frame_to_start.frame
                    #             new_frame.rgb_32f=frame_to_start.rgb_32f
                    #             new_frame.ray_dirs=frame_to_start.ray_dirs
                    #             new_frame.loader=frame.loader
                    #         #rotate a bit 
                    #         model_matrix = new_frame.frame.tf_cam_world.inverse()
                    #         model_matrix=model_matrix.orbit_y_around_point([0,0,0], 10)
                    #         new_frame.tf_cam_world = model_matrix.inverse()
                    #         # new_frame_subsampled=new_frame.subsample(4)
                    #         new_frame_subsampled=new_frame
                    #         #render new 
                    #         # print("new_frame height and width ", new_frame_subsampled.height, " ", new_frame_subsampled.width)
                    #         frames_close=loader_train.get_close_frames(new_frame, 2)
                    #         rgb_pred, depth_pred=model(new_frame, mesh_full, depth_min, depth_max,frames_close, novel=True)
                    #         rgb_pred_mat=tensor2mat(rgb_pred)
                    #         Gui.show(rgb_pred_mat, "rgb_novel")
                    #         #show new frustum 
                    #         frustum_mesh=new_frame_subsampled.create_frustum_mesh(0.01)
                    #         frustum_mesh.m_vis.m_line_width=1
                    #         frustum_mesh.m_vis.m_line_color=[0.0, 1.0, 0.0]
                    #         Scene.show(frustum_mesh, "frustum_novel" )
                    #         #show points at the end of the ray march
                    #         ray_dirs_mesh=new_frame.pixels2dirs_mesh()
                    #         ray_dirs=torch.from_numpy(ray_dirs_mesh.V.copy()).to("cuda").float() #Nx3
                    #         camera_center=torch.from_numpy( new_frame.pos_in_world() ).to("cuda")
                    #         camera_center=camera_center.view(1,3)
                    #         points3D = camera_center + depth_pred.view(-1,1)*ray_dirs
                    #         #get the point that have a color of black (correspond to background) and put them to zero
                    #         rgb_pred_channels_last=rgb_pred.permute(0,2,3,1) # from n,c,h,w to N,H,W,C
                    #         rgb_pred_zeros=rgb_pred_channels_last.view(-1,3).norm(dim=1, keepdim=True)
                    #         rgb_pred_zeros_mask= rgb_pred_zeros<0.01
                    #         rgb_pred_zeros_mask=rgb_pred_zeros_mask.repeat(1,3) #repeat 3 times for rgb
                    #         points3D[rgb_pred_zeros_mask]=0.0 #MASK the point in the background
                    #         show_3D_points(points3D, "points_3d_novel")





                    ###################### NERF PART 

                    # #forward
                    # with torch.set_grad_enabled(is_training):

                    #     #slowly upsample the frames
                    #     # frame_subsampled=frame.subsample(1)
                    #     # print("frame oriignal has height width is ", frame.height, " ", frame.width)
                    #     # print("frame_subsampled has height width is ", frame_subsampled.height, " ", frame_subsampled.width)

                    #     #selects randomly some pixels on which we train
                    #     pixels_indices=None
                    #     chunck_size= min(30*30, frame.height*frame.width)
                    #     weights = torch.ones([frame.height*frame.width], dtype=torch.float32, device=torch.device("cuda"))  #equal probability to choose each pixel
                    #     #weight depending on gradient
                    #     # grad_x=mat2tensor(frame.grad_x_32f, False)
                    #     # grad_y=mat2tensor(frame.grad_y_32f, False)
                    #     # grad=torch.cat([grad_x,grad_y],1).to("cuda")
                    #     # grad_norm=grad.norm(dim=1, keepdim=True)
                    #     # weights=grad_norm.view(-1)
                    #     # weights=weights+0.001 #rto avoid a probability of zero of getting a certain pixel

                    #     pixels_indices=torch.multinomial(weights, chunck_size, replacement=False)
                    #     pixels_indices=pixels_indices.long()

                    #     #during testing, we run the model multiple times with

                    #     TIME_START("forward")
                    #     use_chunking=True
                    #     rgb_pred, depth_map, acc_map, new_loss =model(frame, mesh_full, depth_min, depth_max, pixels_indices)
                    #     TIME_END("forward")

                    #     #VIS 
                    #     # rgb_pred_mat=tensor2mat(rgb_pred)
                    #     # Gui.show(rgb_pred_mat, "rgb_pred")
           
                    #     rgb_gt=rgb_gt.permute(0,2,3,1) #N,C,H,W to N,H,W,C
                    #     rgb_gt=rgb_gt.view(-1,3)
                    #     rgb_gt=torch.index_select(rgb_gt, 0, pixels_indices)

                    #     loss=0
                    #     rgb_loss=(( rgb_gt-rgb_pred)**2).mean()
                    #     # rgb_loss_l1=(torch.abs(rgb_gt-rgb_pred)).mean()
                    #     loss+=rgb_loss
                     
                      
                    #     #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                    #     if first_time:
                    #         first_time=False
                    #         # model.half()
                    #         # optimizer=RAdam( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                    #         optimizer=torch.optim.AdamW( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                    #         # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.1)
                    #         # scheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)
                    #         # lambda1 = lambda epoch: 0.9999 ** phase.iter_nr
                    #         # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
                    #         optimizer.zero_grad()

                    #     cb.after_forward_pass(loss=rgb_loss.item(), phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 
                    #     # cb.after_forward_pass(loss=0, phase=phase, lr=0) #visualizes the prediction 

                    
                  



                    # #backward
                    # if is_training:
                    #     if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    #         scheduler.step(phase.iter_nr /10000  ) #go to zero every 10k iters
                    #     optimizer.zero_grad()
                    #     cb.before_backward_pass()
                    #     TIME_START("backward")
                    #     loss.backward()
                    #     TIME_END("backward")
                    #     cb.after_backward_pass()
                    #     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
                    #     optimizer.step()

                    # # if is_training and phase.iter_nr%2==0: #we reduce the learning rate when the test iou plateus
                    # #     optimizer.step() # DO it only once after getting gradients for all images
                    # #     optimizer.zero_grad()

                    # TIME_END("all")


                    # #novel views after computing gradients and so on
                    # #show a novel view 
                    # if phase.iter_nr%show_every==0:
                    #     with torch.set_grad_enabled(is_training):
                    #         model.eval()
                    #         #create novel view
                    #         if new_frame==None:
                    #             new_frame=Frame()
                    #             frame_to_start=loader_train.get_frame_at_idx(0)
                    #             new_frame.tf_cam_world=frame_to_start.tf_cam_world
                    #             new_frame.K=frame_to_start.K.copy()
                    #             new_frame.height=frame_to_start.height
                    #             new_frame.width=frame_to_start.width
                    #         #rotate a bit 
                    #         model_matrix = new_frame.tf_cam_world.inverse()
                    #         model_matrix=model_matrix.orbit_y_around_point([1,0,0], 10)
                    #         new_frame.tf_cam_world = model_matrix.inverse()
                    #         # new_frame_subsampled=new_frame.subsample(4)
                    #         new_frame_subsampled=new_frame
                    #         #render new 
                    #         # print("new_frame height and width ", new_frame_subsampled.height, " ", new_frame_subsampled.width)
                    #         nr_chuncks=80
                    #         pixels_indices = torch.arange( new_frame_subsampled.height*new_frame_subsampled.width ).to("cuda")
                    #         pixels_indices=pixels_indices.long()
                    #         pixel_indices_chunks=torch.chunk(pixels_indices, nr_chuncks)
                    #         rgb_pred_list=[]
                    #         chunks_rendered=0
                    #         for pixel_indices_chunk in pixel_indices_chunks:
                    #             # print("rendering chunk", chunks_rendered)
                    #             rgb_pred, depth_map, acc_map, new_loss =model(new_frame_subsampled, mesh_full, depth_min, depth_max, pixel_indices_chunk )
                    #             # print("finished rendering chunk", chunks_rendered)
                    #             rgb_pred_list.append(rgb_pred.detach())
                    #             chunks_rendered+=1
                    #         rgb_pred=torch.cat(rgb_pred_list,0)
                    #         rgb_pred=rgb_pred.view(new_frame_subsampled.height, new_frame_subsampled.width, 3)
                    #         rgb_pred=rgb_pred.permute(2,0,1).unsqueeze(0).contiguous()
                    #         rgb_pred_mat=tensor2mat(rgb_pred)
                    #         Gui.show(rgb_pred_mat, "rgb_novel")
                    #         #show new frustum 
                    #         frustum_mesh=new_frame_subsampled.create_frustum_mesh(0.2)
                    #         frustum_mesh.m_vis.m_line_width=1
                    #         frustum_mesh.m_vis.m_line_color=[0.0, 1.0, 0.0]
                    #         Scene.show(frustum_mesh, "frustum_novel" )





                    if train_params.with_viewer():
                        view.update()

            # finished all the images 
            # pbar.close()
            # if phase.loader.is_finished(): #we reduce the learning rate when the test iou plateus
            if True: #if we reached this point we already read all the images so there is no need to check if the loader is finished 
                # if is_training and phase.iter_nr%10==0: #we reduce the learning rate when the test iou plateus
                #     optimizer.step() # DO it only once after getting gradients for all images
                #     optimizer.zero_grad()
                    # print("what")
                # if is_training:
                    # if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        # scheduler.step(phase.loss_acum_per_epoch) #for ReduceLROnPlateau
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
