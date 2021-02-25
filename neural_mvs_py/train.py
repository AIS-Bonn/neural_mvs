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
from neural_mvs.MS_SSIM_L1_loss import *

from callbacks.callback import *
from callbacks.viewer_callback import *
from callbacks.visdom_callback import *
from callbacks.state_callback import *
from callbacks.phase import *

from optimizers.over9000.radam import *
from optimizers.over9000.lookahead import *
from optimizers.over9000.novograd import *
from optimizers.over9000.ranger import *
from optimizers.over9000.apollo import *

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
random.seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(edgeitems=3)

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
    # experiment_name="s_apol_lr5.0_clipno"
    # experiment_name="s_adam0.001_clipno"
    experiment_name="s_rd0.003noabs"

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
    # loader_train=DataLoaderShapeNetImg(config_path)
    # loader_train=DataLoaderNerf(config_path)
    loader_train=DataLoaderColmap(config_path)
    # loader=DataLoaderVolRef(config_path)
    # loader_test=DataLoaderShapeNetImg(config_path)
    # loader_test=DataLoaderNerf(config_path)
    loader_test=DataLoaderColmap(config_path)
    # loader_test=DataLoaderVolRef(config_path)

    #create phases
    phases= [
        Phase('train', loader_train, grad=True),
        # Phase('test', loader_test, grad=False)
    ]
    #model 
    model=None
    # model=Net().to("cuda")
    # model=SirenNetwork(in_channels=2, out_channels=3).to("cuda")
    # model = VAE(nc=3, ngf=128, ndf=128, latent_variable_size=500).to("cuda")
    # model=DepthPredictor(model_params).to("cuda")
    # model=Net2(model_params).to("cuda")
    model=Net3_SRN(model_params).to("cuda")
    # model.train()
    # model.half()

    loss_fn=torch.nn.MSELoss()
    scheduler=None
    concat_coord=ConcatCoord() 
    smooth = InverseDepthSmoothnessLoss()
    ssim_l1_criterion = MS_SSIM_L1_LOSS()

    # show_every=39
    show_every=10
    # show_every=1

    
    #get the frames into a vector
    # selected_frame_idx=[0,4] #two cameras at like 45 degrees
    # selected_frame_idx=[3,5] #also at 45 but probably a bit better
    # selected_frame_idx=[8,9] #also at 45 but probably a bit better
    # frames_train=[]
    # while len(frames_train)<2:
    #     if(loader_train.has_data() ): 
    #         frame=loader_train.get_next_frame()
    #         if frame.frame_idx in selected_frame_idx:
    #             frames_train.append(frame)
    #     if loader_train.is_finished():
    #         loader_train.reset()
    #         break

    frames_train=[]
    # frames_train.append( frame_0 )
    # frames_train.append( loader_train.get_closest_frame(frame_0) )
    
    #compute 3D 
    sfm=SFM.create()
    # selected_frame_idx=[0,3] 
    # selected_frame_idx=[0] 
    # selected_frame_idx=[1] 
    # selected_frame_idx=[0,1,2,3] 
    # selected_frame_idx=np.arange(7) #For nerf
    selected_frame_idx=np.arange(30) #For colmap
    # selected_frame_idx=[10]
    frames_query_selected=[]
    frames_target_selected=[]
    frames_all_selected=[]
    meshes_for_query_frames=[]
    for i in range(loader_train.nr_samples()):
    # for i in range(1 ):
        # frame_0=loader_train.get_frame_at_idx(i+3) 
        if i in selected_frame_idx:
            frame_query=loader_train.get_frame_at_idx(i) 
            frame_target=loader_train.get_closest_frame(frame_query)
            frames_query_selected.append(frame_query)
            frames_target_selected.append(frame_target)
            frames_all_selected.append(frame_query)
            frames_all_selected.append(frame_target)
            mesh_sparse, keypoints_distances_eigen, keypoints_indices_eigen=sfm.compute_3D_keypoints_from_frames(frame_query, frame_target  )
            meshes_for_query_frames.append(mesh_sparse)
            # Scene.show(mesh_sparse, "mesh_sparse_"+str(i) )

            frustum_mesh=frame_query.create_frustum_mesh(0.01)
            frustum_mesh.m_vis.m_line_width=1
            Scene.show(frustum_mesh, "frustum_"+str(frame_query.frame_idx) )

            frustum_mesh=frame_target.create_frustum_mesh(0.01)
            frustum_mesh.m_vis.m_line_width=1
            frustum_mesh.m_vis.m_line_color=[0.0, 0.0, 1.0]
            Scene.show(frustum_mesh, "frustum_T_"+str(frame_target.frame_idx) )


    #fuse all the meshes into one
    mesh_full=Mesh()
    for mesh in meshes_for_query_frames:
        mesh_full.add(mesh)
    mesh_full.m_vis.m_show_points=True
    mesh_full.m_vis.set_color_pervertcolor()
    Scene.show(mesh_full, "mesh_full" )
    print("scene scale is ", Scene.get_scale())


    #get for each frame_query the distances of the keypoints
    frame_idx2keypoint_data={}
    for i in range(loader_train.nr_samples()):
        frame_query=loader_train.get_frame_at_idx(i) 
        frame_target=loader_train.get_closest_frame(frame_query)
        mesh_sparse, keypoints_distances_eigen, keypoints_indices_eigen=sfm.compute_3D_keypoints_from_frames(frame_query, frame_target  )
        keypoints_distances=torch.from_numpy(keypoints_distances_eigen.copy()).to("cuda")
        keypoints_indices=torch.from_numpy(keypoints_indices_eigen.copy()).to("cuda")
        keypoints_3d =torch.from_numpy(mesh_sparse.V.copy()).float().to("cuda")
        keypoint_data=[keypoints_distances, keypoints_indices, keypoints_3d]
        frame_idx2keypoint_data[frame_query.frame_idx] = keypoint_data
        # Scene.show(mesh_sparse, "mesh_full_"+str(frame_query.frame_idx) )



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
                    model.train(phase.grad)
                    frame=phase.loader.get_random_frame() 
                    # frame=phase.loader.get_frame_at_idx(10) 
                    # pass
                    TIME_START("all")
                    # mask_tensor=mat2tensor(frame.mask, False).to("cuda").repeat(1,3,1,1)
                    rgb_gt=mat2tensor(frame.rgb_32f, False).to("cuda")
                    # rgb_gt=rgb_gt*mask_tensor
                    # rgb_gt=rgb_gt+0.1
                    # rgb_mat=tensor2mat(rgb_gt)
                    # Gui.show(rgb_mat,"rgb_gt")





                    #forward attempt 2 using a network with differetnaible ray march
                    with torch.set_grad_enabled(is_training):
                        
                        rgb_pred, depth_pred=model(frame, mesh_full, depth_min, depth_max)

                        #view pred
                        rgb_pred_mat=tensor2mat(rgb_pred)
                        Gui.show(rgb_pred_mat,"rgb_pred")

                        #loss
                        rgb_gt=mat2tensor(frame.rgb_32f, False).to("cuda")
                        loss=0
                        rgb_loss=(( rgb_gt-rgb_pred)**2).mean()
                        # rgb_loss_ssim_l1 = ssim_l1_criterion(rgb_gt, rgb_pred)
                        # rgb_loss_l1=(torch.abs(rgb_gt-rgb_pred)).mean()
                        loss+=rgb_loss

                        #loss on depth 
                        keypoint_data=frame_idx2keypoint_data[frame.frame_idx]
                        keypoint_distances=keypoint_data[0]
                        keypoint_instances=keypoint_data[1]
                        keypoints_3d=keypoint_data[2]
                        depth_pred=depth_pred.view(-1,1)
                        depth_pred_keypoints= torch.index_select(depth_pred, 0, keypoint_instances.long())
                        loss_depth= (( keypoint_distances- depth_pred_keypoints)**2).mean()
                        loss+=loss_depth
                        # print("loss depth is ", loss_depth)

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
                            optimizer=RAdam( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                            # optimizer=Apollo( model.parameters(), lr=train_params.lr() )
                            # optimizer=Ranger( model.parameters(), lr=train_params.lr() )
                            # optimizer=Novograd( model.parameters(), lr=train_params.lr() )
                            # optimizer=torch.optim.AdamW( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                            # optimizer=torch.optim.AdamW( 
                            #     [
                            #         {'params': model.ray_marcher.parameters()},
                            #         {'params': model.rgb_predictor.parameters(), 'lr': train_params.lr()*0.1 }
                            #     ], lr=train_params.lr(), weight_decay=train_params.weight_decay()

                            #  )
                            optimizer.zero_grad()

                        cb.after_forward_pass(loss=rgb_loss.item(), phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 
                        # cb.after_forward_pass(loss=0, phase=phase, lr=0) #visualizes the predictio



                    #backward
                    if is_training:
                        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                            scheduler.step(phase.iter_nr /10000  ) #go to zero every 10k iters
                        optimizer.zero_grad()
                        cb.before_backward_pass()
                        TIME_START("backward")
                        loss.backward()
                        TIME_END("backward")
                        cb.after_backward_pass()
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)
                        optimizer.step()

                    # if is_training and phase.iter_nr%2==0: #we reduce the learning rate when the test iou plateus
                    #     optimizer.step() # DO it only once after getting gradients for all images
                    #     optimizer.zero_grad()

                    TIME_END("all")


                    #novel view
                    #show a novel view 
                    if phase.iter_nr%show_every==0:
                        with torch.set_grad_enabled(False):
                            model.eval()
                            #create novel view
                            if new_frame==None:
                                new_frame=Frame()
                                frame_to_start=loader_train.get_frame_at_idx(0)
                                new_frame.tf_cam_world=frame_to_start.tf_cam_world
                                new_frame.K=frame_to_start.K.copy()
                                new_frame.height=frame_to_start.height
                                new_frame.width=frame_to_start.width
                            #rotate a bit 
                            model_matrix = new_frame.tf_cam_world.inverse()
                            model_matrix=model_matrix.orbit_y_around_point([0,0,0], 10)
                            new_frame.tf_cam_world = model_matrix.inverse()
                            # new_frame_subsampled=new_frame.subsample(4)
                            new_frame_subsampled=new_frame
                            #render new 
                            # print("new_frame height and width ", new_frame_subsampled.height, " ", new_frame_subsampled.width)
                            rgb_pred, depth_pred=model(new_frame, mesh_full, depth_min, depth_max, novel=True)
                            rgb_pred_mat=tensor2mat(rgb_pred)
                            Gui.show(rgb_pred_mat, "rgb_novel")
                            #show new frustum 
                            frustum_mesh=new_frame_subsampled.create_frustum_mesh(0.01)
                            frustum_mesh.m_vis.m_line_width=1
                            frustum_mesh.m_vis.m_line_color=[0.0, 1.0, 0.0]
                            Scene.show(frustum_mesh, "frustum_novel" )





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
            if phase.loader.is_finished(): #we reduce the learning rate when the test iou plateus
                # if is_training and phase.iter_nr%10==0: #we reduce the learning rate when the test iou plateus
                #     optimizer.step() # DO it only once after getting gradients for all images
                #     optimizer.zero_grad()
                    # print("what")
                # if is_training:
                    # if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        # scheduler.step(phase.loss_acum_per_epoch) #for ReduceLROnPlateau
                cb.epoch_ended(phase=phase, model=model, save_checkpoint=train_params.save_checkpoint(), checkpoint_path=train_params.checkpoint_path() ) 
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
