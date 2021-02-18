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
random.seed(0)
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
    experiment_name="43_less"

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
    loader_train=DataLoaderNerf(config_path)
    # loader=DataLoaderVolRef(config_path)
    # loader_test=DataLoaderShapeNetImg(config_path)
    loader_test=DataLoaderNerf(config_path)
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
    model=DepthPredictor(model_params).to("cuda")
    # model.train()

    loss_fn=torch.nn.MSELoss()
    scheduler=None
    concat_coord=ConcatCoord() 
    smooth = InverseDepthSmoothnessLoss()

    # show_every=39
    show_every=100

    
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
    selected_frame_idx=np.arange(7)
    frames_query_selected=[]
    frames_target_selected=[]
    meshes_for_query_frames=[]
    for i in range(loader_train.nr_samples()):
    # for i in range(1 ):
        # frame_0=loader_train.get_frame_at_idx(i+3) 
        if i in selected_frame_idx:
            frame_query=loader_train.get_frame_at_idx(i) 
            frame_target=loader_train.get_closest_frame(frame_query)
            frames_query_selected.append(frame_query)
            frames_target_selected.append(frame_target)
            mesh_sparse=sfm.compute_3D_keypoints_from_frames(frame_query, frame_target  )
            meshes_for_query_frames.append(mesh_sparse)
            # Scene.show(mesh_sparse, "mesh_sparse_"+str(i) )

            frustum_mesh=frame_query.create_frustum_mesh(0.2)
            frustum_mesh.m_vis.m_line_width=1
            Scene.show(frustum_mesh, "frustum_"+str(frame_query.frame_idx) )

            frustum_mesh=frame_target.create_frustum_mesh(0.2)
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


    

    while True:

        for phase in phases:
            cb.epoch_started(phase=phase)
            cb.phase_started(phase=phase)
            # model.train(phase.grad)
            is_training=phase.grad

            # if loader_test.finished_reading_scene(): #For shapenet
            if True: #for nerf

                # if phase.loader.has_data() and loader_test.has_data():
                # if phase.loader.has_data(): #for nerf
                # if True: #Shapenet IMg always had ata at this point 
                for frame_idx, frame_query in enumerate(frames_query_selected):
                    frame_target=frames_target_selected[frame_idx]
                    # mesh_sparse=meshes_for_query_frames[frame_idx]
                    mask_tensor=mat2tensor(frame_query.mask, False).to("cuda").repeat(1,1,1,1)

                    #DEBUG
                    if frame_idx!=0:
                        continue

                    # DEBUG, do only one iter 
                    # if phase.iter_nr>0 and phase.grad:
                        # continue

                    #alternate between optimizing frame_query and frame_target
                    # if phase.iter_nr%2==0:
                    #     temp = frame_query
                    #     frame_query = frame_target
                    #     frame_target = temp
                    




                    #forward
                    with torch.set_grad_enabled(is_training):

                        depth=model(frame_query, mesh_full)
                        depth_original=depth
                        # print("depth has shape ", depth.shape)
                        # depth=depth*mask_tensor

                        # depth= depth+ 0.3 #added a tiny epsilon because depth of 0 gets an invalid uv tensor afterwards and it just get a black color
                        depth= depth+ 0.8 #added a tiny epsilon because depth of 0 gets an invalid uv tensor afterwards and it just get a black color


                        # print("depth is ", depth)

                        #DEBUG depth map
                        depth=depth*2
                        # print("depth minmax", depth.min().item(), " max ", depth.max().item())
                        depth_mat=tensor2mat(depth)
                        depth_vis=map_range(depth_original, 0.6, 2, 0, 1)
                        depth_mat_vis=tensor2mat(depth_vis)
                        Gui.show(depth_mat_vis, "depth_mat")
                        frame_query.depth=depth_mat
                        reprojected_mesh=frame_query.depth2world_xyz_mesh()
                        mask=mask_tensor.permute(0,2,3,1).squeeze(0).float() #mask goes from N,C,H,W to N,H,W,C
                        mask=mask.view(-1,1)
                        reprojected_points_py=torch.from_numpy( reprojected_mesh.V ).to("cuda")
                        reprojected_points_py=reprojected_points_py*mask
                        reprojected_mesh.V=reprojected_points_py.detach().cpu().numpy()
                        Scene.show(reprojected_mesh, "reprojected_mesh")



                        #DEBUG WTF is happening with the tf matrix 
                        # print("frame_query frame idx s  ",frame_query.frame_idx)
                        # print("frame_query.tf_cam_world is ", frame_query.tf_cam_world.matrix())
                        # print("frame_query.tf_cam_worldlinear is ", frame_query.tf_cam_world.linear() )
                        # print("frame_query.tf_cam_worldinverse is ", frame_query.tf_cam_world.inverse().matrix())
                        # print("frame_query.tf_cam_worldinverse linear is ", frame_query.tf_cam_world.inverse().linear() )
                        # tf_world_cam=frame_query.tf_cam_world.inverse()
                        # print("AGAIN frame_query.tf_cam_worldinverse linear is ", tf_world_cam.linear() )
                        # # exit(1)


                        #put the depth in 3D 
                        tf_world_cam=frame_query.tf_cam_world.inverse() # APARENTLY THERE'S A BUG that happens when chaining eigne transforms like inverse() and linear() in which the results is completely crazy like e+24. The solution seems to be to not chain them but rather save the inverse sand then do linear on it
                        R=torch.from_numpy( tf_world_cam.linear() ).to("cuda")
                        t=torch.from_numpy( tf_world_cam.translation() ).to("cuda")
                        K = torch.from_numpy( frame_query.K ).to("cuda")
                        K_inv = torch.from_numpy( np.linalg.inv(frame_query.K) ).to("cuda")
                        # print("K is ", K, "K inv is ", K_inv)
                        ones=torch.ones([1,1,frame_query.height, frame_query.width], dtype=torch.float32).to("cuda")
                        points_screen=concat_coord(ones) 
                        # print("points_screen minmax", points_screen.min().item(), " max ", points_screen.max().item())
                        points_screen[:, 0:1, :,:]= (points_screen[:, 0:1, :,:]+1)*0.5*frame_query.height  #get it from [-1.1] to the [0,height]
                        points_screen[:, 1:2, :,:]= (points_screen[:, 1:2, :,:]+1)*0.5*frame_query.width  #get it from [-1.1] to the [0,width]
                        # print("points_screen after ranging minmax", points_screen.min().item(), " max ", points_screen.max().item())
                        points_screen=points_screen.permute(0,2,3,1) #go from N,C,H,W to N,H,W,C
                        points_screen=points_screen.view(-1,3) # Nx3
                        # print("points_screen after view minmax", points_screen.min().item(), " max ", points_screen.max().item())
                        points_3D_cam= ( torch.matmul(K_inv,points_screen.transpose(0,1))  ).transpose(0,1) #the points 3D are now Nx3
                        points_3D_cam[:,1]=-points_3D_cam[:,1] #flip
                        # print("points_3D_cam minmax", points_3D_cam.min().item(), " max ", points_3D_cam.max().item())
                        depth=depth.permute(0,2,3,1) #go from N,C,H,W to N,H,W,C
                        depth=depth.view(-1,1)
                        points_3D_cam=points_3D_cam*depth
                        # print("points_3D_cam after depth minmax", points_3D_cam.min().item(), " max ", points_3D_cam.max().item())
                        # print("R is ", R, "direct R is ", frame_query.tf_cam_world.inverse().linear() )
                        # print("t is ", t)
                        points_3D_world=torch.matmul(R, points_3D_cam.transpose(0,1) ).transpose(0,1)  + t.view(1,3)

                        #points 3d world are masked
                        points_3D_world=points_3D_world*mask

                        # print("points_3D_world", points_3D_world)
                        # print("points_3D_world", points_3D_world.min().item(), " max ", points_3D_world.max().item())

                        #attempt 2, we don;t need to actually use the GPU to get the thing in 3D, because we only need to compute the uv tensors which we know that it can be done correctly on CPU
                        #THIS WILL NOT WORK BECAUSE WE cannot backrpopagate through this UV tensor
                        # depth_mat=tensor2mat(depth)
                        # frame_query.depth=depth
                        # reprojected_mesh=depth2world_xyz_mesh()
                        # #get the uvs towards the query and the target
                        # uv_query=frame_query.compute_uv(reprojected_mesh) #uv for projecting this cloud into this frame
                        # uv_query_tensor=torch.from_numpy(uv_query).float().to("cuda")
                        # uv_query_tensor= uv_query_tensor*2 -1
                        # uv_query_tensor[:,1]=-uv_query_tensor[:,1] #flip
                        # #uv_for target 
                        # uv_target=frame_target.compute_uv(reprojected_mesh) #uv for projecting this cloud into this frame
                        # uv_target_tensor=torch.from_numpy(uv_target).float().to("cuda")
                        # uv_target_tensor= uv_target_tensor*2 -1
                        # uv_target_tensor[:,1]=-uv_target_tensor[:,1] #flip

                    

                        #get the uv tensor for query and target
                        uv_query=compute_uv(frame_query, points_3D_world)
                        uv_target=compute_uv(frame_target, points_3D_world)

                       



                        #slice the detph from query and from target colors and compare them
                        rgb_query=mat2tensor(frame_query.rgb_32f, False).to("cuda")
                        # print("rgb_query has shape ", rgb_query.shape)
                        rgb_query=rgb_query.permute(0,2,3,1).squeeze(0) #N,3,H,W to N,H,W,C
                        rgb_target=mat2tensor(frame_target.rgb_32f, False).to("cuda")
                        rgb_target=rgb_target.permute(0,2,3,1).squeeze() #N,3,H,W to N,H,W,C
                        # print("uv query is ". uv_query)
                        # print("rgb_query is ". rgb_query)
                        # print("uv_query is ", uv_query.min(), " ", uv_query.max())
                        # print("uv_query is ", uv_query)

                        loss=0

                        rgb_query_for_slicing= rgb_query.clone().to("cuda")
                        rgb_target_for_slicing= rgb_target.clone().to("cuda")
                        for i in range(7):
                            if i>0:
                                rgb_query_for_slicing= NeuralMVS.subsample(rgb_query_for_slicing, 2, "area").to("cuda") # downsample once with AREA interpolation like in opencV
                                rgb_target_for_slicing= NeuralMVS.subsample(rgb_target_for_slicing, 2, "area").to("cuda")

                            # print("at level ", i, "slicing from ", rgb_query_for_slicing.shape)
                            # _,_, predicted_query =model.slice_texture(rgb_query_for_slicing, uv_query)
                            _,_, predicted_query=model.slice_texture(rgb_target_for_slicing, uv_target) ##slices the right view and with this we try to reconstruct the query

                            #splat the points onto the target 
                            rgb_query= rgb_query.view(-1,3)
                            texture_target= model.splat_texture(rgb_query, uv_target, rgb_query_for_slicing.shape[0]) 
                            val_dim=texture_target.shape[2]-1
                            texture_target=texture_target[:,:,0:val_dim] / (texture_target[:,:,val_dim:val_dim+1] +0.0001)
                            # texture_target=texture_target[:,:,0:val_dim] 

                            #debug, see the texture target 
                            texture_target_vis= texture_target.permute(2,0,1).unsqueeze(0) #from H,W,C to C,H,W
                            texture_target_mat_vis=tensor2mat(texture_target_vis)
                            Gui.show(texture_target_mat_vis, "texture_target_mat_vis")



                            #debug the two images 
                            # rgb_query_for_slicing_vis=rgb_query_for_slicing.unsqueeze(0).permute(0,3,1,2) # N,H,W,C to N,C,H,W
                            # rgb_target_for_slicing_vis=rgb_target_for_slicing.unsqueeze(0).permute(0,3,1,2) # N,H,W,C to N,C,H,W
                            # Gui.show( tensor2mat(rgb_query_for_slicing_vis) ,"rgb_query"+str(i) )
                            # Gui.show( tensor2mat(rgb_target_for_slicing_vis) ,"rgb_target"+str(i) )


                            # #DEBUG 
                            # subsampled_height= int(frame_query.height *  1.0/pow(2,i))
                            # subsampled_width= int(frame_query.width *  1.0/pow(2,i))
                            # # print("frame heigth and width is ", frame_query.height, " ", frame_query.width, " subsampled_height ", subsampled_height, " subsampled_width ", subsampled_width)
                            # # print("predicted_query has shape ", predicted_query.shape)
                            # # predicted_query_vis=predicted_query.view(1, subsampled_height, subsampled_width, 3)
                            # predicted_query_vis=predicted_query.view(1, frame_query.height, frame_query.width , 3)
                            # predicted_query_vis=predicted_query_vis.permute(0,3,1,2) #from N,H,W,3 to N,C,H,W
                            # predicted_query_vis_mat=tensor2mat(predicted_query_vis)
                            # Gui.show(predicted_query_vis_mat,"predicted_query_vis_mat_"+str(i) )
                            # #show predicted target 
                            # # predicted_target_vis=predicted_target.view(1, subsampled_height, subsampled_width, 3)
                            # predicted_target_vis=predicted_target.view(1, frame_query.height, frame_query.width, 3)
                            # predicted_target_vis=predicted_target_vis.permute(0,3,1,2) #from N,H,W,3 to N,C,H,W
                            # predicted_target_vis_mat=tensor2mat(predicted_target_vis)
                            # Gui.show(predicted_target_vis_mat,"predicted_target_vis_mat_"+str(i))


                            mask=mask.view(-1,1)
                            # rgb_query=rgb_query_for_slicing.view(-1,3)
                            #RGB loss
                            # diff_rgb=((predicted_query -predicted_target)**2)*mask
                            # diff_rgb=((predicted_query -predicted_target)**2)
                            diff_rgb=(( rgb_query-predicted_query)**2)
                            diff_rgb_2=((texture_target -rgb_target_for_slicing.squeeze())**2)
                            rgb_loss = diff_rgb.mean() 
                            # rgb_loss =  diff_rgb_2.mean()
                            # rgb_loss = diff_rgb.mean() + diff_rgb_2.mean()*phase.iter_nr*0.0001
                            # rgb_loss = diff_rgb.mean()*0.5 +  diff_rgb_2.mean()*0.5

                            # #debug diff 
                            # diff_rgb_vis=diff_rgb_2.view(1, frame_query.height, frame_query.width, 3)
                            # diff_rgb_vis= diff_rgb_vis.permute(0,3,1,2)
                            # diff_rgb_vis=diff_rgb_vis*10 #increase power
                            # diff_rgb_vis_mat=tensor2mat(diff_rgb_vis)
                            # Gui.show(diff_rgb_vis_mat, "diff_rgb_vis_mat")

                            # weight=1.0/(float(i)+1)
                            # rgb_loss=rgb_loss*weight
                            # if i==0:
                            loss+=rgb_loss

                        #smoothnes loss 
                        rgb_query=mat2tensor(frame_query.rgb_32f, False).to("cuda")
                        smooth_loss= smooth(depth_original, rgb_query) 
                        # loss+=smooth_loss* (0.1 + phase.iter_nr*0.0001 )
                        loss+=smooth_loss* 0.3



                        # _,_, predicted_query =model.slice_texture(rgb_query, uv_query)
                        # _,_, predicted_target=model.slice_texture(rgb_target, uv_target)

                        # print("predicted_query is ", predicted_query)

                        #predicted query is actually just the original RGB image
                        # predicted_query_direct= rgb_query.view(-1,3)
                        # predicted_query= rgb_query.view(-1,3)

                        # #Debug the uv_query why is it flipped
                        # ones=torch.ones([uv_query.shape[0],1], dtype=torch.float32).to("cuda")
                        # uv_query_vis=torch.cat([uv_query, ones], 1)
                        # uv_query_vis=uv_query_vis.view(frame_query.height, frame_query.width, -1)
                        # mask=mask_tensor.permute(0,2,3,1).squeeze(0).float() #mask goes from N,C,H,W to N,H,W,C
                        # uv_query_vis=uv_query_vis*mask 
                        # uv_query_vis=uv_query_vis.view(-1,3)
                        # # print("uv_query_vis ha shape ", uv_query_vis.shape)
                        # uv_query_cpu=uv_query_vis.detach().cpu().numpy()
                        # uv_query_mesh=Mesh()
                        # uv_query_mesh.V=uv_query_cpu 
                        # uv_query_mesh.m_vis.m_show_points=True
                        # Scene.show(uv_query_mesh, " uv_query_mesh ")


                        #DEbug the points3d world
                        points_3d_mesh=Mesh()
                        points_3d_mesh.V=points_3D_world.detach().cpu().numpy() 
                        points_3d_mesh.m_vis.m_show_points=True
                        Scene.show(points_3d_mesh, " points_3d_mesh ")



                        # #DEBUG 
                        # predicted_query_vis=predicted_query.view(0, frame_query.height, frame_query.width, 3)
                        # predicted_query_vis=predicted_query_vis.permute(-1,3,1,2)
                        # predicted_query_vis_mat=tensor1mat(predicted_query_vis)
                        # Gui.show(predicted_query_vis_mat,"predicted_query_vis_mat")
                        # #show predicted target 
                        # predicted_target_vis=predicted_target.view(0, frame_query.height, frame_query.width, 3)
                        # predicted_target_vis=predicted_target_vis.permute(-1,3,1,2)
                        # predicted_target_vis_mat=tensor1mat(predicted_target_vis)
                        # Gui.show(predicted_target_vis_mat,"predicted_target_vis_mat")



                        # # mask=mask_tensor.permute(0,2,3,1).squeeze(0).float() #mask goes from N,C,H,W to N,H,W,C
                        # mask=mask.view(-1,1)
                        # #RGB loss
                        # diff_rgb=((predicted_query -predicted_target)**2)*mask
                        # rgb_loss = diff_rgb.mean()
                        # #depth_average loss
                        # depth_mean=depth.mean() 
                        # depth_loss= ((depth_mean-2.0)**2).mean()
                        # #depth std
                        # depth_std_loss=depth.std()
                        # # loss=rgb_loss + depth_loss
                        # # loss= depth_loss + depth_std_loss
                        # # loss=rgb_loss + depth_loss + depth_std_loss*0.1
                        # loss=rgb_loss
                        print("loss is ", loss)
                    
                    


                        TIME_START("forward")
                        # out_tensor,  depth_map, acc_map, new_loss=model(gt_frame, frames_for_encoding, all_imgs_poses_cam_world_list, gt_frame.tf_cam_world, gt_frame.K, depth_min, depth_max, use_ray_compression )
                        TIME_END("forward")


                        # rgb_loss=( torch.abs(out_tensor-gt_rgb_tensor) ).mean()
                       

                      
                        #SSIM LOSS does not mae things better, it may even make then worse
                        # ssim_loss= 1 - ms_ssim( gt_rgb_tensor, out_tensor, win_size=3, data_range=1.0, size_average=True )
                        # loss=rgb_loss*0.5 + ssim_loss*0.5
                        # loss=rgb_loss
                      
                      
                        #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                        if first_time:
                            first_time=False
                            optimizer=RAdam( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.1)
                            # scheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)
                            # lambda1 = lambda epoch: 0.9999 ** phase.iter_nr
                            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
                            optimizer.zero_grad()

                        cb.after_forward_pass(loss=rgb_loss, phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 
                        # cb.after_forward_pass(loss=0, phase=phase, lr=0) #visualizes the prediction 
                        # pbar.update(1)

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
                        optimizer.step()


                if train_params.with_viewer():
                    view.update()

            # finished all the images 
            # pbar.close()
            if phase.loader.is_finished(): #we reduce the learning rate when the test iou plateus
            # if is_training: #we reduce the learning rate when the test iou plateus
                # optimizer.step() # DO it only once after getting gradients for all images
                # optimizer.zero_grad()
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
