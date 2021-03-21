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
    experiment_name="s1ad_0.001"

    use_ray_compression=False





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
    loader_train=DataLoaderColmap(config_path)
    loader_test=DataLoaderColmap(config_path)
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
        frames_train.append(FramePY(frame))
    for i in range(loader_test.nr_samples()):
        frame=loader_test.get_frame_at_idx(i)
        frames_test.append(FramePY(frame))
    phases[0].frames=frames_train 
    phases[1].frames=frames_test
    #Show only the visdom for the testin
    phases[0].show_visdom=False
    phases[1].show_visdom=True




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

    while True:
        with torch.set_grad_enabled(False):

            # tf_cam_world=frame.tf_cam_world.clone()
            # pos=view.m_camera.position()
            # print("pos", pos)
            # quat=view.m_camera.view_matrix_affine().quat()
            # tf_cam_world=tf_cam_world.set_translation(pos)
            # print("set pos is ", tf_cam_world.translation() )
            # tf_cam_world=tf_cam_world.set_quat(quat)
            # print("set quat is ", tf_cam_world.quat() )
            # # print(tf_cam_world.matrix())
            # frame.tf_cam_world=tf_cam_world.clone()
            # frame.frame.tf_cam_world=tf_cam_world.clone()
            # # print("mat",frame.tf_cam_world.matrix())
            # frame=FramePY(frame.frame)
            # exit(1)


            #get the model matrix of the view and set it to the frame
            cam_tf_world_cam= view.m_camera.model_matrix_affine()
            frame.frame.tf_cam_world=cam_tf_world_cam.inverse()
            frame=FramePY(frame.frame)


            discard_same_idx=False
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


            rgb_pred, rgb_refined, depth_pred, mask_pred, signed_distances_for_marchlvl, std=model(frame, ray_dirs, rgb_close_batch, depth_min, depth_max, frames_close, weights, novel=True)

            if first_time:
                first_time=False
                #TODO load checkpoint
                # now that all the parameters are created we can fill them with a model from a file
                model.load_state_dict(torch.load( "/media/rosu/Data/phd/c_ws/src/phenorob/neural_mvs/saved_models/fine_leaves_home_plant/model_e_900.pt" ))

            rgb_pred_mat=tensor2mat(rgb_pred)
            Gui.show(rgb_pred_mat,"rgb_pred")






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
