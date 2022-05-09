import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from easypbr  import *
import sys
import math
from collections import namedtuple
import random
from typing import Dict, List, Optional, Tuple
from torch.nn.utils.weight_norm import WeightNorm, remove_weight_norm
from torch.nn.modules.utils import _pair
import inspect


from neural_mvs_py.neural_mvs.funcs import *


# from torchmeta.modules.conv import MetaConv2d
# from torchmeta.modules.linear import MetaLinear
# from torchmeta.modules.module import *
# from torchmeta.modules.utils import *
# from torchmeta.modules import (MetaModule, MetaSequential)

from neural_mvs_py.neural_mvs.pac import *
from neural_mvs_py.neural_mvs.deform_conv import *
#from latticenet_py.lattice.lattice_modules import *


from dataloaders import *


DatasetParams = namedtuple('DatasetParams', 'sphere_radius sphere_center estimated_scene_dist_from_origin raymarch_depth_min raymarch_depth_jitter triangulation_type frustum_size use_ndc')


def rand_true(probability_of_true):
    return random.random() < probability_of_true



def nchw2nhwc(x):
    x=x.permute(0,2,3,1)
    return x

def nhwc2nchw(x):
    x=x.permute(0,3,1,2)
    return x

#make from N,C,H,W to N,Nrpixels,C
def nchw2nXc(x):
    nr_feat=x.shape[1]
    nr_batches=x.shape[0]
    x=x.permute(0,2,3,1) #from N,C,H,W to N,H,W,C
    x=x.view(nr_batches, -1, nr_feat)
    return x

#make from N,NrPixels,C to N,C,H,W
def nXc2nchw(x, h, w):
    nr_feat=x.shape[2]
    nr_batches=x.shape[0]
    x=x.view(nr_batches, h, w, nr_feat)
    x=x.permute(0,3,1,2) #from N,H,W,C, N,C,H,W 
    return x

# make from N,C,H,W to Nrpixels,C ONLY works when N is 1
def nchw2lin(x):
    if x.shape[0]!=1:
        print("nchw2lin supposes that the N is 1 however x has shape ", x.shape )
        exit(1)
    nr_feat=x.shape[1]
    nr_batches=x.shape[0]
    x=x.permute(0,2,3,1) #from N,C,H,W to N,H,W,C
    x=x.view(-1, nr_feat)
    return x

#go from nr_pixels, C to 1,C,H,W
def lin2nchw(x, h, w):
    nr_feat=x.shape[1]
    x=x.view(1, h, w, nr_feat)
    x=nhwc2nchw(x)
    return x



#get all the frames from a loader and puts them into frame_py
def get_frames(loader):
    frames=[]
    for i in range(loader.nr_samples()):
        frame=loader.get_frame_at_idx(i)
        frames.append(FramePY(frame, create_subsamples=True))
    return frames

def compute_dataset_params(loader, frames):

    frame_centers, frame_idxs = frames_to_points(frames)
    sphere_center, sphere_radius=SFM.fit_sphere(frame_centers)
    #if ithe shapentimg we put the center to zero because we know where it is
    if isinstance(loader, DataLoaderShapeNetImg) or isinstance(loader, DataLoaderDTU):
        sphere_center= np.array([0,0,0])
        sphere_radius= np.amax(np.linalg.norm(frame_centers- sphere_center, axis=1))
    #for most of the datasets the scene is around the center of the sphere but for LLFF which has front facing cameras we have to manually set the estimated scene_center
    if not isinstance(loader, DataLoaderLLFF):
    #    estimated_scene_center =  sphere_center #for most of the datasets the scene is around the center of the sphere
       estimated_scene_dist_from_origin= np.linalg.norm(sphere_center) # #for most of the datasets the scene is around the center of the sphere. Distance from origin of the world to the sphere is the norm
    else: #if we deal with a LLFF datset we need to set our own estimate of the scene center
        # estimated_scene_center= np.array([0,0,-0.3])
        mean = (frames[0].frame.get_extra_field_float("min_near") + frames[0].frame.get_extra_field_float("max_far")) *0.5
        # mean = (frames[0].frame.get_extra_field_float("min_near")*0.75 + frames[0].frame.get_extra_field_float("max_far")*0.25)*0.5
        # mean = frames[0].frame.get_extra_field_float("max_far") 
        # print("mean si ", mean)
        # print("frames[0].frame.get_extra_field_float(min_near)", frames[0].frame.get_extra_field_float("min_near"))
        # estimated_scene_center= np.array([0,0, mean ])
        # estimated_scene_center= np.array([mean ])
        # estimated_scene_center= np.array([0,0, mean ])
        estimated_scene_dist_from_origin=mean
        # estimated_scene_center= np.array([0,0, frames])
    print("sphere center and raidus ", sphere_center, " radius ", sphere_radius)

    #triangulation type is sphere for all datasets except llff
    triangulation_type="sphere"
    if isinstance(loader, DataLoaderLLFF):
        triangulation_type = "plane"


    #min and jitter
    raymarch_depth_min = 0.15
    raymarch_depth_jitter =  2e-3
    if isinstance(loader, DataLoaderLLFF):
        # raymarch_depth_min=0.005
        # raymarch_depth_jitter =  5e-4
        raymarch_depth_min=frames[0].frame.get_extra_field_float("min_near")
        raymarch_depth_jitter =  0.0
        # raymarch_depth_jitter =  1.0


    #frustum size 
    frustum_size=0.02
    if isinstance(loader, DataLoaderLLFF):
        frustum_size=0.001

    #usage of ndc
    use_ndc=False
    if isinstance(loader, DataLoaderLLFF):
        use_ndc=False
        # use_ndc=True


    params= DatasetParams(sphere_radius=sphere_radius, 
                        sphere_center=sphere_center, 
                        # estimated_scene_center=estimated_scene_center, 
                        estimated_scene_dist_from_origin= estimated_scene_dist_from_origin,
                        raymarch_depth_min=raymarch_depth_min,
                        raymarch_depth_jitter = raymarch_depth_jitter,
                        triangulation_type= triangulation_type,
                        frustum_size=frustum_size,
                        use_ndc=use_ndc )

    return params


# def prepare_data(frame_full_res, frames_close_full_res, frame, frames_close):
#     rgb_gt=mat2tensor(frame.frame.rgb_32f, False).to("cuda")
#     rgb_gt_fullres=mat2tensor(frame_full_res.frame.rgb_32f, False).to("cuda")
#     # mask_tensor=mat2tensor(frame.frame.mask, False).to("cuda")
#     ray_dirs=torch.from_numpy(frame.ray_dirs).to("cuda").float().view(1, frame.height, frame.width, 3).permute(0,3,1,2)
#     rgb_close_batch_list=[]
#     for frame_close in frames_close:
#         rgb_close_frame=mat2tensor(frame_close.frame.rgb_32f, False).to("cuda")
#         rgb_close_batch_list.append(rgb_close_frame)
#     rgb_close_batch=torch.cat(rgb_close_batch_list,0)
#     #make also a batch fo directions
#     raydirs_close_batch_list=[]
#     for frame_close in frames_close:
#         ray_dirs_close=torch.from_numpy(frame_close.ray_dirs).to("cuda").float().view(1, frame.height, frame.width, 3).permute(0,3,1,2)
#         # ray_dirs_close=ray_dirs_close.view(1, frame.height, frame.width, 3)
#         # ray_dirs_close=ray_dirs_close.permute(0,3,1,2) #from N,H,W,C to N,C,H,W
#         raydirs_close_batch_list.append(ray_dirs_close)
#     ray_dirs_close_batch=torch.cat(raydirs_close_batch_list,0)

#     #ray diff in the same way that ibr gets it 
#     ray_dirs_fullres=torch.from_numpy(frame_full_res.ray_dirs).to("cuda").float().view(1, frame_full_res.height, frame_full_res.width, 3).permute(0,3,1,2)
#     raydirs_close_fullres_batch_list=[]
#     for frame_close in frames_close_full_res:
#         # print("frame_close, hw", frame_close.height, " ", frame_close.width)
#         # print("frame_full_res, hw", frame_full_res.height, " ", frame_full_res.width)
#         ray_dirs_close=torch.from_numpy(frame_close.ray_dirs).to("cuda").float().view(1, frame_full_res.height, frame_full_res.width, 3).permute(0,3,1,2)
#         raydirs_close_fullres_batch_list.append(ray_dirs_close)
#     ray_dirs_fullres_close_batch=torch.cat(raydirs_close_fullres_batch_list,0)
#     ray_diff = compute_angle(frame_full_res.height, frame_full_res.width, ray_dirs_fullres, ray_dirs_fullres_close_batch)

#     return rgb_gt_fullres, rgb_gt, ray_dirs, rgb_close_batch, ray_dirs_close_batch, ray_diff


def prepare_data(frame, frames_close):
    rgb_gt=mat2tensor(frame.frame.rgb_32f, False).to("cuda")
    ray_dirs=torch.from_numpy(frame.ray_dirs).to("cuda").float().view(1, frame.height, frame.width, 3).permute(0,3,1,2)
    rgb_close_batch_list=[]
    for frame_close in frames_close:
        rgb_close_frame=mat2tensor(frame_close.frame.rgb_32f, False).to("cuda")
        rgb_close_batch_list.append(rgb_close_frame)
    rgb_close_batch=torch.cat(rgb_close_batch_list,0)
    #make also a batch fo directions
    raydirs_close_batch_list=[]
    for frame_close in frames_close:
        ray_dirs_close=torch.from_numpy(frame_close.ray_dirs).to("cuda").float().view(1, frame.height, frame.width, 3).permute(0,3,1,2)
        # ray_dirs_close=ray_dirs_close.view(1, frame.height, frame.width, 3)
        # ray_dirs_close=ray_dirs_close.permute(0,3,1,2) #from N,H,W,C to N,C,H,W
        raydirs_close_batch_list.append(ray_dirs_close)
    ray_dirs_close_batch=torch.cat(raydirs_close_batch_list,0)

    
    ray_diff = compute_angle(frame.height, frame.width, ray_dirs, rgb_close_batch)

    return  rgb_gt, ray_dirs, rgb_close_batch, ray_dirs_close_batch, ray_diff



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


def get_close_frames_barycentric(frame_py, all_frames_py_list, discard_same_idx, sphere_center, sphere_radius, triangulation_type):

    if discard_same_idx:
        frame_centers, frame_idxs = frames_to_points(all_frames_py_list, discard_frame_with_idx=frame_py.frame_idx)
    else:
        frame_centers, frame_idxs = frames_to_points(all_frames_py_list )

    if triangulation_type=="sphere":
        triangulated_mesh=SFM.compute_triangulation_stegreographic( frame_centers, sphere_center, sphere_radius )
    elif triangulation_type=="plane":
        triangulated_mesh=SFM.compute_triangulation_plane( frame_centers )
    else:
        print("triangulation type ", triangulation_type, " is not a valid type"  )

    face, weights= SFM.compute_closest_triangle( frame_py.frame.pos_in_world(), triangulated_mesh )

    #sort the face indices and the weight in order of the weights
    sorted_indices =np.argsort(-1*weights) #the -1* is just a trick to make it sort in descending order because god forbid numpy would have an option to just sort in descending...
    weights=weights[sorted_indices]
    face=face[sorted_indices]






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

def create_loader(dataset_name, config_path):
    if(dataset_name=="volref"):
        loader_train=DataLoaderVolRef(config_path)
        loader_test=DataLoaderVolRef(config_path)
        # loader_train.set_mode_train()
        # loader_test.set_mode_test()
        loader_train.start()
        loader_test.start()
    elif(dataset_name=="nerf_lego"):
        loader_train=DataLoaderNerf(config_path)
        loader_test=DataLoaderNerf(config_path)
        loader_train.set_mode_train()
        loader_test.set_mode_test()
        loader_train.start()
        loader_test.start()
    elif(dataset_name=="easypbr"):
        loader_train=DataLoaderEasyPBR(config_path)
        loader_test=DataLoaderEasyPBR(config_path)
        loader_train.set_mode_train()
        loader_test.set_mode_test()
        loader_train.start()
        loader_test.start()
    elif dataset_name=="colmap":
        loader_train=DataLoaderColmap(config_path)
        loader_test=DataLoaderColmap(config_path)
        loader_train.set_mode_train()
        loader_test.set_mode_test()
        loader_train.start()
        loader_test.start()
    elif dataset_name=="shapenetimg":
        loader_train=DataLoaderShapeNetImg(config_path)
        loader_test=DataLoaderShapeNetImg(config_path)
        loader_train.set_mode_train()
        loader_test.set_mode_test()
        loader_train.start()
        loader_test.start()
        #wait until we have data
        while True:
            if( loader_train.finished_reading_scene() and  loader_test.finished_reading_scene() ): 
                break
    elif dataset_name=="srn":
        loader_train=DataLoaderSRN(config_path)
        loader_test=DataLoaderSRN(config_path)
        loader_train.set_mode_train()
        loader_test.set_mode_test()
        loader_train.start()
        loader_test.start()
        #wait until we have data
        while True:
            if( loader_train.finished_reading_scene() and  loader_test.finished_reading_scene() ): 
                break
    elif dataset_name=="dtu":
        loader_train=DataLoaderDTU(config_path)
        loader_test=DataLoaderDTU(config_path)
        loader_train.set_mode_train()
        loader_test.set_mode_validation() ###We use the validation as test becuase there is no actualy test set
        loader_train.start()
        loader_test.start()
        #wait until we have data
        while True:
            if( loader_train.finished_reading_scene() and  loader_test.finished_reading_scene() ): 
                break
    elif(dataset_name=="llff"):
        loader_train=DataLoaderLLFF(config_path)
        loader_test=DataLoaderLLFF(config_path)
        loader_train.set_mode_train()
        loader_test.set_mode_test()
        loader_train.start()
        loader_test.start()
    else:
        err="Datset name not recognized. It is " + dataset_name
        sys.exit(err)

    return loader_train, loader_test

class FramePY():
    def __init__(self, frame, create_subsamples=False):
        #get mask 
        self.frame=frame
        self.create_subsamples=create_subsamples
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

        #Ray direction in world coordinates
        self.ray_dirs=None
        if not frame.is_shell:
            ray_dirs_mesh=self.frame.pixels2dirs_mesh()
            self.ray_dirs=ray_dirs_mesh.V.copy() #Nx3
 
        if not frame.is_shell:
            self.load_image_tensors()

            # if frame.mask.empty():
            #     mask_tensor= torch.ones((1,1,frame.height,frame.width))
            #     self.frame.mask=tensor2mat(mask_tensor)
            # #weight and hegiht
            # # self.height=self.rgb_tensor.shape[2]
            # # self.width=self.rgb_tensor.shape[3]
            # self.height=frame.height
            # self.width=frame.width
            # #CHECK that the frame width and hegiht has the same values as the rgb 
            # if frame.height!=frame.rgb_32f.rows or  frame.width!=frame.rgb_32f.cols:
            #     print("frame dimensions and rgb32 doesnt match. frame.height", frame.height, " frame.rgb_32f.rows", frame.rgb_32f.rows, " frame.width ", frame.width, " frame.rgb_32f.cols ", frame.rgb_32f.cols)
            # #Ray direction in world coordinates
            # ray_dirs_mesh=frame.pixels2dirs_mesh()
            # # self.ray_dirs=torch.from_numpy(ray_dirs_mesh.V.copy()).to("cuda").float() #Nx3
            # self.ray_dirs=ray_dirs_mesh.V.copy() #Nx3

        # self.frame.rgb_32f=tensor2mat(rgb_tensor)
        #get tf and K
        self.tf_cam_world=frame.tf_cam_world
        self.K=frame.K
        self.R_tensor=torch.from_numpy( frame.tf_cam_world.linear() ).to("cuda")
        self.t_tensor=torch.from_numpy( frame.tf_cam_world.translation() ).to("cuda")
        self.K_tensor = torch.from_numpy( frame.K ).to("cuda")
        #misc
        self.frame_idx=frame.frame_idx
        # self.loader=loader
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
        if create_subsamples and not frame.is_shell:
            self.subsampled_frames=[]
            for i in range(5):
                if i==0:
                    frame_subsampled=frame.subsample(2.0)
                else:
                    frame_subsampled=frame_subsampled.subsample(2.0)
                self.subsampled_frames.append(FramePY(frame_subsampled, create_subsamples=False))
            #every subsampled frame should have also as subsampled frames the ones that don't include it
            i=1
            for frame_subsampled in self.subsampled_frames:
                frame_subsampled.subsampled_frames = self.subsampled_frames[i:] #get the list without the n first elements
                i+=1



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

    def load_images(self):
        if self.frame.is_shell:
            self.frame.load_images() 

            #load the img tensors
            self.load_image_tensors()

            ray_dirs_mesh=self.frame.pixels2dirs_mesh()
            self.ray_dirs=ray_dirs_mesh.V.copy() #Nx3

        if self.create_subsamples and not self.frame.is_shell:
            self.subsampled_frames=[]
            for i in range(5):
                if i==0:
                    frame_subsampled=self.frame.subsample(2.0)
                else:
                    frame_subsampled=frame_subsampled.subsample(2.0)
                self.subsampled_frames.append(FramePY(frame_subsampled, create_subsamples=False))
            #every subsampled frame should have also as subsampled frames the ones that don't include it
            i=1
            for frame_subsampled in self.subsampled_frames:
                frame_subsampled.subsampled_frames = self.subsampled_frames[i:] #get the list without the n first elements
                i+=1

    def load_image_tensors(self):
        if self.frame.mask.empty():
            mask_tensor= torch.ones((1,1,self.frame.height,self.frame.width))
            self.frame.mask=tensor2mat(mask_tensor)
        #weight and hegiht
        # self.height=self.rgb_tensor.shape[2]
        # self.width=self.rgb_tensor.shape[3]
        self.height=self.frame.height
        self.width=self.frame.width
        #CHECK that the frame width and hegiht has the same values as the rgb 
        if self.frame.height!=self.frame.rgb_32f.rows or  self.frame.width!=self.frame.rgb_32f.cols:
            print("frame dimensions and rgb32 doesnt match. frame.height", self.frame.height, " frame.rgb_32f.rows", self.frame.rgb_32f.rows, " frame.width ", self.frame.width, " frame.rgb_32f.cols ", self.frame.rgb_32f.cols)
        
        


#some default arguments are in here https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/demo.sh

class EDSR_args():
    n_resblocks=4
    n_feats=128
    scale=2
    res_scale=0.1
    rgb_range=255
    n_in_channels=3
    n_out_channels=3

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


def map_range( input_val, input_start, input_end,  output_start,  output_end):
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    # input_clamped=max(input_start, min(input_end, input_val))
    input_clamped=torch.clamp(input_val, input_start, input_end)
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start)

def compute_uv(frame, points_3D_world):
    if points_3D_world.shape[1] != 3:
        print("expecting the points3d to be Nx3 but it is ", points_3D_world.shape)
        exit(1)
    if len(points_3D_world.shape) != 2:
        print("expecting the points3d to have 2 dimensions corresponding to Nx3 but it is ", points_3D_world.shape)
        exit(1)

    # R=torch.from_numpy( frame.tf_cam_world.linear() ).to("cuda")
    # t=torch.from_numpy( frame.tf_cam_world.translation() ).to("cuda")
    # K = torch.from_numpy( frame.K ).to("cuda")

    R=frame.R_tensor
    t=frame.t_tensor
    K = frame.K_tensor

    # points_3D_cam=torch.matmul(R, points_3D_world.transpose(0,1) ).transpose(0,1)  + t.view(1,3)
    # points_screen = torch.matmul(K, points_3D_cam.transpose(0,1) ).transpose(0,1)  


    points_3D_cam=torch.matmul(R, points_3D_world.transpose(0,1).contiguous() ).transpose(0,1).contiguous()  + t.view(1,3)
    points_screen = torch.matmul(K, points_3D_cam.transpose(0,1).contiguous() ).transpose(0,1)  


    # print("points_screen ", points_screen)
    points_2d = points_screen[:, 0:2] / ( points_screen[:, 2:3] +0.0001 )
    # print("points_2d before flip ", points_2d)

    points_2d[:,1] = frame.height- points_2d[:,1] 
    # print("points_2d ", points_2d)

    #shift by half a pixel 
    # points_2d[:,1]=points_2d[:,1]+0.5

    #get in range 0,1
    points_2d[:,0]  = points_2d[:,0]/frame.width 
    points_2d[:,1]  = points_2d[:,1]/frame.height 
    uv_tensor = points_2d
    # print("uv_tensor is ", uv_tensor)
    # exit(1)

    #may be needed 
    uv_tensor= uv_tensor*2 -1 #get in range [-1,1]
    # uv_tensor[:,1]=-uv_tensor[:,1] #flip


    return uv_tensor

# def compute_uv_batched(frames_list, points_3D_world):
def compute_uv_batched(R_batched, t_batched, K_batched, height, width, points_3D_world):
    
    #get the points from N,3,H,W to nr_points,3
    h = points_3D_world.shape[2]
    w = points_3D_world.shape[3]
    nr_frames =  R_batched.shape[0]
    points_3D_world = nchw2lin(points_3D_world)

    if points_3D_world.shape[1] != 3:
        print("expecting the points3d to be Nx3 but it is ", points_3D_world.shape)
        exit(1)
    if len(points_3D_world.shape) != 2:
        print("expecting the points3d to have 2 dimensions corresponding to Nx3 but it is ", points_3D_world.shape)
        exit(1)



    # TIME_START("repeat")
    feat_sliced_per_frame=[]
    feat_sliced_per_frame_manual=[]
    # points3d_world_for_uv=points_3D_world.view(1,-1,3).repeat( len(frames_list) ,1, 1) #Make it into NR_frames x N x 3
    points3d_world_for_uv=points_3D_world.view(1,-1,3).repeat( R_batched.shape[0] ,1, 1) #Make it into NR_frames x N x 3
    # TIME_END("repeat")
    # TIME_START("concat")
    # # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #     R_list=[]
    #     t_list=[]
    #     K_list=[]
    #     for i in range(len(frames_list)):
    #         frame=frames_list[i]
    #         R_list.append( frame.R_tensor.view(1,3,3) )
    #         t_list.append( frame.t_tensor.view(1,1,3) )
    #         K_list.append( frame.K_tensor.view(1,3,3) )
    #     R_batched=torch.cat(R_list,0)
    #     t_batched=torch.cat(t_list,0)
    #     K_batched=torch.cat(K_list,0)
    # # print(prof)
    # TIME_END("concat")
    #project 
    # TIME_START("proj")
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    points_3D_cam=torch.matmul(points3d_world_for_uv, R_batched.transpose(1,2) )  + t_batched
    points_screen = torch.matmul(points_3D_cam, K_batched.transpose(1,2) )  
    points_2d = points_screen[:, :, 0:2] / ( points_screen[:, :, 2:3] +0.0001 )
    # points_2d[:,:,1] = height- points_2d[:,:,1] 

    mask = (points_2d[..., 0] <= width - 1.) & \
               (points_2d[..., 0] >= 0) & \
               (points_2d[..., 1] <= height - 1.) &\
               (points_2d[..., 1] >= 0)
    mask=mask.unsqueeze(2)


    #get in range 0,1
    scaling = torch.tensor([width, height]).cuda().view(1,1,2)   ######WATCH out we assume that all the frames are the same width and height
    uv_tensor=points_2d.div(scaling)
    # #may be needed 
    uv_tensor= uv_tensor*2 -1 #get in range [-1,1]


    #attempt 2
    # scaling = torch.tensor([1.0/width*2, 1.0/height*2]).cuda().view(1,1,2)   ######WATCH out we assume that all the frames are the same width and height
    # minus_one= torch.tensor([-1.0]).cuda().view(1,1,1)
    # uv_tensor = torch.addcmul(minus_one, points_2d,scaling) #-1+ points2d*scaling

    # print(prof)
    # TIME_END("proj")

    uv_tensor=uv_tensor.view(nr_frames, h, w, 2)

    # print("uv_tensor has shape ", uv_tensor.shape)


    return uv_tensor, mask

def compute_uv_batched_original(frames_list,  points_3D_world):
    if points_3D_world.shape[1] != 3:
        print("expecting the points3d to be Nx3 but it is ", points_3D_world.shape)
        exit(1)
    if len(points_3D_world.shape) != 2:
        print("expecting the points3d to have 2 dimensions corresponding to Nx3 but it is ", points_3D_world.shape)
        exit(1)



    feat_sliced_per_frame=[]
    feat_sliced_per_frame_manual=[]
    points3d_world_for_uv=points_3D_world.view(1,-1,3).repeat( len(frames_list) ,1, 1) #Make it into NR_frames x N x 3
    R_list=[]
    t_list=[]
    K_list=[]
    for i in range(len(frames_list)):
        frame=frames_list[i]
        R_list.append( frame.R_tensor.view(1,3,3) )
        t_list.append( frame.t_tensor.view(1,1,3) )
        K_list.append( frame.K_tensor.view(1,3,3) )
    R_batched=torch.cat(R_list,0)
    t_batched=torch.cat(t_list,0)
    K_batched=torch.cat(K_list,0)
    #project 
    points_3D_cam=torch.matmul(R_batched, points3d_world_for_uv.transpose(1,2) ).transpose(1,2)  + t_batched
    points_screen = torch.matmul(K_batched, points_3D_cam.transpose(1,2) ).transpose(1,2)  
    points_2d = points_screen[:, :, 0:2] / ( points_screen[:, :, 2:3] +0.0001 )
    points_2d[:,:,1] = frame.height- points_2d[:,:,1] 


    #get in range 0,1
    points_2d[:,:,0]  = points_2d[:,:,0]/frame.width  ######WATCH out we assume that all the frames are the same width and height
    points_2d[:,:,1]  = points_2d[:,:,1]/frame.height 
    uv_tensor = points_2d

    #may be needed 
    uv_tensor= uv_tensor*2 -1 #get in range [-1,1]


    return uv_tensor

def compute_normal(points3d_img):
    assert len(points3d_img.shape) == 4, points3d_img.shape

    #since the gradient is not defined at the boders, we pad the image with zeros
    # points3d_img_padded = torch.nn.functional.pad(points3d_img, [0,1,0,1], mode='constant', value=0)
    points3d_img_x= torch.nn.functional.pad(points3d_img, [0,1,0,0], mode='constant', value=0)
    points3d_img_y= torch.nn.functional.pad(points3d_img, [0,0,0,1], mode='constant', value=0)
    # print("points3d_img_padded", points3d_img_padded.shape)

    grad_x=points3d_img_x[:, :, :, :-1] - points3d_img_x[:, :, :, 1:]
    grad_y=points3d_img_y[:, :, :-1, :] - points3d_img_y[:, :, 1:, :]
    # print("grad x is ", grad_x.shape)

    #make the gradx and grady in Nx3
    height=points3d_img.shape[2]
    width=points3d_img.shape[3]
    nr_channels=points3d_img.shape[1]
    grad_x=grad_x.permute(0,2,3,1) #from N,C,H,W to N,H,W,C
    grad_y=grad_y.permute(0,2,3,1) #from N,C,H,W to N,H,W,C
    grad_x=grad_x.view(-1, nr_channels)
    grad_y=grad_y.view(-1, nr_channels)
    # cross=torch.cross(grad_x, grad_y, dim=1)
    cross=torch.cross(grad_y, grad_x, dim=1)
    # print("corss x is ", cross.shape)

    normal_norm=cross.norm(dim=1, keepdim=True)
    normal=cross/(normal_norm+0.00001)
    # print("normal is ", normal.shape)


    #make into image
    normal=normal.view(-1, height, width, nr_channels)
    normal=normal.permute(0,3,1,2) #from N,H,W,C to N,C,H,W

    return normal

def frames_to_points(frames, discard_frame_with_idx=None ):

    points_list=[]
    points_frame_idxs_list=[]
    for i in range(len(frames)):
        frame=frames[i]
        if discard_frame_with_idx!=None and frame.frame_idx==discard_frame_with_idx:
            continue
        points_list.append( frame.frame.pos_in_world()  )
        points_frame_idxs_list.append( frame.frame_idx )
            
    points = np.asarray(points_list)
    points_frame_idxs = np.asarray(points_frame_idxs_list)

    return points, points_frame_idxs

#from autoclip https://github.com/pseeth/autoclip/blob/master/autoclip.py
def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 

def fused_mean_variance(x, weight, dim_reduce, dim_concat, use_weights=True):
    if use_weights:
        mean = torch.sum(x*weight, dim=dim_reduce, keepdim=True)
        var = torch.sum(weight * (x - mean)**2, dim=dim_reduce, keepdim=True)
    else:
        mean = torch.mean(x, dim=dim_reduce, keepdim=True)
        var = torch.sum( (x - mean)**2, dim=dim_reduce, keepdim=True)
    mean_var=torch.cat([mean,var], dim_concat )
    return mean_var



#NDC to xyz conversions
#from NERF paper
def xyz_and_dirs2ndc (H , W , fx, fy , near , rays_o , rays_d, project_to_near):
    # Shift ray origins to near plane
    if project_to_near:
        t = -( near + rays_o [... , 2]) / rays_d [... , 2]
        rays_o = rays_o + t[... , None ] * rays_d
    # Projection
    o0 = -1./(W/( 2.* fx ) ) * rays_o [... , 0] / rays_o [... , 2]
    o1 = -1./(H/( 2.* fy ) ) * rays_o [... , 1] / rays_o [... , 2]
    o2 = 1. + 2. * near / rays_o [... , 2]
    d0 = -1./(W/( 2.* fx ) ) * ( rays_d [... , 0]/ rays_d [... , 2] - \
    rays_o [... , 0]/ rays_o [... , 2])
    d1 = -1./(H/( 2.* fy ) ) * ( rays_d [... , 1]/ rays_d [... , 2] - \
    rays_o [... , 1]/ rays_o [... , 2])
    d2 = -2. * near / rays_o [... , 2]
    # print("o0", o0.shape)
    # print("d0", d0.shape)
    # rays_o = tf . stack ([o0 ,o1 , o2], -1)
    # rays_d = tf . stack ([d0 ,d1 , d2], -1)
    rays_o = torch.cat([ o0.unsqueeze(1), o1.unsqueeze(1), o2.unsqueeze(1)    ], 1)
    # print("rays_o", rays_o.shape)
    rays_d = torch.cat([ d0.unsqueeze(1), d1.unsqueeze(1), d2.unsqueeze(1)    ], 1)
    return rays_o , rays_d

#From  https://github.com/bmild/nerf/issues/35
def ndc2xyz(H , W , fx, fy , near , rays_o):

    x_ndc = rays_o[:, 0:1]
    y_ndc = rays_o[:, 1:2]
    z_ndc = rays_o[:, 2:3]
    # print("z_ndc is ", z_ndc)
    # z = 2 / (z_ndc - 1)
    z = 2* near / (z_ndc - 1)
    # z = 1 / (1-z_ndc )
    x = -x_ndc * z * W / 2 / fx
    y = -y_ndc * z * H / 2 / fy
    points_xyz= torch.cat([x,y,z],1)

    return points_xyz


#get the poses from the frames into a vector of Nr_frames, 3x5 where the 3x5 is composed of a 3x4 tf_world_cam matrix which os concatenated with a colum of h,w,focal_length. More about this format is in here: https://github.com/Fyusion/LLFF
def get_frames_to_poses_llff(frames):

    poses_list=[]

    for i in range(len(frames)):
        frame=frames[i].frame
        tf_cam_world=frame.tf_cam_world
        # tf_cam_world.flip_x()
        tf_world_cam = tf_cam_world.inverse()
        tf_world_cam = tf_world_cam.matrix()
        print("tf_world_cam", tf_world_cam)
        tf_world_cam = tf_world_cam[0:3, :]
        # tf_world_cam[:, 2:3] = -tf_world_cam[:, 2:3] 
        print("tf_world_cam 3x4", tf_world_cam)
        hwf = np.array([ frame.height, frame.width, (frame.K[0,0]+frame.K[1,1])*0.5   ])
        print("hwf is ", hwf) 
        print(" tf_world_cam ", tf_world_cam.shape)
        hwf = hwf.reshape((3, 1))
        print(" hwf ", hwf.shape)
        tf_world_cam_hwf = np.concatenate([ tf_world_cam, hwf ], 1)
        print("tf_world_cam_hwf", tf_world_cam_hwf)
        tf_world_cam_hwf= tf_world_cam_hwf.reshape((1,3,5))
        print("tf_world_cam_hwf", tf_world_cam_hwf)
        print("tf_world_cam_hwf", tf_world_cam_hwf.shape)
        poses_list.append(tf_world_cam_hwf)


        print("------------------------------")

    poses= np.concatenate(poses_list, 0)

    print("poses", poses.shape)

    return poses




#from the ibr net code  https://github.com/googleinterns/IBRNet/blob/6c43eb3c83e1c8e851ef9b66364cd30647de16e2/ibrnet/data_loaders/llff_data_utils.py#L200
def make_list_of_poses_on_spiral(frames, path_zflat):

    poses= get_frames_to_poses_llff(frames)

    c2w = poses_avg(poses)
    # print('recentered', c2w.shape)
    # print(c2w[:3, :4])

    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    # close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    close_depth, inf_depth = frames[0].frame.get_extra_field_float("min_near") * .9, frames[0].frame.get_extra_field_float("max_far") * 5.
    # dt = .75 #ho far away have the look at point in the scene
    dt = 1.0 #ho far away have the look at point in the scene
    # dt = 1.0
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    # zdelta = 0.0
    tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    # rads = np.percentile(np.abs(tt), 70, 0) #it you push it to 90 then it would cover almost 90 of the space and move almost the edge of all the frames that you recorder. A lower values make it moves less on the edges
    rads = np.percentile(np.abs(tt), 60, 0) #it you push it to 90 then it would cover almost 90 of the space and move almost the edge of all the frames that you recorder. A lower values make it moves less on the edges
    c2w_path = c2w
    N_views = 120
    N_rots = 2
    if path_zflat:
        #             zloc = np.percentile(tt, 10, 0)[2]
        zloc = -close_depth * .1
        c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
        rads[2] = 0.
        N_rots = 1
        N_views /= 2

    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

    # print("render_poses[0]", render_poses[0].shape)


    return render_poses

def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    # center[2]=0.0
    # print("center is ", center)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)


    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, int(N + 1) )[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, focal, 1.])))
        # print("focal points in wolrd", np.dot(c2w[:3, :4], np.array([0, 0, focal, 1.]))  )
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = - normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    # print("xdotz", np.dot(normalize(vec2), normalize(vec0) ) )
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m



#from ibrnet  https://github.com/googleinterns/IBRNet/blob/6c43eb3c83e1c8e851ef9b66364cd30647de16e2/ibrnet/projection.py#L64
def compute_angle(height, width, ray_dirs, ray_dirs_close_batch):

    # ray_dirs is N,3,H,W
    # ray_dirs_close_batch is N,3,H,W

    # return: [n_views, ..., 4]; The first 3 channels are unit-length vector of the difference between
        # query and target ray directions, the last channel is the inner product of the two directions.
    
    # original_shape = xyz.shape[:2]
    # xyz = xyz.reshape(-1, 3)
    # train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
    # num_views = len(train_poses)
    # query_pose = query_camera[-16:].reshape(-1, 4, 4).repeat(num_views, 1, 1)  # [n_views, 4, 4]
    ray2tar_pose = nchw2nXc(ray_dirs)
    # ray2tar_pose = ray2tar_pose / (torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)
    ray2train_pose =  nchw2nXc(ray_dirs_close_batch)
    # ray2train_pose = ray2train_pose/ (torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6)
    ray_diff = ray2tar_pose - ray2train_pose
    ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
    ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)
    ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
    ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
    ray_diff = nXc2nchw(ray_diff, height, width)
    # print("ray_diff is  ", ray_diff.shape)
    return ray_diff






def check_args_shadowing(name, method, arg_names):
    spec = inspect.getfullargspec(method)
    init_args = {*spec.args, *spec.kwonlyargs}
    for arg_name in arg_names:
        if arg_name in init_args:
            raise TypeError(f"{name} attempted to shadow a wrapped argument: {arg_name}")


# For backward compatibility.
class TensorMappingHook(object):
    def __init__(
        self,
        name_mapping: List[Tuple[str, str]],
        expected_shape: Optional[Dict[str, List[int]]] = None,
    ):
        """This hook is expected to be used with "_register_load_state_dict_pre_hook" to
        modify names and tensor shapes in the loaded state dictionary.
        Args:
            name_mapping: list of string tuples
            A list of tuples containing expected names from the state dict and names expected
            by the module.
            expected_shape: dict
            A mapping from parameter names to expected tensor shapes.
        """
        self.name_mapping = name_mapping
        self.expected_shape = expected_shape if expected_shape is not None else {}

    def __call__(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for old_name, new_name in self.name_mapping:
            if prefix + old_name in state_dict:
                tensor = state_dict.pop(prefix + old_name)
                if new_name in self.expected_shape:
                    tensor = tensor.view(*self.expected_shape[new_name])
                state_dict[prefix + new_name] = tensor


def weight_norm_wrapper(cls, name="weight", g_dim=0, v_dim=0):
    """Wraps a torch.nn.Module class to support weight normalization. The wrapped class
    is compatible with the fuse/unfuse syntax and is able to load state dict from previous
    implementations.
    Args:
        name: str
        Name of the parameter to apply weight normalization.
        g_dim: int
        Learnable dimension of the magnitude tensor. Set to None or -1 for single scalar magnitude.
        Default values for Linear and Conv2d layers are 0s and for ConvTranspose2d layers are 1s.
        v_dim: int
        Of which dimension of the direction tensor is calutated independently for the norm. Set to
        None or -1 for calculating norm over the entire direction tensor (weight tensor). Default
        values for most of the WN layers are None to preserve the existing behavior.
    """

    class Wrap(cls):
        def __init__(self, *args, name=name, g_dim=g_dim, v_dim=v_dim, **kwargs):
            # Check if the extra arguments are overwriting arguments for the wrapped class
            check_args_shadowing(
                "weight_norm_wrapper", super().__init__, ["name", "g_dim", "v_dim"]
            )
            super().__init__(*args, **kwargs)

            # Sanitize v_dim since we are hacking the built-in utility to support
            # a non-standard WeightNorm implementation.
            if v_dim is None:
                v_dim = -1
            self.weight_norm_args = {"name": name, "g_dim": g_dim, "v_dim": v_dim}
            self.is_fused = True
            self.unfuse()

            # For backward compatibility.
            self._register_load_state_dict_pre_hook(
                TensorMappingHook(
                    [(name, name + "_v"), ("g", name + "_g")],
                    {name + "_g": getattr(self, name + "_g").shape},
                )
            )

        def fuse(self):
            if self.is_fused:
                return
            # Check if the module is frozen.
            param_name = self.weight_norm_args["name"] + "_g"
            if hasattr(self, param_name) and param_name not in self._parameters:
                raise ValueError("Trying to fuse frozen module.")
            remove_weight_norm(self, self.weight_norm_args["name"])
            self.is_fused = True

        def unfuse(self):
            if not self.is_fused:
                return
            # Check if the module is frozen.
            param_name = self.weight_norm_args["name"]
            if hasattr(self, param_name) and param_name not in self._parameters:
                raise ValueError("Trying to unfuse frozen module.")
            wn = WeightNorm.apply(
                self, self.weight_norm_args["name"], self.weight_norm_args["g_dim"]
            )
            # Overwrite the dim property to support mismatched norm calculate for v and g tensor.
            if wn.dim != self.weight_norm_args["v_dim"]:
                wn.dim = self.weight_norm_args["v_dim"]
                # Adjust the norm values.
                weight = getattr(self, self.weight_norm_args["name"] + "_v")
                norm = getattr(self, self.weight_norm_args["name"] + "_g")
                norm.data[:] = th.norm_except_dim(weight, 2, wn.dim)
            self.is_fused = False

        def __deepcopy__(self, memo):
            # Delete derived tensor to avoid deepcopy error.
            if not self.is_fused:
                delattr(self, self.weight_norm_args["name"])

            # Deepcopy.
            cls = self.__class__
            result = cls.__new__(cls)
            memo[id(self)] = result
            for k, v in self.__dict__.items():
                setattr(result, k, copy.deepcopy(v, memo))

            if not self.is_fused:
                setattr(result, self.weight_norm_args["name"], None)
                setattr(self, self.weight_norm_args["name"], None)
            return result

    return Wrap

def is_weight_norm_wrapped(module):
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, WeightNorm):
            return True
    return False

# Set default g_dim=0 (Conv2d) or 1 (ConvTranspose2d) and v_dim=None to preserve
# the current weight norm behavior.
LinearWN = weight_norm_wrapper(th.nn.Linear, g_dim=0, v_dim=None)
Conv1dWN = weight_norm_wrapper(th.nn.Conv1d, g_dim=0, v_dim=None)
# Conv1dWNUB = weight_norm_wrapper(Conv1dUB, g_dim=0, v_dim=None)
Conv2dWN = weight_norm_wrapper(th.nn.Conv2d, g_dim=0, v_dim=None)
# Conv2dWNUB = weight_norm_wrapper(Conv2dUB, g_dim=0, v_dim=None)
ConvTranspose1dWN = weight_norm_wrapper(th.nn.ConvTranspose1d, g_dim=1, v_dim=None)
# ConvTranspose1dWNUB = weight_norm_wrapper(ConvTranspose1dUB, g_dim=1, v_dim=None)
ConvTranspose2dWN = weight_norm_wrapper(th.nn.ConvTranspose2d, g_dim=1, v_dim=None)
# ConvTranspose2dWNUB = weight_norm_wrapper(ConvTranspose2dUB, g_dim=1, v_dim=None)




def leaky_relu_init(m, negative_slope=0.2):

    #mport here in rder to avoid circular dependency
    # from instant_ngp_2_py.lattice.lattice_modules import ConvLatticeIm2RowModule
    # from instant_ngp_2_py.lattice.lattice_modules import CoarsenLatticeModule 
    # from instant_ngp_2_py.lattice.lattice_modules import FinefyLatticeModule 


    gain = np.sqrt(2.0 / (1.0 + negative_slope ** 2))

    if isinstance(m, th.nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // 2
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // 8
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    # #LATTICE THINGS
    # elif isinstance(m, ConvLatticeIm2RowModule):
    #     print("init ConvLatticeIm2RowModule")
    #     # print("conv lattice weight is ", m.weight.shape)
    #     n1 = m.in_channels
    #     n2 = m.out_channels
    #     filter_extent=m.filter_extent
    #     # print("filter_extent", filter_extent)
    #     # print("n1", n1)
    #     std = gain * np.sqrt(2.0 / ((n1 + n2) * filter_extent))
    #     # return
    # elif isinstance(m, CoarsenLatticeModule):
    #     print("init CoarsenLatticeModule")
    #     n1 = m.in_channels
    #     n2 = m.out_channels
    #     filter_extent=m.filter_extent
    #     filter_extent=filter_extent//8
    #     std = gain * np.sqrt(2.0 / ((n1 + n2) * filter_extent))
    # elif isinstance(m, FinefyLatticeModule):
    #     print("init FinefyLatticeModule")
    #     n1 = m.in_channels
    #     n2 = m.out_channels
    #     filter_extent=m.filter_extent
    #     filter_extent=filter_extent//8
    #     #since coarsen usually hits empty space, the effective extent of it is actually smaller
    #     std = gain * np.sqrt(2.0 / ((n1 + n2) * filter_extent))
    else:
        return

    is_wnw = is_weight_norm_wrapped(m)
    if is_wnw:
        m.fuse()

    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
    if m.bias is not None:
        m.bias.data.zero_()

    if isinstance(m, th.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if is_wnw:
        m.unfuse()

    # m.weights_initialized=True


def apply_weight_init_fn(m, fn, negative_slope=1.0):

    should_initialize_weight=True
    if not hasattr(m, "weights_initialized"): #if we don't have this then we need to intiialzie
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    elif m.weights_initialized==False: #if we have it but it's set to false
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    else:
        print("skipping weight init on ", m)
        should_initialize_weight=False

    if should_initialize_weight:
        # fn(m, is_linear, scale)
        fn(m,negative_slope)
        # m.weights_initialized=True
        for module in m.children():
            apply_weight_init_fn(module, fn, negative_slope)



