import torch
import torch.nn as nn
import torch.nn.functional as F
from easypbr  import *
import sys
import math
from collections import namedtuple

from neural_mvs_py.neural_mvs.funcs import *


# from torchmeta.modules.conv import MetaConv2d
# from torchmeta.modules.linear import MetaLinear
# from torchmeta.modules.module import *
# from torchmeta.modules.utils import *
# from torchmeta.modules import (MetaModule, MetaSequential)

from neural_mvs_py.neural_mvs.pac import *
from neural_mvs_py.neural_mvs.deform_conv import *
from latticenet_py.lattice.lattice_modules import *


from dataloaders import *


DatasetParams = namedtuple('DatasetParams', 'sphere_radius sphere_center estimated_scene_center raymarch_depth_min raymarch_depth_jitter triangulation_type')


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
       estimated_scene_center =  sphere_center #for most of the datasets the scene is around the center of the sphere
    else: #if we deal with a LLFF datset we need to set our own estimate of the scene center
        estimated_scene_center= np.array([0,0,-0.3])
    print("sphere center and raidus ", sphere_center, " radius ", sphere_radius)

    #triangulation type is sphere for all datasets except llff
    triangulation_type="sphere"
    if isinstance(loader, DataLoaderLLFF):
        triangulation_type = "plane"

    raymarch_depth_min = 0.15
    raymarch_depth_jitter =  2e-2

    params= DatasetParams(sphere_radius=sphere_radius, 
                        sphere_center=sphere_center, 
                        estimated_scene_center=estimated_scene_center, 
                        raymarch_depth_min=raymarch_depth_min,
                        raymarch_depth_jitter = raymarch_depth_jitter,
                        triangulation_type= triangulation_type )

    return params


def prepare_data(frame_full_res, frame, frames_close):
    rgb_gt=mat2tensor(frame.frame.rgb_32f, False).to("cuda")
    rgb_gt_fullres=mat2tensor(frame_full_res.frame.rgb_32f, False).to("cuda")
    # mask_tensor=mat2tensor(frame.frame.mask, False).to("cuda")
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
        ray_dirs_close=ray_dirs_close.view(1, frame.height, frame.width, 3)
        ray_dirs_close=ray_dirs_close.permute(0,3,1,2) #from N,H,W,C to N,C,H,W
        raydirs_close_batch_list.append(ray_dirs_close)
    ray_dirs_close_batch=torch.cat(raydirs_close_batch_list,0)

    return rgb_gt_fullres, rgb_gt, ray_dirs, rgb_close_batch, ray_dirs_close_batch



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

    def load_images(self):
        if self.frame.is_shell:
            self.frame.load_images() 

            #load the img tensors
            self.load_image_tensors()

        if self.create_subsamples and not self.frame.is_shell:
            self.subsampled_frames=[]
            for i in range(2):
                if i==0:
                    frame_subsampled=self.frame.subsample(2)
                else:
                    frame_subsampled=frame_subsampled.subsample(2)
                self.subsampled_frames.append(FramePY(frame_subsampled, create_subsamples=False))

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
        #Ray direction in world coordinates
        ray_dirs_mesh=self.frame.pixels2dirs_mesh()
        # self.ray_dirs=torch.from_numpy(ray_dirs_mesh.V.copy()).to("cuda").float() #Nx3
        self.ray_dirs=ray_dirs_mesh.V.copy() #Nx3
        
        


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
    points_2d[:,:,1] = height- points_2d[:,:,1] 

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

def fused_mean_variance(x, weight, dim, use_weights=True):
    if use_weights:
        mean = torch.sum(x*weight, dim=dim, keepdim=True)
        var = torch.sum(weight * (x - mean)**2, dim=dim, keepdim=True)
    else:
        mean = torch.sum(x, dim=dim, keepdim=True)
        var = torch.sum( (x - mean)**2, dim=dim, keepdim=True)
    mean_var=torch.cat([mean,var],-1)
    return mean_var