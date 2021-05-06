import torch
import torch.nn as nn
import torch.nn.functional as F
from easypbr  import *
import sys
import math

from neural_mvs_py.neural_mvs.funcs import *


from torchmeta.modules.conv import MetaConv2d
from torchmeta.modules.linear import MetaLinear
from torchmeta.modules.module import *
from torchmeta.modules.utils import *
from torchmeta.modules import (MetaModule, MetaSequential)

from neural_mvs_py.neural_mvs.pac import *
from neural_mvs_py.neural_mvs.deform_conv import *
from latticenet_py.lattice.lattice_modules import *


from dataloaders import *



# def get_close_frames(loader, frame_py, all_frames_py_list, nr_frames_close, discard_same_idx):
#     frames_close=loader.get_close_frames(frame_py.frame, nr_frames_close, discard_same_idx)
#     # print("frames_close in py is ", len(frames_close))
#     # print("all_frames_py_list", len(all_frames_py_list))
#     #fromt his frame close get the frames from frame_py_list with the same indexes
#     frames_selected=[]
#     for frame in frames_close:
#         frame_idx=frame.frame_idx
#         # print("looking for a frame with frame_idx", frame_idx)
#         #find in all_frames_py_list the one with this frame idx
#         for frame_py in all_frames_py_list:
#             # print("cur frame has frmaeidx", frame_py.frame_idx)
#             if frame_py.frame_idx==frame_idx:
#                 frames_selected.append(frame_py)

#     return frames_selected


# def get_close_frames_barycentric(frame_py, all_frames_py_list, discard_same_idx, sphere_center, sphere_radius, triangulation_type):

#     if discard_same_idx:
#         frame_centers, frame_idxs = frames_to_points(all_frames_py_list, discard_frame_with_idx=frame_py.frame_idx)
#     else:
#         frame_centers, frame_idxs = frames_to_points(all_frames_py_list )

#     if triangulation_type=="sphere":
#         triangulated_mesh=SFM.compute_triangulation_stegreographic( frame_centers, sphere_center, sphere_radius )
#     elif triangulation_type=="plane":
#         triangulated_mesh=SFM.compute_triangulation_plane( frame_centers )
#     else:
#         print("triangulation type ", triangulation_type, " is not a valid type"  )

#     face, weights= SFM.compute_closest_triangle( frame_py.frame.pos_in_world(), triangulated_mesh )

#     #sort the face indices and the weight in order of the weights
#     sorted_indices =np.argsort(-1*weights) #the -1* is just a trick to make it sort in descending order because god forbid numpy would have an option to just sort in descending...
#     weights=weights[sorted_indices]
#     face=face[sorted_indices]






#     #from the face get the vertices that we triangulated
#     # print("frame idx ", frame_idx_0, frame_idx_1, frame_idx_2 )
#     frame_idx_0= frame_idxs[face[0]]
#     frame_idx_1= frame_idxs[face[1]]
#     frame_idx_2= frame_idxs[face[2]]

#     selected_frames=[]
#     frame_0=None
#     frame_1=None
#     frame_2=None
#     #We have to add them in the same order, so first frame0 andthen 1 and then 2
#     for frame in all_frames_py_list:
#         if frame.frame_idx==frame_idx_0:
#             frame_0=frame
#         if frame.frame_idx==frame_idx_1:
#             frame_1=frame
#         if frame.frame_idx==frame_idx_2:
#             frame_2=frame
#     selected_frames.append(frame_0)
#     selected_frames.append(frame_1)
#     selected_frames.append(frame_2)
    

#     return selected_frames, weights

# def create_loader(dataset_name, config_path):
#     if(dataset_name=="volref"):
#         loader_train=DataLoaderVolRef(config_path)
#         loader_test=DataLoaderVolRef(config_path)
#         # loader_train.set_mode_train()
#         # loader_test.set_mode_test()
#         loader_train.start()
#         loader_test.start()
#     elif(dataset_name=="nerf_lego"):
#         loader_train=DataLoaderNerf(config_path)
#         loader_test=DataLoaderNerf(config_path)
#         loader_train.set_mode_train()
#         loader_test.set_mode_test()
#         loader_train.start()
#         loader_test.start()
#     elif(dataset_name=="easypbr"):
#         loader_train=DataLoaderEasyPBR(config_path)
#         loader_test=DataLoaderEasyPBR(config_path)
#         loader_train.set_mode_train()
#         loader_test.set_mode_test()
#         loader_train.start()
#         loader_test.start()
#     elif dataset_name=="colmap":
#         loader_train=DataLoaderColmap(config_path)
#         loader_test=DataLoaderColmap(config_path)
#         loader_train.set_mode_train()
#         loader_test.set_mode_test()
#         loader_train.start()
#         loader_test.start()
#     elif dataset_name=="shapenetimg":
#         loader_train=DataLoaderShapeNetImg(config_path)
#         loader_test=DataLoaderShapeNetImg(config_path)
#         loader_train.set_mode_train()
#         loader_test.set_mode_test()
#         loader_train.start()
#         loader_test.start()
#         #wait until we have data
#         while True:
#             if( loader_train.finished_reading_scene() and  loader_test.finished_reading_scene() ): 
#                 break
#     elif dataset_name=="srn":
#         loader_train=DataLoaderSRN(config_path)
#         loader_test=DataLoaderSRN(config_path)
#         loader_train.set_mode_train()
#         loader_test.set_mode_test()
#         loader_train.start()
#         loader_test.start()
#         #wait until we have data
#         while True:
#             if( loader_train.finished_reading_scene() and  loader_test.finished_reading_scene() ): 
#                 break
#     elif dataset_name=="dtu":
#         loader_train=DataLoaderDTU(config_path)
#         loader_test=DataLoaderDTU(config_path)
#         loader_train.set_mode_train()
#         loader_test.set_mode_validation() ###We use the validation as test becuase there is no actualy test set
#         loader_train.start()
#         loader_test.start()
#         #wait until we have data
#         while True:
#             if( loader_train.finished_reading_scene() and  loader_test.finished_reading_scene() ): 
#                 break
#     elif(dataset_name=="llff"):
#         loader_train=DataLoaderLLFF(config_path)
#         loader_test=DataLoaderLLFF(config_path)
#         loader_train.set_mode_train()
#         loader_test.set_mode_test()
#         loader_train.start()
#         loader_test.start()
#     else:
#         err="Datset name not recognized. It is " + dataset_name
#         sys.exit(err)

#     return loader_train, loader_test

# class FramePY():
#     def __init__(self, frame, create_subsamples=False):
#         #get mask 
#         self.frame=frame
#         self.create_subsamples=create_subsamples
#         # #We do NOT store the tensors on the gpu and rather load them whenever is endessary. This is because we can have many frames and can easily run out of memory
#         # if not frame.mask.empty():
#         #     mask_tensor=mat2tensor(frame.mask, False).to("cuda").repeat(1,3,1,1)
#         #     self.frame.mask=frame.mask
#         # else:
#         #     mask_tensor= torch.ones((1,1,frame.height,frame.width), device=torch.device("cuda") )
#         #     self.frame.mask=tensor2mat(mask_tensor)
#         # #get rgb with mask applied 
#         # rgb_tensor=mat2tensor(frame.rgb_32f, False).to("cuda")
#         # rgb_tensor=rgb_tensor*mask_tensor

 
#         if not frame.is_shell:
#             self.load_image_tensors()

#             # if frame.mask.empty():
#             #     mask_tensor= torch.ones((1,1,frame.height,frame.width))
#             #     self.frame.mask=tensor2mat(mask_tensor)
#             # #weight and hegiht
#             # # self.height=self.rgb_tensor.shape[2]
#             # # self.width=self.rgb_tensor.shape[3]
#             # self.height=frame.height
#             # self.width=frame.width
#             # #CHECK that the frame width and hegiht has the same values as the rgb 
#             # if frame.height!=frame.rgb_32f.rows or  frame.width!=frame.rgb_32f.cols:
#             #     print("frame dimensions and rgb32 doesnt match. frame.height", frame.height, " frame.rgb_32f.rows", frame.rgb_32f.rows, " frame.width ", frame.width, " frame.rgb_32f.cols ", frame.rgb_32f.cols)
#             # #Ray direction in world coordinates
#             # ray_dirs_mesh=frame.pixels2dirs_mesh()
#             # # self.ray_dirs=torch.from_numpy(ray_dirs_mesh.V.copy()).to("cuda").float() #Nx3
#             # self.ray_dirs=ray_dirs_mesh.V.copy() #Nx3

#         # self.frame.rgb_32f=tensor2mat(rgb_tensor)
#         #get tf and K
#         self.tf_cam_world=frame.tf_cam_world
#         self.K=frame.K
#         self.R_tensor=torch.from_numpy( frame.tf_cam_world.linear() ).to("cuda")
#         self.t_tensor=torch.from_numpy( frame.tf_cam_world.translation() ).to("cuda")
#         self.K_tensor = torch.from_numpy( frame.K ).to("cuda")
#         #misc
#         self.frame_idx=frame.frame_idx
#         # self.loader=loader
#         #lookdir and cam center
#         self.camera_center=torch.from_numpy( frame.pos_in_world() ).to("cuda")
#         self.camera_center=self.camera_center.view(1,3)
#         self.look_dir=torch.from_numpy( frame.look_dir() ).to("cuda")
#         self.look_dir=self.look_dir.view(1,3)
#         #create tensor to store the bound in z near and zfar for every pixel of this image
#         # self.znear_zfar = torch.nn.Parameter(  torch.ones([1,2,self.height,self.width], dtype=torch.float32, device=torch.device("cuda"))  )
#         # with torch.no_grad():
#         #     self.znear_zfar[:,0,:,:]=znear
#         #     self.znear_zfar[:,1,:,:]=zfar
#         # self.znear_zfar.requires_grad=True
#         # self.cloud=frame.depth2world_xyz_mesh()
#         # self.cloud=frame.assign_color(self.cloud)
#         # self.cloud.remove_vertices_at_zero()

#         #make a list of subsampled frames
#         if create_subsamples and not frame.is_shell:
#             self.subsampled_frames=[]
#             for i in range(3):
#                 if i==0:
#                     frame_subsampled=frame.subsample(2)
#                 else:
#                     frame_subsampled=frame_subsampled.subsample(2)
#                 self.subsampled_frames.append(FramePY(frame_subsampled, create_subsamples=False))

#     def create_frustum_mesh(self, scale):
#         frame=Frame()
#         frame.K=self.K
#         frame.tf_cam_world=self.tf_cam_world
#         frame.width=self.width
#         frame.height=self.height
#         cloud=frame.create_frustum_mesh(scale)
#         return cloud
#     def compute_uv(self, cloud):
#         frame=Frame()
#         frame.rgb_32f=self.rgb_32f
#         frame.K=self.K
#         frame.tf_cam_world=self.tf_cam_world
#         frame.width=self.width
#         frame.height=self.height
#         uv=frame.compute_uv(cloud)
#         return uv
#     def compute_uv_with_assign_color(self, cloud):
#         frame=Frame()
#         frame.rgb_32f=self.rgb_32f
#         frame.K=self.K
#         frame.tf_cam_world=self.tf_cam_world
#         frame.width=self.width
#         frame.height=self.height
#         cloud=frame.assign_color(cloud)
#         return cloud.UV.copy()

#     def load_images(self):
#         if self.frame.is_shell:
#             self.frame.load_images() 

#             #load the img tensors
#             self.load_image_tensors()

#         if self.create_subsamples and not self.frame.is_shell:
#             self.subsampled_frames=[]
#             for i in range(2):
#                 if i==0:
#                     frame_subsampled=self.frame.subsample(2)
#                 else:
#                     frame_subsampled=frame_subsampled.subsample(2)
#                 self.subsampled_frames.append(FramePY(frame_subsampled, create_subsamples=False))

#     def load_image_tensors(self):
#         if self.frame.mask.empty():
#             mask_tensor= torch.ones((1,1,self.frame.height,self.frame.width))
#             self.frame.mask=tensor2mat(mask_tensor)
#         #weight and hegiht
#         # self.height=self.rgb_tensor.shape[2]
#         # self.width=self.rgb_tensor.shape[3]
#         self.height=self.frame.height
#         self.width=self.frame.width
#         #CHECK that the frame width and hegiht has the same values as the rgb 
#         if self.frame.height!=self.frame.rgb_32f.rows or  self.frame.width!=self.frame.rgb_32f.cols:
#             print("frame dimensions and rgb32 doesnt match. frame.height", self.frame.height, " frame.rgb_32f.rows", self.frame.rgb_32f.rows, " frame.width ", self.frame.width, " frame.rgb_32f.cols ", self.frame.rgb_32f.cols)
#         #Ray direction in world coordinates
#         ray_dirs_mesh=self.frame.pixels2dirs_mesh()
#         # self.ray_dirs=torch.from_numpy(ray_dirs_mesh.V.copy()).to("cuda").float() #Nx3
#         self.ray_dirs=ray_dirs_mesh.V.copy() #Nx3
        
        


# #some default arguments are in here https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/demo.sh

# class EDSR_args():
#     n_resblocks=4
#     n_feats=128
#     scale=2
#     res_scale=0.1
#     rgb_range=255
#     n_in_channels=3
#     n_out_channels=3

# def gelu(x):
#     return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


# def map_range( input_val, input_start, input_end,  output_start,  output_end):
#     # input_clamped=torch.clamp(input_val, input_start, input_end)
#     # input_clamped=max(input_start, min(input_end, input_val))
#     input_clamped=torch.clamp(input_val, input_start, input_end)
#     return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start)

# def compute_uv(frame, points_3D_world):
#     if points_3D_world.shape[1] != 3:
#         print("expecting the points3d to be Nx3 but it is ", points_3D_world.shape)
#         exit(1)
#     if len(points_3D_world.shape) != 2:
#         print("expecting the points3d to have 2 dimensions corresponding to Nx3 but it is ", points_3D_world.shape)
#         exit(1)

#     # R=torch.from_numpy( frame.tf_cam_world.linear() ).to("cuda")
#     # t=torch.from_numpy( frame.tf_cam_world.translation() ).to("cuda")
#     # K = torch.from_numpy( frame.K ).to("cuda")

#     R=frame.R_tensor
#     t=frame.t_tensor
#     K = frame.K_tensor

#     # points_3D_cam=torch.matmul(R, points_3D_world.transpose(0,1) ).transpose(0,1)  + t.view(1,3)
#     # points_screen = torch.matmul(K, points_3D_cam.transpose(0,1) ).transpose(0,1)  


#     points_3D_cam=torch.matmul(R, points_3D_world.transpose(0,1).contiguous() ).transpose(0,1).contiguous()  + t.view(1,3)
#     points_screen = torch.matmul(K, points_3D_cam.transpose(0,1).contiguous() ).transpose(0,1)  


#     # print("points_screen ", points_screen)
#     points_2d = points_screen[:, 0:2] / ( points_screen[:, 2:3] +0.0001 )
#     # print("points_2d before flip ", points_2d)

#     points_2d[:,1] = frame.height- points_2d[:,1] 
#     # print("points_2d ", points_2d)

#     #shift by half a pixel 
#     # points_2d[:,1]=points_2d[:,1]+0.5

#     #get in range 0,1
#     points_2d[:,0]  = points_2d[:,0]/frame.width 
#     points_2d[:,1]  = points_2d[:,1]/frame.height 
#     uv_tensor = points_2d
#     # print("uv_tensor is ", uv_tensor)
#     # exit(1)

#     #may be needed 
#     uv_tensor= uv_tensor*2 -1 #get in range [-1,1]
#     # uv_tensor[:,1]=-uv_tensor[:,1] #flip


#     return uv_tensor

# # def compute_uv_batched(frames_list, points_3D_world):
# def compute_uv_batched(R_batched, t_batched, K_batched, height, width, points_3D_world):
#     if points_3D_world.shape[1] != 3:
#         print("expecting the points3d to be Nx3 but it is ", points_3D_world.shape)
#         exit(1)
#     if len(points_3D_world.shape) != 2:
#         print("expecting the points3d to have 2 dimensions corresponding to Nx3 but it is ", points_3D_world.shape)
#         exit(1)



#     # TIME_START("repeat")
#     feat_sliced_per_frame=[]
#     feat_sliced_per_frame_manual=[]
#     # points3d_world_for_uv=points_3D_world.view(1,-1,3).repeat( len(frames_list) ,1, 1) #Make it into NR_frames x N x 3
#     points3d_world_for_uv=points_3D_world.view(1,-1,3).repeat( R_batched.shape[0] ,1, 1) #Make it into NR_frames x N x 3
#     # TIME_END("repeat")
#     # TIME_START("concat")
#     # # with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     #     R_list=[]
#     #     t_list=[]
#     #     K_list=[]
#     #     for i in range(len(frames_list)):
#     #         frame=frames_list[i]
#     #         R_list.append( frame.R_tensor.view(1,3,3) )
#     #         t_list.append( frame.t_tensor.view(1,1,3) )
#     #         K_list.append( frame.K_tensor.view(1,3,3) )
#     #     R_batched=torch.cat(R_list,0)
#     #     t_batched=torch.cat(t_list,0)
#     #     K_batched=torch.cat(K_list,0)
#     # # print(prof)
#     # TIME_END("concat")
#     #project 
#     # TIME_START("proj")
#     # with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     points_3D_cam=torch.matmul(points3d_world_for_uv, R_batched.transpose(1,2) )  + t_batched
#     points_screen = torch.matmul(points_3D_cam, K_batched.transpose(1,2) )  
#     points_2d = points_screen[:, :, 0:2] / ( points_screen[:, :, 2:3] +0.0001 )
#     points_2d[:,:,1] = height- points_2d[:,:,1] 

#     mask = (points_2d[..., 0] <= width - 1.) & \
#                (points_2d[..., 0] >= 0) & \
#                (points_2d[..., 1] <= height - 1.) &\
#                (points_2d[..., 1] >= 0)
#     mask=mask.unsqueeze(2)


#     #get in range 0,1
#     scaling = torch.tensor([width, height]).cuda().view(1,1,2)   ######WATCH out we assume that all the frames are the same width and height
#     uv_tensor=points_2d.div(scaling)
#     # #may be needed 
#     uv_tensor= uv_tensor*2 -1 #get in range [-1,1]


#     #attempt 2
#     # scaling = torch.tensor([1.0/width*2, 1.0/height*2]).cuda().view(1,1,2)   ######WATCH out we assume that all the frames are the same width and height
#     # minus_one= torch.tensor([-1.0]).cuda().view(1,1,1)
#     # uv_tensor = torch.addcmul(minus_one, points_2d,scaling) #-1+ points2d*scaling

#     # print(prof)
#     # TIME_END("proj")


#     return uv_tensor, mask

# def compute_uv_batched_original(frames_list,  points_3D_world):
#     if points_3D_world.shape[1] != 3:
#         print("expecting the points3d to be Nx3 but it is ", points_3D_world.shape)
#         exit(1)
#     if len(points_3D_world.shape) != 2:
#         print("expecting the points3d to have 2 dimensions corresponding to Nx3 but it is ", points_3D_world.shape)
#         exit(1)



#     feat_sliced_per_frame=[]
#     feat_sliced_per_frame_manual=[]
#     points3d_world_for_uv=points_3D_world.view(1,-1,3).repeat( len(frames_list) ,1, 1) #Make it into NR_frames x N x 3
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
#     #project 
#     points_3D_cam=torch.matmul(R_batched, points3d_world_for_uv.transpose(1,2) ).transpose(1,2)  + t_batched
#     points_screen = torch.matmul(K_batched, points_3D_cam.transpose(1,2) ).transpose(1,2)  
#     points_2d = points_screen[:, :, 0:2] / ( points_screen[:, :, 2:3] +0.0001 )
#     points_2d[:,:,1] = frame.height- points_2d[:,:,1] 


#     #get in range 0,1
#     points_2d[:,:,0]  = points_2d[:,:,0]/frame.width  ######WATCH out we assume that all the frames are the same width and height
#     points_2d[:,:,1]  = points_2d[:,:,1]/frame.height 
#     uv_tensor = points_2d

#     #may be needed 
#     uv_tensor= uv_tensor*2 -1 #get in range [-1,1]


#     return uv_tensor

# def compute_normal(points3d_img):
#     assert len(points3d_img.shape) == 4, points3d_img.shape

#     #since the gradient is not defined at the boders, we pad the image with zeros
#     # points3d_img_padded = torch.nn.functional.pad(points3d_img, [0,1,0,1], mode='constant', value=0)
#     points3d_img_x= torch.nn.functional.pad(points3d_img, [0,1,0,0], mode='constant', value=0)
#     points3d_img_y= torch.nn.functional.pad(points3d_img, [0,0,0,1], mode='constant', value=0)
#     # print("points3d_img_padded", points3d_img_padded.shape)

#     grad_x=points3d_img_x[:, :, :, :-1] - points3d_img_x[:, :, :, 1:]
#     grad_y=points3d_img_y[:, :, :-1, :] - points3d_img_y[:, :, 1:, :]
#     # print("grad x is ", grad_x.shape)

#     #make the gradx and grady in Nx3
#     height=points3d_img.shape[2]
#     width=points3d_img.shape[3]
#     nr_channels=points3d_img.shape[1]
#     grad_x=grad_x.permute(0,2,3,1) #from N,C,H,W to N,H,W,C
#     grad_y=grad_y.permute(0,2,3,1) #from N,C,H,W to N,H,W,C
#     grad_x=grad_x.view(-1, nr_channels)
#     grad_y=grad_y.view(-1, nr_channels)
#     # cross=torch.cross(grad_x, grad_y, dim=1)
#     cross=torch.cross(grad_y, grad_x, dim=1)
#     # print("corss x is ", cross.shape)

#     normal_norm=cross.norm(dim=1, keepdim=True)
#     normal=cross/(normal_norm+0.00001)
#     # print("normal is ", normal.shape)


#     #make into image
#     normal=normal.view(-1, height, width, nr_channels)
#     normal=normal.permute(0,3,1,2) #from N,H,W,C to N,C,H,W

#     return normal

# def frames_to_points(frames, discard_frame_with_idx=None ):

#     points_list=[]
#     points_frame_idxs_list=[]
#     for i in range(len(frames)):
#         frame=frames[i]
#         if discard_frame_with_idx!=None and frame.frame_idx==discard_frame_with_idx:
#             continue
#         points_list.append( frame.frame.pos_in_world()  )
#         points_frame_idxs_list.append( frame.frame_idx )
            
#     points = np.asarray(points_list)
#     points_frame_idxs = np.asarray(points_frame_idxs_list)

#     return points, points_frame_idxs

# #from autoclip https://github.com/pseeth/autoclip/blob/master/autoclip.py
# def get_grad_norm(model):
#     total_norm = 0
#     for p in model.parameters():
#         if p.grad is not None:
#             param_norm = p.grad.data.norm(2)
#             total_norm += param_norm.item() ** 2
#     total_norm = total_norm ** (1. / 2)
#     return total_norm 

# def fused_mean_variance(x, weight, dim, use_weights=True):
#     if use_weights:
#         mean = torch.sum(x*weight, dim=dim, keepdim=True)
#         var = torch.sum(weight * (x - mean)**2, dim=dim, keepdim=True)
#     else:
#         mean = torch.sum(x, dim=dim, keepdim=True)
#         var = torch.sum( (x - mean)**2, dim=dim, keepdim=True)
#     mean_var=torch.cat([mean,var],-1)
#     return mean_var

#wraps module, and changes them to become a torchscrip version of them during inference
class TorchScriptTraceWrapper(torch.nn.Module):
    def __init__(self, module):
        super(TorchScriptTraceWrapper, self).__init__()

        self.module=module
        self.module_traced=None

    def forward(self, *args):
        args_list=[]
        for arg in args:
            args_list.append(arg)
        if self.module_traced==None:
                self.module_traced = torch.jit.trace(self.module, args_list )
        return self.module_traced(*args)

class FrameWeightComputer(torch.nn.Module):

    def __init__(self ):
        super(FrameWeightComputer, self).__init__()

        # self.s_weight = torch.nn.Parameter(torch.randn(1))  #from equaiton 3 here https://arxiv.org/pdf/2010.08888.pdf
        # with torch.set_grad_enabled(False):
        #     # self.s_weight.fill_(0.5)
        #     self.s_weight.fill_(10.0)

        ######CREATING A NEW parameter for the s_weight for some reason destroys the rest of the network and it doesnt optimize anymore. Either way, it barely changes so we just set it to 10
        self.s_weight=10

    def forward(self, frame, frames_close):
        cur_dir=frame.look_dir
        exponential_weight_towards_neighbour=[]
        for i in range(len(frames_close)):
            dir_neighbour=frames_close[i].look_dir
            dot= torch.dot( cur_dir.view(-1), dir_neighbour.view(-1) )
            s_dot= self.s_weight*(dot-1)
            exp=torch.exp(s_dot)
            exponential_weight_towards_neighbour.append(exp.view(1))
        all_exp=torch.cat(exponential_weight_towards_neighbour)
        exp_minimum= all_exp.min()
        unnormalized_weights=[]
        for i in range(len(frames_close)):
            cur_exp= exponential_weight_towards_neighbour[i]
            exp_sub_min= cur_exp-exp_minimum
            unnormalized_weight= torch.relu(exp_sub_min)
            unnormalized_weights.append(unnormalized_weight)
            # print("unnormalized_weight", unnormalized_weight)
        all_unormalized_weights=torch.cat(unnormalized_weights)
        weight_sum=all_unormalized_weights.sum()
        weights=[]
        for i in range(len(frames_close)):
            unnormalized_weight= unnormalized_weights[i]
            weight= unnormalized_weight/weight_sum
            weights.append(weight)
        weights=torch.cat(weights)

        # ##attempt 2 by just using barycentric coords
        # frames_close_list=[]
        # for framepy in frames_close:
        #     frames_close_list.append(framepy.frame)
        # weights_vec=SFM.compute_frame_weights(frame.frame, frames_close_list)
        # # print("weigrs vec is ", weights_vec)
        # weights=torch.from_numpy( np.array(weights_vec) ).float().to("cuda")
        # #clamp them
        # weights=torch.clamp(weights,0.0, 1.0)


        return weights


class FeatureAgregator(torch.nn.Module):

    def __init__(self ):
        super(FeatureAgregator, self).__init__()


    def forward(self, feat_sliced_per_frame, weights, mask, use_mask=True, novel=False):

        #similar to https://ibrnet.github.io/static/paper.pdf
        # feat_sliced_per_frame is Nr_frames x N x FEATDIM
        # print("feat_sliced_per_frame", feat_sliced_per_frame.shape)
        weights=weights.view(-1,1,1)
          
        img_features_concat_weighted=feat_sliced_per_frame*weights
        if(use_mask):
            img_features_concat_weighted = img_features_concat_weighted*mask

        img_features_mean= img_features_concat_weighted.sum(dim=0)

        # STD https://stats.stackexchange.com/a/6536
        img_features_normalized=  (feat_sliced_per_frame-img_features_mean.unsqueeze(0))**2 #xi- mu
        img_features_normalized_weighted= img_features_normalized*weights
        if use_mask:
            img_features_normalized_weighted = img_features_normalized_weighted *mask
        std= img_features_normalized_weighted.sum(dim=0) #this is just the nominator but the denominator is probably not needed since it's just 1
        # print("stdm in", std.min())
        std=torch.sqrt(std+0.0001) #adding a small espilon to avoid sqrt(negative number) which then cuases nans

        # #similar to what vladnet has
        # diff_to_mean = (feat_sliced_per_frame-img_features_mean.unsqueeze(0)) #xi- mu
        # img_features_normalized_weighted= diff_to_mean*weights
        # std= img_features_normalized_weighted.sum(dim=0) #Nxfead_dim
        # #we normalize columnswise as vladnet does it
        # norm=std.norm(dim=0, keepdim=True)
        # std=std/norm


        final_feat=torch.cat([img_features_mean, std],1)

        

        return final_feat


#passes the features through a linear layer instead of calculating the mean and variance, so it's kinda dependant on the ordering of the frames now
class FeatureAgregatorLinear(torch.nn.Module):

    def __init__(self ):
        super(FeatureAgregatorLinear, self).__init__()


        self.pred=MetaSequential( 
            BlockNerf(activ=torch.nn.GELU(), in_channels=32*3, out_channels=64,  bias=True ).cuda(),
            BlockNerf(activ=torch.nn.GELU(), in_channels=64, out_channels=64,  bias=True ).cuda(),
            BlockNerf(activ=None, in_channels=64, out_channels=64,  bias=True ).cuda()
            )

    def forward(self, feat_sliced_per_frame, weights, novel=False):


        # feat_sliced_per_frame is Nr_frames x N x FEATDIM
        nr_frames= feat_sliced_per_frame.shape[0]
        nr_pixels= feat_sliced_per_frame.shape[1]
        feat_dim= feat_sliced_per_frame.shape[2]
        weights=weights.view(-1,1,1)

        # #since we don't want the network to overfit to a perticular  order of the Nr_frames, we randomize their order 
        # if not novel:
        #     randperm=torch.randperm(nr_frames)
        #     feat_sliced_per_frame=feat_sliced_per_frame[randperm,:,:]
        #     weights=weights[randperm,:,:]


        img_features_concat_weighted=feat_sliced_per_frame*weights

        #get the features to be N x featdim*Nr_frames
        img_features_concat_weighted=img_features_concat_weighted.permute(1,0,2).view(nr_pixels, nr_frames*feat_dim) # from Nr_frames x N x FEATDIM to  NxNr_framesxFatudim
        x=self.pred( img_features_concat_weighted )
        

        return x

#fuse the features using an invariant functional like in pointnet
class FeatureAgregatorInvariant(torch.nn.Module):

    def __init__(self ):
        super(FeatureAgregatorInvariant, self).__init__()

        self.last_channels=32
        self.pred=MetaSequential( 
            # BlockNerf(activ=torch.nn.GELU(), in_channels=16, out_channels=32,  bias=True ).cuda(),
            BlockNerf(activ=torch.nn.GELU(), in_channels=16, out_channels=self.last_channels,  bias=True ).cuda()
            )

    def forward(self, feat_sliced_per_frame, weights, novel=False):


        # feat_sliced_per_frame is Nr_frames x N x FEATDIM
        nr_frames= feat_sliced_per_frame.shape[0]
        nr_pixels= feat_sliced_per_frame.shape[1]
        feat_dim= feat_sliced_per_frame.shape[2]
        weights=weights.view(-1,1,1)


        img_features_concat_weighted=feat_sliced_per_frame*weights
        # img_features_concat_weighted=img_features_concat_weighted.contiguous().view(-1, feat_dim) # from Nr_frames x nr_pixels x FEATDIM to  -xxFatudim

        #get the features to be N * featdim x Nr_frames
        x=self.pred( img_features_concat_weighted )

        # x=x.view(nr_frames, nr_pixels, self.last_channels)
        # x,_=x.max(dim=0)
        # x=x.mean(dim=0)
        x,_=x.min(dim=0) #this if effectivlly encoding for the minimum safe distance that the ray can advance
       

        

        return x


class FeatureAgregatorIBRNet(torch.nn.Module):

    def __init__(self ):
        super(FeatureAgregatorIBRNet, self).__init__()


        self.base_fc = nn.Sequential(nn.Linear(32*3, 64),
                                     nn.ELU(inplace=False),
                                     nn.Linear(64, 32),
                                     nn.ELU(inplace=False)
                                     )

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    nn.ELU(inplace=False),
                                    nn.Linear(32, 33),
                                    nn.ELU(inplace=False),
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     nn.ELU(inplace=False),
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

    def forward(self, feat_sliced_per_frame, weights, mask, novel=False):

        weights=weights.view(-1,1,1)
        
        # feat_sliced_per_frame is Nr_frames x N x FEATDIM
        nr_frames= feat_sliced_per_frame.shape[0]
        nr_pixeles= feat_sliced_per_frame.shape[1]
        feat_dim= feat_sliced_per_frame.shape[2]

        mean_var=fused_mean_variance(feat_sliced_per_frame,weights, 0, use_weights=False)
        mean_var_batched = mean_var.view(1,nr_pixeles, -1).repeat(nr_frames,1,1)

        feat_mu_var=torch.cat([feat_sliced_per_frame,mean_var_batched],2)
        feat_mu_var_reduced = self.base_fc(feat_mu_var)


        #from ibnrnet 
        x=feat_mu_var_reduced
        weight=weights

        # print("x is ", x.shape)
        # print("mask is ", mask.shape)


        x_vis = self.vis_fc(x * mask)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis)
        x = x + x_res
        vis = self.vis_fc2(x * vis)
        weight = vis / (torch.sum(vis, dim=0, keepdim=True) + 1e-8)

        # print("x", x.shape)
        # print("weight", weight.shape)

        mean_var = fused_mean_variance(x, weight, 0)
        # print("mean_var_before lat", mean_var.shape)
        globalfeat = torch.cat([mean_var.squeeze(0), weight.mean(dim=0)], dim=-1)  # [n_rays, n_samples, 32*2+1]

        # print("global ", globalfeat.shape)

        

        return globalfeat










class Block(MetaModule):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.relu, init=None, do_norm=False, is_first_layer=False ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.GELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.sin, init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.SELU(inplace=False), init=None ):
        super(Block, self).__init__()
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.bias=bias 
        self.conv= None
        self.norm= None
        # self.relu=torch.nn.ReLU(inplace=False)
        self.activ=activ
        self.with_dropout=with_dropout
        self.transposed=transposed
        # self.cc=ConcatCoord()
        self.init=init
        self.do_norm=do_norm
        self.is_first_layer=is_first_layer

        if with_dropout:
            self.drop=torch.nn.Dropout2d(0.2)

        self.relu=torch.nn.ReLU(inplace=True)

        # self.conv=None

        # self.norm = torch.nn.BatchNorm2d(in_channels, momentum=0.01).cuda()
        # self.norm = torch.nn.BatchNorm2d(self.out_channels, momentum=0.01).cuda()
        # self.norm = torch.nn.GroupNorm(1, in_channels).cuda()

        # self.sine_scale=torch.nn.Parameter(torch.Tensor(1)).cuda()
        # self.sine_scale=torch.nn.Parameter(torch.randn(3,6)).cuda()
        # torch.nn.init.constant_(self.sine_scale, 30)
        # with torch.set_grad_enabled(False):
            # self.sine_scale=30
        # self.sine_scale.requires_grad = True
        # self.wtf=torch.nn.Linear(10,10)
        # self.weight = torch.nn.Parameter(torch.Tensor(10, 10))
        # torch.nn.init.uniform_(self.sine_scale, -1, 1)
        # self.W = torch.nn.Parameter(torch.randn(3,4,5))
        self.sine_scale = torch.nn.Parameter(torch.randn(1))
        torch.nn.init.constant_(self.sine_scale, 30)
        # self.W.requires_grad = True

        if not self.transposed:
            self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
        else:
            self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )

        if self.init=="zero":
                torch.nn.init.zeros_(self.conv[-1].weight) 
        if self.activ==torch.sin:
            with torch.no_grad():
                # print(":we are usign sin")
                # print("in channels is ", in_channels, " and conv weight size 1 is ", self.conv.weight.size(-1) )
                # num_input = self.conv.weight.size(-1)
                num_input = in_channels
                # num_input = self.out_channels
                # See supplement Sec. 1.5 for discussion of factor 30
                if self.is_first_layer:
                    # self.conv[-1].weight.uniform_(-1 / num_input, 1 / num_input)
                    # self.conv[-1].weight.uniform_(-1 / num_input*2, 1 / num_input*2)
                    self.conv[-1].weight.uniform_(-1 / num_input, 1 / num_input)
                    # print("conv 1 is ", self.conv[-1].weight )
                else:
                    self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input)/30 , np.sqrt(6 / num_input)/30 )
                    # self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
                    # self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input)/7 , np.sqrt(6 / num_input)/7 )
                    # print("conv any other is ", self.conv[-1].weight )
        if self.activ==torch.relu:
            print("initializing with kaiming uniform")
            torch.nn.init.kaiming_uniform_(self.conv[-1].weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv[-1].weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.conv[-1].bias, -bound, bound)
        

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        # print("params is", params)

        # x=self.cc(x)

        in_channels=x.shape[1]


        #create modules if they are not created
        # if self.norm is None:
        #     # self.norm = torch.nn.GroupNorm(in_channels, in_channels).cuda()
        #     nr_groups=32
        #     nr_params=in_channels
        #     if nr_params<=32:
        #         nr_groups=int(nr_params/2)
            # self.norm = torch.nn.GroupNorm(nr_groups, nr_params).cuda()
            # self.norm = torch.nn.GroupNorm(16, self.out_channels).cuda()
            # self.norm = torch.nn.GroupNorm(1, in_channels).cuda()
            # self.norm = torch.nn.GroupNorm(in_channels, in_channels).cuda()
            # self.norm = torch.nn.GroupNorm(self.out_channels, self.out_channels).cuda()
            # self.norm = torch.nn.BatchNorm2d(in_channels, momentum=0.01).cuda()
            # self.norm = torch.nn.BatchNorm2d(self.out_channels, momentum=0.01).cuda()
        # if self.conv is None:
        #     # self.net=[]
        #     if not self.transposed:
        #         self.conv= MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  
        #     else:
        #         self.conv= MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda() 

        #     if self.init=="zero":
        #         torch.nn.init.zeros_(self.conv.weight) 
        #     if self.activ==torch.sin:
        #         with torch.no_grad():
        #             # print(":we are usign sin")
        #             # print("in channels is ", in_channels, " and conv weight size 1 is ", self.conv.weight.size(-1) )
        #             # num_input = self.conv.weight.size(-1)
        #             num_input = in_channels
        #             # num_input = self.out_channels
        #             # See supplement Sec. 1.5 for discussion of factor 30
        #             if self.is_first_layer:
        #                 self.conv.weight.uniform_(-1 / num_input, 1 / num_input)
        #             else:
        #                 self.conv.weight.uniform_(-np.sqrt(6 / num_input)/30 , np.sqrt(6 / num_input)/30 )

      


        #pass the tensor through the modules
        # if self.do_norm:
            # x=self.norm(x) # TODO The vae seems to work a lot better without any normalization but more testing might be needed
        # if self.do_norm:
            # x=self.norm(x) # TODO The vae seems to work a lot better without any normalization but more testing might be needed
        # x=gelu(x)
        # x=self.activ(x)
        # if self.with_dropout:
            # x = self.drop(x)
        # if self.activ==torch.sin:
            # print("am in a sin acitvation, x before conv is ", x.shape)
            # print("am in a sin acitvation, conv has params with shape ", self.conv[-1].weight.shape)
        # x=self.norm(x) # TODO The vae seems to work a lot better without any normalization but more testing might be needed

        # if self.is_first_layer:
        #     x=30*x
        

        # print("before conv, x has mean and std " , x.mean() , " std ", x.std() )
        x = self.conv(x, params=get_subdict(params, 'conv') )
        # if self.do_norm:
            # print("norm")
            # if(x.shape[1]%16==0):
                # x=self.norm(x) # TODO The vae seems to work a lot better without any normalization but more testing might be needed
            # x=self.norm(x) # TODO The vae seems to work a lot better without any normalization but more testing might be needed
        # x=self.relu(x)
        # x=self.norm(x) # TODO The vae seems to work a lot better without any normalization but more testing might be needed
        if self.activ==torch.sin:
            # x=30*x
            x=30*x
            # x=self.sine_scale*x
            # print("self.sine_scale", self.sine_scale)
            # print("before activ, x has mean and std " , x.mean() , " std ", x.std() )
            # print("before activ, x*30 has mean and std " , (x*30).mean() , " std ", (x*30).std() )
            # x=self.activ(30*x)
            x=self.activ(x)
            # print("after activ, x has mean and std " , x.mean() , " std ", x.std() )
        elif self.activ is not None:
            # x=self.activ(x)
            x=self.relu(x)
            # print("after activ, x has mean and std " , x.mean() , " std ", x.std() )
        # x=gelu(x)
        # x=torch.sin(x)
        # x=torch.sigmoid(x)

        return x


class BlockLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels,  bias,  activ=torch.relu ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.GELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.sin, init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.SELU(inplace=False), init=None ):
        super(BlockLinear, self).__init__()

        self.activ=activ 
        self.conv=torch.nn.Linear(  in_features=in_channels, out_features=out_channels, bias=bias ) 
       
        if self.activ==torch.relu:
            # print("initializing with kaiming uniform")
            torch.nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
            if bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.conv.bias, -bound, bound)
        

    def forward(self, x):
       
        x = self.conv(x )

        if self.activ is not None: 
            x=self.activ(x)
         
        return x

class BNReluConv(MetaModule):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.relu, init=None, do_norm=False, is_first_layer=False ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.GELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.sin, init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.SELU(inplace=False), init=None ):
        super(BNReluConv, self).__init__()
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.bias=bias 
        self.conv= None
        self.norm= None
        # self.relu=torch.nn.ReLU(inplace=False)
        self.activ=activ
        self.with_dropout=with_dropout
        self.transposed=transposed
        # self.cc=ConcatCoord()
        self.init=init
        self.do_norm=do_norm
        self.is_first_layer=is_first_layer

        if with_dropout:
            self.drop=torch.nn.Dropout2d(0.2)

        if do_norm:
            self.norm = torch.nn.BatchNorm2d(in_channels).cuda()
        # self.norm = torch.nn.GroupNorm(1, in_channels).cuda()
        self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
        

       
        print("initializing with kaiming uniform")
        torch.nn.init.kaiming_uniform_(self.conv[-1].weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
        if self.bias is not False:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv[-1].weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.conv[-1].bias, -bound, bound)
    

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        # print("params is", params)

        # x=self.cc(x)

        in_channels=x.shape[1]

        if self.do_norm:
            x=self.norm(x)
        if self.activ !=None: 
            x=self.activ(x)
        x = self.conv(x, params=get_subdict(params, 'conv') )
       

        return x


class GNReluConv(MetaModule):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.relu, init=None, do_norm=False, is_first_layer=False ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.GELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.sin, init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.SELU(inplace=False), init=None ):
        super(GNReluConv, self).__init__()
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.bias=bias 
        self.conv= None
        self.norm= None
        # self.relu=torch.nn.ReLU(inplace=False)
        self.activ=activ
        self.with_dropout=with_dropout
        self.transposed=transposed
        # self.cc=ConcatCoord()
        self.init=init
        self.do_norm=do_norm
        self.is_first_layer=is_first_layer

        if with_dropout:
            self.drop=torch.nn.Dropout2d(0.2)

        if do_norm:
            nr_groups=32
            #if the groups is not diivsalbe so for example if we have 80 params
            if in_channels%nr_groups!=0:
                nr_groups= int(in_channels/4)
            if in_channels==32:
                nr_groups= int(in_channels/4)
            # print("nr groups is ", nr_groups, " in channels ", in_channels)
            self.norm = torch.nn.GroupNorm(nr_groups, in_channels).cuda()
            # self.norm = torch.nn.GroupNorm(in_channels, in_channels).cuda()
            # self.norm = torch.nn.GroupNorm( int(in_channels/4), in_channels).cuda()
        # self.norm = torch.nn.GroupNorm(1, in_channels).cuda()
        self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
        

       
        print("initializing with kaiming uniform")
        torch.nn.init.kaiming_uniform_(self.conv[-1].weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
        if self.bias is not False:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv[-1].weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.conv[-1].bias, -bound, bound)
    

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        # print("params is", params)

        # x=self.cc(x)

        in_channels=x.shape[1]

        if self.do_norm:
            x=self.norm(x)
        if self.activ !=None: 
            x=self.activ(x)
        x = self.conv(x, params=get_subdict(params, 'conv') )
       

        return x


class WNReluConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.relu, init=None, do_norm=False, is_first_layer=False ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.GELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.sin, init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.SELU(inplace=False), init=None ):
        super(WNReluConv, self).__init__()
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.bias=bias 
        self.conv= None
        self.norm= None
        # self.relu=torch.nn.ReLU(inplace=False)
        self.activ=activ
        self.with_dropout=with_dropout
        self.transposed=transposed
        # self.cc=ConcatCoord()
        self.init=init
        self.do_norm=do_norm
        self.is_first_layer=is_first_layer

        if with_dropout:
            self.drop=torch.nn.Dropout2d(0.2)

        # if do_norm:
        #     nr_groups=32
        #     #if the groups is not diivsalbe so for example if we have 80 params
        #     if in_channels%nr_groups!=0:
        #         nr_groups= int(in_channels/4)
        #     if in_channels==32:
        #         nr_groups= int(in_channels/4)
        #     # print("nr groups is ", nr_groups, " in channels ", in_channels)
        #     self.norm = torch.nn.GroupNorm(nr_groups, in_channels).cuda()
            # self.norm = torch.nn.GroupNorm(in_channels, in_channels).cuda()
            # self.norm = torch.nn.GroupNorm( int(in_channels/4), in_channels).cuda()
        # self.norm = torch.nn.GroupNorm(1, in_channels).cuda()
        if do_norm:
            if self.transposed:
                self.conv= torch.nn.utils.weight_norm( torch.nn.ConvTranspose2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
            else:
                self.conv= torch.nn.utils.weight_norm( torch.nn.Conv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
                # self.conv=  DeformConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias).cuda()  
        else: 
            if self.transposed:
                self.conv=torch.nn.ConvTranspose2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()
            else:
                self.conv= torch.nn.Conv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()
        
        
        # self.conv=  torch.nn.Conv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  
        

       
        # print("initializing with kaiming uniform")
        # torch.nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
        # if self.bias is not False:
        #     fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     torch.nn.init.uniform_(self.conv.bias, -bound, bound)
    

    def forward(self, x):
        # if params is None:
            # params = OrderedDict(self.named_parameters())
        # print("params is", params)

        # x=self.cc(x)

        in_channels=x.shape[1]

        # if self.do_norm:
            # x=self.norm(x)
        if self.activ !=None: 
            x=self.activ(x)
        x = self.conv(x )
       

        return x


class WNGatedConvRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.relu, init=None, do_norm=False, is_first_layer=False ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.GELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.sin, init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.SELU(inplace=False), init=None ):
        super(WNGatedConvRelu, self).__init__()
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.bias=bias 
        self.conv= None
        self.norm= None
        # self.relu=torch.nn.ReLU(inplace=False)
        self.activ=activ
        self.with_dropout=with_dropout
        self.transposed=transposed
        # self.cc=ConcatCoord()
        self.init=init
        self.do_norm=do_norm
        self.is_first_layer=is_first_layer

        if with_dropout:
            self.drop=torch.nn.Dropout2d(0.2)

        # if do_norm:
        #     nr_groups=32
        #     #if the groups is not diivsalbe so for example if we have 80 params
        #     if in_channels%nr_groups!=0:
        #         nr_groups= int(in_channels/4)
        #     if in_channels==32:
        #         nr_groups= int(in_channels/4)
        #     # print("nr groups is ", nr_groups, " in channels ", in_channels)
        #     self.norm = torch.nn.GroupNorm(nr_groups, in_channels).cuda()
            # self.norm = torch.nn.GroupNorm(in_channels, in_channels).cuda()
            # self.norm = torch.nn.GroupNorm( int(in_channels/4), in_channels).cuda()
        # self.norm = torch.nn.GroupNorm(1, in_channels).cuda()
        if do_norm:
            if self.transposed:
                self.conv= torch.nn.utils.weight_norm( torch.nn.ConvTranspose2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
                self.maskconv= torch.nn.utils.weight_norm( torch.nn.ConvTranspose2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
            else:
                self.conv= torch.nn.utils.weight_norm( torch.nn.Conv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
                self.maskconv= torch.nn.utils.weight_norm( torch.nn.Conv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
                # self.conv=  DeformConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias).cuda()  
        else: 
            if self.transposed:
                self.conv=torch.nn.ConvTranspose2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()
                self.maskconv=torch.nn.ConvTranspose2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()
            else:
                self.conv= torch.nn.Conv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()
                self.maskconv= torch.nn.Conv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()
        
        
        # self.conv=  torch.nn.Conv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  
        

       
        # print("initializing with kaiming uniform")
        # torch.nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
        # if self.bias is not False:
        #     fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     torch.nn.init.uniform_(self.conv.bias, -bound, bound)
        # torch.nn.init.kaiming_uniform_(self.maskconv.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='sigmoid')
        # if self.bias is not False:
        #     fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.maskconv.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     torch.nn.init.uniform_(self.maskconv.bias, -bound, bound)
    

    def forward(self, x):
        # if params is None:
            # params = OrderedDict(self.named_parameters())
        # print("params is", params)

        # x=self.cc(x)

        in_channels=x.shape[1]

        # if self.do_norm:
            # x=self.norm(x)
        # if self.activ !=None: 
            # x=self.activ(x)
        xconv = self.conv(x )
        xmask = self.maskconv(x )

        if self.activ !=None:
            x = self.activ(xconv) * torch.sigmoid(xmask)
        else:
            x = xconv * torch.sigmoid(xmask)


        return x


class BlockPAC(torch.nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.relu, init=None, do_norm=False, is_first_layer=False ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.GELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.sin, init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.SELU(inplace=False), init=None ):
        super(BlockPAC, self).__init__()
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.bias=bias 
        self.conv= None
        self.norm= None
        # self.relu=torch.nn.ReLU(inplace=False)
        self.activ=activ
        self.with_dropout=with_dropout
        self.transposed=transposed
        # self.cc=ConcatCoord()
        self.init=init
        self.do_norm=do_norm
        self.is_first_layer=is_first_layer

        if with_dropout:
            self.drop=torch.nn.Dropout2d(0.2)

       
        # self.norm = torch.nn.BatchNorm2d(in_channels).cuda()
        if do_norm:
            self.norm = torch.nn.GroupNorm( int(in_channels/4), in_channels).cuda()
        self.conv=PacConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias).cuda() 
        # self.conv=torch.nn.Conv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias).cuda() 
        

       
        # print("initializing with kaiming uniform")
        # torch.nn.init.kaiming_uniform_(self.conv[-1].weight, a=math.sqrt(5), mode='fan_out', nonlinearity='tanh')
        # if self.bias is not None:
        #     fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv[-1].weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     torch.nn.init.uniform_(self.conv[-1].bias, -bound, bound)
    

    def forward(self, x, guide):
        # if params is None:
            # params = OrderedDict(self.named_parameters())
        # print("params is", params)

        # x=self.cc(x)

        in_channels=x.shape[1]


        if self.do_norm:
            x=self.norm(x)

        if self.activ !=None: 
            x=self.activ(x)

        x = self.conv(x, guide )
        # x = self.conv(x )

       

        return x

# class BlockSiren(MetaModule):
#     def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ReLU(inplace=False), init=None, do_norm=False, is_first_layer=False ):
#     # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.GELU(), init=None ):
#     # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.sin, init=None ):
#     # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ELU(), init=None ):
#     # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), init=None ):
#     # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.SELU(inplace=False), init=None ):
#         super(BlockSiren, self).__init__()
#         self.out_channels=out_channels
#         self.kernel_size=kernel_size
#         self.stride=stride
#         self.padding=padding
#         self.dilation=dilation
#         self.bias=bias 
#         self.conv= None
#         self.norm= None
#         # self.relu=torch.nn.ReLU(inplace=False)
#         self.activ=activ
#         self.with_dropout=with_dropout
#         self.transposed=transposed
#         # self.cc=ConcatCoord()
#         self.init=init
#         self.do_norm=do_norm
#         self.is_first_layer=is_first_layer

#         if with_dropout:
#             self.drop=torch.nn.Dropout2d(0.2)

#         self.relu=torch.nn.ReLU()
#         self.leaky_relu=torch.nn.LeakyReLU(negative_slope=0.1)

#         # self.conv=None

#         # self.norm = torch.nn.BatchNorm2d(in_channels, momentum=0.01).cuda()
#         # self.norm = torch.nn.BatchNorm2d(self.out_channels, momentum=0.01).cuda()
#         # self.norm = torch.nn.GroupNorm(1, in_channels).cuda()

#         # self.sine_scale=torch.nn.Parameter(torch.Tensor(1)).cuda()
#         # self.sine_scale=torch.nn.Parameter(torch.randn(3,6)).cuda()
#         # torch.nn.init.constant_(self.sine_scale, 30)
#         # with torch.set_grad_enabled(False):
#             # self.sine_scale=30
#         # self.sine_scale.requires_grad = True
#         # self.wtf=torch.nn.Linear(10,10)
#         # self.weight = torch.nn.Parameter(torch.Tensor(10, 10))
#         # torch.nn.init.uniform_(self.sine_scale, -1, 1)
#         # self.W = torch.nn.Parameter(torch.randn(3,4,5))
#         # self.sine_scale = torch.nn.Parameter(torch.randn(1))
#         # torch.nn.init.constant_(self.sine_scale, 30)
#         # self.W.requires_grad = True

#         # if not self.transposed:
#         # self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
#         self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias ).cuda()  )
#             # self.conv_alt= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
#         # else:
#             # self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )

#         if self.init=="zero":
#                 torch.nn.init.zeros_(self.conv[-1].weight) 
#         if self.activ==torch.sin:
#             with torch.no_grad():
#                 # print(":we are usign sin")
#                 # print("in channels is ", in_channels, " and conv weight size 1 is ", self.conv.weight.size(-1) )
#                 # num_input = self.conv.weight.size(-1)
#                 num_input = in_channels
#                 # num_input = self.out_channels
#                 # See supplement Sec. 1.5 for discussion of factor 30
#                 if self.is_first_layer:
#                     # self.conv[-1].weight.uniform_(-1 / num_input, 1 / num_input)
#                     self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
#                     # print("conv 1 is ", self.conv[-1].weight )
#                 else:
#                     # self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input)/30 , np.sqrt(6 / num_input)/30 )
#                     self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
#                     # print("conv any other is ", self.conv[-1].weight )
#                 # self.conv[-1].bias.zero_()
#                 # print("siren weight init has mean and varaicne ", self.conv[-1].weight.mean(), " std", self.conv[-1].weight.std() )

#         if self.activ==torch.relu or self.activ==None:
#             print("initializing with kaiming uniform")
#             torch.nn.init.kaiming_uniform_(self.conv[-1].weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
#             if self.bias is not None:
#                 fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv[-1].weight)
#                 bound = 1 / math.sqrt(fan_in)
#                 torch.nn.init.uniform_(self.conv[-1].bias, -bound, bound)

#         self.iter=1

#     def forward(self, x, params=None):
#         if params is None:
#             params = OrderedDict(self.named_parameters())
#         # else: 
#             # print("params is not none")
       

#         in_channels=x.shape[1]

#         x_input=x


#         # print("BLOCK SIREN: before conv, x has mean " , x.mean().item() , " var ", x.var().item(), " std", x.std().item() )
#         x = self.conv(x, params=get_subdict(params, 'conv') )
#         # x_relu=self.conv_alt(x_input,  params=get_subdict(params, 'conv_alt') )
#         # x_relu=self.leaky_relu(x_relu)

#         # #page 3 at the top of principled initialization hypernetwork https://openreview.net/pdf?id=H1lma24tPB
#         # var_xi=x_input.var()
#         # w= get_subdict(params, 'conv')["0.weight"]
#         # bias= get_subdict(params, 'conv')["0.bias"]
#         # var_w=w.var()
#         # var_b=bias.var()
#         # # w= self.conv[-1].weight
#         # dj=w.shape[1]
#         # print("dj is ", dj)
#         # print("w mean is", w.mean() )
#         # print("w var is", w.var() )
#         # print("b var is", bias.var() )
#         # var_yi=x.var()
#         # var_predicted= dj*var_w*var_xi + var_b
#         # print("var predicted, ", var_predicted.item(), " var_yi ", var_yi.item())

#         # print("exiting")
#         # exit(1)


#         if self.activ==torch.sin:
#             # print("before 30x, x has mean and std " , x.mean().item() , " std ", x.std().item(), " min: ", x.min().item(),  "max ", x.max().item() )
#             if self.is_first_layer: 
#                 # x_conv_scaled=30*x_conv
#                 # x=x*(5+self.iter*0.01)
#                 x=1*x
#             # x_conv_scaled=x_conv
#             else: 
#                 x=x*1
#             # print("before activ, x has mean " , x.mean().item() , " var ", x.var().item(), " min: ", x.min().item(),  "max ", x.max().item() )
#             # mean=x.mean()
#             # std=x.std()
#             # x=(x-mean)/std
#             x=self.activ(x)
#             # x_relu=self.relu(x_conv)
#             #each x will map into a certain period of the sine depending on their value, the network has to be aware of which sine it will activate
#             # x_pos = x/30
#             # x=torch.cat( [x_sine, x_relu],1)
#             # x=torch.cat( [x_sine, x_pos],1)
#             # x=x_sine+x_conv%(3.14)
#             # x=x_sine + x_relu
#             # x=x_sine
#             # print("after activ, x has mean and std " , x.mean().item() , " std ", x.std().item(), " min: ", x.min().item(),  "max ", x.max().item() )


#             # # if self.is_first_layer:
#             # #check the layer
#             # print("x.shape ", x.shape)
#             # nr_layers=x.shape[1]
#             # for i in range(10):
#             #     layer=x[:,i:i+1, :, :]
#             #     layer=(layer+1.0)*0.5
#             #     layer_mat=tensor2mat(layer)
#             #     Gui.show(layer_mat, "layer_"+str(i))


#         elif self.activ is not None:
#             x=self.activ(x)
#         # elif self.activ is None:
#             # x=x_conv

#         # if(self.activ==torch.sin):
#             # print("after activ, x has mean and std " , x.mean().item() , " var ", x.var().item(), " min: ", x.min().item(),  "max ", x.max().item() )
#         # print("x has shape ", x.shape)

#         # x=x+x_relu
#         # print("returning x with mean " , x.mean().item() , " var ", x.var().item(), " min: ", x.min().item(),  "max ", x.max().item() )

#         return x


class BlockSiren(MetaModule):
    def __init__(self, in_channels, out_channels, bias, activ=torch.nn.ReLU(inplace=False), is_first_layer=False, scale_init=90 ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.GELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.sin, init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.SELU(inplace=False), init=None ):
        super(BlockSiren, self).__init__()
        self.bias=bias 
        self.activ=activ
        self.is_first_layer=is_first_layer
        self.scale_init=scale_init
        

        self.relu=torch.nn.ReLU()
        self.leaky_relu=torch.nn.LeakyReLU(negative_slope=0.1)

      

        self.conv= MetaSequential( MetaLinear(in_channels, out_channels, bias=self.bias ).cuda()  )



        if self.activ==torch.sin or self.activ==None:
            with torch.no_grad():
                num_input = in_channels
                # See supplement Sec. 1.5 for discussion of factor 30
                if self.is_first_layer:
                    self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
                else:
                    self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )

        # if self.activ==torch.relu or self.activ==None:
        #     print("initializing with kaiming uniform")
        #     torch.nn.init.kaiming_uniform_(self.conv[-1].weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
        #     if self.bias is not None:
        #         fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv[-1].weight)
        #         bound = 1 / math.sqrt(fan_in)
        #         torch.nn.init.uniform_(self.conv[-1].bias, -bound, bound)


    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        x = self.conv(x, params=get_subdict(params, 'conv') )


        if self.activ==torch.sin:
            if self.is_first_layer: 
                x=self.scale_init*x
            else: 
                x=x*1
            x=self.activ(x)
 


        elif self.activ is not None:
            x=self.activ(x)
     

        return x

class BlockNerf(MetaModule):
    def __init__(self, in_channels, out_channels, bias, activ=torch.nn.ReLU(inplace=False), init="default" ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.GELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.sin, init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ELU(), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), init=None ):
    # def __init__(self, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.SELU(inplace=False), init=None ):
        super(BlockNerf, self).__init__()
        self.bias=bias 
        self.activ=activ
        self.init=init
        

        self.relu=torch.nn.ReLU()
        self.leaky_relu=torch.nn.LeakyReLU(negative_slope=0.1)

      

        self.conv= MetaSequential( MetaLinear(in_channels, out_channels, bias=self.bias ).cuda()  )


        ##if the init is set, it will take precedence over the initializaion from the corresponding activ
        
        if init=="default":
            if self.activ==torch.nn.GELU() or self.activ==torch.nn.ReLU() or self.activ==None:
                torch.nn.init.kaiming_uniform_(self.conv[-1].weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
            if self.activ==torch.sigmoid:
                torch.nn.init.kaiming_uniform_(self.conv[-1].weight, a=math.sqrt(5), mode='fan_in', nonlinearity='sigmoid')
            if self.activ==torch.tanh:
                torch.nn.init.kaiming_uniform_(self.conv[-1].weight, a=math.sqrt(5), mode='fan_in', nonlinearity='tanh')
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv[-1].weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.conv[-1].bias, -bound, bound)
        elif init=="sigmoid":
            torch.nn.init.kaiming_uniform_(self.conv[-1].weight, a=math.sqrt(5), mode='fan_in', nonlinearity='sigmoid')
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv[-1].weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.conv[-1].bias, -bound, bound)
        elif init=="tanh":
            torch.nn.init.kaiming_uniform_(self.conv[-1].weight, a=math.sqrt(5), mode='fan_in', nonlinearity='tanh')
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv[-1].weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.conv[-1].bias, -bound, bound)
        else: 
            print("unknown initialization", init)
            exit(1)


    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        x = self.conv(x, params=get_subdict(params, 'conv') )
     
        if self.activ is not None:
            x=self.activ(x)
        

        return x

class BlockForResnet(MetaModule):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, dilation, bias, with_dropout, transposed, activ=torch.nn.ReLU(inplace=False), init=None, do_norm=False, is_first_layer=False ):
        super(BlockForResnet, self).__init__()
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.bias=bias 
        self.conv= None
        self.norm= None
        # self.relu=torch.nn.ReLU(inplace=False)
        self.activ=activ
        self.with_dropout=with_dropout
        self.transposed=transposed
        # self.cc=ConcatCoord()
        self.init=init
        self.do_norm=do_norm
        self.is_first_layer=is_first_layer

        if with_dropout:
            self.drop=torch.nn.Dropout2d(0.2)

        # self.conv=None

        # self.norm = torch.nn.BatchNorm2d(in_channels, momentum=0.01).cuda()
        self.norm = torch.nn.BatchNorm2d(in_channels).cuda()
        # self.norm = torch.nn.BatchNorm2d(self.out_channels, momentum=0.01).cuda()
        # self.norm = torch.nn.GroupNorm(1, in_channels).cuda()


        if not self.transposed:
            self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )
        else:
            self.conv= MetaSequential( MetaConv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1, bias=self.bias).cuda()  )

        if self.init=="zero":
                torch.nn.init.zeros_(self.conv[-1].weight) 
        if self.activ==torch.sin:
            with torch.no_grad():
                # print(":we are usign sin")
                # print("in channels is ", in_channels, " and conv weight size 1 is ", self.conv.weight.size(-1) )
                # num_input = self.conv.weight.size(-1)
                num_input = in_channels
                # num_input = self.out_channels
                # See supplement Sec. 1.5 for discussion of factor 30
                if self.is_first_layer:
                    # self.conv[-1].weight.uniform_(-1 / num_input, 1 / num_input)
                    # self.conv[-1].weight.uniform_(-1 / num_input*2, 1 / num_input*2)
                    self.conv[-1].weight.uniform_(-1 / num_input, 1 / num_input)
                    # print("conv 1 is ", self.conv[-1].weight )
                else:
                    self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input)/30 , np.sqrt(6 / num_input)/30 )
                    # self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
                    # self.conv[-1].weight.uniform_(-np.sqrt(6 / num_input)/7 , np.sqrt(6 / num_input)/7 )
                    # print("conv any other is ", self.conv[-1].weight )
        else : # assume we have a relu
            print("initializing with kaiming uniform")
            torch.nn.init.kaiming_uniform_(self.conv[-1].weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.conv[-1].weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.conv[-1].bias, -bound, bound)

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        # print("params is", params)

        # x=self.cc(x)

        in_channels=x.shape[1]


        x=self.norm(x)
        x=self.activ(x)
        x = self.conv(x, params=get_subdict(params, 'conv') )

        return x






class ResnetBlock2D(torch.nn.Module):

    def __init__(self, out_channels, kernel_size, stride, padding, dilations, biases, with_dropout, do_norm=False, activ=torch.nn.ReLU(inplace=False), is_first_layer=False, block_type=WNReluConv ):
    # def __init__(self, out_channels, kernel_size, stride, padding, dilations, biases, with_dropout, do_norm=False, activ=torch.nn.GELU(), is_first_layer=False ):
    # def __init__(self, out_channels, kernel_size, stride, padding, dilations, biases, with_dropout, do_norm=False, activ=torch.nn.GELU(), is_first_layer=False ):
        super(ResnetBlock2D, self).__init__()

        #again with bn-relu-conv
        # self.conv1=GnReluConv(out_channels, kernel_size=3, stride=1, padding=1, dilation=dilations[0], bias=biases[0], with_dropout=False, transposed=False)
        # self.conv2=GnReluConv(out_channels, kernel_size=3, stride=1, padding=1, dilation=dilations[0], bias=biases[0], with_dropout=with_dropout, transposed=False)

        # self.conv1=BlockForResnet(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=False, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=is_first_layer )
        # self.conv2=BlockForResnet(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=with_dropout, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=False )

        # self.conv1=BlockPAC(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=False, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=is_first_layer )
        # self.conv2=BlockPAC(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=with_dropout, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=False )

        self.conv1=block_type(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=False, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=is_first_layer )
        self.conv2=block_type(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=with_dropout, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=False )

    def forward(self, x):
        identity=x
        # x=self.conv1(x, x)
        # x=self.conv2(x, x)
        x=self.conv1(x)
        x=self.conv2(x)
        x+=identity
        return x


class ResnetBlockNerf(torch.nn.Module):

    def __init__(self, out_channels, kernel_size, stride, padding, dilations, biases, with_dropout, do_norm=False, activ=torch.nn.ReLU(inplace=False), is_first_layer=False ):
    # def __init__(self, out_channels, kernel_size, stride, padding, dilations, biases, with_dropout, do_norm=False, activ=torch.nn.GELU(), is_first_layer=False ):
        super(ResnetBlockNerf, self).__init__()

        #again with bn-relu-conv
        # self.conv1=GnReluConv(out_channels, kernel_size=3, stride=1, padding=1, dilation=dilations[0], bias=biases[0], with_dropout=False, transposed=False)
        # self.conv2=GnReluConv(out_channels, kernel_size=3, stride=1, padding=1, dilation=dilations[0], bias=biases[0], with_dropout=with_dropout, transposed=False)

        # self.conv1=BlockForResnet(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=False, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=is_first_layer )
        # self.conv2=BlockForResnet(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=with_dropout, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=False )

        self.conv1=BlockSiren(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=False, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=is_first_layer )
        self.conv2=BlockSiren(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilations[0], bias=biases[0], with_dropout=with_dropout, transposed=False, do_norm=do_norm, activ=activ, is_first_layer=False )

        # self.conv1=ConvRelu(out_channels, kernel_size=3, stride=1, padding=1, dilation=dilations[0], bias=biases[0], with_dropout=False, transposed=False)
        # self.conv2=ConvRelu(out_channels, kernel_size=3, stride=1, padding=1, dilation=dilations[0], bias=biases[0], with_dropout=with_dropout, transposed=False)

    def forward(self, x):
        identity=x
        x=self.conv1(x)
        x=self.conv2(x)
        x+=identity
        return x




class ConcatCoord(torch.nn.Module):
    def __init__(self):
        super(ConcatCoord, self).__init__()

    def forward(self, x):

        #concat the coordinates in x an y as in coordconv https://github.com/Wizaron/coord-conv-pytorch/blob/master/coord_conv.py
        image_height=x.shape[2]
        image_width=x.shape[3]
        x_coords = 2.0 * torch.arange(image_width).unsqueeze(
            0).expand(image_height, image_width) / (image_width - 1.0) - 1.0
        y_coords = 2.0 * torch.arange(image_height).unsqueeze(
            1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        coords = torch.stack((x_coords, y_coords), dim=0).float()
        coords=coords.unsqueeze(0)
        coords=coords.repeat(x.shape[0],1,1,1)
        # print("coords have size ", coords.size())
        x_coord = torch.cat((coords.to("cuda"), x), dim=1)

        return x_coord

class LearnedPE(MetaModule):
    def __init__(self, in_channels, num_encoding_functions, logsampling ):
        super(LearnedPE, self).__init__()
        self.num_encoding_functions=num_encoding_functions
        self.logsampling=logsampling

        out_channels=in_channels*self.num_encoding_functions*2
       
        # self.conv= torch.nn.Linear(in_channels, out_channels, bias=True).cuda()  
        self.conv= MetaLinear(in_channels, int(out_channels/2), bias=True).cuda()  #in the case we set the weight ourselves

        with torch.no_grad():
            num_input = in_channels
            self.conv.weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input) )
            print("weight is ", self.conv.weight.shape) #60x3
            
            #we make the same as the positonal encoding, which is mutiplying each coordinate with this linespaced frequencies
            lin=2.0 ** torch.linspace(
                0.0,
                num_encoding_functions - 1,
                num_encoding_functions,
                dtype=torch.float32,
                device=torch.device("cuda"),
            )
            lin_size=lin.shape[0]
            weight=torch.zeros([in_channels, num_encoding_functions*in_channels], dtype=torch.float32, device=torch.device("cuda") )
            for i in range(in_channels):
                weight[i:i+1,   i*lin_size:i*lin_size+lin_size ] = lin

            weight=weight.t().contiguous()

            #set the new weights = 
            self.conv.weight=torch.nn.Parameter(weight)
            # self.conv.weight.requires_grad=False
            print("weight is", weight.shape)
            print("bias is", self.conv.bias.shape)
            print("weight is", weight)


    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        if len(x.shape)!=2:
            print("LeanerPE forward: x should be a NxM matrix so 2 dimensions but it actually has ", x.shape,  " so the lenght is ", len(x.shape) )
            exit(1)
        
        # x_input=x

        # print("self.conv.weight", self.conv.weight)

        # print("x ", x.shape)
        x_proj = self.conv(x, params=get_subdict(params, 'conv'), incremental=True)
        # print("learned pe is ", self.conv.weight)
        # print("after conv", x.shape)
        # x=90*x
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj), x], 1)


#the lerned gaussian pe as in the work of Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
#should work better than just a PE on each axis.
class LearnedPEGaussian(MetaModule):
    def __init__(self, in_channels, out_channels, std ):
        super(LearnedPEGaussian, self).__init__()

        # self.b = torch.nn.Parameter(torch.randn(in_channels, int(out_channels/2) ))
        # self.bias = torch.nn.Parameter(torch.randn(1, int(out_channels/2) ))
        # torch.nn.init.normal_(self.b, 0.0, std)


        #with conv so we can control it using the hyernet
        self.conv= MetaLinear(in_channels, int(out_channels/2), bias=True).cuda()  #in the case we set the weight ourselves
        torch.nn.init.normal_(self.conv.weight, 0.0, std*2.0*3.141592)


    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        if len(x.shape)!=2:
            print("LeanerPE forward: x should be a NxM matrix so 2 dimensions but it actually has ", x.shape,  " so the lenght is ", len(x.shape) )
            exit(1)

        # # b=self.b.repeat(self.in_channels ,1)
        # mat=2.0*3.141592*self.b
        # x_proj = torch.matmul(x, mat)
        # x_proj=x_proj+self.bias
        # # print("x is ", x.shape)
        # # print("xproj is ", x_proj.shape)
        # return torch.cat([torch.sin(x_proj), torch.cos(x_proj), x], 1)



        #with conv so we can control it with hypernet
        x_proj = self.conv(x, params=get_subdict(params, 'conv'), incremental=True)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj), x], 1)


class LearnedPEGaussian2(MetaModule):
    def __init__(self, in_channels, out_channels, std, num_encoding_functions, logsampling  ):
        super(LearnedPEGaussian2, self).__init__()

        # self.in_channels=in_channels

        # self.conv= torch.nn.Linear(in_channels, out_channels, bias=True).cuda()  
        # self.conv= MetaLinear(in_channels, out_channels bias=True).cuda()  #in the case we set the weight ourselves
        # self.b = torch.nn.Parameter(torch.randn(in_channels, int(out_channels/2) ))
        self.b = torch.nn.Parameter(torch.randn(in_channels, int(out_channels/2) ))
        torch.nn.init.normal_(self.b, 0.0, std)

        self.learnedpe=LearnedPE(in_channels=in_channels, num_encoding_functions=num_encoding_functions, logsampling=logsampling)


    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        if len(x.shape)!=2:
            print("LeanerPE forward: x should be a NxM matrix so 2 dimensions but it actually has ", x.shape,  " so the lenght is ", len(x.shape) )
            exit(1)

        # b=self.b.repeat(self.in_channels ,1)
        mat=2.0*3.141592*self.b
        x_proj = torch.matmul(x, mat)
        # print("x is ", x.shape)
        # print("xproj is ", x_proj.shape)
        gx=torch.cat([torch.sin(x_proj), torch.cos(x_proj), x], 1)

        pex=self.learnedpe(x)

        x=torch.cat([gx,pex],1)

        return x




class SplatTextureModule(torch.nn.Module):
    def __init__(self):
    # def __init__(self, nr_filters, neighbourhood_size, dilation=1):
        super(SplatTextureModule, self).__init__()
        self.first_time=True
        # self.neural_mvs=NeuralMVS.create()
        # print("creating neural_mvs----------")


    def forward(self, values_tensor, uv_tensor, texture_size):

        # homogeneous=torch.ones(values_tensor.shape[0],1).to("cuda")
        # values_with_homogeneous=torch.cat([values_tensor, homogeneous],1)

        texture = SplatTexture.apply(values_tensor, uv_tensor, texture_size)
        # texture = SplatTexture.apply(values_with_homogeneous, uv_tensor, texture_size)
        
        return texture


class SliceTextureModule(torch.nn.Module):
    def __init__(self):
    # def __init__(self, nr_filters, neighbourhood_size, dilation=1):
        super(SliceTextureModule, self).__init__()
        self.first_time=True
        # self.neural_mvs=NeuralMVS.create()

    def forward(self, texture, uv_tensor):

        values_not_normalized = SliceTexture.apply(texture, uv_tensor)

        #normalize by the homogeneous coord
        val_dim = values_not_normalized.shape[1] - 1
        homogeneous = values_not_normalized[:, val_dim:val_dim+1].clone()
        values=values_not_normalized[:, 0:val_dim] / ( homogeneous + 1e-5)

        
        return values, homogeneous, values_not_normalized




#LATTICE 

class ConvRelu(torch.nn.Module):
    def __init__(self, nr_filters, dilation, bias, with_dropout):
        super(ConvRelu, self).__init__()
        self.nr_filters=nr_filters
        self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias)
        self.relu = torch.nn.ReLU(inplace=False)
        self.with_dropout=with_dropout
        if with_dropout:
            self.drop=DropoutLattice(0.2)
        # self.relu = torch.nn.ReLU()
    def forward(self, lv, ls):

        ls.set_values(lv)

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        # lv=gelu(lv)
        if self.with_dropout:
            lv = self.drop(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.conv(lv, ls)
        lv_1=self.relu(lv_1)


        ls_1.set_values(lv_1)

        return lv_1, ls_1


class CoarsenRelu(torch.nn.Module):
    def __init__(self, nr_filters, bias):
        super(CoarsenRelu, self).__init__()
        self.nr_filters=nr_filters
        self.coarse=CoarsenLatticeModule(nr_filters=nr_filters, bias=bias)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, lv, ls, concat_connection=None):

        ls.set_values(lv)

        #similar to densenet and resnet: bn, relu, conv
        ls.set_values(lv)
        lv_1, ls_1 = self.coarse(lv, ls)
        lv_1=self.relu(lv_1)

        return lv_1, ls_1


class FinefyRelu(torch.nn.Module):
    def __init__(self, nr_filters, bias):
        super(FinefyRelu, self).__init__()
        self.nr_filters=nr_filters
        self.fine=FinefyLatticeModule(nr_filters=nr_filters, bias=bias)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, lv_coarse, ls_coarse, ls_fine):

        ls_coarse.set_values(lv_coarse)

        #similar to densenet and resnet: bn, relu, conv
        ls_coarse.set_values(lv_coarse)
        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)
        lv_1=self.relu(lv_1)

        ls_1.set_values(lv_1)

        return lv_1, ls_1





class ReluConv(torch.nn.Module):
    def __init__(self, nr_filters, dilation, bias, with_dropout):
        super(ReluConv, self).__init__()
        self.nr_filters=nr_filters
        self.conv=ConvLatticeModule(nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias)
        self.relu = torch.nn.ReLU(inplace=False)
        self.with_dropout=with_dropout
        if with_dropout:
            self.drop=DropoutLattice(0.2)
        # self.relu = torch.nn.ReLU()
    def forward(self, lv, ls):

        ls.set_values(lv)
        lv=self.relu(lv)

        #similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        # lv=gelu(lv)
        if self.with_dropout:
            lv = self.drop(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.conv(lv, ls)


        ls_1.set_values(lv_1)

        return lv_1, ls_1


class ReluCoarsen(torch.nn.Module):
    def __init__(self, nr_filters, bias):
        super(ReluCoarsen, self).__init__()
        self.nr_filters=nr_filters
        self.coarse=CoarsenLatticeModule(nr_filters=nr_filters, bias=bias)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, lv, ls, concat_connection=None):

        ls.set_values(lv)
        lv=self.relu(lv)

        #similar to densenet and resnet: bn, relu, conv
        ls.set_values(lv)
        lv_1, ls_1 = self.coarse(lv, ls)

        return lv_1, ls_1


class ReluFinefy(torch.nn.Module):
    def __init__(self, nr_filters, bias):
        super(ReluFinefy, self).__init__()
        self.nr_filters=nr_filters
        self.fine=FinefyLatticeModule(nr_filters=nr_filters, bias=bias)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, lv_coarse, ls_coarse, ls_fine):

        ls_coarse.set_values(lv_coarse)
        lv_coarse=self.relu(lv_coarse)


        #similar to densenet and resnet: bn, relu, conv
        ls_coarse.set_values(lv_coarse)
        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)

        ls_1.set_values(lv_1)

        return lv_1, ls_1







class FinefyOnly(torch.nn.Module):
    def __init__(self, nr_filters, bias):
        super(FinefyOnly, self).__init__()
        self.nr_filters=nr_filters
        self.fine=FinefyLatticeModule(nr_filters=nr_filters, bias=bias)
    def forward(self, lv_coarse, ls_coarse, ls_fine):

        ls_coarse.set_values(lv_coarse)

        #similar to densenet and resnet: bn, relu, conv
        ls_coarse.set_values(lv_coarse)
        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)

        ls_1.set_values(lv_1)

        return lv_1, ls_1





class ResnetBlockConvReluLattice(torch.nn.Module):

    def __init__(self, nr_filters, dilations, biases, with_dropout):
        super(ResnetBlockConvReluLattice, self).__init__()
        
        #again with bn-relu-conv
        self.conv1=ConvRelu(nr_filters, dilations[0], biases[0], with_dropout=False)
        self.conv2=ConvRelu(nr_filters, dilations[1], biases[1], with_dropout=with_dropout)

        # self.conv1=GnReluDepthwiseConv(nr_filters, dilations[0], biases[0], with_dropout=False)
        # self.conv2=GnReluDepthwiseConv(nr_filters, dilations[1], biases[1], with_dropout=with_dropout)

        # self.residual_gate  = torch.nn.Parameter( torch.ones( 1 ).to("cuda") ) #gate for the skip connection https://openreview.net/pdf?id=Sywh5KYex


    def forward(self, lv, ls):
      

        identity=lv

        ls.set_values(lv)

        # print("conv 1")
        lv, ls=self.conv1(lv,ls)
        # print("conv 2")
        lv, ls=self.conv2(lv,ls)
        # print("finished conv 2")
        # lv=lv*self.residual_gate
        lv+=identity
        ls.set_values(lv)
        return lv, ls


class ResnetBlockReluConvLattice(torch.nn.Module):

    def __init__(self, nr_filters, dilations, biases, with_dropout):
        super(ResnetBlockReluConvLattice, self).__init__()
        
        #again with bn-relu-conv
        self.conv1=ReluConv(nr_filters, dilations[0], biases[0], with_dropout=False)
        self.conv2=ReluConv(nr_filters, dilations[1], biases[1], with_dropout=with_dropout)

        # self.conv1=GnReluDepthwiseConv(nr_filters, dilations[0], biases[0], with_dropout=False)
        # self.conv2=GnReluDepthwiseConv(nr_filters, dilations[1], biases[1], with_dropout=with_dropout)

        # self.residual_gate  = torch.nn.Parameter( torch.ones( 1 ).to("cuda") ) #gate for the skip connection https://openreview.net/pdf?id=Sywh5KYex


    def forward(self, lv, ls):
      

        identity=lv

        ls.set_values(lv)

        # print("conv 1")
        lv, ls=self.conv1(lv,ls)
        # print("conv 2")
        lv, ls=self.conv2(lv,ls)
        # print("finished conv 2")
        # lv=lv*self.residual_gate
        lv+=identity
        ls.set_values(lv)
        return lv, ls