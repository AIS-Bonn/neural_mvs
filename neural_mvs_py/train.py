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
    experiment_name="6"

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
    model=Net2(model_params).to("cuda")
    # model.train()
    # model.half()

    loss_fn=torch.nn.MSELoss()
    scheduler=None
    concat_coord=ConcatCoord() 
    smooth = InverseDepthSmoothnessLoss()

    # show_every=39
    show_every=10

    
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


    #depth min max for nerf 
    depth_min=2
    depth_max=5
    #depth min max for home photos
    depth_min=4.3
    depth_max=11.5

    new_frame=None

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
                # for frame_idx, frame in enumerate(frames_all_selected):
                for i in range(phase.loader.nr_samples()):
                    frame=phase.loader.get_random_frame() 
                    # pass
                    TIME_START("all")
                    # mask_tensor=mat2tensor(frame.mask, False).to("cuda").repeat(1,1,1,1)
                    rgb_gt=mat2tensor(frame.rgb_32f, False).to("cuda")



                    #forward
                    with torch.set_grad_enabled(is_training):

                        TIME_START("forward")
                        rgb_pred, depth_map, acc_map, new_loss =model(frame, mesh_full, depth_min, depth_max, use_chunking=False)
                        TIME_END("forward")

                        #VIS 
                        # rgb_pred_mat=tensor2mat(rgb_pred)
                        # Gui.show(rgb_pred_mat, "rgb_pred")
           

                        loss=0
                        rgb_loss=(( rgb_gt-rgb_pred)**2).mean()
                        loss+=rgb_loss
                     
                      
                        #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                        if first_time:
                            first_time=False
                            # model.half()
                            # optimizer=RAdam( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                            optimizer=torch.optim.AdamW( model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay() )
                            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.1)
                            # scheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)
                            # lambda1 = lambda epoch: 0.9999 ** phase.iter_nr
                            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
                            optimizer.zero_grad()

                        cb.after_forward_pass(loss=rgb_loss.item(), phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 
                        # cb.after_forward_pass(loss=0, phase=phase, lr=0) #visualizes the prediction 

                    
                  



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

                    TIME_END("all")


                    #novel views after computing gradients and so on
                    #show a novel view 
                    if phase.iter_nr%show_every==0:
                        with torch.set_grad_enabled(is_training):
                            #create novel view
                            if new_frame==None:
                                new_frame=Frame()
                                frame_to_start=loader_train.get_frame_at_idx(0)
                                new_frame.tf_cam_world=frame_to_start.tf_cam_world
                                new_frame.K=frame_to_start.K.copy()
                                new_frame.height=frame_to_start.height
                                new_frame.width=frame_to_start.width
                                # translate a bit at the beggining so it starts more centered
                                # model_matrix = new_frame.tf_cam_world.inverse()
                                # model_matrix=model_matrix.translate([1.0, 0.0, 0.0])
                                # new_frame.tf_cam_world = model_matrix.inverse()
                            #rotate a bit 
                            # model_matrix = new_frame.tf_cam_world.inverse()
                            # model_matrix=model_matrix.rotate_axis_angle([0,1,0], 10)
                            # new_frame.tf_cam_world = model_matrix.inverse()
                            #attempt 2 to rotate 
                            model_matrix = new_frame.tf_cam_world.inverse()
                            model_matrix=model_matrix.orbit_y_around_point([1,0,0], 10)
                            new_frame.tf_cam_world = model_matrix.inverse()
                            #render new 
                            rgb_pred, depth_map, acc_map, new_loss =model(new_frame, mesh_full, depth_min, depth_max, use_chunking=False)
                            rgb_pred_mat=tensor2mat(rgb_pred)
                            Gui.show(rgb_pred_mat, "rgb_novel")
                            #show new frustum 
                            frustum_mesh=new_frame.create_frustum_mesh(0.2)
                            frustum_mesh.m_vis.m_line_width=1
                            frustum_mesh.m_vis.m_line_color=[0.0, 1.0, 0.0]
                            Scene.show(frustum_mesh, "frustum_novel" )





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
