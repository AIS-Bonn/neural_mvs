#!/usr/bin/env python3.6

import os
import numpy as np
import sys
try:
  import torch
except ImportError:
    pass
from easypbr  import *
from dataloaders import *
# np.set_printoptions(threshold=sys.maxsize)

config_file="train.cfg"

config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../config', config_file)
view=Viewer.create(config_path) #first because it needs to init context




def show_3D_points(points_3d_tensor, color=None):
    mesh=Mesh()
    mesh.V=points_3d_tensor.detach().double().reshape((-1, 3)).cpu().numpy()

    if color is not None:
        color_channels_last=color.permute(0,2,3,1).detach() # from n,c,h,w to N,H,W,C
        color_channels_last=color_channels_last.view(-1,3).contiguous()
        # color_channels_last=color_channels_last.permute() #from bgr to rgb
        color_channels_last=torch.index_select(color_channels_last, 1, torch.LongTensor([2,1,0]).cuda() ) #switch the columns so that we grom from bgr to rgb
        mesh.C=color_channels_last.detach().double().reshape((-1, 3)).cpu().numpy()
        mesh.m_vis.set_color_pervertcolor()

    mesh.m_vis.m_show_points=True
    # Scene.show(mesh, name)

    return mesh


def ndc_rays (H , W , fx, fy , near , rays_o , rays_d, project_to_near):
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


# def ndc_rays(H, W, focal, near, rays_o, rays_d):
#     # UNTESTED, but fairly sure.

#     # Shift rays origins to near plane
#     t = -(near + rays_o[..., 2]) / rays_d[..., 2]
#     rays_o = rays_o + t[..., None] * rays_d

#     # Projection
#     o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
#     o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
#     o2 = 1.0 + 2.0 * near / rays_o[..., 2]

#     d0 = (
#         -1.0
#         / (W / (2.0 * focal))
#         * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
#     )
#     d1 = (
#         -1.0
#         / (H / (2.0 * focal))
#         * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
#     )
#     d2 = -2.0 * near / rays_o[..., 2]

#     rays_o = torch.stack([o0, o1, o2], -1)
#     rays_d = torch.stack([d0, d1, d2], -1)

#     return rays_o, rays_d

def transform_to_ndc(cloud, frame, near, far):

    cloud.apply_model_matrix_to_cpu(False)
    cloud.remove_vertices_at_zero()
    # xyz2NDC = intrinsics_to_opengl_proj(frame_color.K, frame_color.width, frame_color.height, 0.01, 10)
    n=near
    f=far
    # r= frame.width #right
    # t= frame.height #top
    # r= 0.22 #right and top bounds of the scene at the near clipping plane #TODO gotte calculate them correctly
    # t= 0.22
    # r= 0.0974 #right and top bounds of the scene at the near clipping plane #TODO gotte calculate them correctly
    # t= 0.0760
    # xyz2NDC = np.matrix([
    #                 [n/r, 0, 0, 0], 
    #                 [0, n/t, 0, 0], 
    #                 [0, 0, -(f+n)/(f-n), 2*f*n/(f-n) ], 
    #                 [0, 0, -1, 0], 
    #                 ])
    xyz2NDC = intrinsics_to_opengl_proj(frame.K, frame.width, frame.height, near, far)
    V_xyzw = np.c_[ cloud.V, np.ones(cloud.V.shape[0]) ]
    V_clip = np.matmul(V_xyzw, xyz2NDC)
    print("V_clip", V_clip)
    print("V_clip has min max", np.min(V_clip), " ", np.max(V_clip)  )
    V_ndc= V_clip[:,0:3]/(V_clip[:,3:4])
    print("V_ndc", V_ndc)
    # V_ndc /= frame_color.width # get it from the [0,frame.width] to [0,1]

    #also rotate the directions 
    NV_xyzw = np.c_[ cloud.NV, np.zeros(cloud.NV.shape[0]) ]
    NV_clip = np.matmul(NV_xyzw, xyz2NDC)
    NV_ndc= NV_clip[:,0:3]/(NV_clip[:,3:4])
    # NV_ndc= NV_clip[:,0:3]
    #flip y and z
    # V_ndc[:, 1] = -V_ndc[:, 1] 
    # V_ndc[:, 2] = -V_ndc[:, 2] 
    # V_ndc[:, 0] = -V_ndc[:, 0] 
    print("V_ndc has min max", np.min(V_ndc), " ", np.max(V_ndc)  )
    print("V_ndc, xy has min max", np.min(V_ndc[:,0:2]), " ", np.max(V_ndc[:,0:2])  )
    cloudNDC=Mesh()
    cloudNDC.V=V_ndc
    cloudNDC.NV=NV_ndc
    cloudNDC.C = cloud.C.copy()
    cloudNDC.m_vis.m_show_points=True
    cloudNDC.m_vis.set_color_pervertcolor()

    return cloudNDC


def test_ndc():
    loader=DataLoaderVolRef(config_path)
    loader.start()

    first=False


    #make a cube of size 2
    box_ndc = Mesh()
    box_ndc.create_box_ndc()
    # Scene.show(box_ndc, "box_ndc")

    while True:
        if(loader.has_data() and not first ): 
        # if(loader.has_data()  ): 

            first=True

            #volref 
            # print("got frame")
            frame_color=loader.get_color_frame()
            frame_depth=loader.get_depth_frame()

            print("frame_depth has height and width", frame_depth.height, " ", frame_depth.width)

            #move the frame forwards so that we start with the clouds at z0 and not in the negative part
            # tf_cam_world=frame_color.tf_cam_world.clone()
            # tf_world_cam = tf_cam_world.inverse()
            # tf_world_cam.translate([0,0,-1.5])
            # frame_color.tf_cam_world= tf_world_cam.inverse()
            # #same for the depth frame 
            # tf_cam_world=frame_depth.tf_cam_world.clone()
            # tf_world_cam = tf_cam_world.inverse()
            # tf_world_cam.translate([0,0,-1.5])
            # frame_depth.tf_cam_world= tf_world_cam.inverse()

            #just set the pose to identity 
            frame_color.tf_cam_world.set_identity()
            frame_depth.tf_cam_world.set_identity()





            Gui.show(frame_color.rgb_32f, "rgb")

            rgb_with_valid_depth=frame_color.rgb_with_valid_depth(frame_depth)
            Gui.show(rgb_with_valid_depth, "rgb_valid")


            frustum_mesh=frame_depth.create_frustum_mesh(0.02)
            frustum_mesh.m_vis.m_line_width=3
            frustum_name="frustum"
            Scene.show(frustum_mesh, frustum_name)
            # Scene.show(frustum_mesh, "frustum_"+str(frame_color.frame_idx) )

            cloud=frame_depth.depth2world_xyz_mesh()
            frame_color.assign_color(cloud) #project the cloud into this frame and creates a color matrix for it
            #move the frame in front a bit so that the clouds actually starts at zero bcause currently we have part of it in the negative z and part in the positive z

            # near =0.001
            # far = 9999

            near =0.2
            # far = 9999

            show_ndc = False
            if show_ndc:
                V=cloud.V.copy()
                V[:,2:3] -=2
                cloud.V=V
                Scene.show(cloud, "cloud")

                #make the clouds into NDC
                cloud.remove_vertices_at_zero()
                # xyz2NDC = intrinsics_to_opengl_proj(frame_color.K, frame_color.width, frame_color.height, 0.01, 10)
                # n=0.001 #near
                # f=10 #far
                # # r= frame_color.width #right
                # # t= frame_color.height #top
                # r= 1.0 #right
                # t= 1.0
                # xyz2NDC = np.matrix([
                #                 [n/r, 0, 0, 0], 
                #                 [0, n/t, 0, 0], 
                #                 [0, 0, -(f+n)/(f-n), 2*f*n/(f-n) ], 
                #                 [0, 0, -1, 0], 
                #                 ])
                # max_size = np.maximum(frame_color.width, frame_color.height)
                xyz2NDC = intrinsics_to_opengl_proj(frame_depth.K, frame_depth.width, frame_depth.height, near, far)
                # xyz2NDC = intrinsics_to_opengl_proj(frame_color.K, frame_color.width/max_size, frame_color.height/max_size, 0.001, 10)
                V_xyzw = np.c_[ V, np.ones(cloud.V.shape[0]) ]
                V_clip = np.matmul(V_xyzw, xyz2NDC)
                print("V_clip has min max", np.min(V_clip), " ", np.max(V_clip)  )
                V_ndc= V_clip[:,0:3]/(V_clip[:,3:4])
                V_ndc /= frame_color.width # get it from the [0,frame.width] to [0,1]
                #flip y and z
                # V_ndc[:, 1] = -V_ndc[:, 1] 
                # V_ndc[:, 2] = -V_ndc[:, 2] 
                # V_ndc[:, 0] = -V_ndc[:, 0] 
                print("V_ndc has min max", np.min(V_ndc), " ", np.max(V_ndc)  )
                print("V_ndc, xy has min max", np.min(V_ndc[:,0:2]), " ", np.max(V_ndc[:,0:2])  )
                cloudNDC=Mesh()
                cloudNDC.V=V_ndc
                cloudNDC.C = cloud.C.copy()
                cloudNDC.m_vis.m_show_points=True
                cloudNDC.m_vis.set_color_pervertcolor()
                # Scene.show(cloudNDC, "cloudNDC")


                # #go from NDC back to original cloud
                # NDC2xyz = np.linalg.inv(xyz2NDC)
                # V_ndc = V_ndc*frame_color.width
                # V_ndc_xyzw = np.c_[ V_ndc, np.ones(cloud.V.shape[0]) ]
                # V_xyzw= np.matmul(V_ndc_xyzw, NDC2xyz) 
                # V_xyz= V_xyzw[:,0:3]/(V_xyzw[:,3:4])
                # roundback = Mesh()
                # roundback.V=V_xyz
                # roundback.C = roundback.C.copy()
                # roundback.m_vis.m_show_points=True
                # roundback.m_vis.set_color_pervertcolor()
                # Scene.show(roundback, "roundback")


            #get rays and show them
            Scene.show(cloud, "cloud")
            ray_meshes_list=[]
            ray_dirs_mesh=frame_depth.pixels2dirs_mesh()
            # ray_dirs=ray_dirs_mesh.V.copy()
            ray_dirs=torch.from_numpy(ray_dirs_mesh.V.copy()).float()
            ray_dirs= -ray_dirs
            depth_per_pixel =   torch.ones([frame_depth.height* frame_depth.width, 1], dtype=torch.float32) 
            depth_per_pixel.fill_(near)
            camera_center=torch.from_numpy( frame_depth.pos_in_world() )
            camera_center=camera_center.view(1,3)
            nr_layers=30
            layer_spacing=0.15
            for i in range(nr_layers):
                points3D = camera_center + depth_per_pixel*ray_dirs #N,3,H,W
                if i == 0:
                    print("for the first layer the min max x and y is")
                    min_x = points3D[:,0:1].min()
                    max_x = points3D[:,0:1].max()
                    min_y = points3D[:,1:2].min()
                    max_y = points3D[:,1:2].max()
                    print("min x,", min_x, "max x", max_x, " min_y ", min_y, " max_y ", max_y)
                rays_vis = show_3D_points(points3D)
                rays_vis.NV = ray_dirs.detach().double().reshape((-1, 3)).cpu().numpy()
                rays_vis.m_vis.m_show_normals=True
                rays_vis.C =  np.ones( (cloud.V.shape[0],3) )* i/nr_layers
                ray_meshes_list.append(rays_vis)
                depth_per_pixel+=layer_spacing
            #apppend all of the meshes 
            rays_vis=Mesh()
            rays_vis.m_vis.m_show_points=True
            # rays_vis.m_vis.m_show_normals=True
            for i in range(nr_layers):
                rays_vis.add(ray_meshes_list[i])
            Scene.show(rays_vis, "rays" )

            far= (nr_layers+1)*layer_spacing
            all_points= torch.from_numpy(rays_vis.V.copy())
            point_dist = all_points.norm(dim=1)
            near = point_dist.min()
            far = point_dist.max()
            print("point_dist", point_dist)
            print("far is ", far)
            print("near is ", near)
            print("depth_per_pixel is ", depth_per_pixel)

            #show the NDC of the rays 
            # rays_ndc = transform_to_ndc(rays_vis, frame_depth, near, far)
            # rays_ndc.m_vis.m_show_normals=True
            # Scene.show(rays_ndc, "rays_ndc" )

            #make te origins and direcitons similar to what nerf does in here, at the end is pytorch code ndc_derivation.pdf
            rays_o = torch.from_numpy(rays_vis.V)
            rays_d = torch.from_numpy(rays_vis.NV)
            fx= frame_depth.K[0,0]
            fy= frame_depth.K[1,1]
            ndc_origins, ndc_dirs = ndc_rays (frame_depth.height , frame_depth.width , fx, fy, near, rays_o , rays_d , project_to_near=False )
            # ndc_origins, ndc_dirs = ndc_rays (frame_depth.height , frame_depth.width , frame_depth.K[0,0],  near, rays_o , rays_d )
            # print("ndc_dirs", ndc_dirs)
            NDC_rays_vis = show_3D_points(ndc_origins)
            NDC_rays_vis.NV = ndc_dirs.detach().double().reshape((-1, 3)).cpu().numpy()
            NDC_rays_vis.m_vis.m_show_normals=True
            # Scene.show(NDC_rays_vis, "NDC_rays_vis" )


            #project back from ndc to xyz
            # print("ndc_origins", ndc_origins.shape)
            x_ndc = ndc_origins[:, 0:1]
            y_ndc = ndc_origins[:, 1:2]
            z_ndc = ndc_origins[:, 2:3]
            print("z_ndc is ", z_ndc)
            # z = 2 / (z_ndc - 1)
            z = 2* near / (z_ndc - 1)
            # z = 1 / (1-z_ndc )
            x = -x_ndc * z * frame_depth.width / 2 / fx
            y = -y_ndc * z * frame_depth.height / 2 / fy
            points_xyz= torch.cat([x,y,z],1)
            rounback_xyz_mesh = show_3D_points(points_xyz)
            rounback_xyz_mesh.m_vis.m_point_size=5.0
            rounback_xyz_mesh.m_vis.m_point_color=[0.0, 1.0, 0.0]
            Scene.show(rounback_xyz_mesh, "rounback_xyz_mesh" )



            



        if loader.is_finished():
            # print("resetting")
            loader.reset()
        
        view.update()


test_ndc()