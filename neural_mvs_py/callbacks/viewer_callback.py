from callbacks.callback import *
from callbacks.vis import * #for the map_range
from easypbr import Scene
from easypbr import Gui
from easypbr import Frame
from easypbr import tensor2mat, mat2tensor
import numpy as np

class ViewerCallback(Callback):

    def __init__(self):
        pass


    def after_forward_pass(self, **kwargs):
        pass

        #show the input
        # self.show_rgb(input_frame_color)
        # self.show_depth(input_frame_depth)
        # self.show_frustum(input_frame_depth)

        #show the rgb valid
        # rgb_valid=input_frame_color.rgb_with_valid_depth(input_frame_depth)
        # Gui.show(rgb_valid, "rgb_valid")

        #show the output color from the network
        # output_color_mat=tensor2mat(output_color) #output color is in bgr
        # Gui.show(output_color_mat, "output_color_mat")

        # #show the output
        # frame_depth_output= input_frame_depth
        # depth_tensor_clamped=map_range(output_tensor, 1.2, 1.7, 0.0, 1.0)
        # frame_depth_output.depth=tensor2mat(depth_tensor_clamped)
        # Gui.show(frame_depth_output.depth, "depth_output")

        # #show the mesh with the depth predicted by the net
        # frame_depth_output.depth=tensor2mat(output_tensor)
        # self.show_colored_cloud(input_frame_color, frame_depth_output, "cloud_output")




    def show_rgb(self, frame_color):
        Gui.show(frame_color.rgb_32f, "rgb")

    def show_depth(self, frame_depth):
        # depth_tensor=frame_depth.depth2tensor() #tensor fo size h,w,1
        depth_tensor=mat2tensor(frame_depth.depth, False)
        depth_tensor_clamped=map_range(depth_tensor, 1.2, 1.7, 0.0, 1.0)
        frame_depth.depth=tensor2mat(depth_tensor)
        # frame_depth.tensor2depth(depth_tensor_clamped) 
        Gui.show(frame_depth.depth, "depth")
        frame_depth.depth=tensor2mat(depth_tensor)
        # frame_depth.tensor2depth(depth_tensor)

    def show_frustum(self, frame):
        frustum_mesh=frame.create_frustum_mesh(0.1)
        frustum_mesh.m_vis.m_line_width=3
        Scene.show(frustum_mesh, "frustum")

    def show_colored_cloud(self, frame_color, frame_depth, name):
        cloud=frame_depth.backproject_depth()
        frame_color.assign_color(cloud) #project the cloud into this frame and creates a color matrix for it
        Scene.show(cloud, name)



