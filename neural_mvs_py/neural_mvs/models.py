import torch
from neural_mvs.modules import *

import torch
import torch.nn as nn

from neural_mvs.utils import *



class UNet(torch.nn.Module):
    def __init__(self, nr_channels_start, nr_channels_output, nr_stages, max_nr_channels=999990, block_type=WNConvActiv ):
        super(UNet, self).__init__()



        #params
        self.start_nr_channels=nr_channels_start
        self.nr_stages=nr_stages
        self.compression_factor=1.0
        self.block_type=block_type



        #DELAYED creation 
        self.first_conv=None
        cur_nr_channels=self.start_nr_channels


        self.down_stages_list = torch.nn.ModuleList([])
        self.coarsen_list = torch.nn.ModuleList([])
        self.nr_layers_ending_stage=[]
        for i in range(self.nr_stages):
            # print("cur nr_channels ", cur_nr_channels)
            print("down adding blocks with ", cur_nr_channels)
            self.down_stages_list.append( nn.Sequential(
             
                TwoBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=True, block_type=block_type  ),
                
            ))
            self.nr_layers_ending_stage.append(cur_nr_channels)
            after_coarsening_nr_channels=int(cur_nr_channels*2*self.compression_factor)
            if after_coarsening_nr_channels> max_nr_channels:
                after_coarsening_nr_channels=max_nr_channels
            print("down adding coarsen with ", after_coarsening_nr_channels)
            self.coarsen_list.append(  block_type(in_channels=cur_nr_channels, out_channels=after_coarsening_nr_channels, kernel_size=2, stride=2, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.Mish(), is_first_layer=False )  )
            cur_nr_channels= after_coarsening_nr_channels


        print("adding bottleneck ", cur_nr_channels)
        self.bottleneck=nn.Sequential(
                TwoBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=True, block_type=block_type  ),
            )

        self.up_stages_list = torch.nn.ModuleList([])
        self.squeeze_list = torch.nn.ModuleList([])
        for i in range(self.nr_stages):
            after_finefy_nr_channels=int(cur_nr_channels/2)
            print("up adding finefy with ", after_finefy_nr_channels)
            self.squeeze_list.append(  
                torch.nn.Sequential(
                    block_type(in_channels=cur_nr_channels, out_channels=after_finefy_nr_channels, kernel_size=4, stride=2, padding=1, dilation=1, bias=True, with_dropout=False, transposed=True, do_norm=True, activ=torch.nn.Mish(), is_first_layer=False )  
                )
                
                )
            #we now concat the features from the corresponding stage
            cur_nr_channels=after_finefy_nr_channels
            cur_nr_channels+= self.nr_layers_ending_stage.pop()
            print("up adding resnet with ", cur_nr_channels)
            self.up_stages_list.append( nn.Sequential(
                TwoBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=True, block_type=block_type  ),
            ))
            print("up stage which outputs nr of layers ", cur_nr_channels)



        self.last_conv = Conv2dWN(cur_nr_channels, nr_channels_output, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True).cuda() 

        self.relu=torch.nn.ReLU(inplace=False)
        self.concat_coord=ConcatCoord() 


        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.last_conv, negative_slope=1.0)

    # @profile
    # @profile_every(1)
    def forward(self, x):
        if self.first_conv==None:
            self.first_conv = self.block_type( x.shape[1], self.start_nr_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=None, is_first_layer=False ).cuda() 

        


        x=self.first_conv(x)

        features_ending_stage=[]
        #down
        for i in range(self.nr_stages):
            x=self.down_stages_list[i](x)
            features_ending_stage.append(x)
            x=self.coarsen_list[i](x)

        #bottleneck 
        x=self.bottleneck(x)

        #up
        multi_res_features=[]
        for i in range(self.nr_stages):
            
            vertical_feats= features_ending_stage.pop()
            x=self.squeeze_list[i](x) #upsample resolution and reduced the channels
            if x.shape[2]!=vertical_feats.shape[2] or x.shape[3]!=vertical_feats.shape[3]:
                x = torch.nn.functional.interpolate(x,size=(vertical_feats.shape[2], vertical_feats.shape[3]), mode='bilinear',  align_corners=False) #to make sure that the sized between the x and vertical feats match because the transposed conv may not neceserraly create the same size of the image as the one given as input
            x=torch.cat([x,vertical_feats],1)

            x=self.up_stages_list[i](x)
            multi_res_features.append(x)

        x=self.last_conv(x)

        
        return x, multi_res_features


#inspired from the ray marcher from https://github.com/vsitzmann/scene-representation-networks/blob/master/custom_layers.py
class DifferentiableRayMarcher(torch.nn.Module):
    def __init__(self):
        super(DifferentiableRayMarcher, self).__init__()

        num_encodings=8
        self.pe=PositionalEncoding(in_channels=3, num_encoding_functions=num_encodings)

        #model 
        self.lstm_hidden_size = 16
        self.lstm=None #Create this later, the volumentric feature can maybe change and therefore the features that get as input to the lstm will be different
        self.out_layer = BlockNerf(activ=None, in_channels=self.lstm_hidden_size, out_channels=1,  bias=True ).cuda()
        
        self.conv1= WNReluConv(in_channels=3+3*num_encodings*2+ 64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=None, is_first_layer=False )
        self.conv2= WNReluConv(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.GELU(), is_first_layer=False )

        #activ
        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()

        # self.feature_aggregator=  FeatureAgregatorIBRNet() 

        #params 
        self.nr_iters=20

        #starting depth per pixels 
        self.depth_per_pixel_train=None
        self.depth_per_pixel_test=None

      
    def forward(self, dataset_params, frame, ray_dirs, frames_close, frames_features, multi_res_features, weights,  novel=False):


        if novel:
            depth_per_pixel= torch.ones([1, 1, frame.height, frame.width], dtype=torch.float32, device=torch.device("cuda")) 
            depth_per_pixel.fill_(dataset_params.raymarch_depth_min)
        else:
            if (dataset_params.raymarch_depth_jitter!=0.0):
                depth_per_pixel = torch.zeros((1, 1, frame.height, frame.width), device=torch.device("cuda") ).normal_(mean=dataset_params.raymarch_depth_min, std=dataset_params.raymarch_depth_jitter)
            else: 
                depth_per_pixel= torch.ones([1, 1, frame.height, frame.width], dtype=torch.float32, device=torch.device("cuda")) 
                depth_per_pixel.fill_(dataset_params.raymarch_depth_min)


        # depth_per_pixel= torch.ones([frame.height*frame.width,1], dtype=torch.float32, device=torch.device("cuda")) 
        # depth_per_pixel.fill_(depth_min)   #randomize the deptha  bith


        nr_nearby_frames=len(frames_close)
        R_list=[]
        t_list=[]
        K_list=[]
        for i in range(len(frames_close)):
            frame_selectd=frames_close[i]
            R_list.append( frame_selectd.R_tensor.view(1,3,3) )
            t_list.append( frame_selectd.t_tensor.view(1,1,3) )
            K_list.append( frame_selectd.K_tensor.view(1,3,3) )
        R_batched=torch.cat(R_list,0)
        t_batched=torch.cat(t_list,0)
        K_batched=torch.cat(K_list,0)
        #####when we project we assume all the frames have the same size
        height=frames_close[0].height
        width=frames_close[0].width
        fx= frame.K[0,0]
        fy= frame.K[1,1]


        #attempt 2 unproject to 3D 
        camera_center=torch.from_numpy( frame.frame.pos_in_world() ).to("cuda")
        camera_center=camera_center.view(1,3,1,1)
        points3D = camera_center + depth_per_pixel*ray_dirs #N,3,H,W

        ray_dirs_original=ray_dirs #make a copy of the ray dirs because in ndc they will be differ

        points3d_mesh=show_3D_points( nchw2lin(points3D))
        Scene.show(points3d_mesh, "points_3d_init")



        #if we use ndc, we must convert the points and the rays
        if dataset_params.use_ndc:
            near= dataset_params.raymarch_depth_min - dataset_params.raymarch_depth_jitter*3
            # print("near is ", near)
            #transform from xyz to ndc
            # ray_dirs= - ray_dirs_original 
            points3D_lin = nchw2lin(points3D)
            ray_dirs_lin = nchw2lin(ray_dirs)
            points3D_lin, ray_dirs_lin = xyz_and_dirs2ndc (frame.height , frame.width , fx, fy, near, points3D_lin , ray_dirs_lin , project_to_near=False )
            points3D = lin2nchw(points3D_lin, frame.height , frame.width)
            ray_dirs = lin2nchw(ray_dirs_lin, frame.height , frame.width)

            # #show the points in ndc
            # NDC_rays_vis = show_3D_points(points3D_lin)
            # NDC_rays_vis.NV = ray_dirs_lin.detach().double().reshape((-1, 3)).cpu().numpy()
            # NDC_rays_vis.m_vis.m_show_normals=True
            # Scene.show(NDC_rays_vis, "NDC_rays_vis" )


            # #transform from ndc to xyz
            # points3D_lin = nchw2lin(points3D)
            # points3D_lin = ndc2xyz(frame.height , frame.width , fx, fy, near, points3D_lin)
            # points3D = lin2nchw(points3D_lin, frame.height , frame.width)
            # ray_dirs=ray_dirs_original

            # #show the points in xyz
            # NDC_rounback = show_3D_points(points3D_lin)
            # Scene.show(NDC_rounback, "NDC_rounback" )



        init_world_coords=points3D
        initial_depth=depth_per_pixel
        world_coords = [init_world_coords]
        depths = [initial_depth]
        signed_distances_for_marchlvl=[]
        states = [None]


        # print("")
        for iter_nr in range(self.nr_iters):
            # print("iter is ", iter_nr)
            TIME_START("raymarch_iter")
        
            #compute the features at this position 
            TIME_START("raymarch_pe")
            #using positional encoding is beneficial weather you are generalizing or overfitting. In the case of overfitting it makes the geoemtry sharper because we have more frequency to work with and in the case of generaliation it manages to get structures in the dataseet like the fact that the floor in DTU is always eithr white or brown and it doesnt need to infer this from just RGB features which can be noisy or can have problems when we have occlusion
            pos_encoded_linear=self.learned_pe(  nchw2lin(world_coords[-1]) ) 
            pos_encoded=lin2nchw(pos_encoded_linear, frame.height, frame.width).contiguous()
            TIME_END("raymarch_pe")

            # TIME_START("raymarch_uv")
            TIME_START("rm_get_and_aggr")

            if dataset_params.use_ndc:
                #transform from NDC to xyz
                world_coords[-1] = nchw2lin(world_coords[-1])
                points3D_lin = ndc2xyz(frame.height , frame.width , fx, fy, near, world_coords[-1])
                world_coords[-1] = lin2nchw(points3D_lin, frame.height , frame.width)
                ray_dirs=ray_dirs_original

                # # #show the points in xyz
                # layer_mesh = show_3D_points(points3D_lin)
                # layer_mesh.C = np.ones( (layer_mesh.V.shape[0],3) )* iter_nr/self.nr_iters
                # layer_mesh.m_vis.set_color_pervertcolor()
                # Scene.show(layer_mesh, "layer_mesh_"+str(iter_nr) )


            uv_tensor, mask=compute_uv_batched(R_batched, t_batched, K_batched, height, width, world_coords[-1] )
            # TIME_END("raymarch_uv")


            if dataset_params.use_ndc:
                #transform from xyz to ndc
                points3D_lin = nchw2lin(world_coords[-1])
                ray_dirs_lin = nchw2lin(ray_dirs)
                points3D_lin, ray_dirs_lin = xyz_and_dirs2ndc (frame.height , frame.width , fx, fy, near, points3D_lin , ray_dirs_lin , project_to_near=False )
                world_coords[-1] = lin2nchw(points3D_lin, frame.height , frame.width)
                ray_dirs = lin2nchw(ray_dirs_lin, frame.height , frame.width)

                # # #show the points in ndc
                layer_mesh = show_3D_points(points3D_lin)
                layer_mesh.C = np.ones( (layer_mesh.V.shape[0],3) )* iter_nr/self.nr_iters
                layer_mesh.m_vis.set_color_pervertcolor()
                layer_mesh.NV = ray_dirs_lin.detach().double().reshape((-1, 3)).cpu().numpy()
                layer_mesh.m_vis.m_show_normals=True
                Scene.show(layer_mesh, "ndc_layer_mesh_"+str(iter_nr) )


            # slice with grid_sample
            sliced_feat_batched=torch.nn.functional.grid_sample( frames_features, uv_tensor, align_corners=False, mode="bilinear", padding_mode="zeros" ) #sliced features is N,C,H,W
         
            
            #attempt 2 
            # img_features_aggregated= self.feature_aggregator(sliced_feat_batched, weights, novel) 
            mean=sliced_feat_batched.mean(dim=0, keepdim=True)
            std=sliced_feat_batched.std(dim=0,keepdim=True)
            img_features_aggregated=torch.cat([mean,std],1)
            TIME_END("rm_get_and_aggr")

            #conv the img featues and than concat with position and then some 1x1 convs 
            TIME_START("raymarch_fuse")
            img_features_aggregated=torch.cat([pos_encoded,img_features_aggregated],1)
            img_features_aggregated=self.conv1(img_features_aggregated)
            img_features_aggregated=self.conv2(img_features_aggregated)
            img_features_aggregated=torch.relu(img_features_aggregated)
            feat=img_features_aggregated
            TIME_END("raymarch_fuse")
            
            


            #create the lstm if not created 
            TIME_START("raymarch_lstm")
            if self.lstm==None:
                self.lstm = torch.nn.LSTMCell(input_size=feat.shape[1], hidden_size=self.lstm_hidden_size ).to("cuda")
                self.lstm.apply(init_recurrent_weights)
                lstm_forget_gate_init(self.lstm)
                # self.lstm = torch.nn.Sequential(
                #     BlockNerf(activ=torch.nn.GELU(), in_channels=feat.shape[1], out_channels=32,  bias=True ).cuda(),
                #     BlockNerf(activ=torch.nn.GELU(), in_channels=32, out_channels=32,  bias=True ).cuda(),
                #     BlockNerf(activ=None, in_channels=32, out_channels=16,  bias=True ).cuda(),
                # )

            #run through the lstm
            state = self.lstm( nchw2lin(feat), states[-1])
            # state = self.lstm(feat)
            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-10, max=10))
            # if state.requires_grad:
                # state.register_hook(lambda x: x.clamp(min=-10, max=10))

            signed_distance= self.out_layer(state[0])
            signed_distance = lin2nchw(signed_distance, frame.height, frame.width)
            # signed_distance= self.out_layer(state)
            signed_distance= torch.abs(signed_distance)
            TIME_END("raymarch_lstm")
            depth_scaling=1.0/(1.0*self.nr_iters) #1.0 is the scene scale and we expect on average that every step will do a movement of 0.5, maybe the average movement is more like 0.25 idunno
            signed_distance=signed_distance*depth_scaling
            # print("signed_distance iter", iter_nr, " is ", signed_distance.mean())
            

            new_world_coords = world_coords[-1] + ray_dirs * signed_distance
            states.append(state)
            world_coords.append(new_world_coords)
            signed_distances_for_marchlvl.append(signed_distance)

            TIME_END("raymarch_iter")


        if dataset_params.use_ndc: 
            #finally get them to xyz 
            new_world_coords = nchw2lin(new_world_coords)
            points3D_lin = ndc2xyz(frame.height , frame.width , fx, fy, near, new_world_coords)
            new_world_coords = lin2nchw(points3D_lin, frame.height , frame.width)
            ray_dirs=ray_dirs_original

        #get the depth at this final 3d position
        depth= (new_world_coords-camera_center).norm(dim=1, keepdim=True)

        #return also the world coords at every march
        # world_coords.pop(0)


      

        return new_world_coords, depth 


class DifferentiableRayMarcherHierarchical(torch.nn.Module):
    def __init__(self):
        super(DifferentiableRayMarcherHierarchical, self).__init__()

        num_encodings=8
        # self.learned_pe=LearnedPE(in_channels=3, num_encoding_functions=num_encodings, logsampling=True)
        self.learned_pe=PositionalEncoding(in_channels=3, num_encoding_functions=num_encodings)
        # cur_nr_channels = in_channels + 3*num_encodings*2

        #model 
        self.lstm_hidden_size = 16
        self.lstm=None #Create this later, the volumentric feature can maybe change and therefore the features that get as input to the lstm will be different
        # self.out_layer = BlockNerf(activ=None, in_channels=self.lstm_hidden_size, out_channels=5,  bias=True ).cuda()
        self.out_layer = torch.nn.Sequential(
            WNConvActiv(in_channels=self.lstm_hidden_size, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=None, is_first_layer=False ),
        )

        self.conv1= WNConvActiv(in_channels=3+3*num_encodings*2+ 64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.Mish(), is_first_layer=False )
        self.conv2= WNConvActiv(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.Mish(), is_first_layer=False )


        apply_weight_init_fn(self.conv1, leaky_relu_init, negative_slope=0.0)
        apply_weight_init_fn(self.conv2, leaky_relu_init, negative_slope=0.0) #conv2 is followed by a relu
        leaky_relu_init(self.out_layer, negative_slope=1.0)
      

        self.compress_feat = torch.nn.ModuleList([])
        # self.compress_feat.append(
        #     [WNReluConv(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.GELU(), is_first_layer=False ),
        #     WNReluConv(in_channels=112, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.GELU(), is_first_layer=False ),
        #     WNReluConv(in_channels=88, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.GELU(), is_first_layer=False ),
        #     WNReluConv(in_channels=60, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.GELU(), is_first_layer=False )
        #     ]
        # )
        for i in range(3):
            self.compress_feat.append(None)



        # self.compress_feat.append(WNReluConv(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.GELU(), is_first_layer=False ))
        # self.compress_feat.append(WNReluConv(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.GELU(), is_first_layer=False ))
        # self.compress_feat.append(WNReluConv(in_channels=40, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.GELU(), is_first_layer=False ))
        # self.compress_feat.append(WNReluConv(in_channels=60, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.GELU(), is_first_layer=False ))


        # self.compress_feat.append(WNReluConv(in_channels=96*2, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.GELU(), is_first_layer=False ))
        # self.compress_feat.append(WNReluConv(in_channels=112*2, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.GELU(), is_first_layer=False ))
        # self.compress_feat.append(WNReluConv(in_channels=88*2, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.GELU(), is_first_layer=False ))
        # self.compress_feat.append(WNReluConv(in_channels=60*2, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.GELU(), is_first_layer=False ))


        # self.down_stages_list = torch.nn.ModuleList([])
        # self.down_stages_list.append( nn.Sequential(
        #     ResnetBlock2D(cur_nr_channels, kernel_size=3, stride=1, padding=1, dilations=[1,1], biases=[True, True], with_dropout=False, do_norm=True, block_type=block_type  ),
        # ))


        self.base_fc = nn.Sequential(
            WNConvActiv( 32*3, 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, is_first_layer=False ).cuda(),
            WNConvActiv( 16, 8, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=None, is_first_layer=False ).cuda(),
                                     )
        apply_weight_init_fn(self.base_fc, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.base_fc[-1], negative_slope=1.0)
        

        self.vis_fc = nn.Sequential(
            WNConvActiv( 8, 8, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True,  is_first_layer=False ).cuda(),
            WNConvActiv( 8, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=None, is_first_layer=False ).cuda(),
                                     )
        apply_weight_init_fn(self.vis_fc, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.vis_fc[-1], negative_slope=1.0)


        #activ
        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()

        #params 
        self.nr_iters_per_res=[10,5,3]
        self.total_nr_iters=sum(self.nr_iters_per_res)
        self.nr_resolutions=2
        self.use_dynamic_weight=False

      
    def forward(self, dataset_params, frame, ray_dirs, frames_close, frames_features, multi_res_features, weights,  novel=False):

        
        #make a series of frames subsaqmpled
        frames_subsampled=[]
        frames_subsampled.append(frame)
        for i in range(self.nr_resolutions):
            frames_subsampled.append(frame.subsampled_frames[i])
        frames_subsampled.reverse()

        #debug the frames
        # for i in range(len(frames_subsampled)):
            # frame_subsampled=frames_subsampled[i]
            # print("frames is ", frame_subsampled.height)



        nr_nearby_frames=len(frames_close)
        R_list=[]
        t_list=[]
        K_list=[]
        for i in range(len(frames_close)):
            frame_selectd=frames_close[i]
            R_list.append( frame_selectd.R_tensor.view(1,3,3) )
            t_list.append( frame_selectd.t_tensor.view(1,1,3) )
            K_list.append( frame_selectd.K_tensor.view(1,3,3) )
        R_batched=torch.cat(R_list,0)
        t_batched=torch.cat(t_list,0)
        K_batched=torch.cat(K_list,0)


        points_3d_for_each_res=[] #stores the world coords after each res stage
        points_3d_for_each_step=[] #stores the world coords after each step of the ray marcher


        #go from each level of the hierarchy and ray march from there 
        for res_iter in range(len(frames_subsampled)):
            frame_subsampled=frames_subsampled[res_iter]

            # print("res iter", res_iter)
            # print("height and width is ", frame_subsampled.height, " ", frame_subsampled.width)

            if res_iter==0:
                if novel:
                    depth_per_pixel= torch.ones([1, 1, frame_subsampled.height, frame_subsampled.width], dtype=torch.float32, device=torch.device("cuda")) 
                    depth_per_pixel.fill_(dataset_params.raymarch_depth_min)
                else:
                    # depth_per_pixel = torch.zeros((1, 1, frame_subsampled.height, frame_subsampled.width ), device=torch.device("cuda") ).normal_(mean=dataset_params.raymarch_depth_min, std=2e-2)
                    if (dataset_params.raymarch_depth_jitter!=0.0):
                        depth_per_pixel = torch.zeros((1, 1, frame_subsampled.height, frame_subsampled.width), device=torch.device("cuda") ).normal_(mean=dataset_params.raymarch_depth_min, std=dataset_params.raymarch_depth_jitter)
                    else: 
                        depth_per_pixel= torch.ones([1, 1, frame_subsampled.height, frame_subsampled.width], dtype=torch.float32, device=torch.device("cuda")) 
                        depth_per_pixel.fill_(dataset_params.raymarch_depth_min)
            else: 
                ## if any other level above the coarsest one, then we upsample the depth
                depth_per_pixel =depth
                depth_per_pixel = torch.nn.functional.interpolate(depth_per_pixel ,size=(frame_subsampled.height, frame_subsampled.width ), mode='bicubic')

            ray_dirs=torch.from_numpy(frame_subsampled.ray_dirs).float().cuda().view(1, frame_subsampled.height, frame_subsampled.width, 3).permute(0,3,1,2)

            # print("ray_dirs is ", ray_dirs.shape)

            #attempt 2 unproject to 3D 
            camera_center=torch.from_numpy( frame_subsampled.frame.pos_in_world() ).to("cuda")
            camera_center=camera_center.view(1,3,1,1)
            points3D = camera_center + depth_per_pixel*ray_dirs

            #start runnign marches 
            init_world_coords=points3D
            initial_depth=depth_per_pixel
            world_coords = [init_world_coords]
            depths = [initial_depth]
            signed_distances_for_marchlvl=[]
            states = [None]



            #get the frame_featues at this correct level
            # print("frame subsampled is ", frame_subsampled.height, " ", frame_subsampled.width)
            idx=-1+res_iter-self.nr_resolutions
            # print("res iter, ", res_iter, " idx is ", idx)
            frames_features = multi_res_features[idx]
            # print("frames_features", frames_features.shape)
            if (frames_features.shape[2]!=frame_subsampled.height or frames_features.shape[3]!=frame_subsampled.width ):
                frames_features= torch.nn.functional.interpolate(frames_features ,size=(frame_subsampled.height, frame_subsampled.width ), mode='bilinear')
            # print("frames_features", frames_features.shape)
            if self.compress_feat[res_iter]==None:
                self.compress_feat[res_iter]=WNConvActiv(in_channels=frames_features.shape[1], out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.Mish(), is_first_layer=False )
                apply_weight_init_fn(self.compress_feat[res_iter], leaky_relu_init, negative_slope=0.0)

            frames_features = self.compress_feat[res_iter](frames_features)
            
            # rgb = torch.cat([rgb0, rgb1, rgb2],0)
            # print("rgb is ", rgb.shape)
            # frames_features = torch.cat([frames_features, rgb],1)
            # print("frames_features", frames_features.shape)
            #get the K of this subsampled frame 
            K_list=[]
            for i in range(len(frames_close)):
                K_list.append( frame_subsampled.K_tensor.view(1,3,3) )
            K_batched=torch.cat(K_list,0)


            # weights=self.frame_weights_computer(frame, frames_close)

            nr_iters=self.nr_iters_per_res[res_iter]
            # for iter_nr in range(self.nr_iters):
            for iter_nr in range(nr_iters):
                TIME_START("raymarch_iter")
            
                #compute the features at this position 
                # feat=self.learned_pe(world_coords[-1])
                pos_encoded_linear=self.learned_pe(  nchw2lin(world_coords[-1]) ) 
                pos_encoded=lin2nchw(pos_encoded_linear, frame_subsampled.height, frame_subsampled.width).contiguous()

                # uv_tensor=compute_uv_batched(frames_close, world_coords[-1] )
                uv_tensor, mask=compute_uv_batched(R_batched, t_batched, K_batched, frame_subsampled.height, frame_subsampled.width, world_coords[-1] )


                # slice with grid_sample
                # uv_tensor=uv_tensor.view(-1, frame_subsampled.height, frame_subsampled.width, 2)
                sliced_feat_batched=torch.nn.functional.grid_sample( frames_features, uv_tensor, align_corners=False, mode="bilinear", padding_mode="zeros" ) #sliced features is N,C,H,W



                mean=sliced_feat_batched.mean(dim=0, keepdim=True)
                std=sliced_feat_batched.std(dim=0,keepdim=True)
                img_features_aggregated=torch.cat([mean,std],1)

                if self.use_dynamic_weight:
                    globalfeat= img_features_aggregated

                    #concat each rgb_feat with the mean and var
                    x = torch.cat([globalfeat.expand(nr_nearby_frames, -1, -1, -1), sliced_feat_batched], dim=1)  # N,C,H,W
                    # print("x before is ", x.shape)
                    x = self.base_fc(x) #reduces to 32

                    #gets the weights for each of the frames
                    vis = self.vis_fc( x * weights.view(-1,1,1,1) )
                    img_features_aggregated =  fused_mean_variance(sliced_feat_batched, vis, dim_reduce=0, dim_concat=1, use_weights=True)

                # img_features_aggregated = self.compress_feat[res_iter](img_features_aggregated)

                #conv the img featues and than concat with position and then some 1x1 convs 
                TIME_START("raymarch_fuse")
                img_features_aggregated=torch.cat([pos_encoded,img_features_aggregated],1)
                img_features_aggregated=self.conv1(img_features_aggregated)
                img_features_aggregated=self.conv2(img_features_aggregated)
                # img_features_aggregated=torch.relu(img_features_aggregated)
                feat=img_features_aggregated
                TIME_END("raymarch_fuse")
                
            
                
                #create the lstm if not created 
                TIME_START("raymarch_lstm")
                if self.lstm==None:
                    self.lstm = torch.nn.LSTMCell(input_size=feat.shape[1], hidden_size=self.lstm_hidden_size ).to("cuda")
                    self.lstm.apply(init_recurrent_weights)
                    lstm_forget_gate_init(self.lstm)

                #run through the lstm
                state = self.lstm(nchw2lin(feat), states[-1])
                if state[0].requires_grad:
                    state[0].register_hook(lambda x: x.clamp(min=-100, max=100))


                #Doing it like SMD-net 
                sdf_like_smd=False
                if sdf_like_smd:
                    activation = nn.Sigmoid()
                    input_out_layer= lin2nchw(state[0], frame_subsampled.height, frame_subsampled.width)
                    pred= self.out_layer(input_out_layer)

                    eps = 1e-2 #1e-3 in case of gaussian distribution
                    # mu0 = activation(torch.unsqueeze(pred[:,0],1))
                    # mu1 = activation(torch.unsqueeze(pred[:,1],1))
                    mu0 = torch.unsqueeze(pred[:,0,:,:],1)
                    mu1 = torch.unsqueeze(pred[:,1,:,:],1)

                    sigma0 =  torch.clamp(activation(torch.unsqueeze(pred[:,2,:,:],1)), eps, 1.0)
                    sigma1 =  torch.clamp(activation(torch.unsqueeze(pred[:,3,:,:],1)), eps, 1.0)

                    pi0 = activation(torch.unsqueeze(pred[:,4,:,:],1))
                    pi1 = 1. - pi0

                    # Mode with the highest density value as final prediction
                    mask = (pi0 / sigma0  >   pi1 / sigma1).float()
                    disp = mu0 * mask + mu1 * (1. - mask)
                    signed_distance = disp
                else:
                    input_out_layer= lin2nchw(state[0], frame_subsampled.height, frame_subsampled.width)
                    signed_distance= self.out_layer(input_out_layer)


                TIME_END("raymarch_lstm")
                # print("signed_distance iter", iter_nr, " is ", signed_distance.mean())
                # signed_distance = lin2nchw(signed_distance, frame_subsampled.height, frame_subsampled.width)
                #the output of the lstm after abs will probably be on average around 0.5 (because before the abs it was zero meaned and kinda spread around [-1,1])
                # however, doing nr_steps*0.5 will likely put the depth above the scene scale which is normally 1.0
                # therefore we expect each step to be 1.0/nr_steps so for 10 steps each steps should to 0.1
                # depth_scaling=1.0/(1.0*self.nr_iters*self.nr_resolutions) #1.0 is the scene scale and we expect on average that every step will do a movement of 0.5, maybe the average movement is more like 0.25 idunno
                depth_scaling=1.0/(1.0*self.total_nr_iters) #1.0 is the scene scale and we expect on average that every step will do a movement of 0.5, maybe the average movement is more like 0.25 idunno
                signed_distance=signed_distance*depth_scaling
                # signed_distance= torch.abs(signed_distance) #NOT having the abs is definitelly better, it seems that being able to predict negative values is quite important
                # print("signed_distance iter", iter_nr, " is ", signed_distance.mean())
                


                new_world_coords = world_coords[-1] + ray_dirs * signed_distance
                states.append(state)
                world_coords.append(new_world_coords)
                signed_distances_for_marchlvl.append(signed_distance)
                points_3d_for_each_step.append( new_world_coords )

                # if iter_nr==self.nr_iters-1:
                    # show_3D_points(new_world_coords, "points_3d_"+str(res_iter))



            #get the depth at this final 3d position
            # depth= (new_world_coords-camera_center).norm(dim=1, keepdim=True)
            depth= (new_world_coords-camera_center).norm(dim=1, keepdim=True) #this does not have a sign so if the points are somehow behind the camera, the norm is the same
            # depth= ( (new_world_coords-camera_center)/ (ray_dirs+1e-5)  ).mean(dim=1, keepdim=True) #because the ray is expressed as x=orig+dir*t, then the t is t=(x-orig)/dir and since each component xyz is scaled the same then we can just take the first one or just to have better gradient we just average
            #in order to get the sign of the depth we do NOT use the previous line which divides by ray_dirs because that can give you division by zero and instability ever with an epsilon, rather, we just multiply by the dot product between the ray_dirs and the vector made by this new positions
            pos_dir= F.normalize((new_world_coords-camera_center), dim=1)
            dot=torch.sum(pos_dir * ray_dirs, dim=1, keepdim=True)
            depth=depth*dot

            # print("new_world_coords", new_world_coords.shape)
            points_3d_for_each_res.append( new_world_coords )


        return new_world_coords, depth, points_3d_for_each_res, points_3d_for_each_step



class Net(torch.nn.Module):
    def __init__(self, predict_confidence_map, multi_res_loss):
        super(Net, self).__init__()

        self.predict_confidence_map=predict_confidence_map
        self.multi_res_loss=multi_res_loss

        self.first_time=True

        #models
        self.unet_extract_feat=UNet( nr_channels_start=8, nr_channels_output=32, nr_stages=4, max_nr_channels=128)

        
        out_nr=3
        if self.predict_confidence_map:
            out_nr=4
        self.unet_pred_output=UNet( nr_channels_start=16, nr_channels_output=out_nr, nr_stages=1, max_nr_channels=32)

      
        self.compress_features=nn.Sequential( 
            WNConvActiv( 32, 8, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, is_first_layer=False ).cuda(),
            WNConvActiv( 8, 8, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=None, is_first_layer=False ).cuda(),
        )
        apply_weight_init_fn(self.compress_features, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.compress_features[-1], negative_slope=1.0)

        # self.compute_blending_weights=UNet( nr_channels_start=16, nr_channels_output=1, nr_stages=1, max_nr_channels=32, block_type=WNReluConv)

        # self.ray_marcher=DifferentiableRayMarcher()
        self.ray_marcher=DifferentiableRayMarcherHierarchical()
        # self.ray_marcher=DifferentiableRayMarcherHierarchicalNoLSTM()
        # self.feature_aggregator= FeatureAgregator()
        # self.feature_aggregator= FeatureAgregatorLinear()
        # self.feature_aggregator= FeatureAgregatorIBRNet()


        #activ
        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()

        #params


        
        # self.slice_texture= SliceTextureModule()
        # self.splat_texture= SplatTextureModule()
        # self.concat_coord=ConcatCoord() 



        #ibrnet things 
        self.ray_dir_fc = nn.Sequential( 
            WNConvActiv( 4, 16, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True,is_first_layer=False ).cuda(),
            WNConvActiv( 16, 8+3, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=None, is_first_layer=False ).cuda(),
            )
        apply_weight_init_fn(self.ray_dir_fc, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.ray_dir_fc[-1], negative_slope=1.0)

        self.base_fc = nn.Sequential(
            WNConvActiv( (8+3)*3, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, is_first_layer=False ).cuda(),
            WNConvActiv( 32, 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=None, is_first_layer=False ).cuda(),
                                     )
        apply_weight_init_fn(self.base_fc, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.base_fc[-1], negative_slope=1.0)

        self.vis_fc = nn.Sequential(
            WNConvActiv( 16, 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, is_first_layer=False ).cuda(),
            WNConvActiv( 16, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=None, is_first_layer=False ).cuda(),
                                     )
        apply_weight_init_fn(self.vis_fc, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.vis_fc[-1], negative_slope=1.0)
        
        # self.vis_fc2 = nn.Sequential(
        #     WNReluConv( 32, 32, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=None, is_first_layer=False ).cuda(),
        #     torch.nn.ReLU(),
        #     WNReluConv( 32, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.ReLU(), is_first_layer=False ).cuda(),
        #     torch.nn.Sigmoid()
        #                              )
        
        # self.rgb_fc = nn.Sequential(
        #     WNReluConv( 37, 16, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=None, is_first_layer=False ).cuda(),
        #     torch.nn.ReLU(),
        #     WNReluConv( 16, 8, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.ReLU(), is_first_layer=False ).cuda(),
        #     WNReluConv( 8, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=torch.nn.ReLU(), is_first_layer=False ).cuda(),
        #                             )

        self.rgb_pred_per_res=torch.nn.ModuleList([])
        for i in range(3):
            self.rgb_pred_per_res.append(None)


      
    def forward(self, dataset_params, frame, ray_dirs, rgb_close_batch, rgb_close_fullres_batch, ray_dirs_close_batch, ray_diff, frame_full_res, frames_close, weights, novel=False):

       
        #pass through unet 
        TIME_START("unet")
        frames_features, multi_res_features=self.unet_extract_feat( rgb_close_batch )
        frames_features=torch.nn.Mish()(frames_features) #the multires features are also obtained after a mish, so in order to maintain the same gradient flow we also apply a mish to these
        TIME_END("unet")

       
        TIME_START("ray_march")
        point3d, depth, points3d_for_each_res, points3d_for_each_step  = self.ray_marcher(dataset_params, frame, ray_dirs, frames_close, frames_features, multi_res_features, weights, novel)
        TIME_END("ray_march")


        ######################################### 
        #for each lvl of the points 3d, slice the features at the corresponding lvl of unet and predict an RGB
        points3d_for_each_res.reverse() #this makes it from HR to LR
        multi_res_features.reverse() #this makes it from HR to LR
        rgb_loss_multires=0
        if self.multi_res_loss:
            for i in range(len(points3d_for_each_res)-1):
                point3d_LR =  points3d_for_each_res[i+1]
                frame_LR=frame.subsampled_frames[i]
                feat =  multi_res_features[i+1]

                if feat.shape[2]!=frame_LR.height or feat.shape[3]!=frame_LR.width:
                    feat = torch.nn.functional.interpolate(feat, size=(frame_LR.height, frame_LR.width ), mode='bilinear',  align_corners=False) #3,c,h,w

                #get gt 
                rgb_gt=mat2tensor(frame_LR.frame.rgb_32f, True).to("cuda")

                #get camera params
                nr_nearby_frames=len(frames_close)
                R_list=[]
                t_list=[]
                K_list=[]
                for frame_idx in range(len(frames_close)):
                    frame_selected=frames_close[frame_idx].subsampled_frames[i]
                    R_list.append( frame_selected.R_tensor.view(1,3,3) )
                    t_list.append( frame_selected.t_tensor.view(1,1,3) )
                    K_list.append( frame_selected.K_tensor.view(1,3,3) )
                R_batched=torch.cat(R_list,0)
                t_batched=torch.cat(t_list,0)
                K_batched=torch.cat(K_list,0)
                ######when we project we assume all the frames have the same size
                height=frame_LR.height
                width=frame_LR.width


                #slice 
                uv_tensor, mask=compute_uv_batched(R_batched, t_batched, K_batched, height, width,  point3d_LR )
                sliced_feat_batched=torch.nn.functional.grid_sample( feat, uv_tensor, align_corners=False, mode="bilinear",  padding_mode="zeros"  ) #sliced features is N,C,H,W

                mean_var = fused_mean_variance(sliced_feat_batched, weights.view(-1,1,1,1), dim_reduce=0, dim_concat=1, use_weights=True)


                if self.rgb_pred_per_res[i]==None:
                    self.rgb_pred_per_res[i]=torch.nn.Sequential(
                        WNConvActiv( mean_var.shape[1], 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, is_first_layer=False ).cuda(),
                        WNConvActiv( 16, 8, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, is_first_layer=False ).cuda(),
                        WNConvActiv( 8, 3, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, with_dropout=False, transposed=False, do_norm=True, activ=None, is_first_layer=False ).cuda()
                    )
                    apply_weight_init_fn(self.rgb_pred_per_res[i], leaky_relu_init, negative_slope=0.0)
                    leaky_relu_init(self.rgb_pred_per_res[i][-1], negative_slope=1.0)
                rgb_pred_LR=self.rgb_pred_per_res[i](mean_var)

                weight= 1/(4*(i+1) ) #the lower the resolution the lower the weight, so half of resolution has a 1/4 of the weight and quarter res has 1/8 of res because it has 1/8 of pixels

                rgb_loss_l1= ((rgb_gt- rgb_pred_LR).abs()).mean()
                rgb_loss_multires+= rgb_loss_l1*weight



        #rgb gets other features than the ray marcher 
        frames_features_rgb=frames_features
   


        nr_nearby_frames=len(frames_close)
        R_list=[]
        t_list=[]
        K_list=[]
        for i in range(len(frames_close)):
            frame_selected=frames_close[i]
            R_list.append( frame_selected.R_tensor.view(1,3,3) )
            t_list.append( frame_selected.t_tensor.view(1,1,3) )
            K_list.append( frame_selected.K_tensor.view(1,3,3) )
        R_batched=torch.cat(R_list,0)
        t_batched=torch.cat(t_list,0)
        K_batched=torch.cat(K_list,0)
        ######when we project we assume all the frames have the same size
        height=frames_close[0].height
        width=frames_close[0].width


        uv_tensor, mask=compute_uv_batched(R_batched, t_batched, K_batched, height, width,  point3d )
        sliced_feat_batched_img=torch.nn.functional.grid_sample( frames_features_rgb, uv_tensor, align_corners=False, mode="bilinear",  padding_mode="zeros"  ) #sliced features is N,C,H,W
          




        #get the weights similar to ibrnnet but thne do 3x3 convs similar to what I do --------------------------------------------------------------------
        TIME_START("unet_predrgb")

        sliced_feat_compressed = self.compress_features(sliced_feat_batched_img)
        full_res_height=rgb_close_fullres_batch.shape[2]
        full_res_width=rgb_close_fullres_batch.shape[3]
        sliced_feat_compressed = torch.nn.functional.interpolate(sliced_feat_compressed, size=(full_res_height, full_res_width ), mode='bilinear',  align_corners=False) #3,c,h,w
        #slice also from the high res images and concat that too 
        uv_tensor=uv_tensor.view(nr_nearby_frames, frame.height, frame.width, 2) ####for upsamplign the uv, make some conv layers
        uv_tensor=uv_tensor.permute(0,3,1,2) # from N,H,W,C to N,C,H,W
        uv_tensor_hr= torch.nn.functional.interpolate(uv_tensor,size=(full_res_height, full_res_width ), mode='bilinear',  align_corners=False)
        uv_tensor_hr=uv_tensor_hr.permute(0,2,3,1) #from N,C,H,W to N,H,W,C
        sliced_color_HR=torch.nn.functional.grid_sample( rgb_close_fullres_batch, uv_tensor_hr, align_corners=False, mode="bilinear",  padding_mode="zeros",  ) #sliced features is N,C,H,W
        rgb_feat=torch.cat([sliced_color_HR, sliced_feat_compressed],1)

        #get the ray_diff
        direction_feat = self.ray_dir_fc(ray_diff)

        #RGB_feat + direction_Feat
        rgb_feat = rgb_feat + direction_feat
        mean_var_HR = fused_mean_variance(rgb_feat, weights.view(-1,1,1,1), dim_reduce=0, dim_concat=1, use_weights=True)
        globalfeat= mean_var_HR

        #concat each rgb_feat with the mean and var
        x = torch.cat([globalfeat.expand(nr_nearby_frames, -1, -1, -1), rgb_feat], dim=1)  # N,C,H,W
        x = self.base_fc(x)

        #computation 
        vis = self.vis_fc( x * weights.view(-1,1,1,1) )

        #get weighted mean and var from both colors and feat
        rgb_feat_mean_var =  fused_mean_variance(rgb_feat, vis, dim_reduce=0, dim_concat=1, use_weights=True)
        input_last_unet = rgb_feat_mean_var

        
        input_last_unet=torch.cat([input_last_unet, sliced_color_HR.view(1,-1,full_res_height, full_res_width)],1)
        rgb_pred, multi_res_features=self.unet_pred_output(input_last_unet )

        confidence_map=None 
        if self.predict_confidence_map:
            confidence_map=rgb_pred[:,0:1,:,:]
            rgb_pred=rgb_pred[:,1:4,:,:]
       
        TIME_END("unet_predrgb")



        depth=depth.view(frame.height, frame.width,1)
        depth=depth.permute(2,0,1).unsqueeze(0)



        depth_for_each_res=[]
        for i in range(len(points3d_for_each_res)):
            camera_center=torch.from_numpy( frame.frame.pos_in_world() ).to("cuda")
            camera_center=camera_center.view(1,3,1,1)
            point3d_LR =  points3d_for_each_res[i]
            depth_for_res= (point3d_LR-camera_center).norm(dim=1, keepdim=True)
            depth_for_each_res.append(depth_for_res)

        depth_for_each_step=[]
        for i in range(len(points3d_for_each_step)):
            camera_center=torch.from_numpy( frame.frame.pos_in_world() ).to("cuda")
            camera_center=camera_center.view(1,3,1,1)
            point3d_LR =  points3d_for_each_step[i]
            depth_for_step= (point3d_LR-camera_center).norm(dim=1, keepdim=True)
            depth_for_each_step.append(depth_for_step)

        return rgb_pred, depth, point3d, rgb_loss_multires, depth_for_each_res, confidence_map, depth_for_each_step


class PCA(torch.autograd.Function):
    # @staticmethod
    def forward(ctx, sv): #sv corresponds to the slices values, it has dimensions nr_positions x val_full_dim

        # http://agnesmustar.com/2017/11/01/principal-component-analysis-pca-implemented-pytorch/


        X=sv.detach().cpu()#we switch to cpu of memory issues when doing svd on really big imaes
        k=3
        # print("x is ", X.shape)
        X_mean = torch.mean(X,0)
        # print("x_mean is ", X_mean.shape)
        X = X - X_mean.expand_as(X)

        U,S,V = torch.svd(torch.t(X)) 
        C = torch.mm(X,U[:,:k])
        # print("C has shape ", C.shape)
        # print("C min and max is ", C.min(), " ", C.max() )
        C-=C.min()
        C/=C.max()
        # print("after normalization C min and max is ", C.min(), " ", C.max() )

        return C

