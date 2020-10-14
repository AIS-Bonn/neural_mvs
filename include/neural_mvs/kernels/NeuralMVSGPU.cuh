#pragma once

// #include "lattice_net/kernels/HashTableGPU.cuh"

#ifndef __CUDACC_RTC__ 
    #include "neural_mvs/jitify_helper/jitify_helper.cuh"
#endif

#if !defined(__CUDACC_RTC__)
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>
    #include "device_launch_parameters.h" //needed for threadIdx and blockDim 
#endif

#ifndef __CUDACC_RTC__ 
//Add this header after we add all cuda stuff because we need the profiler to have cudaDeviceSyncronize defined
#define ENABLE_CUDA_PROFILING 1
#include "Profiler.h" 
#endif


#define BLOCK_SIZE 256 

class NeuralMVSGPU { 
public:

    //during nvrtc compilation we do not want to compile this code as it does not work due to not including vector definition
    #ifndef __CUDACC_RTC__ 
        NeuralMVSGPU(){
            create_program_handles();
        }

        //it uses Jittify to get a handle for the programs. The programs can contain more than one kernel.. It doesnt actually compile them, they will get jit compiled the first time you run them
        void create_program_handles(){
            m_program=create_jitify_program( std::string(CMAKE_SOURCE_DIR)+"/include/neural_mesh/kernels/NeuralMVSGPU.cuh" );
        }


        // void splat_texture(float* texture, const float* values, const float* uv, const int nr_values, const int val_dim, const int texture_size){
   
        //     dim3 blocks((nr_values - 1) / BLOCK_SIZE + 1, 1, 1);
        //     dim3 blockSize(BLOCK_SIZE, 1, 1);
        //     CUresult res= m_program.kernel("splat_texture")
        //                 .instantiate(nr_values, val_dim, texture_size)
        //                 .configure(blocks, blockSize)
        //                 .launch( texture, values, uv );
        //     CUDA_CHECK_CURESULT(res);
        //     CUDA_CHECK_ERROR();

        // }


       


        jitify::Program m_program;

    #endif



   

};


#if defined(__CUDACC_RTC__)

#define CLIP_COORDINATES(in, out, clip_limit) out = min((clip_limit-1), max(in, 0))

template<int nr_values, int val_dim, int texture_size>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
splat_texture(float* texture, const float* values, const float* uv ){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_values){ //don't go out of bounds
        return;
    }

    //possibly needed variables
    int nr_channels_texture=val_dim+1; // in the texture we store also a homogeneous coordinate, so we have a +1

    //grab pointer to the current values we are procesing
    const float* cur_val = values+idx*val_dim;
    const float* cur_uv = uv+idx*2;

    // //the uvs are supposed to be in range [-1, 1], now we get them in range [0, texture_size-1]
    // float x=(cur_uv[0]+1)*0.5*(texture_size-1);
    // float y=(cur_uv[1]+1)*0.5*(texture_size-1);

    // //get the coordiantes of the neighbouring 4 pixels according to the wikipedia convention  https://en.wikipedia.org/wiki/Bilinear_interpolation
    // int x1=(int)x;
    // int y1=(int)y;
    // int x2=min(x1+1, texture_size-1 ); //the min is in order to avoid accesing outside of the boundaries of the texture
    // int y2=min(y1+1, texture_size-1);

    // // if (x2>=texture_size){
    // //     printf("wtf x2\n");
    // // }
    // // if (y2>=texture_size){
    // //     printf("wtf y2\n");
    // // }
    // // if (x1>=texture_size){
    // //     printf("wtf x1\n");
    // // }
    // // if (y1>=texture_size){
    // //     printf("wtf y1 is %d and y is %f because v from uv is %f \n", y1, y, cur_uv[1] );
    // // }
    // // if (x1<0){
    // //     printf("wtf negative x1\n");
    // // }
    // // if (y1<0){
    // //     printf("wtf nergative y1\n");
    // // }
    
    

    // //get a pointer to the 4 pixels onto which we splat following the convention of wikipedia https://en.wikipedia.org/wiki/Bilinear_interpolation
    // float* q11 = texture + x1*nr_channels_texture + y1*texture_size*nr_channels_texture;
    // float* q12 = texture + x1*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // float* q22 = texture + x2*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // float* q21 = texture + x2*nr_channels_texture + y1*texture_size*nr_channels_texture;

    // //get the weigthings of the pixels
    // // float denom = 1.0/( (x2 - x1)*(y2-y1)  + 1e-7 ); //the denominator is mostly just to normalize for the size of the pixel but we can assume a pixel of size 1 and just drop this whole term
    // float wq11=(x2-x)*(y2-y);
    // float wq21=(x-x1)*(y2-y);
    // float wq12=(x2-x)*(y-y1);
    // float wq22=(x-x1)*(y-y1);






    //the uvs are supposed to be in range [-1, 1], now we get them in range [0, texture_size-1]
    float ix=(cur_uv[0]+1)*0.5*(texture_size-1);
    float iy=(cur_uv[1]+1)*0.5*(texture_size-1);
    // printf("ix %f and uv is %f \n", ix, cur_uv[0] );

    //get the coordiantes of the neighbouring 4 pixels according to the wikipedia convention  https://en.wikipedia.org/wiki/Bilinear_interpolation
    //get the coordiantes of the neighbouring 4 pixels according to the convention of https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c
    // int x1=(int)ix;
    // int y1=(int)iy;
    // int x2=min(x1+1, texture_size-1 ); //the min is in order to avoid accesing outside of the boundaries of the texture
    // int y2=min(y1+1, texture_size-1);

    int ix_nw = floor(ix);
    int iy_nw = floor(iy);
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    //clip 
    CLIP_COORDINATES(ix_nw, ix_nw, texture_size);
    CLIP_COORDINATES(iy_nw, iy_nw, texture_size);
    CLIP_COORDINATES(ix_ne, ix_ne, texture_size);
    CLIP_COORDINATES(iy_ne, iy_ne, texture_size);
    CLIP_COORDINATES(ix_sw, ix_sw, texture_size);
    CLIP_COORDINATES(iy_sw, iy_sw, texture_size);
    CLIP_COORDINATES(ix_se, ix_se, texture_size);
    CLIP_COORDINATES(iy_se, iy_se, texture_size);



    //get a pointer to the 4 pixels onto which we splat following the convention of wikipedia https://en.wikipedia.org/wiki/Bilinear_interpolation
    // const float* q11 = texture + x1*nr_channels_texture + y1*texture_size*nr_channels_texture;
    // const float* q12 = texture + x1*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // const float* q22 = texture + x2*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // const float* q21 = texture + x2*nr_channels_texture + y1*texture_size*nr_channels_texture;

    float* nw_val = texture + ix_nw*nr_channels_texture + iy_nw*texture_size*nr_channels_texture;
    float* ne_val = texture + ix_ne*nr_channels_texture + iy_ne*texture_size*nr_channels_texture;
    float* sw_val = texture + ix_sw*nr_channels_texture + iy_sw*texture_size*nr_channels_texture;
    float* se_val = texture + ix_se*nr_channels_texture + iy_se*texture_size*nr_channels_texture;

    //get the weigthings of the pixels
    // float denom = 1.0/( (x2 - x1)*(y2-y1)  + 1e-7 ); //the denominator is mostly just to normalize for the size of the pixel but we can assume a pixel of size 1 and just drop this whole term
    // float wq11=(x2-x)*(y2-y);
    // float wq21=(x-x1)*(y2-y);
    // float wq12=(x2-x)*(y-y1);
    // float wq22=(x-x1)*(y-y1);
    float nw = (ix_se - ix)    * (iy_se - iy);
    float ne = (ix    - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix)    * (iy    - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);






    //splat onto the 4 pixels the weighted cur_val
    for(int i=0; i<val_dim; i++){
        atomicAdd(nw_val + i, cur_val[i]*nw );
        atomicAdd(ne_val + i, cur_val[i]*ne );
        atomicAdd(sw_val + i, cur_val[i]*sw );
        atomicAdd(se_val + i, cur_val[i]*se );
    }
    //splat also a homogeneous coord used later for normalization
    atomicAdd(nw_val + val_dim, nw );
    atomicAdd(ne_val + val_dim, ne );
    atomicAdd(sw_val + val_dim, sw );
    atomicAdd(se_val + val_dim, se );

    


}



#endif

