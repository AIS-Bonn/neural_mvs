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

#include "EasyCuda/UtilsCuda.h"


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
            VLOG(1) << "compiling cuda code";
            m_program=create_jitify_program( std::string(CMAKE_SOURCE_DIR)+"/include/neural_mvs/kernels/NeuralMVSGPU.cuh" );
        }



         void splat_texture(float* texture, const float* values, const float* uv, const int nr_values, const int val_dim, const int texture_height, const int texture_width){
   
            dim3 blocks((nr_values - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            CUresult res= m_program.kernel("splat_texture")
                        .instantiate(val_dim, texture_height, texture_width)
                        .configure(blocks, blockSize)
                        .launch(nr_values, texture, values, uv );
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();

        }


        void slice_texture(float* values_not_normalized_tensor, const float* texture, const float* uv, const int nr_values, const int nr_channels_texture, const int texture_height, const int texture_width){
   
            dim3 blocks((nr_values - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            CUresult res= m_program.kernel("slice_texture")
                        .instantiate(nr_channels_texture, texture_height, texture_width)
                        .configure(blocks, blockSize)
                        .launch(nr_values, values_not_normalized_tensor, texture, uv );
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();

        }


   
        void splat_texture_backward(float* grad_values, float* grad_uv, const float* grad_texture, const float* values, const float* uv, const int nr_values, const int val_dim, const int texture_height, const int texture_width){
   
            dim3 blocks((nr_values - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            CUresult res= m_program.kernel("splat_texture_backward")
                        .instantiate( val_dim, texture_height, texture_width)
                        .configure(blocks, blockSize)
                        .launch( nr_values, grad_values, grad_uv, grad_texture, values, uv );
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();

        }

        //  m_impl->slice_texture_backward( grad_texture.data_ptr<float>(), grad_uv.data_ptr<float>(). //output
        //                             grad_values.data_ptr<float>(), texture.data_ptr<float>(), uv_tensor.data_ptr<float>(), //input
        //                             nr_values, nr_channels_texture, texture_size); //constant

        void slice_texture_backward(float* grad_texture, float* grad_uv, const float* grad_values_not_normalized, const float* texture, const float* uv, const int nr_values, const int nr_channels_texture, const int texture_height, const int texture_width){
   
            dim3 blocks((nr_values - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            CUresult res= m_program.kernel("slice_texture_backward")
                        .instantiate( nr_channels_texture, texture_height, texture_width)
                        .configure(blocks, blockSize)
                        .launch( nr_values, grad_texture, grad_uv, grad_values_not_normalized, texture, uv );
            CUDA_CHECK_CURESULT(res);
            CUDA_CHECK_ERROR();

        }


       


        jitify::Program m_program;

    #endif



   

};


#if defined(__CUDACC_RTC__)

#define CLIP_COORDINATES(in, out, clip_limit) out = min((clip_limit-1), max(in, 0))



template< int val_dim, int texture_height, int texture_width>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
splat_texture(int nr_values, float* texture, const float* values, const float* uv ){

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
    float ix=(cur_uv[0]+1)*0.5*(texture_width-1);
    float iy=(cur_uv[1]+1)*0.5*(texture_height-1);
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
    CLIP_COORDINATES(ix_nw, ix_nw, texture_width);
    CLIP_COORDINATES(iy_nw, iy_nw, texture_height);
    CLIP_COORDINATES(ix_ne, ix_ne, texture_width);
    CLIP_COORDINATES(iy_ne, iy_ne, texture_height);
    CLIP_COORDINATES(ix_sw, ix_sw, texture_width);
    CLIP_COORDINATES(iy_sw, iy_sw, texture_height);
    CLIP_COORDINATES(ix_se, ix_se, texture_width);
    CLIP_COORDINATES(iy_se, iy_se, texture_height);



    //get a pointer to the 4 pixels onto which we splat following the convention of wikipedia https://en.wikipedia.org/wiki/Bilinear_interpolation
    // const float* q11 = texture + x1*nr_channels_texture + y1*texture_size*nr_channels_texture;
    // const float* q12 = texture + x1*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // const float* q22 = texture + x2*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // const float* q21 = texture + x2*nr_channels_texture + y1*texture_size*nr_channels_texture;

    float* nw_val = texture + ix_nw*nr_channels_texture + iy_nw*texture_width*nr_channels_texture;
    float* ne_val = texture + ix_ne*nr_channels_texture + iy_ne*texture_width*nr_channels_texture;
    float* sw_val = texture + ix_sw*nr_channels_texture + iy_sw*texture_width*nr_channels_texture;
    float* se_val = texture + ix_se*nr_channels_texture + iy_se*texture_width*nr_channels_texture;

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

    // int i=0;
    


}



template<int nr_channels_texture, int texture_height, int texture_width>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_texture( int nr_values, float* values_not_normalized, const float* texture, const float* uv ){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_values){ //don't go out of bounds
        return;
    }

    //possibly needed variables
    // int val_dim = nr_channels_texture-1; //in the texture we also store a homogeneous coords so the val dim will be -1 


    //grab pointer to the current values we are procesing
    float* cur_val_not_normalized = values_not_normalized+idx*nr_channels_texture;
    const float* cur_uv = uv+idx*2;

    //the uvs are supposed to be in range [-1, 1], now we get them in range [0, texture_size-1]
    float ix=(cur_uv[0]+1)*0.5*(texture_width-1);
    float iy=(cur_uv[1]+1)*0.5*(texture_height-1);
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
    CLIP_COORDINATES(ix_nw, ix_nw, texture_width);
    CLIP_COORDINATES(iy_nw, iy_nw, texture_height);
    CLIP_COORDINATES(ix_ne, ix_ne, texture_width);
    CLIP_COORDINATES(iy_ne, iy_ne, texture_height);
    CLIP_COORDINATES(ix_sw, ix_sw, texture_width);
    CLIP_COORDINATES(iy_sw, iy_sw, texture_height);
    CLIP_COORDINATES(ix_se, ix_se, texture_width);
    CLIP_COORDINATES(iy_se, iy_se, texture_height);



    //get a pointer to the 4 pixels onto which we splat following the convention of wikipedia https://en.wikipedia.org/wiki/Bilinear_interpolation
    // const float* q11 = texture + x1*nr_channels_texture + y1*texture_size*nr_channels_texture;
    // const float* q12 = texture + x1*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // const float* q22 = texture + x2*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // const float* q21 = texture + x2*nr_channels_texture + y1*texture_size*nr_channels_texture;

    // printf("accesing at x %d and y %d\n", ix_nw, iy_nw );

    const float* nw_val = texture + ix_nw*nr_channels_texture + iy_nw*texture_width*nr_channels_texture;
    const float* ne_val = texture + ix_ne*nr_channels_texture + iy_ne*texture_width*nr_channels_texture;
    const float* sw_val = texture + ix_sw*nr_channels_texture + iy_sw*texture_width*nr_channels_texture;
    const float* se_val = texture + ix_se*nr_channels_texture + iy_se*texture_width*nr_channels_texture;

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

    //retreive the value weighted between the 4 pixels
    // float val_not_normalized[nr_channels_texture]{0};
    for(int i=0; i<nr_channels_texture; i++){
        // cur_val_not_normalized[i]+=q11[i]*wq11;
        // cur_val_not_normalized[i]+=q12[i]*wq12;
        // cur_val_not_normalized[i]+=q22[i]*wq22;
        // cur_val_not_normalized[i]+=q21[i]*wq21;
        cur_val_not_normalized[i]+=nw_val[i]*nw;
        cur_val_not_normalized[i]+=ne_val[i]*ne;
        cur_val_not_normalized[i]+=sw_val[i]*sw;
        cur_val_not_normalized[i]+=se_val[i]*se;
    }

    // normalize by the homogenenous coordinate
    // for(int i=0; i<val_dim; i++){
        // cur_val[i] = val_not_normalized[i] / val_not_normalized[val_dim];
    // }


}


template< int val_dim, int texture_height, int texture_width>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
splat_texture_backward( int nr_values, float* grad_values, float* grad_uv, const float* grad_texture, const float* values, const float* uv ){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_values){ //don't go out of bounds
        return;
    }

    //possibly needed variables
    int nr_channels_texture=val_dim+1; // in the texture we store also a homogeneous coordinate, so we have a +1

    //grab pointer to the current values we are procesing
    const float* cur_val = values+idx*val_dim;
    const float* cur_uv = uv+idx*2;
    float* cur_grad_uv = grad_uv + idx*2;
    float* cur_grad_values = grad_values + idx*val_dim;

    //the uvs are supposed to be in range [-1, 1], now we get them in range [0, texture_size-1]
    // float x=(cur_uv[0]+1)*0.5*(texture_size-1);
    // float y=(cur_uv[1]+1)*0.5*(texture_size-1);

    // //get the coordiantes of the neighbouring 4 pixels according to the wikipedia convention  https://en.wikipedia.org/wiki/Bilinear_interpolation
    // int x1=(int)x;
    // int y1=(int)y;
    // int x2=min(x1+1, texture_size-1 ); //the min is in order to avoid accesing outside of the boundaries of the texture
    // int y2=min(y1+1, texture_size-1);
    

    // //get a pointer to the 4 pixels onto which we splat following the convention of wikipedia https://en.wikipedia.org/wiki/Bilinear_interpolation
    // const float* dq11 = grad_texture + x1*nr_channels_texture + y1*texture_size*nr_channels_texture;
    // const float* dq12 = grad_texture + x1*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // const float* dq22 = grad_texture + x2*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // const float* dq21 = grad_texture + x2*nr_channels_texture + y1*texture_size*nr_channels_texture;

    // //get the weigthings of the pixels
    // // float denom = 1.0/( (x2 - x1)*(y2-y1)  + 1e-7 ); //the denominator is mostly just to normalize for the size of the pixel but we can assume a pixel of size 1 and just drop this whole term
    // float wq11=(x2-x)*(y2-y);
    // float wq21=(x-x1)*(y2-y);
    // float wq12=(x2-x)*(y-y1);
    // float wq22=(x-x1)*(y-y1);



    //the uvs are supposed to be in range [-1, 1], now we get them in range [0, texture_size-1]
    float ix=(cur_uv[0]+1)*0.5*(texture_width-1);
    float iy=(cur_uv[1]+1)*0.5*(texture_height-1);
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
    CLIP_COORDINATES(ix_nw, ix_nw, texture_width);
    CLIP_COORDINATES(iy_nw, iy_nw, texture_height);
    CLIP_COORDINATES(ix_ne, ix_ne, texture_width);
    CLIP_COORDINATES(iy_ne, iy_ne, texture_height);
    CLIP_COORDINATES(ix_sw, ix_sw, texture_width);
    CLIP_COORDINATES(iy_sw, iy_sw, texture_height);
    CLIP_COORDINATES(ix_se, ix_se, texture_width);
    CLIP_COORDINATES(iy_se, iy_se, texture_height);



    //get a pointer to the 4 pixels onto which we splat following the convention of wikipedia https://en.wikipedia.org/wiki/Bilinear_interpolation
    // const float* q11 = texture + x1*nr_channels_texture + y1*texture_size*nr_channels_texture;
    // const float* q12 = texture + x1*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // const float* q22 = texture + x2*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // const float* q21 = texture + x2*nr_channels_texture + y1*texture_size*nr_channels_texture;

    const float* d_nw_val = grad_texture + ix_nw*nr_channels_texture + iy_nw*texture_width*nr_channels_texture;
    const float* d_ne_val = grad_texture + ix_ne*nr_channels_texture + iy_ne*texture_width*nr_channels_texture;
    const float* d_sw_val = grad_texture + ix_sw*nr_channels_texture + iy_sw*texture_width*nr_channels_texture;
    const float* d_se_val = grad_texture + ix_se*nr_channels_texture + iy_se*texture_width*nr_channels_texture;

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








    ///GRAD UV
    float grad_u=0;
    float grad_v=0;
    for(int i=0; i<val_dim; i++){
        grad_u+=cur_val[i]*(-1)*(iy_se - iy)*d_nw_val[i];
        grad_u+=cur_val[i]*(iy_sw - iy)*d_ne_val[i];
        grad_u+=cur_val[i]*(-1)*(iy    - iy_ne)*d_sw_val[i];
        grad_u+=cur_val[i]*(iy - iy_nw)*d_se_val[i];

        grad_v+=cur_val[i]*(-1)*(ix_se - ix)*d_nw_val[i];
        grad_v+=cur_val[i]*(-1)*(ix    - ix_sw)*d_ne_val[i];
        grad_v+=cur_val[i]*(ix_ne - ix)*d_sw_val[i];
        grad_v+=cur_val[i]*(ix - ix_nw)*d_se_val[i];
    }
    //TODO gradients with respect to the homogeneous coord is as if we splatted a value of 1
    grad_u+=1*(-1)*(iy_se - iy)*d_nw_val[val_dim];
    grad_u+=1*(iy_sw - iy)*d_ne_val[val_dim];
    grad_u+=1*(-1)*(iy    - iy_ne)*d_sw_val[val_dim];
    grad_u+=1*(iy - iy_nw)*d_se_val[val_dim];

    grad_v+=1*(-1)*(ix_se - ix)*d_nw_val[val_dim];
    grad_v+=1*(-1)*(ix    - ix_sw)*d_ne_val[val_dim];
    grad_v+=1*(ix_ne - ix)*d_sw_val[val_dim];
    grad_v+=1*(ix - ix_nw)*d_se_val[val_dim];

    //unnormalize the grad_uv back to the [-1.1] constraint
    // https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c
    grad_u = grad_u * (texture_width - 1) *0.5;
    grad_v = grad_v * (texture_height - 1) *0.5;

    //put them in the tensor
    cur_grad_uv[0]=grad_u;
    cur_grad_uv[1]=grad_v;


    // //GRAD VALUES
    // for(int i=0; i<val_dim; i++){
    //     cur_grad_values[i]+=dq11[i]*wq11;
    //     cur_grad_values[i]+=dq12[i]*wq12;
    //     cur_grad_values[i]+=dq22[i]*wq22;
    //     cur_grad_values[i]+=dq21[i]*wq21;
    // }


    //GRAD VALUES
    for(int i=0; i<val_dim; i++){
        cur_grad_values[i]+=d_nw_val[i]*nw;
        cur_grad_values[i]+=d_ne_val[i]*ne;
        cur_grad_values[i]+=d_sw_val[i]*sw;
        cur_grad_values[i]+=d_se_val[i]*se;
    }




}



template< int nr_channels_texture, int texture_height, int texture_width>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_texture_backward(int nr_values, float* grad_texture, float* grad_uv, const float* grad_values_not_normalized, const float* texture, const float* uv ){


    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_values){ //don't go out of bounds
        return;
    }

    //possibly needed variables
    // int val_dim=nr_channels_texture-1; // in the texture we store also a homogeneous coordinate, so we have a -1

    //grab pointer to the current values we are procesing
    const float* cur_grad_val = grad_values_not_normalized+idx*nr_channels_texture;
    const float* cur_uv = uv+idx*2;
    float* cur_grad_uv = grad_uv + idx*2;


    //the uvs are supposed to be in range [-1, 1], now we get them in range [0, texture_size-1]
    float ix=(cur_uv[0]+1)*0.5*(texture_width-1);
    float iy=(cur_uv[1]+1)*0.5*(texture_height-1);

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
    CLIP_COORDINATES(ix_nw, ix_nw, texture_width);
    CLIP_COORDINATES(iy_nw, iy_nw, texture_height);
    CLIP_COORDINATES(ix_ne, ix_ne, texture_width);
    CLIP_COORDINATES(iy_ne, iy_ne, texture_height);
    CLIP_COORDINATES(ix_sw, ix_sw, texture_width);
    CLIP_COORDINATES(iy_sw, iy_sw, texture_height);
    CLIP_COORDINATES(ix_se, ix_se, texture_width);
    CLIP_COORDINATES(iy_se, iy_se, texture_height);


    //GRAD TEXTURE 
    //get a pointer to the 4 pixels onto which we splat following the convention of wikipedia https://en.wikipedia.org/wiki/Bilinear_interpolation
    // float* dq11 = grad_texture + x1*nr_channels_texture + y1*texture_size*nr_channels_texture;
    // float* dq12 = grad_texture + x1*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // float* dq22 = grad_texture + x2*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // float* dq21 = grad_texture + x2*nr_channels_texture + y1*texture_size*nr_channels_texture;

    float* d_nw_val = grad_texture + ix_nw*nr_channels_texture + iy_nw*texture_width*nr_channels_texture;
    float* d_ne_val = grad_texture + ix_ne*nr_channels_texture + iy_ne*texture_width*nr_channels_texture;
    float* d_sw_val = grad_texture + ix_sw*nr_channels_texture + iy_sw*texture_width*nr_channels_texture;
    float* d_se_val = grad_texture + ix_se*nr_channels_texture + iy_se*texture_width*nr_channels_texture;

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
    for(int i=0; i<nr_channels_texture; i++){
        // atomicAdd(dq11 + i, cur_grad_val[i]*wq11 );
        // atomicAdd(dq12 + i, cur_grad_val[i]*wq12 );
        // atomicAdd(dq22 + i, cur_grad_val[i]*wq22 );
        // atomicAdd(dq21 + i, cur_grad_val[i]*wq21 );
        atomicAdd(d_nw_val + i, cur_grad_val[i]*nw );
        atomicAdd(d_ne_val + i, cur_grad_val[i]*ne );
        atomicAdd(d_sw_val + i, cur_grad_val[i]*sw );
        atomicAdd(d_se_val + i, cur_grad_val[i]*se );
    }


    // //GRAD UV
    // //get a pointer to the 4 pixels onto which we splat following the convention of wikipedia https://en.wikipedia.org/wiki/Bilinear_interpolation
    // const float* q11 = texture + x1*nr_channels_texture + y1*texture_size*nr_channels_texture;
    // const float* q12 = texture + x1*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // const float* q22 = texture + x2*nr_channels_texture + y2*texture_size*nr_channels_texture;
    // const float* q21 = texture + x2*nr_channels_texture + y1*texture_size*nr_channels_texture;
    // float grad_u=0;
    // float grad_v=0;
    // for(int i=0; i<nr_channels_texture; i++){
    //     grad_u+=q11[i]*(-1)*(y2-y)*cur_grad_val[i];
    //     grad_u+=q21[i]*(y2-y)*cur_grad_val[i];
    //     grad_u+=q12[i]*(-1)*(y-y1)*cur_grad_val[i];
    //     grad_u+=q22[i]*(y-y1)*cur_grad_val[i];

    //     grad_v+=q11[i]*(-1)*(x2-x)*cur_grad_val[i];
    //     grad_v+=q21[i]*(-1)*(x-x1)*cur_grad_val[i];
    //     grad_v+=q12[i]*(x2-x)*cur_grad_val[i];
    //     grad_v+=q22[i]*(x-x1)*cur_grad_val[i];
    // }

    
    // //unnormalize the grad_uv back to the [-1.1] constraint
    // // https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c
    // grad_u = grad_u * (texture_size - 1) *0.5;
    // grad_v = grad_v * (texture_size - 1) *0.5;

    // //put them in the tensor
    // cur_grad_uv[0]=grad_u;
    // cur_grad_uv[1]=grad_v;

    
    //GRAD UV
    //get a pointer to the 4 pixels onto which we splat following the convention of wikipedia https://en.wikipedia.org/wiki/Bilinear_interpolation
    const float* nw_val = texture + ix_nw*nr_channels_texture + iy_nw*texture_width*nr_channels_texture;
    const float* ne_val = texture + ix_ne*nr_channels_texture + iy_ne*texture_width*nr_channels_texture;
    const float* sw_val = texture + ix_sw*nr_channels_texture + iy_sw*texture_width*nr_channels_texture;
    const float* se_val = texture + ix_se*nr_channels_texture + iy_se*texture_width*nr_channels_texture;
    float grad_u=0;
    float grad_v=0;
    for(int i=0; i<nr_channels_texture; i++){
        grad_u-=nw_val[i] * (iy_se - iy)*cur_grad_val[i];
        grad_u+=ne_val[i] * (iy_sw - iy)*cur_grad_val[i];
        grad_u-=sw_val[i] * (iy - iy_ne)*cur_grad_val[i];
        grad_u+=se_val[i] * (iy - iy_nw)*cur_grad_val[i];

        grad_v-=nw_val[i] * (ix_se - ix) *cur_grad_val[i];
        grad_v-=ne_val[i] * (ix - ix_sw)*cur_grad_val[i];
        grad_v+=sw_val[i] * (ix_ne - ix)*cur_grad_val[i];
        grad_v+=se_val[i] * (ix - ix_nw)*cur_grad_val[i];
    }

    
    //unnormalize the grad_uv back to the [-1.1] constraint
    // https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c
    grad_u = grad_u * (texture_width - 1) *0.5;
    grad_v = grad_v * (texture_height - 1) *0.5;

    //put them in the tensor
    cur_grad_uv[0]=grad_u;
    cur_grad_uv[1]=grad_v;


}



#endif

