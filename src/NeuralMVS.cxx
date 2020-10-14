#include "neural_mvs/NeuralMVS.cuh"

//c++
#include <string>

// #include "EasyPytorch/UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it
#include "EasyCuda/UtilsCuda.h"
#include "string_utils.h"
#include "numerical_utils.h"
#include "eigen_utils.h"
#include "easy_pbr/Mesh.h"


// #include "igl/adjacency_list.h"

//my stuff
// #include "lattice_net/HashTable.cuh"
// #include "surfel_renderer/lattice/kernels/HashTableGPU.cuh"
#include "neural_mvs/kernels/NeuralMVSGPU.cuh"

//jitify
#define JITIFY_PRINT_INSTANTIATION 1
#define JITIFY_PRINT_SOURCE 1
#define JITIFY_PRINT_LOG 1
#define JITIFY_PRINT_PTX 1
#define JITIFY_PRINT_LAUNCH 1

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp> //needs to be added after torch.h otherwise loguru stops printing for some reason

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;
//Add this header after we add all cuda stuff because we need the profiler to have cudaDeviceSyncronize defined
#define ENABLE_CUDA_PROFILING 1
#include "Profiler.h" 

//boost
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

//jitify
using jitify::reflection::type_of;

// #define BLOCK_SIZE 128 //TODO no actually need for it. It can be a parameter. And the one kernel that needs to read this inside it's code can just use BLOCKdim.x
// #define BLOCK_SIZE 64 //TODO no actually need for it. It can be a parameter. And the one kernel that needs to read this inside it's code can just use BLOCKdim.x


using torch::Tensor;
// using namespace easy_pbr::utils;






//CPU code that calls the kernels
NeuralMVS::NeuralMVS():
    m_impl( new NeuralMVSGPU() )
    {


}

NeuralMVS::~NeuralMVS(){
    // LOG(WARNING) << "Deleting lattice: " << m_name;
}


// // positions_tensor, uv_tensor, texture_size
// Tensor NeuralSDF::splat_texture( const torch::Tensor& values_tensor, const torch::Tensor& uv_tensor, const int& texture_size){

//     CHECK(values_tensor.dim()==2 ) << "values tensor should have 2 dimensions correponding to N x val_dim";
//     CHECK(uv_tensor.dim()==2 ) << "UV tensor should have 2 dimensions correponding to N x 2";
//     CHECK(uv_tensor.size(1)==2 ) << "UV tensor last dimensions to have 2 channels";

//     int nr_values=values_tensor.size(0);
//     int val_dim=values_tensor.size(1);
//     int nr_channels_texture = val_dim+1; // we have a +1 because we store also a homogeneous value

//     Tensor texture = torch::zeros({ texture_size, texture_size, nr_channels_texture }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

   
//     m_impl->splat_texture( texture.data_ptr<float>(),   //output
//                            values_tensor.data_ptr<float>(), uv_tensor.data_ptr<float>(), //input
//                            nr_values, val_dim, texture_size); //constant

//     return texture;

// }

// }