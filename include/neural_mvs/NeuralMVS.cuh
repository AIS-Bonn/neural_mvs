#pragma once

//ceres (ADDING THIS first before any loguru stuff because otherwise ceres overwrites the LOG functions)
#include "ceres/ceres.h"
#include "ceres/rotation.h"
using namespace ceres;

#include <memory>
#include <stdarg.h>
#include <tuple>

#include <cuda.h>


#include "torch/torch.h"

#include "neural_mvs/jitify_helper/jitify_options.hpp" //Needs to be added BEFORE jitify because this defined the include paths so that the kernels cna find each other
#include "jitify/jitify.hpp"
#include <Eigen/Dense>
// #include "neural_mvs/kernels/NeuralMVSGPU.cuh"

#include "easy_pbr/Frame.h"

#include "Shader.h"
#include "GBuffer.h"

namespace easy_pbr{
    class Mesh;
}

class NeuralMVSGPU;

// class Lattice : public torch::autograd::Variable, public std::enable_shared_from_this<Lattice>{
// class Lattice : public at::Tensor, public std::enable_shared_from_this<Lattice>{
class NeuralMVS : public std::enable_shared_from_this<NeuralMVS>{
// class Lattice :public THPVariable, public std::enable_shared_from_this<Lattice>{
public:
    template <class ...Args>
    static std::shared_ptr<NeuralMVS> create( Args&& ...args ){
        return std::shared_ptr<NeuralMVS>( new NeuralMVS(std::forward<Args>(args)...) );
    }
    ~NeuralMVS();

    void compile_shaders();
    void init_opengl();


    //renders the mesh into a camera frame and returns a vector of Nx1 where a 1 indices that the vertex is visible and a 0 indives that the vertex is not rendered into the view or is occluded
    Eigen::MatrixXi depth_test(const std::shared_ptr<easy_pbr::Mesh> mesh_core, const Eigen::Affine3d tf_cam_world, const Eigen::Matrix3d K); 

    //forward functions
    static torch::Tensor splat_texture(torch::Tensor& values_tensor, torch::Tensor& uv_tensor, const int texture_height, const int texture_width); //uv tensor is Mx2 and in range [-1,1]
    static torch::Tensor slice_texture(torch::Tensor& texture, torch::Tensor& uv_tensor);
    //backward functions
    static std::tuple<torch::Tensor, torch::Tensor> splat_texture_backward(torch::Tensor& grad_texture, torch::Tensor& values_tensor, torch::Tensor& uv_tensor );
    static std::tuple<torch::Tensor, torch::Tensor> slice_texture_backward(torch::Tensor& grad_values, torch::Tensor& texture, torch::Tensor& uv_tensor );

    // static torch::Tensor subsample( const torch::Tensor& tensor, const int subsample_factor, const std::string subsample_type ); //assumes the tensor input is in format H,W,C
    // static std::vector<float> compute_frame_weights( const easy_pbr::Frame& frame, std::vector<easy_pbr::Frame>& close_frames);
    // static void compute_triangulation(std::vector<easy_pbr::Frame>& frames);


private:
    NeuralMVS();

    gl::Shader m_depth_test_shader;

    gl::GBuffer m_pos_buffer; 

    static std::shared_ptr<NeuralMVSGPU> m_impl;
    // std::shared_ptr<NeuralMVSGPU> m_impl;
    bool m_opengl_initialized;
};


