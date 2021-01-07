#pragma once

#include <memory>
#include <stdarg.h>
#include <tuple>

#include <cuda.h>


#include "torch/torch.h"

#include "neural_mvs/jitify_helper/jitify_options.hpp" //Needs to be added BEFORE jitify because this defined the include paths so that the kernels cna find each other
#include "jitify/jitify.hpp"
#include <Eigen/Dense>

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


    //renders the mesh into a camera frame and returns a vector of Nx1 where a 1 indices that the vertex is visible and a 0 indives that the vertex is not rendered into the view or is occluded
    // Eigen::MatriXi depth_test(const std::shared_ptr<easy_pbr::Mesh> mesh, const Eigen::Affine3d tf_cam_world. const Eigen::Matrix3d K); 

    //forward functions
    torch::Tensor splat_texture(const torch::Tensor& values_tensor, const torch::Tensor& uv_tensor, const int& texture_size);
    torch::Tensor slice_texture(const torch::Tensor& texture, const torch::Tensor& uv_tensor);
    //backward functions
    std::tuple<torch::Tensor, torch::Tensor> splat_texture_backward(const torch::Tensor& grad_texture, const torch::Tensor& values_tensor, const torch::Tensor& uv_tensor );
    std::tuple<torch::Tensor, torch::Tensor> slice_texture_backward(const torch::Tensor& grad_values, const torch::Tensor& texture, const torch::Tensor& uv_tensor );


private:
    NeuralMVS();

    std::shared_ptr<NeuralMVSGPU> m_impl;
};


