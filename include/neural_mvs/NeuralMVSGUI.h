#pragma once

//ceres (ADDING THIS first before any loguru stuff because otherwise ceres overwrites the LOG functions)
#include "ceres/ceres.h"
#include "ceres/rotation.h"


#include <memory>
#include <stdarg.h>
#include <tuple>


#include <Eigen/Dense>
// #include "neural_mvs/kernels/NeuralMVSGPU.cuh"
#include "easy_pbr/Frame.h"

namespace easy_pbr{
    class Mesh;
    class Viewer;
}


// class Lattice : public torch::autograd::Variable, public std::enable_shared_from_this<Lattice>{
// class Lattice : public at::Tensor, public std::enable_shared_from_this<Lattice>{
class NeuralMVSGUI : public std::enable_shared_from_this<NeuralMVSGUI>{
// class Lattice :public THPVariable, public std::enable_shared_from_this<Lattice>{
public:
    template <class ...Args>
    static std::shared_ptr<NeuralMVSGUI> create( Args&& ...args ){
        return std::shared_ptr<NeuralMVSGUI>( new NeuralMVSGUI(std::forward<Args>(args)...) );
    }
    ~NeuralMVSGUI();

    bool m_show_rgb;
    bool m_show_depth;
    bool m_show_normal;
    float m_min_depth;
    float m_max_depth;
    bool m_control_secondary_cam;

private:
    NeuralMVSGUI(const std::shared_ptr<easy_pbr::Viewer>& view);

    std::shared_ptr<easy_pbr::Viewer> m_view;


    void install_callbacks(const std::shared_ptr<easy_pbr::Viewer>& view); //installs some callbacks that will be called by the viewer after it finishes an update

    //post draw callbacks
    void post_draw(easy_pbr::Viewer& view);
    
   
};
