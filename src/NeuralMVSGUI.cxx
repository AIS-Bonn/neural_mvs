#include "neural_mvs/NeuralMVSGUI.h"



//c++
#include <string>

//easypbr
#include "easy_pbr/Gui.h"
#include "easy_pbr/Scene.h"


//opencv 
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
// #include "opencv2/sfm/triangulation.hpp"
#include <opencv2/core/eigen.hpp>


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


#include "string_utils.h"
#include "numerical_utils.h"
#include "eigen_utils.h"
#include "easy_pbr/Mesh.h"
#include "easy_pbr/Gui.h"
#include "easy_pbr/Scene.h"
#include "easy_pbr/Viewer.h"

#include <igl/triangle/triangulate.h>





//boost
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;




NeuralMVSGUI::NeuralMVSGUI(const std::shared_ptr<easy_pbr::Viewer>& view):
    m_show_rgb(true),
    m_show_depth(false),
    m_show_normal(false),
    m_min_depth(0.0),
    m_max_depth(1.0),
    m_control_secondary_cam(false),
    m_view(view)
    {

    install_callbacks(view);

}

NeuralMVSGUI::~NeuralMVSGUI(){
    LOG(WARNING) << "Deleting NeuralMVSGUI";
}

void NeuralMVSGUI::install_callbacks(const std::shared_ptr<easy_pbr::Viewer>& view){
    //pre draw functions (can install multiple functions and they will be called in order)

    //post draw functions
    view->add_callback_post_draw( [this]( easy_pbr::Viewer& v ) -> void{ this->post_draw(v); }  );
}


void NeuralMVSGUI::post_draw(easy_pbr::Viewer& view){
    //get the final render as a opencv mat
    // cv::Mat mat = view.rendered_tex_no_gui(false).download_to_cv_mat();

    // //the opencv mat can now be written to disk or even rendered in the GUI as a texture
    // cv::flip(mat, mat, 0);
    // cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    // Gui::show(mat, "mat");

    //draw some gui elements 
    ImGuiWindowFlags window_flags = 0;
    ImGui::Begin("NeuralMVSGUI", nullptr, window_flags);

    // if (ImGui::Button("My new button")){
        // VLOG(1) << "Clicked button";
    // }

    ImGui::Checkbox("Show rgb", &m_show_rgb); 
    ImGui::Checkbox("Show depth", &m_show_depth); 
    ImGui::Checkbox("Show normal", &m_show_normal); 
    ImGui::Checkbox("Control secondary cam", &m_control_secondary_cam); 
    ImGui::SliderFloat("Min Depth", &m_min_depth, 0.0f, 5.0f);
    ImGui::SliderFloat("Max Depth", &m_max_depth, 0.0f, 5.0f);

    ImGui::End();

    // m_iter++;
}



