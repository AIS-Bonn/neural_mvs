#include "neural_mvs/PyBridge.h"


//my stuff 
#include "easy_pbr/Mesh.h"
#include "easy_pbr/Viewer.h"
#include "neural_mvs/NeuralMVSGUI.h"
#include "neural_mvs/SFM.h"
#include "neural_mvs/TrainParams.h"



namespace py = pybind11;




PYBIND11_MODULE(neuralmvs, m) {

 
    //TrainParams
    py::class_<TrainParams, std::shared_ptr<TrainParams>   > (m, "TrainParams")
    .def_static("create", &TrainParams::create<const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def("dataset_name",  &TrainParams::dataset_name )
    .def("with_viewer",  &TrainParams::with_viewer )
    .def("with_visdom",  &TrainParams::with_visdom )
    .def("with_tensorboard",  &TrainParams::with_tensorboard )
    .def("with_debug_output",  &TrainParams::with_debug_output )
    .def("with_error_checking",  &TrainParams::with_error_checking )
    .def("lr",  &TrainParams::lr )
    .def("weight_decay",  &TrainParams::weight_decay )
    .def("max_training_epochs",  &TrainParams::max_training_epochs )
    .def("save_checkpoint",  &TrainParams::save_checkpoint )
    .def("checkpoint_path",  &TrainParams::checkpoint_path )
    .def("save_every_x_epoch",  &TrainParams::save_every_x_epoch )
    ;


    //NeuralMVSGUI
    py::class_<NeuralMVSGUI, std::shared_ptr<NeuralMVSGUI>   > (m, "NeuralMVSGUI")
    .def_static("create",  &NeuralMVSGUI::create<const std::shared_ptr<easy_pbr::Viewer>& > ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def_readwrite("m_show_rgb", &NeuralMVSGUI::m_show_rgb )
    .def_readwrite("m_show_depth", &NeuralMVSGUI::m_show_depth )
    .def_readwrite("m_show_normal", &NeuralMVSGUI::m_show_normal )
    .def_readwrite("m_min_depth", &NeuralMVSGUI::m_min_depth )
    .def_readwrite("m_max_depth", &NeuralMVSGUI::m_max_depth )
    .def_readwrite("m_control_secondary_cam", &NeuralMVSGUI::m_control_secondary_cam )
    ;


    
    //SFM
    py::class_<SFM, std::shared_ptr<SFM>   > (m, "SFM")
    .def_static("create", &SFM::create<> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def_static("fit_sphere", &SFM::fit_sphere)
    .def_static("compute_triangulation_stegreographic", &SFM::compute_triangulation_stegreographic)
    .def_static("compute_triangulation_plane", &SFM::compute_triangulation_plane)
    .def_static("compute_closest_triangle", &SFM::compute_closest_triangle)
    ;

    

}