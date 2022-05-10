#include "neural_mvs/NeuralMVS.h"

// //ceres 
// #include "ceres/ceres.h"
// #include "ceres/rotation.h"
// using namespace ceres;

#include <glad/glad.h> // Initialize with gladLoadGL()
// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

//c++
#include <string>

#include "UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it
// #include "EasyCuda/UtilsCuda.h"
// #include "EasyPytorch/UtilsPytorch.h"
#include "string_utils.h"
#include "numerical_utils.h"
#include "eigen_utils.h"
#include "easy_pbr/Mesh.h"
#include "easy_pbr/MeshGL.h"
#include "easy_pbr/Gui.h"
#include "easy_gl/UtilsGL.h"


// //ceres 
// #include "ceres/ceres.h"
// #include "ceres/rotation.h"
// using namespace ceres;


// #include "igl/adjacency_list.h"

//my stuff
// #include "lattice_net/HashTable.cuh"
// #include "surfel_renderer/lattice/kernels/HashTableGPU.cuh"
// #include "neural_mvs/kernels/NeuralMVSGPU.cuh"

//jitify
// #define JITIFY_PRINT_INSTANTIATION 1
// #define JITIFY_PRINT_SOURCE 1
// #define JITIFY_PRINT_LOG 1
// #define JITIFY_PRINT_PTX 1
// #define JITIFY_PRINT_LAUNCH 1

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
// using jitify::reflection::type_of;

// #define BLOCK_SIZE 128 //TODO no actually need for it. It can be a parameter. And the one kernel that needs to read this inside it's code can just use BLOCKdim.x
// #define BLOCK_SIZE 64 //TODO no actually need for it. It can be a parameter. And the one kernel that needs to read this inside it's code can just use BLOCKdim.x


using torch::Tensor;
// using namespace easy_pbr::utils;


// std::shared_ptr<NeuralMVSGPU> NeuralMVS::m_impl(new NeuralMVSGPU());



//CPU code that calls the kernels
NeuralMVS::NeuralMVS():
    // m_impl( new NeuralMVSGPU() ),
    m_opengl_initialized(false)
    {
    
    VLOG(1) << "init neural mvs";

    //we do not compile the things here because this object might be created before the viewer and therefore before the opengl context
    // compile_shaders();
    // init_opengl();



}

NeuralMVS::~NeuralMVS(){
    LOG(WARNING) << "Deleting NeuralMVS";
}


void NeuralMVS::compile_shaders(){
    if(!m_depth_test_shader){
        m_depth_test_shader = std::make_shared<gl::Shader>();
    }
    m_depth_test_shader->compile( std::string(CMAKE_SOURCE_DIR)+"/shaders/points_vert.glsl", std::string(CMAKE_SOURCE_DIR)+"/shaders/points_frag.glsl" ) ;
}

void NeuralMVS::init_opengl(){
    compile_shaders();

    if(!m_pos_buffer){
        m_pos_buffer=std::make_shared<gl::GBuffer>();
    }

    GL_C( m_pos_buffer->set_size(256, 256 ) ); //established what will be the size of the textures attached to this framebuffer
    GL_C( m_pos_buffer->add_texture("pos_gtex", GL_RGB32F, GL_RGB, GL_FLOAT) );  
    GL_C( m_pos_buffer->add_depth("depth_gtex") );
    m_pos_buffer->sanity_check();
}

Eigen::MatrixXi NeuralMVS::depth_test(const std::shared_ptr<easy_pbr::Mesh> mesh_core, const Eigen::Affine3d tf_cam_world, const Eigen::Matrix3d K){

    if(!m_opengl_initialized){
        VLOG(1) << "Initializing opengl";
        init_opengl();
        m_opengl_initialized=true;
    }

    // params
    int subsample_factor=1;

    //upload to meshgl
    easy_pbr::MeshGLSharedPtr mesh=easy_pbr::MeshGL::create();
    mesh->assign_core(mesh_core);
    mesh->upload_to_gpu();
    mesh->sanity_check(); //check that we have for sure all the normals for all the vertices and faces and that everything is correct



    // make a viewport of a certain size like 256x256
    int viewport_width=K(0,2)*2;
    int viewport_height=K(1,2)*2;
    glViewport(0.0f , 0.0f, viewport_width/subsample_factor, viewport_height/subsample_factor ); 

    // make a texture to store the XYZ positions 
    if(viewport_width/subsample_factor!=m_pos_buffer->width() || viewport_height/subsample_factor!=m_pos_buffer->height()){
        m_pos_buffer->set_size(viewport_width/subsample_factor, viewport_height/subsample_factor);
    }
    m_pos_buffer->bind_for_draw();
    m_pos_buffer->clear();


    //setup
    gl::Shader &shader = *m_depth_test_shader;
    if(mesh->m_core->V.size()){
        mesh->vao.vertex_attribute(shader, "position", mesh->V_buf, 3);
    }
    if(mesh->m_core->F.size()){
        mesh->vao.indices(mesh->F_buf); //Says the indices with we refer to vertices, this gives us the triangles
    }


    //matrices setuo
    // Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f M=mesh->m_core->model_matrix().cast<float>().matrix();
    Eigen::Matrix4f V = tf_cam_world.matrix().cast<float>();
    Eigen::Matrix4f P = intrinsics_to_opengl_proj(K.cast<float>(), m_pos_buffer->width(), m_pos_buffer->height());
    // Eigen::Matrix4f MV = V*M;
    Eigen::Matrix4f MVP = P*V*M;
 
    //shader setup
    shader.use();
    shader.uniform_4x4(MVP, "MVP");
  


    m_pos_buffer->bind_for_draw();
    shader.draw_into( *m_pos_buffer,
                                    {
                                    std::make_pair("pos_out", "pos_gtex"),
                                    }
                                    ); //makes the shaders draw into the buffers we defines in the gbuffer

    // draw
    mesh->vao.bind(); 
    // glDrawElements(GL_TRIANGLES, mesh->m_core->F.size(), GL_UNSIGNED_INT, 0);
    glPointSize(3.0);
    glDrawArrays(GL_POINTS, 0, mesh->m_core->V.rows());


    GL_C( glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0) );

    //get the texture to cpu 
    cv::Mat pos_mat= m_pos_buffer->tex_with_name("pos_gtex").download_to_cv_mat();
    easy_pbr::Gui::show(pos_mat, "pos_mat");

    //compare each vertex from the mesh to the stored xyz in the texture, if it's close then we deem it as visible
    Eigen::MatrixXi is_visible;
    is_visible.resize(mesh_core->V.rows(),1);
    is_visible.setZero();

    for(int i=0; i<mesh_core->V.rows(); i++){
        Eigen::Vector3d point = mesh_core->V.row(i);
        Eigen::Vector3d point_cam_coords=tf_cam_world*point;
        Eigen::Vector3d point2d = K*point_cam_coords;

        point2d.x()/=point2d.z();
        point2d.y()/=point2d.z();
        int x=point2d.x();
        // int y=m_pos_buffer.height()-point2d.y();
        int y=point2d.y();

        if (y>=0 && y< pos_mat.rows && x>=0 && x<pos_mat.cols ){

            // VLOG(1) << "Accessing at y,x" << y << " " << x << " of pos mat which has rows and cols " << pos_mat.rows<<" " << pos_mat.cols;
            cv::Vec3f pixel= pos_mat.at<cv::Vec3f>(y,x); 
            Eigen::Vector3d pixel_xyz;
            pixel_xyz << pixel[0], pixel[1], pixel[2];
            float diff=(point-pixel_xyz).norm();
            if(diff<0.05){
                is_visible(i,0)=1;
            }
        }
    }


    //AFTERrendering we set again the m_is_dirty so that the next scene.show actually uplaod the data again to the gpu
    mesh_core->m_is_dirty=true;


    return is_visible;

}

// Tensor NeuralMVS::splat_texture( torch::Tensor& values_tensor, torch::Tensor& uv_tensor, const int texture_height, const int texture_width){

//     CHECK(values_tensor.dim()==2 ) << "values tensor should have 2 dimensions correponding to N x val_dim";
//     CHECK(uv_tensor.dim()==2 ) << "UV tensor should have 2 dimensions correponding to N x 2";
//     CHECK(uv_tensor.size(1)==2 ) << "UV tensor last dimensions to have 2 channels";
//     CHECK(values_tensor.scalar_type()==torch::kFloat32 ) << "Values should be float";
//     CHECK(uv_tensor.scalar_type()==torch::kFloat32 ) << "UVs should be float";
//     CHECK(values_tensor.device().is_cuda() ) << "Values should be on the GPU but it has " << values_tensor.device();
//     CHECK(uv_tensor.device().is_cuda() ) << "UVs should be on the GPU but it has " << uv_tensor.device();

//     int nr_values=values_tensor.size(0);
//     int val_dim=values_tensor.size(1);
//     int nr_channels_texture = val_dim+1; // we have a +1 because we store also a homogeneous value

//     values_tensor=values_tensor.contiguous();
//     uv_tensor=uv_tensor.contiguous();

//     TIME_START("splat_cuda");
//     Tensor texture = torch::zeros({ texture_height, texture_width, nr_channels_texture }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

   
//     m_impl->splat_texture( texture.data_ptr<float>(),   //output
//                            values_tensor.data_ptr<float>(), uv_tensor.data_ptr<float>(), //input
//                            nr_values, val_dim, texture_height, texture_width); //constant
//     TIME_END("splat_cuda");

//     return texture;

// }

// Tensor NeuralMVS::slice_texture( torch::Tensor& texture, torch::Tensor& uv_tensor){

//     CHECK(uv_tensor.dim()==2 ) << "UV tensor should have 2 dimensions correponding to N x 2";
//     CHECK(uv_tensor.size(1)==2 ) << "UV tensor last dimensions to have 2 channels";
//     CHECK(texture.dim()==3 ) << "texture should have 3 dimensions correponding to HxWxC";
//     // CHECK( texture.size(0)==texture.size(1) ) << "We are currently assuming that the height and width are the same. At some points I have to implement properly non-square textures";
//     CHECK(texture.scalar_type()==torch::kFloat32 ) << "Texture should be float";
//     CHECK(uv_tensor.scalar_type()==torch::kFloat32 ) << "UVs should be float";
//     CHECK(texture.device().is_cuda() ) << "Texture should be on the GPU but it has " << texture.device();
//     CHECK(uv_tensor.device().is_cuda() ) << "UVs should be on the GPU but it has " << uv_tensor.device();


//     int nr_values=uv_tensor.size(0);
//     int nr_channels_texture=texture.size(2);
//     int texture_height=texture.size(0);
//     int texture_width=texture.size(1);
//     // int val_dim = nr_channels_texture-1; // we have a +1 because we store also a homogeneous value

//     texture=texture.contiguous();
//     uv_tensor=uv_tensor.contiguous();


//     Tensor values_not_normalized_tensor = torch::zeros({ nr_values, nr_channels_texture}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

//     m_impl->slice_texture( values_not_normalized_tensor.data_ptr<float>(), //output
//                            texture.data_ptr<float>(), uv_tensor.data_ptr<float>(), //input 
//                            nr_values, nr_channels_texture, texture_height, texture_width); //constant

//     return values_not_normalized_tensor;
// }


// std::tuple<Tensor, Tensor> NeuralMVS::splat_texture_backward( torch::Tensor& grad_texture, torch::Tensor& values_tensor, torch::Tensor&  uv_tensor ){

//     CHECK(values_tensor.dim()==2 ) << "values tensor should have 2 dimensions correponding to N x val_dim";
//     CHECK(uv_tensor.dim()==2 ) << "UV tensor should have 2 dimensions correponding to N x 2";
//     CHECK(uv_tensor.size(1)==2 ) << "UV tensor last dimensions to have 2 channels";

//     int nr_values=values_tensor.size(0);
//     int val_dim=values_tensor.size(1);
//     int texture_height=grad_texture.size(0);
//     int texture_width=grad_texture.size(1);
//     // int nr_channels_texture = val_dim+1; // we have a +1 because we store also a homogeneous value

//     grad_texture=grad_texture.contiguous();
//     values_tensor=values_tensor.contiguous();
//     uv_tensor=uv_tensor.contiguous();

//     Tensor grad_values = torch::zeros({ nr_values, val_dim }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     Tensor grad_uv = torch::zeros({ nr_values, 2 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

   
//     m_impl->splat_texture_backward( grad_values.data_ptr<float>(), grad_uv.data_ptr<float>(),  //output
//                                     grad_texture.data_ptr<float>(), values_tensor.data_ptr<float>(), uv_tensor.data_ptr<float>(), //input
//                                     nr_values, val_dim, texture_height, texture_width); //constant

//     return std::make_tuple(grad_values, grad_uv);

// }


// std::tuple<torch::Tensor, torch::Tensor> NeuralMVS::slice_texture_backward( torch::Tensor& grad_values_not_normalized, torch::Tensor& texture, torch::Tensor&  uv_tensor ){


//     CHECK(uv_tensor.dim()==2 ) << "UV tensor should have 2 dimensions correponding to N x 2";
//     CHECK(uv_tensor.size(1)==2 ) << "UV tensor last dimensions to have 2 channels";

//     int nr_values=uv_tensor.size(0);
//     int nr_channels_texture=texture.size(2);
//     int texture_height=texture.size(0);
//     int texture_width=texture.size(1);
//     // int val_dim = nr_channels_texture-1; // we have a +1 because we store also a homogeneous value

//     grad_values_not_normalized=grad_values_not_normalized.contiguous();
//     texture=texture.contiguous();
//     uv_tensor=uv_tensor.contiguous();

//     Tensor grad_texture = torch::zeros({ texture_height, texture_width, nr_channels_texture }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     Tensor grad_uv = torch::zeros({ nr_values, 2 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

//     m_impl->slice_texture_backward( grad_texture.data_ptr<float>(), grad_uv.data_ptr<float>(), //output
//                                     grad_values_not_normalized.data_ptr<float>(), texture.data_ptr<float>(), uv_tensor.data_ptr<float>(), //input
//                                     nr_values, nr_channels_texture, texture_height, texture_width); //constant

//     return std::make_tuple(grad_texture, grad_uv);

// }

// torch::Tensor NeuralMVS::subsample( const torch::Tensor& tensor, const int subsample_factor, const std::string subsample_type ){

//     if( subsample_type!="area" && subsample_type!="nearest"){
//         LOG(FATAL) << " Subsample type " << subsample_type << " is not a known type";
//     }
    
//     torch::Tensor tensor_in=tensor.permute({2, 0, 1}).unsqueeze(0).contiguous(); //from H,W,C to N,C,H,W

//     cv::Mat img= tensor2mat(tensor_in);
//     cv::Mat img_resized;
//     if(subsample_type=="area"){
//         cv::resize(img, img_resized, cv::Size(), 1.0/subsample_factor, 1.0/subsample_factor, cv::INTER_AREA);
//     }else if(subsample_type=="nearest"){
//         cv::resize(img, img_resized, cv::Size(), 1.0/subsample_factor, 1.0/subsample_factor, cv::INTER_NEAREST);
//     }

//     torch::Tensor tensor_out=mat2tensor(img_resized, false);

//     //permute from N,C,H,W to H,W,C
//     tensor_out=tensor_out.squeeze(0).permute({1,2,0});

//     return tensor_out;



// }


// //compute weights 
// std::vector<float> NeuralMVS::compute_frame_weights( const easy_pbr::Frame& frame, std::vector<easy_pbr::Frame>& close_frames){
//     // https://people.cs.clemson.edu/~dhouse/courses/404/notes/barycentric.pdf
//     // https://stackoverflow.com/questions/2924795/fastest-way-to-compute-point-to-triangle-distance-in-3d
//     // https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle

//     //to compute the weights we use barycentric coordinates. 
//     //this has several steps, first project the current frame into the triangle defiend by the close_frames. 
//     //compute barycentric coords
//     //if the barycentric coords are not within [0,1], clamp them

//     //checks
//     CHECK(close_frames.size()==3) <<"This assumes we are using 3 frames as close frames because we want to compute barycentric coords";

//     //make triangle
//     Eigen::Vector3d cur_pos= frame.pos_in_world().cast<double>();
//     Eigen::Vector3d p1= close_frames[0].pos_in_world().cast<double>();
//     Eigen::Vector3d p2= close_frames[1].pos_in_world().cast<double>();
//     Eigen::Vector3d p3= close_frames[2].pos_in_world().cast<double>();

//     //get barycentirc coords of the projection https://math.stackexchange.com/a/544947
//     Eigen::Vector3d u=p2-p1;
//     Eigen::Vector3d v=p3-p1;
//     Eigen::Vector3d n=u.cross(v);
//     Eigen::Vector3d w=cur_pos-p1;

//     float w_p3= u.cross(w).dot(n)/ (n.dot(n));
//     float w_p2= w.cross(v).dot(n)/ (n.dot(n));
//     float w_p1= 1.0-w_p2-w_p3;

//     //to get weights as if the point was inside the triangle, we clamp the barycentric coordinates (I don't know if this is needed yeat)

//     //return tha values
//     std::vector<float> vals;
//     vals.push_back(w_p1);
//     vals.push_back(w_p2);
//     vals.push_back(w_p3);

//     return vals;


// }

//compute triangulation 
// https://www.redblobgames.com/x/1842-delaunay-voronoi-sphere/
// void NeuralMVS::compute_triangulation(std::vector<easy_pbr::Frame>& frames){
//     //we assume the frames are laid in a somewhat sphere.
//     // we want to compute a delauany triangulation of them. For this we folow   https://www.redblobgames.com/x/1842-delaunay-voronoi-sphere/
//     // which says we can use steregraphic projection and then do a delaunay in 2D and then lift it back to 3D which will yield a valid triangulation of the points on the sphere. 
    
//     //get all the points from the frames into a EigenMatrix
//     Eigen::MatrixXf points;
//     points.resize(frames.size(),3);
//     for (size_t i=0; i<frames.size(); i++){
//         points.row(i) = frames[i].pos_in_world();
//     }

//     //get an approximate center and radius
//     Eigen::MatrixXf init_center=points.colwise().mean();
//     std::cout << "init center" << init_center << std::endl;
//     VLOG(1) << "init center " << init_center;

//     //compute the radius adn center of the sphere
    
// }

//compute closest frame 
// https://www.redblobgames.com/x/1842-delaunay-voronoi-sphere/



