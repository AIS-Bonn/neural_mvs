#pragma once

//ceres (ADDING THIS first before any loguru stuff because otherwise ceres overwrites the LOG functions)
#include "ceres/ceres.h"
#include "ceres/rotation.h"
using namespace ceres;

#include <memory>
#include <stdarg.h>
#include <tuple>


#include <Eigen/Dense>
// #include "neural_mvs/kernels/NeuralMVSGPU.cuh"
#include "easy_pbr/Frame.h"

namespace easy_pbr{
    class Mesh;
}


// class Lattice : public torch::autograd::Variable, public std::enable_shared_from_this<Lattice>{
// class Lattice : public at::Tensor, public std::enable_shared_from_this<Lattice>{
class SFM : public std::enable_shared_from_this<SFM>{
// class Lattice :public THPVariable, public std::enable_shared_from_this<Lattice>{
public:
    template <class ...Args>
    static std::shared_ptr<SFM> create( Args&& ...args ){
        return std::shared_ptr<SFM>( new SFM(std::forward<Args>(args)...) );
    }
    ~SFM();

    // std::shared_ptr<easy_pbr::Mesh> compute_3D_keypoints_from_frames(const std::vector<easy_pbr::Frame>& frames);
    std::tuple< easy_pbr::MeshSharedPtr, Eigen::VectorXf, Eigen::VectorXi> compute_3D_keypoints_from_frames(const easy_pbr::Frame& frame_query, const easy_pbr::Frame& frame_target);

    static std::vector<float> compute_frame_weights( const easy_pbr::Frame& frame, std::vector<easy_pbr::Frame>& close_frames);
    static void compute_triangulation(std::vector<easy_pbr::Frame>& frames);

   

private:
    SFM();

    std::pair< std::vector<cv::KeyPoint>,   cv::Mat > compute_keypoints_and_descriptor( const easy_pbr::Frame& frame );
    // std::vector<cv::DMatch> filter_matches_lowe_ratio ( std::vector< std::vector<cv::DMatch> >& knn_matches );
    // std::vector<cv::DMatch> filter_cross_check( const std::vector<cv::DMatch>& query_matches, const std::vector<cv::DMatch>& target_matches );
    std::vector<bool> filter_matches_lowe_ratio ( std::vector< std::vector<cv::DMatch> >& knn_matches );
    std::vector<bool> filter_min_max_movement (  const std::vector< std::vector<cv::DMatch> >& knn_query_matches, const std::vector< std::vector<cv::DMatch> >& knn_target_matches, std::vector<cv::KeyPoint>& query_keypoints, 
std::vector<cv::KeyPoint>& target_keypoints,  const int img_size, std::vector<bool>& is_query_match_good_prev  );
    std::vector<bool> filter_cross_check(  const std::vector< std::vector<cv::DMatch> >& knn_query_matches, const std::vector< std::vector<cv::DMatch> >& knn_target_matches,  std::vector<bool>& is_query_match_good_prev );
    Eigen::Vector3f compute_direction_from_2d_point(const Eigen::Vector2d& point_2d, const easy_pbr::Frame& frame );
    Eigen::Vector3f intersect_rays( const Eigen::Vector3f& origin_query, const Eigen::Vector3f& dir_query, const Eigen::Vector3f& origin_target, const Eigen::Vector3f& dir_target);

    void debug_by_projecting(const easy_pbr::Frame& frame, std::shared_ptr<easy_pbr::Mesh> mesh, std::vector<cv::KeyPoint>& keypoints, const std::string name);

    static std::tuple<Eigen::Vector3d, double> fit_sphere( const Eigen::MatrixXd& points);

   
};
