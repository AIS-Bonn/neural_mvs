#include "neural_mvs/SFM.h"

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


// //ceres 
// #include "ceres/ceres.h"
// #include "ceres/rotation.h"
// using namespace ceres;


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



//boost
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;



//in this contructor we put the things that are not going to be optimized
struct ReprojectionError {
  ReprojectionError(Eigen::Vector2d point_observed, Eigen::Matrix3d K, Eigen::Affine3d tf_cam_world)
      : m_point_observed(point_observed), m_K(K), m_tf_cam_world(tf_cam_world)  {}


  //things that will be optimized go here
  template <typename T>
  bool operator()(
                  const T* const point,
                  T* residuals) const {


    //typedefs
    typedef Eigen::Matrix<T, 3, 1> Vec3;
    typedef Eigen::Transform<T, 3, Eigen::Affine> Affine3d;
    // typedef Eigen::Quaternion<T> Quaterniond;
    // typedef Eigen::Matrix<T, 4, 3> Mat4x3;
    typedef Eigen::Matrix<T, 3, 3> Mat3x3;


    //we need these things as jet object
    Affine3d tf_cam_world;
    // tf_cam_world.linear()=m_tf_cam_world.linear();
    // tf_cam_world.translation()=m_tf_cam_world.translation();
    tf_cam_world=m_tf_cam_world.cast<T>();


    //K matrix because we need it as Jet object
    Mat3x3 K;
    K=m_K.cast<T>();
    // K(0,0)=T(m_K(0,0));
    // K(0,1)=T(m_K(0,1));
    // K(0,2)=T(m_K(0,2));
    // //
    // K(1,0)=T(m_K(1,0));
    // K(1,1)=T(m_K(1,1));
    // K(1,2)=T(m_K(1,2));
    // //
    // K(2,0)=T(m_K(2,0));
    // K(2,1)=T(m_K(2,1));
    // K(2,2)=T(m_K(2,2));

    //make the point in 3D
    Vec3 point_world; 
    point_world.x()=point[0];
    point_world.y()=point[1];
    point_world.z()=point[2];

    //project the 3d into 2D 
    Vec3 point_cam=tf_cam_world*point_world;
    // point_cam.x()/=point_cam.z();
    // point_cam.y()/=point_cam.z();
    // point_cam.z()=T(1.0);
    Vec3 point_2d = K*point_cam;
    point_2d.x()/=point_2d.z();
    point_2d.y()/=point_2d.z();
    point_2d.z()=T(1.0);

    // std::cout << "m_point observed is at "  << m_point_observed << " point 2d projected is " <<point_2d.x() << " " << point_2d.y() << " point world is "<< point_world.transpose() << std::endl;
    T error_x= T(m_point_observed.x()) - point_2d.x();
    T error_y= T(m_point_observed.y()) - point_2d.y();

    residuals[0] = error_x;
    residuals[1] = error_y;


    return true;
  }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create( Eigen::Vector2d point_observed, Eigen::Matrix3d K, Eigen::Affine3d tf_cam_world) {
     return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3>( //the first number is the number of residuals we output,  the rest of the number is the dimensions that we optimize of the inputs to the operator() 
                 new ReprojectionError( point_observed, K, tf_cam_world )));
   }
 

    Eigen::Vector2d m_point_observed;
    Eigen::Matrix3d m_K;
    Eigen::Affine3d m_tf_cam_world;
};

//in this contructor we put the things that are not going to be optimized
struct SphereFitError {
  SphereFitError(Eigen::Vector3d point3d )
      : m_point3d(point3d)   {}


  //things that will be optimized go here
  template <typename T>
  bool operator()(
                  const T* const sphere_center,
                  const T* const sphere_radius,
                  T* residuals) const {


    //typedefs
    typedef Eigen::Matrix<T, 3, 1> Vec3;
    // typedef Eigen::Transform<T, 3, Eigen::Affine> Affine3d;
    // typedef Eigen::Matrix<T, 3, 3> Mat3x3;


    Vec3 sphere_center_eigen; 
    sphere_center_eigen.x()=sphere_center[0];
    sphere_center_eigen.y()=sphere_center[1];
    sphere_center_eigen.z()=sphere_center[2];

    T sphere_radius_eigen=sphere_radius[0];

    // ( |P_i - C|^2 - r^2 )^2
    T distance_to_sphere=  (m_point3d.cast<T>()-sphere_center_eigen).norm() - sphere_radius_eigen;

   

    residuals[0] = distance_to_sphere;


    return true;
  }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create( Eigen::Vector3d point3d) {
     return (new ceres::AutoDiffCostFunction<SphereFitError, 1, 3, 1>( //the first number is the number of residuals we output,  the rest of the number is the dimensions that we optimize of the inputs to the operator() 
                 new SphereFitError( point3d )));
   }
 

    Eigen::Vector3d m_point3d;
};





SFM::SFM()
    {


}

SFM::~SFM(){
    LOG(WARNING) << "Deleting SFM";
}

std::pair< std::vector<cv::KeyPoint>,   cv::Mat > SFM::compute_keypoints_and_descriptor( const easy_pbr::Frame& frame ){


    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    //extract features
    // auto orb = cv::ORB::create(1000);
    //akaze
    auto feat_extractor = cv::AKAZE::create();
    feat_extractor->setThreshold(0.00001);
    // feat_extractor->setNOctaveLayers(2);
    // feat_extractor->setNOctaves(2);
    feat_extractor->detectAndCompute( frame.rgb_8u, cv::noArray(), keypoints, descriptors);

    //for using flannbased matcher we need to conver the descriptor to float 32 
    descriptors.convertTo(descriptors, CV_32F);

    // keypoints_per_frame.push_back(keypoints);
    // descriptors_per_frame.push_back(descriptors);


    //draw extracted features
    // cv::Mat keypoints_img;
    // cv::drawKeypoints(frame.gray_8u, keypoints, keypoints_img);
    // easy_pbr::Gui::show(keypoints_img, "keypoints_img"+std::to_string(i) );
    // easy_pbr::Gui::show(frame.rgb_32f, "img"+std::to_string(i) );

    auto ret = std::make_pair( keypoints, descriptors);
    return ret;

}


std::vector<bool> SFM::filter_matches_lowe_ratio ( std::vector< std::vector<cv::DMatch> >& knn_matches ){

    // https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
    const float ratio_thresh = 0.7f;
    std::vector<bool> is_match_good(knn_matches.size(), false);
    for (size_t i = 0; i < knn_matches.size(); i++){
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
            // good_matches.push_back(knn_matches[i][0]);
            is_match_good[i] = true;
        }
    }

    return is_match_good;

}

std::vector<bool> SFM::filter_min_max_movement (  const std::vector< std::vector<cv::DMatch> >& knn_query_matches, const std::vector< std::vector<cv::DMatch> >& knn_target_matches, std::vector<cv::KeyPoint>& query_keypoints, 
std::vector<cv::KeyPoint>& target_keypoints, const int img_size, std::vector<bool>& is_query_match_good_prev  ){

    // https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
    const float max_movement_ratio = 0.1f;
    // const float min_movement_ratio = 0.01f;
    const float min_movement_ratio = 0.0f;
    std::vector<bool> is_match_good(knn_query_matches.size(), false);
    for (size_t i = 0; i < knn_query_matches.size(); i++){
        if(is_query_match_good_prev[i]){
            int query_idx=knn_query_matches[i][0].queryIdx;
            int target_idx= knn_target_matches[ knn_query_matches[i][0].trainIdx ][0].queryIdx;

            cv::Point2f point_query = query_keypoints[query_idx].pt;
            cv::Point2f point_target = target_keypoints[target_idx].pt;

            Eigen::Vector2f pt1, pt2;
            pt1 <<point_query.x, point_query.y;
            pt2 <<point_target.x, point_target.y;

            float dist= (pt1-pt2).norm();
            float dist_ratio= dist/img_size;

            if(dist_ratio>max_movement_ratio || dist_ratio<min_movement_ratio){
                //it's too far away or the points are too close together for a good triangulation in the other image os it;s a bad match
            }else{
                is_match_good[i]=true;
            }
        }

    }

    return is_match_good;

}

std::vector<bool> SFM::filter_cross_check( const std::vector< std::vector<cv::DMatch> >& knn_query_matches, const std::vector< std::vector<cv::DMatch> >& knn_target_matches, std::vector<bool>& is_query_match_good_prev ){

    // std::vector<cv::DMatch> good_query_matches;
    std::vector<bool> is_query_match_good(knn_query_matches.size(), false);

    // int nr_cross_check_passed=0;

    for (size_t i = 0; i < knn_query_matches.size(); i++){
        cv::DMatch query_match=knn_query_matches[i][0];
        int query_idx= query_match.queryIdx;
        int train_idx= query_match.trainIdx;

        //get the match that the target image says it has at the target points
        cv::DMatch target_match=knn_target_matches[train_idx][0];
        //if the cross check if correct
        // VLOG(1) << " query_idx is "<< query_idx << " train idx is " <<target_match.trainIdx;
        if (query_idx==target_match.trainIdx ){ //the target of the target should be this point
            // good_query_matches.push_back(query_match);
            if (is_query_match_good_prev[i]){
                is_query_match_good[i]=true;
                // nr_cross_check_passed++;
            }
        }
    }

    // VLOG(1) << "Cross check left " << nr_cross_check_passed << " out of a possible of " << knn_query_matches.size();


    return is_query_match_good;


}


//return a mesh with the cloud that we get from the sparse point. also returns the distances of those points to the frame_query and the indexes of the pixels corresponding to the keypoints
std::tuple< easy_pbr::MeshSharedPtr, Eigen::VectorXf, Eigen::VectorXi> SFM::compute_3D_keypoints_from_frames(const easy_pbr::Frame& frame_query, const easy_pbr::Frame& frame_target){

    easy_pbr::MeshSharedPtr mesh=easy_pbr::Mesh::create();

    //get the distance of each of the triangulated keypoints towards the camera and also the index that it has in the camera (x+y*width)
    std::vector<float> keypoint_distances;
    std::vector<int> keypoint_indices;


    //attempt 2 a bit nicer

    //get keypoints and descritpro
    auto keypoints_and_descriptors_query = compute_keypoints_and_descriptor(frame_query);
    auto keypoints_and_descriptors_target = compute_keypoints_and_descriptor(frame_target);
    std::vector<cv::KeyPoint> query_keypoints=keypoints_and_descriptors_query.first; 
    cv::Mat query_descriptors=keypoints_and_descriptors_query.second; 
    std::vector<cv::KeyPoint> target_keypoints=keypoints_and_descriptors_target.first; 
    cv::Mat target_descriptors=keypoints_and_descriptors_target.second; 


    //match 
    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches_query_towards_target;
    std::vector< std::vector<cv::DMatch> > knn_matches_target_towards_query;
    matcher->knnMatch( query_descriptors, target_descriptors, knn_matches_query_towards_target, 2 );
    matcher->knnMatch( target_descriptors, query_descriptors, knn_matches_target_towards_query, 2 );

    
    std::vector<bool> is_query_match_good = filter_matches_lowe_ratio ( knn_matches_query_towards_target );
    std::vector<bool> is_target_match_good = filter_matches_lowe_ratio ( knn_matches_target_towards_query );

    int img_size=std::max(frame_query.width, frame_query.height); 
    is_query_match_good=filter_min_max_movement(knn_matches_query_towards_target, knn_matches_target_towards_query, query_keypoints, target_keypoints,  img_size, is_query_match_good);
    is_target_match_good=filter_min_max_movement(knn_matches_target_towards_query, knn_matches_query_towards_target, query_keypoints, target_keypoints, img_size, is_target_match_good);

    is_query_match_good = filter_cross_check( knn_matches_query_towards_target, knn_matches_target_towards_query, is_query_match_good );


    //actually remove the matches that are not good
    std::vector<cv::DMatch> matches;
    for (size_t i = 0; i < knn_matches_query_towards_target.size(); i++){
        if (is_query_match_good[i]){
            matches.push_back(knn_matches_query_towards_target[i][0]);
        }
    }





     //draw matches 
    cv::Mat matches_img;
    cv::drawMatches (frame_query.gray_8u, query_keypoints, frame_target.gray_8u, target_keypoints, matches, matches_img);
    easy_pbr::Gui::show(matches_img, "matches" );




    //triangulate https://stackoverflow.com/a/16299909
    // https://answers.opencv.org/question/171898/sfm-triangulatepoints-input-array-assertion-failed/
    // VLOG(1) << "traingulate";


    // //attempt 2
    // cv::Mat cam_0_points(2, matches.size(), CV_64FC1);
    // cv::Mat cam_1_points(2, matches.size(), CV_64FC1);
    // for(size_t m_idx=0; m_idx<matches.size(); m_idx++){
    //     cv::DMatch match=matches[m_idx];
    //     int query_idx=match.queryIdx;
    //     int target_idx=match.trainIdx;

    //     cam_0_points.at<double>(0, m_idx)= query_keypoints[query_idx].pt.x;
    //     cam_0_points.at<double>(1, m_idx)= query_keypoints[query_idx].pt.y;

    //     cam_1_points.at<double>(0, m_idx)= target_keypoints[target_idx].pt.x;
    //     cam_1_points.at<double>(1, m_idx)= target_keypoints[target_idx].pt.y;

    // }
    


    // // VLOG(1) << points1Mat.rows << " x " << points1Mat.cols;
    // // VLOG(1) << "cam_0_points " << cam_0_points;
    // //matrices of each cam
    // cv::Mat cam_0_matrix;
    // cv::Mat cam_1_matrix;
    // // Eigen::MatrixXd cam_0_matrix_eigen= ( frame_query.K * frame_query.tf_cam_world.matrix().block<3,4>(0,0) ).cast<double>();
    // // Eigen::MatrixXd cam_1_matrix_eigen= ( frame_target.K * frame_target.tf_cam_world.matrix().block<3,4>(0,0) ).cast<double>();

    // //with cma query being at identity
    // Eigen::Affine3f tf_cam_query_target=  frame_query.tf_cam_world* frame_target.tf_cam_world.inverse(); //target to world and world to query
    // Eigen::Affine3f id=tf_cam_query_target;
    // id.setIdentity();
    // Eigen::MatrixXd cam_0_matrix_eigen= ( frame_query.K * id.matrix().block<3,4>(0,0) ).cast<double>();
    // Eigen::MatrixXd cam_1_matrix_eigen= ( frame_target.K * tf_cam_query_target.matrix().block<3,4>(0,0) ).cast<double>();
    // cv::eigen2cv(cam_0_matrix_eigen, cam_0_matrix);
    // cv::eigen2cv(cam_1_matrix_eigen, cam_1_matrix);
    // // VLOG(1) << "cam eigen is " << cam_0_matrix;
    // // VLOG(1) << "cam mat is " << cam_0_matrix;
    // std::vector<cv::Mat> cam_matrices;
    // cam_matrices.push_back(cam_0_matrix);
    // cam_matrices.push_back(cam_1_matrix);

    // // std::vector<std::vector<cv::Point2f>> points2d_for_each_frame;
    // std::vector<cv::Mat> points2d_for_each_frame;
    // points2d_for_each_frame.push_back(cam_0_points);
    // points2d_for_each_frame.push_back(cam_1_points);
    // //DEBUG 
    // // points2d_for_each_frame.push_back(points1Mat);
    // // points2d_for_each_frame.push_back(points1Mat);


    // // cv::sfm::triangulatePoints(dummy, dummy, dummy);
    // cv::Mat dummy;
    // // std::vector<cv::Point3d> points_3d;
    // cv::Mat points_3d;
    // cv::sfm::triangulatePoints(points2d_for_each_frame, cam_matrices, points_3d);

    // //put the points to V 
    // int nr_points=points_3d.cols;
    // mesh->V.resize( nr_points, 3 );
    // for(int p_idx=0; p_idx<nr_points; p_idx++){
    //     mesh->V(p_idx, 0) = points_3d.at<double>(0, p_idx);
    //     mesh->V(p_idx, 1) = -points_3d.at<double>(1, p_idx);
    //     mesh->V(p_idx, 2) = points_3d.at<double>(2, p_idx);
    // }




    //attempt 3 using ceres
    // ceres::Problem problem;

    // std::vector<double> points_3d( matches.size()*3, 0); 

    // for(size_t m_idx=0; m_idx<matches.size(); m_idx++){
    //     cv::DMatch match=matches[m_idx];
    //     int query_idx=match.queryIdx;
    //     int target_idx=match.trainIdx;

    //     Eigen::Vector2d query_observed;
    //     query_observed << query_keypoints[query_idx].pt.x, query_keypoints[query_idx].pt.y;
    //     // query_observed << target_keypoints[target_idx].pt.x, target_keypoints[target_idx].pt.y;

    //     Eigen::Vector2d target_observed;
    //     target_observed <<  target_keypoints[target_idx].pt.x, target_keypoints[target_idx].pt.y;
    //     // target_observed << query_keypoints[query_idx].pt.x, query_keypoints[query_idx].pt.y;

    //     ceres::CostFunction* cost_function_query = ReprojectionError::Create( query_observed, frame_query.K.cast<double>(), frame_query.tf_cam_world.cast<double>() );
    //     ceres::CostFunction* cost_function_target = ReprojectionError::Create( target_observed, frame_target.K.cast<double>(), frame_target.tf_cam_world.cast<double>() );

    //     problem.AddResidualBlock(cost_function_query,
    //                            NULL /* squared loss */,
    //                         // loss_function,
    //                         points_3d.data()+m_idx*3
    //                         );
    //     problem.AddResidualBlock(cost_function_target,
    //                            NULL /* squared loss */,
    //                         // loss_function,
    //                         points_3d.data()+m_idx*3
    //                         );

    // }

    // //solve
    // ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.minimizer_progress_to_stdout = true;
    // options.max_num_iterations=400;
    // // options.max_num_iterations=1;
    // ceres::Solver::Summary summary;
    // ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << "\n";
    // VLOG_S(1) << summary.FullReport();

    // //put the points to V 
    // int nr_points=matches.size();
    // mesh->V.resize( nr_points, 3 );
    // for(int p_idx=0; p_idx<nr_points; p_idx++){
    //     mesh->V(p_idx, 0) = points_3d[p_idx*3+0];
    //     mesh->V(p_idx, 1) = points_3d[p_idx*3+1];
    //     mesh->V(p_idx, 2) = points_3d[p_idx*3+2];
    // }



    //attempt 4 using ray intersection 
    // int nr_points=matches.size();
    // mesh->V.resize( nr_points, 3 );
    std::vector<Eigen::VectorXd> points_3d;
    // https://stackoverflow.com/questions/29188686/finding-the-intersect-location-of-two-rays
    for(size_t m_idx=0; m_idx<matches.size(); m_idx++){
        cv::DMatch match=matches[m_idx];
        int query_idx=match.queryIdx;
        int target_idx=match.trainIdx;

        Eigen::Vector2d query_observed;
        query_observed << query_keypoints[query_idx].pt.x, query_keypoints[query_idx].pt.y;
        Eigen::Vector2d target_observed;
        target_observed <<  target_keypoints[target_idx].pt.x, target_keypoints[target_idx].pt.y;

        //calculate the directions
        Eigen::Vector3f dir_query = compute_direction_from_2d_point(query_observed, frame_query);
        Eigen::Vector3f dir_target = compute_direction_from_2d_point(target_observed, frame_target);
        Eigen::Vector3f origin_query=frame_query.tf_cam_world.inverse().translation();
        Eigen::Vector3f origin_target=frame_target.tf_cam_world.inverse().translation();

        // if (m_idx==0){
        //     //show ray 
        //     easy_pbr::MeshSharedPtr ray_query = easy_pbr::Mesh::create();
        //     ray_query->V.resize(2,3);
        //     ray_query->V.row(0) = origin_query.cast<double>();
        //     ray_query->V.row(1) = origin_query.cast<double>()  +dir_query.cast<double>()*10;
        //     ray_query->E.resize(1,2);
        //     ray_query->E.row(0) << 0,1;
        //     ray_query->m_vis.m_show_lines=true;
        //     easy_pbr::Scene::show(ray_query, "ray_query");
        //     //ray target 
        //     easy_pbr::MeshSharedPtr ray_target = easy_pbr::Mesh::create();
        //     ray_target->V.resize(2,3);
        //     ray_target->V.row(0) = origin_target.cast<double>();
        //     ray_target->V.row(1) = origin_target.cast<double>()  +dir_target.cast<double>()*10;
        //     ray_target->E.resize(1,2);
        //     ray_target->E.row(0) << 0,1;
        //     ray_target->m_vis.m_show_lines=true;
        //     easy_pbr::Scene::show(ray_target, "ray_target");
        // }


        // https://stackoverflow.com/questions/29188686/finding-the-intersect-location-of-two-rays
        Eigen::Vector3f point_intersection;
        Eigen::Vector3f d3= dir_query.cross(dir_target);

        // auto ray_dirs_mesh=frame_query.pixels2dirs_mesh();

        if (d3.norm()<0.00001){
            std::cout<< "lines are in the same direction. Cannot compute intersection" << std::endl;
        }else{
            point_intersection=intersect_rays(origin_query, dir_query, origin_target, dir_target);
            if (!point_intersection.isZero()){
                points_3d.push_back( point_intersection.cast<double>() );
                float dist=  (point_intersection - frame_query.tf_cam_world.inverse().translation()).norm();
                keypoint_distances.push_back(dist);
                // keypoint_indices.push_back( query_observed.x() + (frame_query.height-query_observed.y())*frame_query.width);
                // keypoint_indices.push_back( query_observed.x() + query_observed.y()*frame_query.width);
                keypoint_indices.push_back( int(query_observed.x()) + int(query_observed.y())*frame_query.width);

                // //debug why the heck does reprojecting this does not give me the point_intersection
                // Eigen::Vector3f point_screen;
                // //the point is not at x,y but at x, heght-y. That's because we got the depth from the depth map at x,y and we have to take into account that opencv mat has origin at the top left. However the camera coordinate system actually has origin at the bottom left (same as the origin of the uv space in opengl) So the point in screen coordinates will be at x, height-y
                // point_screen << query_observed.x(),frame_query.height-query_observed.y(),1.0;
                // Eigen::Vector3f point_cam_coords;
                // point_cam_coords=frame_query.K.inverse()*point_screen;
                // Eigen::Vector3f point_world_coords;
                // point_world_coords=frame_query.tf_cam_world.inverse()*point_cam_coords;
                // Eigen::Vector3f dir=(point_world_coords-frame_query.tf_cam_world.inverse().translation()).normalized();
                // Eigen::Vector3f point_reprojected= frame_query.tf_cam_world.inverse().translation() + dir*dist; 
                // float error= (point_reprojected-point_intersection).norm();
                // if (frame_query.frame_idx==10 && points_3d.size()==1){
                //     std::cout << " point_intersection "<< point_intersection.transpose() << " point_reprojected "<< point_reprojected.transpose() <<  " error is " << error << std::endl;
                //     std::cout << "dir is " << dir.transpose() << " dist is " << dist << " center is " << frame_query.tf_cam_world.inverse().translation().transpose() << std::endl;
                //     std::cout << "query_observed.x() " <<  query_observed.x() << " query_observed.y() " <<  query_observed.y() << " frame_width " << frame_query.width <<std::endl;
                //     // std::cout << "pushed_index " <<  query_observed.x() + query_observed.y()*frame_query.width << std::endl;
                //     // int pushed_index= query_observed.x() + query_observed.y()*frame_query.width;
                //     int pushed_index= keypoint_indices.back();
                //     std::cout << "pushed_index (INT) " <<  pushed_index << std::endl;

                //     //find which is the index that this dir has
                //     for(int i=0; i<ray_dirs_mesh->V.rows(); i++){
                //         Eigen::Vector3d dir_from_mesh=ray_dirs_mesh->V.row(i);
                //         float error= (dir_from_mesh- dir.cast<double>()).norm();
                //         int y= i/frame_query.width;
                //         int x = i-y*frame_query.width;
                //         if(error<0.01){
                //             std::cout << " found dir_from mesh " <<dir_from_mesh.transpose() << " at idx " << i << " error is "<< error <<std::endl;
                //             std::cout  <<" this linear index correspnds to x " << x << " and y " << y << std::endl;
                //         }
                //         if(i==pushed_index){
                //             std::cout << "==== at Pushed_inndex dir_from mesh " <<dir_from_mesh.transpose() << " at idx " << i << " error is "<< error <<std::endl;
                //             std::cout  <<" this linear index correspnds to x " << x << " and y " << y << std::endl;
                //         }

                //     }

                // }


            }
        }


    }

    mesh->V=radu::utils::vec2eigen(points_3d);
    
    Eigen::VectorXf keypoint_distances_eigen=radu::utils::vec2eigen(keypoint_distances);
    Eigen::VectorXi keypoint_indices_eigen=radu::utils::vec2eigen(keypoint_indices);



    




    mesh=frame_query.assign_color(mesh);
    mesh->m_vis.set_color_pervertcolor();




    mesh->m_vis.m_show_points=true;
    // mesh->set_model_matrix( frame_query.tf_cam_world.inverse().cast<double>() );
    // mesh->apply_model_matrix_to_cpu(false);


    //DEBUG by projecting the 3d points into the frame
    //project in query img 
    // debug_by_projecting(frame_query, mesh, query_keypoints, "debug_query");
    // debug_by_projecting(frame_target, mesh, target_keypoints, "debug_target");
    // cv::Mat img =frame_query.rgb_32f;
    // Eigen::Affine3d trans=frame_query.tf_cam_world.cast<double>();
    // Eigen::Matrix3d K=frame_query.K.cast<double>();
    // for (int i = 0; i < mesh->V.rows(); i++){
    //     Eigen::Vector3d point_world= Eigen::Vector3d(mesh->V.row(i));
    //     Eigen::Vector3d point_cam = trans.linear()*point_world + trans.translation();
    //     point_cam.y()=-point_cam.y();
    //     Eigen::Vector3d point_img=K*point_cam;
    //     int x= point_img.x()/ point_img.z();
    //     int y= point_img.y()/ point_img.z();

    //     cv::Point pt = cv::Point(x, y);
    //     cv::circle(img, pt, 1, cv::Scalar(0, 255, 0));
    // }
    // easy_pbr::Gui::show(img, "img_debug");


    return {mesh, keypoint_distances_eigen, keypoint_indices_eigen} ;
}


Eigen::Vector3f SFM::compute_direction_from_2d_point(const Eigen::Vector2d& point_2d,  const easy_pbr::Frame& frame){

    //calculate the directions
    Eigen::Vector3f point_screen;
    point_screen << point_2d.x(), frame.height - point_2d.y(),1.0;  //the point is not at x,y but at x, heght-y. That's because we got the depth from the depth map at x,y and we have to take into account that opencv mat has origin at the top left. However the camera coordinate system actually has origin at the bottom left (same as the origin of the uv space in opengl) So the point in screen coordinates will be at x, height-y
    Eigen::Vector3f point_cam_coords;
    point_cam_coords=frame.K.inverse()*point_screen;
    Eigen::Vector3f point_world_coords;
    point_world_coords=frame.tf_cam_world.inverse()*point_cam_coords;
    Eigen::Vector3f dir=(point_world_coords-frame.tf_cam_world.inverse().translation()).normalized();

    return dir;

}


Eigen::Vector3f SFM::intersect_rays( const Eigen::Vector3f& origin_query, const Eigen::Vector3f& dir_query, const Eigen::Vector3f& origin_target, const Eigen::Vector3f& dir_target){


    // Matrix32 A; A << T_search_ref.linear() * f_ref, f_cur;
    // const Matrix2 AtA = A.transpose()*A;
    // if(AtA.determinant() < 0.000001)
    //     return false;
    // const Vector2 depth2 = - AtA.inverse()*A.transpose()*T_search_ref.translation();
    // depth = fabs(depth2[0]);
    // if ( ! std::isfinite( depth ) ) throw std::runtime_error("hickup");
    // return true;

    Eigen::Vector3f point_intersection;

    Eigen::Vector3f origin_diff = origin_query - origin_target;

    // https://stackoverflow.com/a/34604574
    float a = dir_query.dot(dir_query);
    float b = dir_query.dot(dir_target);
    float c = dir_target.dot(dir_target);
    float d = dir_query.dot(origin_diff);
    float e = dir_target.dot(origin_diff);

    //find discriminant
    float discriminant = a * c- b * b;
    if( std::fabs(discriminant)<1e-5 ){ //lines are parallel so there is no intersection
        point_intersection.setZero(); //INVALID
    }else{
        float tt = (b * e - c * d) / discriminant;
        float uu = (a * e - b * d) / discriminant;

        Eigen::Vector3f intersection_along_query= origin_query + tt * dir_query;
        Eigen::Vector3f intersection_along_target= origin_target + uu * dir_target;
        float dist = (intersection_along_query-intersection_along_target).norm();
        float dist_thresh=0.1;
        if(dist>dist_thresh){
            point_intersection.setZero(); //INVALID, the distance is too great
        }else{
            // middle point between the two points along the ray
            point_intersection= (intersection_along_query+intersection_along_target)/2.0;
        }
    }


    return point_intersection;

}



void SFM::debug_by_projecting(const easy_pbr::Frame& frame, std::shared_ptr<easy_pbr::Mesh> mesh, std::vector<cv::KeyPoint>& keypoints,  const std::string name){

    cv::Mat img =frame.rgb_32f.clone();
    // Eigen::Affine3d trans=frame.tf_cam_world.cast<double>();
    Eigen::Matrix3d K=frame.K.cast<double>();
    for (int i = 0; i < mesh->V.rows(); i++){
        Eigen::Vector3d point_world= Eigen::Vector3d(mesh->V.row(i));
        // Eigen::Vector3d point_cam = trans.linear()*point_world + trans.translation();
        Eigen::Vector3d point_cam = frame.tf_cam_world.cast<double>()*point_world;
        point_cam.y()=-point_cam.y();
        Eigen::Vector3d point_img=K*point_cam;
        float x= point_img.x()/ point_img.z();
        float y= point_img.y()/ point_img.z();

        // std::cout << " Point2d is " << x << " " << y << " point3d is " << point_world.transpose() << std::endl;

        cv::Point pt = cv::Point( std::round(x), std::round(y) );
        cv::circle(img, pt, 4, cv::Scalar(0, 0, 255), 3); //projected in red
    }

    for (size_t i = 0; i < keypoints.size(); i++){
        cv::Point pt = keypoints[i].pt;
        cv::circle(img, pt, 1, cv::Scalar(0, 255, 0)); //keypoints in blue
    }


    easy_pbr::Gui::show(img, name);

}


//compute weights 
std::vector<float> SFM::compute_frame_weights( const easy_pbr::Frame& frame, std::vector<easy_pbr::Frame>& close_frames){
    // https://people.cs.clemson.edu/~dhouse/courses/404/notes/barycentric.pdf
    // https://stackoverflow.com/questions/2924795/fastest-way-to-compute-point-to-triangle-distance-in-3d
    // https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle

    //to compute the weights we use barycentric coordinates. 
    //this has several steps, first project the current frame into the triangle defiend by the close_frames. 
    //compute barycentric coords
    //if the barycentric coords are not within [0,1], clamp them

    //checks
    CHECK(close_frames.size()==3) <<"This assumes we are using 3 frames as close frames because we want to compute barycentric coords";

    //make triangle
    Eigen::Vector3d cur_pos= frame.pos_in_world().cast<double>();
    Eigen::Vector3d p1= close_frames[0].pos_in_world().cast<double>();
    Eigen::Vector3d p2= close_frames[1].pos_in_world().cast<double>();
    Eigen::Vector3d p3= close_frames[2].pos_in_world().cast<double>();

    //get barycentirc coords of the projection https://math.stackexchange.com/a/544947
    Eigen::Vector3d u=p2-p1;
    Eigen::Vector3d v=p3-p1;
    Eigen::Vector3d n=u.cross(v);
    Eigen::Vector3d w=cur_pos-p1;

    float w_p3= u.cross(w).dot(n)/ (n.dot(n));
    float w_p2= w.cross(v).dot(n)/ (n.dot(n));
    float w_p1= 1.0-w_p2-w_p3;

    //to get weights as if the point was inside the triangle, we clamp the barycentric coordinates (I don't know if this is needed yeat)

    //return tha values
    std::vector<float> vals;
    vals.push_back(w_p1);
    vals.push_back(w_p2);
    vals.push_back(w_p3);

    return vals;


}


std::tuple<Eigen::Vector3d, double> SFM::fit_sphere( const Eigen::MatrixXd& points){

    //get an approximate center and radius
    Eigen::VectorXd init_center=points.colwise().mean();
    VLOG(1) << "init center " << init_center;
    Eigen::VectorXd min_point = points.colwise().minCoeff();   
    Eigen::VectorXd max_point = points.colwise().maxCoeff();   
    float init_radius= 0.5* (max_point-min_point).norm();
    VLOG(1) << "radius is " << init_radius; 

    
    //establish a error function similar to https://stackoverflow.com/q/10344119
    ceres::Problem problem;
    std::vector<double> sphere_center(3);
    std::vector<double> sphere_radius(1);
    sphere_center[0] =init_center.x();
    sphere_center[1] =init_center.y();
    sphere_center[2] =init_center.z();
    sphere_radius[0]= init_radius;
    for (int i=0; i<points.rows(); i++){
        Eigen::Vector3d point3d= points.row(i);
        ceres::CostFunction* cost_function = SphereFitError::Create( point3d );
        problem.AddResidualBlock(cost_function,
                               NULL /* squared loss */,
                            sphere_center.data(),
                            sphere_radius.data());

        
    }

    //solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations=400;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    VLOG(1) << summary.FullReport();


    //read the sphere params
    Eigen::Vector3d sphere_center_final;
    double sphere_radius_final;
    sphere_center_final.x()=sphere_center[0];
    sphere_center_final.y()=sphere_center[1];
    sphere_center_final.z()=sphere_center[2];
    sphere_radius_final= sphere_radius[0];

    return std::make_tuple(sphere_center_final, sphere_radius_final);


}

// compute triangulation 
// https://www.redblobgames.com/x/1842-delaunay-voronoi-sphere/
void SFM::compute_triangulation(std::vector<easy_pbr::Frame>& frames){
    //we assume the frames are laid in a somewhat sphere.
    // we want to compute a delauany triangulation of them. For this we folow   https://www.redblobgames.com/x/1842-delaunay-voronoi-sphere/
    // which says we can use steregraphic projection and then do a delaunay in 2D and then lift it back to 3D which will yield a valid triangulation of the points on the sphere. 
    
    //get all the points from the frames into a EigenMatrix
    Eigen::MatrixXd points;
    points.resize(frames.size(),3);
    for (size_t i=0; i<frames.size(); i++){
        points.row(i) = frames[i].pos_in_world().cast<double>();
    }

    auto sphere_params=fit_sphere(points);
    Eigen::Vector3d sphere_center = std::get<0>(sphere_params);
    double sphere_radius = std::get<1>(sphere_params);

    //make sphere and check that it looks ok
    auto sphere= easy_pbr::Mesh::create();
    sphere->create_sphere(sphere_center, sphere_radius);
    sphere->m_vis.m_show_mesh=false;
    sphere->m_vis.m_show_points=true;
    easy_pbr::Scene::show(sphere,"sphere");


    VLOG(1) << "sphere center_final " << sphere_center; 
    VLOG(1) << "sphere radius_final " << sphere_radius; 

    
}



//attempt 2 
// compute_3D_keypoints_from_frames
// {

    //for each frame compute keypoints and descriptors

    //for each frame, get the potential mathing ones, which are the ones that have a frustum that points in a somewhat similar direction

    //for each frame, compute the matches towards the potential frames

    //for each frame 
    //  for each keypoint in the frame 
    //      if keypoint_has_no_3d associated{
    //        instantiate a 3D point at origin
    //        record that this keypoint for this frame has a 3D associated
    //        associate a reprojection error function for this 3D point
    //      }
    //      for each match 
    //          frame_target=frame_in which the match was found
    //          record that this keypoint for this frame has a 3D associated
    //          associate a reprojection error function for this 3D point
    //
    //bundle adjust with known poses          


// }



//attempt 3 with colmap 
//compute_3D_keypoints_from_frames 
// {
    
    //run colmap

// }



