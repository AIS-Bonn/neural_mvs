#include "neural_mvs/SFM.h"

//c++
#include <string>



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

//opencv 
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/sfm/triangulation.hpp"
#include <opencv2/core/eigen.hpp>


//boost
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;




//CPU code that calls the kernels
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
    feat_extractor->detectAndCompute( frame.gray_8u, cv::noArray(), keypoints, descriptors);

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


easy_pbr::MeshSharedPtr SFM::compute_3D_keypoints_from_frames(const easy_pbr::Frame& frame_query, const easy_pbr::Frame& frame_target){

    easy_pbr::MeshSharedPtr mesh=easy_pbr::Mesh::create();




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
    VLOG(1) << "traingulate";


    //attempt 2
    cv::Mat cam_0_points(2, matches.size(), CV_64FC1);
    cv::Mat cam_1_points(2, matches.size(), CV_64FC1);
    for(size_t m_idx=0; m_idx<matches.size(); m_idx++){
        cv::DMatch match=matches[m_idx];
        int query_idx=match.queryIdx;
        int target_idx=match.trainIdx;

        cam_0_points.at<double>(0, m_idx)= query_keypoints[query_idx].pt.x;
        cam_0_points.at<double>(1, m_idx)= query_keypoints[query_idx].pt.y;

        cam_1_points.at<double>(0, m_idx)= target_keypoints[target_idx].pt.x;
        cam_1_points.at<double>(1, m_idx)= target_keypoints[target_idx].pt.y;

    }
    


    // VLOG(1) << points1Mat.rows << " x " << points1Mat.cols;
    // VLOG(1) << "cam_0_points " << cam_0_points;
    //matrices of each cam
    cv::Mat cam_0_matrix;
    cv::Mat cam_1_matrix;
    Eigen::MatrixXd cam_0_matrix_eigen= ( frame_query.K * frame_query.tf_cam_world.matrix().block<3,4>(0,0) ).cast<double>();
    Eigen::MatrixXd cam_1_matrix_eigen= ( frame_target.K * frame_target.tf_cam_world.matrix().block<3,4>(0,0) ).cast<double>();
    cv::eigen2cv(cam_0_matrix_eigen, cam_0_matrix);
    cv::eigen2cv(cam_1_matrix_eigen, cam_1_matrix);
    VLOG(1) << "cam eigen is " << cam_0_matrix;
    VLOG(1) << "cam mat is " << cam_0_matrix;
    std::vector<cv::Mat> cam_matrices;
    cam_matrices.push_back(cam_0_matrix);
    cam_matrices.push_back(cam_1_matrix);

    // std::vector<std::vector<cv::Point2f>> points2d_for_each_frame;
    std::vector<cv::Mat> points2d_for_each_frame;
    points2d_for_each_frame.push_back(cam_0_points);
    points2d_for_each_frame.push_back(cam_1_points);
    //DEBUG 
    // points2d_for_each_frame.push_back(points1Mat);
    // points2d_for_each_frame.push_back(points1Mat);


    // cv::sfm::triangulatePoints(dummy, dummy, dummy);
    cv::Mat dummy;
    // std::vector<cv::Point3d> points_3d;
    cv::Mat points_3d;
    cv::sfm::triangulatePoints(points2d_for_each_frame, cam_matrices, points_3d);

    //put the points to V 
    int nr_points=points_3d.cols;
    mesh->V.resize( nr_points, 3 );
    for(int p_idx=0; p_idx<nr_points; p_idx++){
        mesh->V(p_idx, 0) = points_3d.at<double>(0, p_idx);
        mesh->V(p_idx, 1) = points_3d.at<double>(1, p_idx);
        mesh->V(p_idx, 2) = points_3d.at<double>(2, p_idx);
    }










    // //for each image match against the others  https://github.com/opencv/opencv/blob/2.4/samples/cpp/matching_to_many_images.cpp
    // // for(size_t i=0; i<frames.size(); i++){
    // for(size_t i=0; i<1; i++){ //ONLY go for the first frame
    //     const easy_pbr::Frame& frame_query=frames[i];
    //     std::vector<cv::KeyPoint> query_keypoints=keypoints_per_frame[i]; 
    //     cv::Mat query_descriptors=descriptors_per_frame[i]; 


    //     for(size_t j=0; j<frames.size(); j++){
    //         if(i==j){
    //             continue;
    //         }
    //         VLOG(1) << "i and j are "<< i << " " << j;

    //         const easy_pbr::Frame& frame_target=frames[j];
    //         std::vector<cv::KeyPoint> target_keypoints=keypoints_per_frame[j]; 
    //         cv::Mat target_descriptors=descriptors_per_frame[j]; 

    //         // auto matcher = cv::DescriptorMatcher::create("FlannBased");
    //         auto matcher = cv::DescriptorMatcher::create("BruteForce");
    //         std::vector<cv::DMatch> matches;
    //         matcher->add( target_descriptors );
    //         matcher->train();
    //         matcher->match( query_descriptors, matches );

    //         //draw matches 
    //         cv::Mat matches_img;
    //         cv::drawMatches (frame_query.gray_8u, query_keypoints, frame_target.gray_8u, target_keypoints, matches, matches_img);
    //         easy_pbr::Gui::show(matches_img, "matches" );

    //         //triangulate https://stackoverflow.com/a/16299909
    //         // https://answers.opencv.org/question/171898/sfm-triangulatepoints-input-array-assertion-failed/
    //         VLOG(1) << "traingulate";
    //         //fill the matrices of 2d points
    //         // cv::Mat cam_0_points(2, query_keypoints.size(), CV_64FC1);
    //         // cv::Mat cam_1_points(2, target_keypoints.size(), CV_64FC1);
    //         // cv::Mat cam_0_points( query_keypoints.size(), 2, CV_64FC1);
    //         // cv::Mat cam_1_points( target_keypoints.size(), 2, CV_64FC1);
    //         // std::vector<cv::Point2f> cam_0_points;
    //         // std::vector<cv::Point2f> cam_1_points;

    //         // cv::Mat points1Mat = (cv::Mat_<double>(2,1) << 1, 1);
    //         // cv::Mat points2Mat = (cv::Mat_<double>(2,1) << 1, 1);
    //         // for(size_t p_idx=0; p_idx<query_keypoints.size(); p_idx++){
    //             // cam_0_points.at<double>(0, p_idx)= query_keypoints[p_idx].pt.x;
    //             // cam_0_points.at<double>(1, p_idx)= query_keypoints[p_idx].pt.y;
    //             // cam_0_points.push_back(query_keypoints[p_idx].pt);
    //             // cam_0_points.at<double>(p_idx, 0)= query_keypoints[p_idx].pt.x;
    //             // cam_0_points.at<double>(p_idx, 1)= query_keypoints[p_idx].pt.y;

    //             // cv::Point2f myPoint1 = query_keypoints[p_idx].pt;
    //             // cv::Mat matPoint1 = (cv::Mat_<double>(2,1) << myPoint1.x, myPoint1.y);
    //             // cv::hconcat(points1Mat, matPoint1, points1Mat);
    //         // }
    //         // for(size_t p_idx=0; p_idx<target_keypoints.size(); p_idx++){
    //             // cam_1_points.at<double>(0, p_idx)= target_keypoints[p_idx].pt.x;
    //             // cam_1_points.at<double>(1, p_idx)= target_keypoints[p_idx].pt.y;
    //             // cam_1_points.push_back(target_keypoints[p_idx].pt);
    //             // cam_1_points.at<double>(p_idx, 0)= target_keypoints[p_idx].pt.x;
    //             // cam_1_points.at<double>(p_idx, 1)= target_keypoints[p_idx].pt.y;
    //         // }


    //         //attempt 2
    //         cv::Mat cam_0_points(2, matches.size(), CV_64FC1);
    //         cv::Mat cam_1_points(2, matches.size(), CV_64FC1);
    //         for(size_t m_idx=0; m_idx<matches.size(); m_idx++){
    //             cv::DMatch match=matches[m_idx];
    //             int query_idx=match.queryIdx;
    //             int target_idx=match.trainIdx;

    //             cam_0_points.at<double>(0, m_idx)= query_keypoints[query_idx].pt.x;
    //             cam_0_points.at<double>(1, m_idx)= query_keypoints[query_idx].pt.y;

    //             cam_1_points.at<double>(0, m_idx)= target_keypoints[target_idx].pt.x;
    //             cam_1_points.at<double>(1, m_idx)= target_keypoints[target_idx].pt.y;

    //         }
            


    //         // VLOG(1) << points1Mat.rows << " x " << points1Mat.cols;
    //         // VLOG(1) << "cam_0_points " << cam_0_points;
    //         //matrices of each cam
    //         cv::Mat cam_0_matrix;
    //         cv::Mat cam_1_matrix;
    //         Eigen::MatrixXd cam_0_matrix_eigen= ( frame_query.K * frame_query.tf_cam_world.matrix().block<3,4>(0,0) ).cast<double>();
    //         Eigen::MatrixXd cam_1_matrix_eigen= ( frame_target.K * frame_target.tf_cam_world.matrix().block<3,4>(0,0) ).cast<double>();
    //         cv::eigen2cv(cam_0_matrix_eigen, cam_0_matrix);
    //         cv::eigen2cv(cam_1_matrix_eigen, cam_1_matrix);
    //         VLOG(1) << "cam eigen is " << cam_0_matrix;
    //         VLOG(1) << "cam mat is " << cam_0_matrix;
    //         std::vector<cv::Mat> cam_matrices;
    //         cam_matrices.push_back(cam_0_matrix);
    //         cam_matrices.push_back(cam_1_matrix);

    //         // std::vector<std::vector<cv::Point2f>> points2d_for_each_frame;
    //         std::vector<cv::Mat> points2d_for_each_frame;
    //         points2d_for_each_frame.push_back(cam_0_points);
    //         points2d_for_each_frame.push_back(cam_1_points);
    //         //DEBUG 
    //         // points2d_for_each_frame.push_back(points1Mat);
    //         // points2d_for_each_frame.push_back(points1Mat);


    //         // cv::sfm::triangulatePoints(dummy, dummy, dummy);
    //         cv::Mat dummy;
    //         // std::vector<cv::Point3d> points_3d;
    //         cv::Mat points_3d;
    //         cv::sfm::triangulatePoints(points2d_for_each_frame, cam_matrices, points_3d);

    //         //put the points to V 
    //         int nr_points=points_3d.cols;
    //         mesh->V.resize( nr_points, 3 );
    //         for(int p_idx=0; p_idx<nr_points; p_idx++){
    //             mesh->V(p_idx, 0) = points_3d.at<double>(0, p_idx);
    //             mesh->V(p_idx, 1) = points_3d.at<double>(1, p_idx);
    //             mesh->V(p_idx, 2) = points_3d.at<double>(2, p_idx);
    //         }


    //     }
    // }


    mesh->m_vis.m_show_points=true;

    return mesh;
}
