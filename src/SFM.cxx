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


easy_pbr::MeshSharedPtr SFM::compute_3D_keypoints_from_frames(const std::vector<easy_pbr::Frame>& frames){

    easy_pbr::MeshSharedPtr mesh=easy_pbr::Mesh::create();

    VLOG(1) << "got nr of frames " << frames.size();

    //extraxct features for all images
    std::vector< std::vector<cv::KeyPoint> > keypoints_per_frame;
    std::vector< cv::Mat > descriptors_per_frame;
    for(size_t i=0; i<frames.size(); i++){
        const easy_pbr::Frame& frame=frames[i];
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        //extract features
        // auto orb = cv::ORB::create(1000);
        //akaze
        auto feat_extractor = cv::AKAZE::create();
        feat_extractor->detectAndCompute( frame.gray_8u, cv::noArray(), keypoints, descriptors);

        //for using flannbased matcher we need to conver the descriptor to float 32 
        descriptors.convertTo(descriptors, CV_32F);

        keypoints_per_frame.push_back(keypoints);
        descriptors_per_frame.push_back(descriptors);


        //draw extracted features
        cv::Mat keypoints_img;
        cv::drawKeypoints(frame.gray_8u, keypoints, keypoints_img);
        easy_pbr::Gui::show(keypoints_img, "keypoints_img"+std::to_string(i) );
        easy_pbr::Gui::show(frame.rgb_32f, "img"+std::to_string(i) );

    }


    //for each image match against the others  https://github.com/opencv/opencv/blob/2.4/samples/cpp/matching_to_many_images.cpp
    for(size_t i=0; i<frames.size(); i++){
        const easy_pbr::Frame& frame_query=frames[i];
        std::vector<cv::KeyPoint> query_keypoints=keypoints_per_frame[i]; 
        cv::Mat query_descriptors=descriptors_per_frame[i]; 


        for(size_t j=0; j<frames.size(); j++){
            if(i==j){
                continue;
            }

            const easy_pbr::Frame& frame_target=frames[j];
            std::vector<cv::KeyPoint> target_keypoints=keypoints_per_frame[j]; 
            cv::Mat target_descriptors=descriptors_per_frame[j]; 

            auto matcher = cv::DescriptorMatcher::create("FlannBased");
            std::vector<cv::DMatch> matches;
            matcher->add( target_descriptors );
            matcher->train();
            matcher->match( query_descriptors, matches );

            //draw matches 
            cv::Mat matches_img;
            cv::drawMatches (frame_query.gray_8u, query_keypoints, frame_target.gray_8u, target_keypoints, matches, matches_img);
            easy_pbr::Gui::show(matches_img, "matches" );

            //triangulate https://stackoverflow.com/a/16299909
            VLOG(1) << "traingulate";
            //fill the matrices of 2d points
            // cv::Mat cam_0_points(2, query_keypoints.size(), CV_64FC2);
            // cv::Mat cam_1_points(2, target_keypoints.size(), CV_64FC2);
            std::vector<cv::Point2f> cam_0_points;
            std::vector<cv::Point2f> cam_1_points;
            for(size_t p_idx=0; p_idx<query_keypoints.size(); p_idx++){
                // cam_0_points.at<dobule>(0, p_idx)= query_keypoints[p_idx].pt.x;
                // cam_0_points.at<dobule>(1, p_idx)= query_keypoints[p_idx].pt.y;
                cam_0_points.push_back(query_keypoints[p_idx].pt);
            }
            for(size_t p_idx=0; p_idx<target_keypoints.size(); p_idx++){
                // cam_1_points.at<dobule>(0, p_idx)= target_keypoints[p_idx].pt.x;
                // cam_1_points.at<dobule>(1, p_idx)= target_keypoints[p_idx].pt.y;
                cam_1_points.push_back(target_keypoints[p_idx].pt);
            }
            //matrices of each cam
            cv::Mat cam_0_matrix;
            cv::Mat cam_1_matrix;
            Eigen::MatrixXf cam_0_matrix_eigen= frame_query.K * frame_query.tf_cam_world.matrix().block<3,3>(0,0);
            Eigen::MatrixXf cam_1_matrix_eigen= frame_target.K * frame_target.tf_cam_world.matrix().block<3,3>(0,0);
            cv::eigen2cv(cam_0_matrix_eigen, cam_0_matrix);
            cv::eigen2cv(cam_1_matrix_eigen, cam_1_matrix);
            VLOG(1) << "cam eigen is " << cam_0_matrix;
            VLOG(1) << "cam mat is " << cam_0_matrix;
            std::vector<cv::Mat> cam_matrices;
            cam_matrices.push_back(cam_0_matrix);
            cam_matrices.push_back(cam_1_matrix);

            std::vector<std::vector<cv::Point2f>> points2d_for_each_frame;
            points2d_for_each_frame.push_back(cam_0_points);
            points2d_for_each_frame.push_back(cam_1_points);

            // cv::sfm::triangulatePoints(dummy, dummy, dummy);
            cv::Mat dummy;
            std::vector<cv::Point3f> points_3d;
            cv::sfm::triangulatePoints(points2d_for_each_frame, cam_matrices, points_3d);


        }
    }


    return mesh;
}
