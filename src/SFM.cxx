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
            // https://answers.opencv.org/question/171898/sfm-triangulatepoints-input-array-assertion-failed/
            VLOG(1) << "traingulate";
            //fill the matrices of 2d points
            // cv::Mat cam_0_points(2, query_keypoints.size(), CV_64FC1);
            // cv::Mat cam_1_points(2, target_keypoints.size(), CV_64FC1);
            // cv::Mat cam_0_points( query_keypoints.size(), 2, CV_64FC1);
            // cv::Mat cam_1_points( target_keypoints.size(), 2, CV_64FC1);
            // std::vector<cv::Point2f> cam_0_points;
            // std::vector<cv::Point2f> cam_1_points;

            // cv::Mat points1Mat = (cv::Mat_<double>(2,1) << 1, 1);
            // cv::Mat points2Mat = (cv::Mat_<double>(2,1) << 1, 1);
            // for(size_t p_idx=0; p_idx<query_keypoints.size(); p_idx++){
                // cam_0_points.at<double>(0, p_idx)= query_keypoints[p_idx].pt.x;
                // cam_0_points.at<double>(1, p_idx)= query_keypoints[p_idx].pt.y;
                // cam_0_points.push_back(query_keypoints[p_idx].pt);
                // cam_0_points.at<double>(p_idx, 0)= query_keypoints[p_idx].pt.x;
                // cam_0_points.at<double>(p_idx, 1)= query_keypoints[p_idx].pt.y;

                // cv::Point2f myPoint1 = query_keypoints[p_idx].pt;
                // cv::Mat matPoint1 = (cv::Mat_<double>(2,1) << myPoint1.x, myPoint1.y);
                // cv::hconcat(points1Mat, matPoint1, points1Mat);
            // }
            // for(size_t p_idx=0; p_idx<target_keypoints.size(); p_idx++){
                // cam_1_points.at<double>(0, p_idx)= target_keypoints[p_idx].pt.x;
                // cam_1_points.at<double>(1, p_idx)= target_keypoints[p_idx].pt.y;
                // cam_1_points.push_back(target_keypoints[p_idx].pt);
                // cam_1_points.at<double>(p_idx, 0)= target_keypoints[p_idx].pt.x;
                // cam_1_points.at<double>(p_idx, 1)= target_keypoints[p_idx].pt.y;
            // }


            //attempt 2
            cv::Mat cam_0_points(2, matches.size(), CV_64FC1);
            cv::Mat cam_1_points(2, matches.size(), CV_64FC1);
            for(size_t m_idx=0; m_idx<matches.size(); m_idx++){
                cv::DMatch match=matches[m_idx];
                int query_idx=match.queryIdx;
                int target_idx=match.queryIdx;

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
            for(size_t p_idx=0; p_idx<nr_points; p_idx++){
                mesh->V(p_idx, 0) = points_3d.at<double>(0, p_idx);
                mesh->V(p_idx, 1) = points_3d.at<double>(1, p_idx);
                mesh->V(p_idx, 2) = points_3d.at<double>(2, p_idx);
            }


        }
    }


    mesh->m_vis.m_show_points=true;

    return mesh;
}
