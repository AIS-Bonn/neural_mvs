#include "neural_mvs/TinyLoader.h"

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;

#include "eigen_utils.h"
using namespace radu::utils;

//boost
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


TinyLoader::TinyLoader(const std::string config_file){

    init_params(config_file);
    read_poses(m_pose_file_path);
    read_imgs();
}

void TinyLoader::init_params(const std::string config_file){

    //read all the parameters
    Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    Config loader_config=cfg["tiny_loader"];

    m_imgs_path=(std::string)loader_config["imgs_path"];
    m_pose_file_path=(std::string)loader_config["pose_file_path"];
    m_subsample_factor=loader_config["subsample_factor"];


}

void TinyLoader::read_poses(std::string pose_file){
    Config cfg = configuru::parse_file(pose_file, CFG);

    //get the nr of cameras
    int nr_cams=cfg.object_size();

    for (int i = 0; i < nr_cams; i++){
        Config cam=cfg["cam_"+std::to_string(i)];

        //read the transform from the cam and the world
        Eigen::VectorXf tf_world_cam_vec = cam["tf_world_cam"];
        Eigen::Affine3f tf_world_cam = tf_vec2matrix(tf_world_cam_vec);
        m_poses_world_cam.push_back(tf_world_cam);

        //read the intrinsics 
        // m_intrinsics_jpeg.setIdentity();
        Eigen::VectorXf K_vec = cam["intrinsics"];
        Eigen::Matrix3f K = K_vec2matrix(K_vec);
        m_intrinsics.push_back(K);

        // CHECK(intrinsics_vec.rows()==4) << "We should have 4 tokens for the intrininscs corresponding to fx fy cx cy";
        // m_intrinsics_jpeg(0,0)= intrinsics_vec[0];
        // m_intrinsics_jpeg(1,1)= intrinsics_vec[1];
        // m_intrinsics_jpeg(0,2)= intrinsics_vec[2];
        // m_intrinsics_jpeg(1,2)= intrinsics_vec[3];


        // m_tf_world_corner.setIdentity();
        // m_tf_world_corner.translation().x() = tf_world_corner_vec[0];
        // m_tf_world_corner.translation().y() = tf_world_corner_vec[1];
        // m_tf_world_corner.translation().z() = tf_world_corner_vec[2];
        // Eigen::Quaterniond q; 
        // q.x() = tf_world_corner_vec[3];
        // q.y() = tf_world_corner_vec[4];
        // q.z() = tf_world_corner_vec[5];
        // q.w() = tf_world_corner_vec[6];
        // q.normalize();
        // m_tf_world_corner.linear()=q.toRotationMatrix();
    }
    

    
    
}

void TinyLoader::read_imgs(){

    //read all the images into frames

    //see how many images we have and read the files paths into a vector
    std::vector<fs::path> rgb_filenames_all;
    for (fs::directory_iterator itr(m_imgs_path); itr!=fs::directory_iterator(); ++itr){
        //chekc fi we should read this file
        std::string extension = boost::filesystem::extension(itr->path().filename() );
        if(     fs::is_regular_file( itr->path()) && 
                ( extension==".png" || extension==".jpeg" || extension==".jpg" )
            ){
            VLOG(1) << "adding filename " << itr->path();
            rgb_filenames_all.push_back(itr->path());
        }
    }

    m_frames.resize(rgb_filenames_all.size());

    //read the files which we assume that they have as names 0, 1, 2, 3 etc
    for (size_t i = 0; i < rgb_filenames_all.size(); i++){
        fs::path img_path=rgb_filenames_all[i];

        //check that the name of it is a number because we will use that number to index into the poses and get them
        int cam_idx=-1;
        try {
            std::string filename = img_path.stem().string();
            VLOG(1) << "filename is " <<filename;
            cam_idx = std::stoi(filename); 
        }catch(std::exception const & e){
            LOG(FATAL) << "Filename cannot be parsed as integer. Please change it to be an integer because we will use it to index into the camera poses " << e.what();
        }

        //Read the mat 
        cv::Mat img=cv::imread(img_path.string());
    

        //read the intrinsics
        auto K = m_intrinsics[cam_idx];

        //read pose 
        auto tf_world_cam=m_poses_world_cam[cam_idx];

        //resize if necesary
        //resize if the downsample factor is anything ther than 1
        if (m_subsample_factor!=1){
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(), 1.0/m_subsample_factor, 1.0/m_subsample_factor, cv::INTER_LANCZOS4 );
            img=resized;
            K/=m_subsample_factor;
            K(2,2)=1.0;
        }

        //put everything into a frame;
        easy_pbr::Frame frame;
        frame.rgb_8u=img;
        frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC3, 1.0/255.0);
        frame.width=frame.rgb_32f.cols;
        frame.height=frame.rgb_32f.rows;
        frame.K=K;
        frame.tf_cam_world=tf_world_cam.inverse();

        m_frames[cam_idx]=frame;

    }

    
}

easy_pbr::Frame TinyLoader::get_frame(const int idx){
    CHECK(idx<m_frames.size()) << "Trying to acces frame with idx " << idx << " but we only have stored " << m_frames.size() << " frames";
    return m_frames[idx];
}

int TinyLoader::nr_frames(){
    return m_frames.size();
}


