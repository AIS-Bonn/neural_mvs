#pragma once 

#include <memory>
#include <stdarg.h>

#include "easy_pbr/Frame.h"


class TinyLoader: public std::enable_shared_from_this<TinyLoader>
{
public:
    template <class ...Args>
    static std::shared_ptr<TinyLoader> create( Args&& ...args ){
        return std::shared_ptr<TinyLoader>( new TinyLoader(std::forward<Args>(args)...) );
    }


    easy_pbr::Frame get_frame(const int idx);
    int nr_frames();

private:
    TinyLoader(const std::string config_file);
    void init_params(const std::string config_file);
    void read_poses(std::string pose_file);
    void read_imgs();

    //params 
    std::string m_imgs_path;
    std::string m_pose_file_path;
    int m_subsample_factor;

    std::vector<easy_pbr::Frame> m_frames;
    std::vector<Eigen::Matrix3f> m_intrinsics;
    std::vector<Eigen::Affine3f> m_poses_world_cam;
};