#pragma once

//ceres (ADDING THIS first before any loguru stuff because otherwise ceres overwrites the LOG functions)
#include "ceres/ceres.h"
#include "ceres/rotation.h"


#include <memory>
#include <stdarg.h>
#include <tuple>


#include <Eigen/Dense>
#include "easy_pbr/Frame.h"

namespace easy_pbr{
    class Mesh;
}


class SFM : public std::enable_shared_from_this<SFM>{
public:
    template <class ...Args>
    static std::shared_ptr<SFM> create( Args&& ...args ){
        return std::shared_ptr<SFM>( new SFM(std::forward<Args>(args)...) );
    }
    ~SFM();

   
    static std::tuple<Eigen::Vector3d, double> fit_sphere( const Eigen::MatrixXd& points);
    static easy_pbr::MeshSharedPtr compute_triangulation_stegreographic( const Eigen::MatrixXd& points,  const Eigen::Vector3d& sphere_center, double sphere_radius  );  //return triangulation 
    static easy_pbr::MeshSharedPtr compute_triangulation_plane( const Eigen::MatrixXd& points );  //return triangulation
    static std::tuple<Eigen::Vector3i, Eigen::Vector3d> compute_closest_triangle(  const Eigen::Vector3d& point, const easy_pbr::MeshSharedPtr& triangulated_mesh3d); //returns the closest face and the barycentric coords of the point projected onto that closest triangle
    static Eigen::Vector3d compute_barycentric_weights_from_triangle_points( const Eigen::Vector3d& point,  const Eigen::Matrix3d& vertices_for_face );
    static Eigen::Vector3d compute_barycentric_weights_from_face_and_mesh_points( const Eigen::Vector3d& point,  const Eigen::Vector3i& face, const Eigen::Matrix3d& points_mesh ); //convenience func
    static Eigen::Vector3d compute_barycentric_coordinates_of_closest_point_inside_triangle(const Eigen::Matrix3d& vertices_for_face,  const Eigen::Vector3d& weights ); //The barycentric coordinates computed for a triangle by the other two function can return the barucentric coordinates corresponding to a point outside of the triangle. This function gives me the barycentric coordiantes of the point that would be closest to the triangle from the query point

   

private:
    SFM();

    std::pair< std::vector<cv::KeyPoint>,   cv::Mat > compute_keypoints_and_descriptor( const easy_pbr::Frame& frame );
   

    static Eigen::Vector3d stereographic_projection(const Eigen::Vector3d& point3d, const Eigen::Vector3d& sphere_center, const double sphere_radius);
    static easy_pbr::MeshSharedPtr create_normalized_sphere();


    static easy_pbr::MeshSharedPtr m_sphere_normalized;
   
};
