#include "neural_mvs/SFM.h"



//c++
#include <string>

//easypbr
#include "easy_pbr/Gui.h"
#include "easy_pbr/Scene.h"


//opencv 
#include "opencv2/highgui/highgui.hpp"
// #include "opencv2/features2d/features2d.hpp"
#include <opencv2/core/eigen.hpp>


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

#include <igl/triangle/triangulate.h>





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



//initialize static members
// Eigen::Vector3d sphere_center=Eigen::Vector3d::Zero();
easy_pbr::MeshSharedPtr SFM::m_sphere_normalized= SFM::create_normalized_sphere();
// easy_pbr::MeshSharedPtr SFM::m_sphere_normalized->create_sphere(sphere_center, 1.0);



SFM::SFM()
    {

    // m_sphere_normalized= easy_pbr::Mesh::create();
    // Eigen::Vector3d sphere_center;
    // sphere_center.setZero();
    // m_sphere_normalized->create_sphere(sphere_center, 1.0);

}

SFM::~SFM(){
    LOG(WARNING) << "Deleting SFM";
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



Eigen::Vector3d SFM::stereographic_projection(const Eigen::Vector3d& point3d, const Eigen::Vector3d& sphere_center, const double sphere_radius){

    //project sphere using stereographic projection https://en.wikipedia.org/wiki/Stereographic_projection
    //we use the bottom points of the sphere to start the projection from
    Eigen::Vector3d bottom_point=sphere_center;
    bottom_point.y()-=sphere_radius;
    //plane is the plane that runs through the center of the sphere and has normal pointing up (so int he psoitive y axis), Also a hyperplane in 3d is a 2d plane
    Eigen::Hyperplane<double,3> plane = Eigen::Hyperplane<double,3>(Eigen::Vector3d::UnitY(), sphere_center.y());
    Eigen::ParametrizedLine<double,3> line = Eigen::ParametrizedLine<double,3>::Through(bottom_point, point3d );
    Eigen::Vector3d point_intersection = line.intersectionPoint( plane ) ;

    return point_intersection;

}

// compute triangulation 
// https://www.redblobgames.com/x/1842-delaunay-voronoi-sphere/
easy_pbr::MeshSharedPtr SFM::compute_triangulation_stegreographic( const Eigen::MatrixXd& points,  const Eigen::Vector3d& sphere_center, double sphere_radius ){
    //we assume the frames are laid in a somewhat sphere.
    // we want to compute a delauany triangulation of them. For this we folow   https://www.redblobgames.com/x/1842-delaunay-voronoi-sphere/
    // which says we can use steregraphic projection and then do a delaunay in 2D and then lift it back to 3D which will yield a valid triangulation of the points on the sphere. 
    
    // //get all the points from the frames into a EigenMatrix
    // Eigen::MatrixXd points;
    // points.resize(frames.size(),3);
    // for (size_t i=0; i<frames.size(); i++){
    //     points.row(i) = frames[i].pos_in_world().cast<double>();
    // }

    // //fit sphere
    // auto sphere_params=fit_sphere(points);
    // Eigen::Vector3d sphere_center = std::get<0>(sphere_params);
    // double sphere_radius = std::get<1>(sphere_params);

    // make sphere and check that it looks ok
    // auto sphere=std::make_shared<easy_pbr::Mesh>(  m_sphere_normalized->clone() );
    // for (int i = 0; i < sphere->V.rows(); i++) {
    //     sphere->V.row(i) = Eigen::Vector3d( sphere->V.row(i))*sphere_radius+sphere_center;
    // }
    // sphere->m_vis.m_show_mesh=false;
    // sphere->m_vis.m_show_points=true;
    // easy_pbr::Scene::show(sphere,"sphere");

    //project sphere using stereographic projection https://en.wikipedia.org/wiki/Stereographic_projection
    //we use the bottom points of the sphere to start the projection from
    //plane is the plane that runs through the center of the sphere and has normal pointing up (so int he psoitive y axis), Also a hyperplane in 3d is a 2d plane
    Eigen::MatrixXd points_intesection(points.rows(),3);
    for (int i=0; i<points.rows(); i++){
        Eigen::Vector3d point=points.row(i);
        Eigen::Vector3d point_intersection = stereographic_projection(point, sphere_center, sphere_radius) ;
        points_intesection.row(i) = point_intersection;
    }
    //show the intersection points
    // auto intersect_mesh= easy_pbr::Mesh::create();
    // intersect_mesh->V=points_intesection;
    // intersect_mesh->m_vis.m_show_points=true;
    // easy_pbr::Scene::show(intersect_mesh,"intersect_mesh");


    //triangulate 
    Eigen::MatrixXd points_intesection_2d(points.rows(),2); //get the intersection points from the xz plane to just xy so we can run delaunay triangulation
    for (int i=0; i<points.rows(); i++){
        Eigen::Vector2d point;
        point.x()= points_intesection.row(i).x();
        point.y()= points_intesection.row(i).z();
        points_intesection_2d.row(i)=point;
    }
    auto triangulated_mesh = easy_pbr::Mesh::create();
    // std::string params="Q"; //DO NOT add "v" for verbose output, for some reason it breaks now and it segment faults somewhere inside. Maybe due to pybind I dunno
    //Flag Q is for quiet and c is for adding edges along the convex hull of the points otherwise we end up with no triangles in our triangulation
    std::string params="Qc"; //DO NOT add "v" for verbose output, for some reason it breaks now and it segment faults somewhere inside. Maybe due to pybind I dunno
    Eigen::MatrixXi E_empty;
    Eigen::MatrixXd H_empty;
    Eigen::MatrixXi F_out;
    Eigen::MatrixXd V_out;
    igl::triangle::triangulate(points_intesection_2d, E_empty ,H_empty ,params, triangulated_mesh->V, triangulated_mesh->F); 

    // triangulated_mesh->V=points;
    // triangulated_mesh->m_vis.m_show_mesh=false;
    // triangulated_mesh->m_vis.m_show_wireframe=true;
    // easy_pbr::Scene::show(triangulated_mesh, "triangulated_mesh"); 


    


    // VLOG(1) << "sphere center_final " << sphere_center; 
    // VLOG(1) << "sphere radius_final " << sphere_radius; 

    return triangulated_mesh;

    
}



easy_pbr::MeshSharedPtr SFM::compute_triangulation_plane( const Eigen::MatrixXd& points ){
    //we assume the frames are laid in plane similar to our xy axis so right and up.
   
   
    //triangulate 
    Eigen::MatrixXd points_intesection_2d(points.rows(),2); //get the intersection points from the xz plane to just xy so we can run delaunay triangulation
    for (int i=0; i<points.rows(); i++){
        Eigen::Vector2d point;
        point.x()= points.row(i).x();
        point.y()= points.row(i).y();
        points_intesection_2d.row(i)=point;
    }
    auto triangulated_mesh = easy_pbr::Mesh::create();
    // std::string params="Q"; //DO NOT add "v" for verbose output, for some reason it breaks now and it segment faults somewhere inside. Maybe due to pybind I dunno
    //Flag Q is for quiet and c is for adding edges along the convex hull of the points otherwise we end up with no triangles in our triangulation
    std::string params="Qc"; //DO NOT add "v" for verbose output, for some reason it breaks now and it segment faults somewhere inside. Maybe due to pybind I dunno
    Eigen::MatrixXi E_empty;
    Eigen::MatrixXd H_empty;
    Eigen::MatrixXi F_out;
    Eigen::MatrixXd V_out;
    igl::triangle::triangulate(points_intesection_2d, E_empty ,H_empty ,params, triangulated_mesh->V, triangulated_mesh->F); 

    // VLOG(1) <<triangulated_mesh->V.rows();
    // VLOG(1) <<points.rows();
    // VLOG(1) << "nr of faces generated" << triangulated_mesh->F.rows();
    triangulated_mesh->V=points;
    triangulated_mesh->m_vis.m_show_mesh=false;
    triangulated_mesh->m_vis.m_show_wireframe=true;
    easy_pbr::Scene::show(triangulated_mesh, "triangulated_mesh"); 


    


    // VLOG(1) << "sphere center_final " << sphere_center; 
    // VLOG(1) << "sphere radius_final " << sphere_radius; 

    return triangulated_mesh;

    
}

//compute_closest_triangle of a triangulated surface using stegraphic projection
std::tuple<Eigen::Vector3i, Eigen::Vector3d> SFM::compute_closest_triangle(  const Eigen::Vector3d& point, const easy_pbr::MeshSharedPtr& triangulated_mesh3d){



    //brute force through all the faces and calcualte the projected point on all the faces, finally we get the closest face
    double lowest_dist=std::numeric_limits<double>::max();
    Eigen::Vector3i closest_face;
    closest_face.fill(-1);
    Eigen::Vector3d closest_projected_point;
    Eigen::MatrixXd vertices_of_closest_face;
    Eigen::Vector3d selected_closest_point_weights;


    for (int i=0; i<triangulated_mesh3d->F.rows(); i++){
        // VLOG(1) << "Face i" << i;
        Eigen::Vector3i face= triangulated_mesh3d->F.row(i);
        // VLOG(1) << "face is " << face.transpose();
        Eigen::Matrix3d vertices_for_face(3,3);
        for (size_t j=0; j<3; j++){
            vertices_for_face.row(j) = triangulated_mesh3d->V.row( face(j) );
        }
        // VLOG(1) << "vertices_for_face" << vertices_for_face;
        // VLOG(1) << "point" << point.transpose();
        Eigen::Vector3d weights= compute_barycentric_weights_from_triangle_points(point, vertices_for_face);
        //since want to restrict the point to be exactly inside the triangle and not just on the plane that defines it we clamp the weights
        Eigen::Vector3d closest_point_weights= compute_barycentric_coordinates_of_closest_point_inside_triangle(vertices_for_face, weights);


        // VLOG(1) << "weights " << weights.transpose();
        Eigen::Vector3d projected_point =  vertices_for_face.row(0)*closest_point_weights(0) +vertices_for_face.row(1)*closest_point_weights(1) + vertices_for_face.row(2)*closest_point_weights(2);
        // VLOG(1) << "projected_point " << projected_point.transpose();
        double dist= (projected_point-point).norm();
        // VLOG(1) << "dist is "<< dist;
        // VLOG(1) << "lowest_dist is "<< lowest_dist;
        if (dist<lowest_dist){
            lowest_dist=dist;
            closest_face=face;
            closest_projected_point=projected_point;
            vertices_of_closest_face= vertices_for_face;
            selected_closest_point_weights=closest_point_weights;
            // VLOG(1) << "----------setting closest face to " <<closest_face.transpose();
        }
    }

    //check
    CHECK( closest_face.minCoeff()!=-1 && closest_face.maxCoeff()<triangulated_mesh3d->V.rows() ) << "Close face was not found and it's " << closest_face << " lowest dist is " << lowest_dist;

    // VLOG(1) << "lowest dist is " << lowest_dist;

    // show the projected point 
    auto projected_mesh= easy_pbr::Mesh::create();
    projected_mesh->V.resize(1,3);
    projected_mesh->V.row(0)=closest_projected_point;
    projected_mesh->m_vis.m_show_points=true;
    projected_mesh->m_vis.m_point_size=10;
    easy_pbr::Scene::show(projected_mesh,"projected_mesh");



    return std::make_tuple(closest_face, selected_closest_point_weights);

   
}

Eigen::Vector3d SFM::compute_barycentric_weights_from_triangle_points( const Eigen::Vector3d& point,  const Eigen::Matrix3d& vertices_for_face ){
    // https://people.cs.clemson.edu/~dhouse/courses/404/notes/barycentric.pdf
    // https://stackoverflow.com/questions/2924795/fastest-way-to-compute-point-to-triangle-distance-in-3d
    // https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle

    //to compute the weights we use barycentric coordinates. 
    //this has several steps, first project the current frame into the triangle defiend by the close_frames. 
    //compute barycentric coords
    //if the barycentric coords are not within [0,1], clamp them

    //checks
    // CHECK(close_frames.size()==3) <<"This assumes we are using 3 frames as close frames because we want to compute barycentric coords";

    //make triangle
    Eigen::Vector3d cur_pos= point;
    Eigen::Vector3d p1= vertices_for_face.row(0);
    Eigen::Vector3d p2= vertices_for_face.row(1);
    Eigen::Vector3d p3= vertices_for_face.row(2);

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
    // std::vector<float> vals;
    // vals.push_back(w_p1);
    // vals.push_back(w_p2);
    // vals.push_back(w_p3);

    Eigen::Vector3d weights;
    weights.x()=w_p1;
    weights.y()=w_p2;
    weights.z()=w_p3;

    return weights;


}

//convenicen function
Eigen::Vector3d SFM::compute_barycentric_weights_from_face_and_mesh_points( const Eigen::Vector3d& point,  const Eigen::Vector3i& face, const Eigen::Matrix3d& points_mesh ){

    Eigen::Matrix3d vertices_for_face(3,3);
    for (size_t j=0; j<3; j++){
        vertices_for_face.row(j) = points_mesh.row( face(j) );
    }

    Eigen::Vector3d weights= compute_barycentric_weights_from_triangle_points(point, vertices_for_face);

    return weights;

}

Eigen::Vector3d SFM::compute_barycentric_coordinates_of_closest_point_inside_triangle( const Eigen::Matrix3d& vertices_for_face, const Eigen::Vector3d& weights ){
    //following https://stackoverflow.com/a/37923949
    double u=weights.x();
    double v=weights.y();
    double w=weights.z();
    Eigen::Vector3d p0=vertices_for_face.row(0);
    Eigen::Vector3d p1=vertices_for_face.row(1);
    Eigen::Vector3d p2=vertices_for_face.row(2);
    Eigen::Vector3d p =  vertices_for_face.row(0)*weights(0) +vertices_for_face.row(1)*weights(1) + vertices_for_face.row(2)*weights(2);

    if ( u < 0){
        double t = (p-p1).dot(p2-p1) / (p2-p1).dot(p2-p1);
        t=radu::utils::clamp(t, 0.0, 1.0);
        return Eigen::Vector3d( 0.0f, 1.0f-t, t );
    }
    else if ( v < 0 ){
        double t =(p-p2).dot(p0-p2) / (p0-p2).dot(p0-p2);
        t=radu::utils::clamp(t, 0.0, 1.0);
        return Eigen::Vector3d( t, 0.0f, 1.0f-t );
    }
    else if ( w < 0 ){
        double t = (p-p0).dot(p1-p0) / (p1-p0).dot(p1-p0);
        t=radu::utils::clamp(t, 0.0, 1.0);
        return Eigen::Vector3d( 1.0f-t, t, 0.0f );
    }
    else{
        return Eigen::Vector3d( u, v, w );
    }

}


easy_pbr::MeshSharedPtr SFM::create_normalized_sphere(){

    auto sphere= easy_pbr::Mesh::create();
    Eigen::Vector3d sphere_center=Eigen::Vector3d::Zero();
    sphere->create_sphere(sphere_center, 1.0);

    return sphere;

}




