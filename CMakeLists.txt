cmake_minimum_required(VERSION 2.8 FATAL_ERROR) # we need at least 3.11 because we use the FetchContent feature to manage dependencies and at least 3.14 because we use FetchContent_MakeAvailable

project(neural_mvs)


###   VARIABLES   ##############################################################
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -O3") 
set(PYTORCH_PATH "/opt/pytorch")
# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/opt/ros/melodic/\;${PYTORCH_PATH}"
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${PYTORCH_PATH}")
set(CMAKE_CXX_STANDARD 17) #we need c++17 because this solves alignment issues with eigen http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1409
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_VERBOSE_MAKEFILE ON)


# add_subdirectory( ${CMAKE_SOURCE_DIR}/deps/pybind11)

#pybind attempt 3
# pybind11_add_module(DataLoaderTest ${CMAKE_SOURCE_DIR}/src/pycom/PyCom.cxx )


####   GLOBAL OPTIONS   ###https://stackoverflow.com/questions/15201064/cmake-conditional-preprocessor-define-on-code
# libigl 
# libigl options: choose between header only and compiled static library
# Header-only is preferred for small projects. For larger projects the static build
# considerably reduces the compilation times
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR       "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL              "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO            "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK              "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"         OFF)
option(LIBIGL_WITH_LIM               "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB            "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK             "Use MOSEK"          OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         OFF)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           OFF)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          OFF)
option(LIBIGL_WITH_PNG               "Use PNG"            OFF)
option(LIBIGL_WITH_PYTHON            "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       ON)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  OFF)
option(LIBIGL_WITH_XML               "Use XML"            OFF)






######   PACKAGES   ############################################################
# find_package(catkin REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs features2d )
find_package(EasyPBR REQUIRED)
find_package(DataLoaders REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_package(Ceres REQUIRED)
#libigl
set(LIBIGL_INCLUDE_DIR "${EASYPBR_SRC_PATH}/deps/libigl/include")
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBIGL
    "\nlibigl not found --- You can add it as a submodule it using:\n\tgit add submodule https://github.com/libigl/libigl.git deps/libigl"
    LIBIGL_INCLUDE_DIR)
mark_as_advanced(LIBIGL_INCLUDE_DIR)
list(APPEND CMAKE_MODULE_PATH "${LIBIGL_INCLUDE_DIR}/../cmake")
include(libigl)
# find_package(LIBIGL REQUIRED)

# add_subdirectory(${PROJECT_SOURCE_DIR}/deps/pybind11)
# find_package(catkin REQUIRED COMPONENTS roscpp std_msgs cv_bridge pcl_ros image_transport )
#opencv contrib things
# set(HAVE_EIGEN TRUE)
# add_subdirectory( ${PROJECT_SOURCE_DIR}/deps/opencv_contrib/modules/sfm )

add_subdirectory( ${EASYPBR_SRC_PATH}/deps/pybind11 ./../../build/pybind11 EXCLUDE_FROM_ALL)
pybind11_add_module(neuralmvs ${PROJECT_SOURCE_DIR}/src/PyBridge.cxx )

###   INCLUDES   #########################################################
# include_directories(${catkin_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/deps)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${EASYPBR_INCLUDE_DIR})
include_directories(${DATALOADERS_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${LIBIGL_INCLUDE_DIR})

# include_directories(${LIBIGL_INCLUDE_DIR})
# catkin_package( INCLUDE_DIRS include  CATKIN_DEPENDS roscpp std_msgs cv_bridge pcl_ros image_transport )


###   SOURCES   #################################################################
set(MY_SRC  
    # ${PROJECT_SOURCE_DIR}/src/Frame.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderVolRef.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderImgRos.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderScanNet.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderShapeNetPartSeg.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataTransformer.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderSemanticKitti.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderStanfordIndoor.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderToyExample.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderCloudRos.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderRueMonge.cxx
    # ${PROJECT_SOURCE_DIR}/src/RosBagPlayer.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderPNG.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderModelNet40.cxx
)
set(DEPS_SRC 
    # ${PROJECT_SOURCE_DIR}/deps/json11/json11.cpp
    )


###   SET ALL THE GLOBAL OPTIONS #########################################


#lib 
# add_library( latticenet_cpp SHARED ${MY_SRC}  ${DEPS_SRC}  )

##    CUDA library  ######################
#configure the include paths of jitify so that the kernels can properly include other ones
configure_file(${PROJECT_SOURCE_DIR}/include/neural_mvs/jitify_helper/jitify_options.hpp.in ${PROJECT_SOURCE_DIR}/include/neural_mvs/jitify_helper/jitify_options.hpp)
install(FILES "${PROJECT_SOURCE_DIR}/include/neural_mvs/jitify_helper/jitify_options.hpp" DESTINATION ${PROJECT_SOURCE_DIR}/include/neural_mvs/jitify_helper/)
# set(CUDA_NVCC_FLAGS "-gencode arch=compute_30 -code=sm_30" CACHE STRING "nvcc flags" FORCE)
# set(CUDA_NVCC_FLAGS "-arch=sm_60 " CACHE STRING "nvcc flags" FORCE)
# message("NVCC flags are ", ${CUDA_NVCC_FLAGS})
# set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE) 
# # set(LIB_TYPE SHARED)
# cuda_add_library(neuralmesh_cu  SHARED  
#                                         ${CMAKE_SOURCE_DIR}/src/NeuralMesh.cu 
#                                         ${CMAKE_SOURCE_DIR}/src/ModelParams.cxx
#                                         ${CMAKE_SOURCE_DIR}/src/TrainParams.cxx
#                                         ${CMAKE_SOURCE_DIR}/src/EvalParams.cxx
#                                         )


add_library( neuralmvs_cu SHARED
                                        ${CMAKE_SOURCE_DIR}/src/NeuralMVS.cxx
                                        ${CMAKE_SOURCE_DIR}/src/SFM.cxx
                                        # ${CMAKE_SOURCE_DIR}/include/neural_mvs/kernels/NeuralMVSGPU.cuh
                                        ${CMAKE_SOURCE_DIR}/src/ModelParams.cxx
                                        ${CMAKE_SOURCE_DIR}/src/TrainParams.cxx
                                        ${CMAKE_SOURCE_DIR}/src/EvalParams.cxx
                                        ${CMAKE_SOURCE_DIR}/src/TinyLoader.cxx
                                        )




########### NEW modules  
# #png required by libmv 
# # find_package(PNG REQUIRED)
# include_directories(${PNG_INCLUDE_DIR})
# #libmv which is needed by the sfm module in opencv_contrib
# # set(WITH_SYSTEM_CERES TRUE)
# set(BUILD_SHARED_LIBS TRUE)
# add_subdirectory(${PROJECT_SOURCE_DIR}/deps/libmv/src)
# include_directories(${PROJECT_SOURCE_DIR}/deps/libmv/src )
# set(LIBMV_LIBS autotrack base camera correspondence descriptor detector image multiview numeric reconstruction simple_pipeline tools tracking)
# #Make the SFM module from the opencv_contrib that we added as a submodule in deps
# include_directories(${PROJECT_SOURCE_DIR}/deps/opencv_contrib/modules/sfm/include/ )
# # message(" including ------------------------------------------------------- " ${PROJECT_SOURCE_DIR}/deps/opencv_contrib/modules/sfm/include/ )
# FILE(GLOB OPENCV_SFM_SRC ${PROJECT_SOURCE_DIR}/deps/opencv_contrib/modules/sfm/src/*.cpp)
# # FILE(GLOB OPENCV_SFM_HDRS 
#                             # ${PROJECT_SOURCE_DIR}/deps/opencv_contrib/modules/sfm/include/opencv2/sfm.hpp 
#                             # ${PROJECT_SOURCE_DIR}/deps/opencv_contrib/modules/sfm/include/opencv2/sfm/*.hpp)
# add_library ( opencv_sfm SHARED 
#                                 ${OPENCV_SFM_SRC}
#             )
# target_link_libraries(opencv_sfm ${LIBMV_LIBS} )
################### 





##pybind
# pybind11_add_module(dataloaders ${PROJECT_SOURCE_DIR}/src/PyBridge.cxx )

###   EXECUTABLE   #######################################
# add_executable(run_data_loaders ${PROJECT_SOURCE_DIR}/src/main.cxx  )


###   LIBS   ###############################################
 #torch 1.5.0 and above mess with pybind and we therefore need to link against libtorch_python.so also
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
if(TORCH_PYTHON_LIBRARY)
    message(STATUS "Linking to torch_python_library")
    set(LIBS ${LIBS} ${TORCH_PYTHON_LIBRARY} )
endif()
# set(LIBS ${LIBS} ${catkin_LIBRARIES} ${EASYPBR_LIBRARY} ${DATALOADERS_LIBRARY} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS} opencv_sfm )
set(LIBS ${LIBS} ${catkin_LIBRARIES} ${EASYPBR_LIBRARY} ${DATALOADERS_LIBRARY} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} igl::triangle )






target_link_libraries(neuralmvs_cu ${LIBS} )
target_link_libraries(neuralmvs PRIVATE neuralmvs_cu )
# target_link_libraries(run_data_loaders PRIVATE data_loaders_cpp  )


#definitions for cmake variables that are necesarry during runtime
target_compile_definitions(neuralmvs_cu PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}") #point to the cmakelist folder of the easy_pbr
target_compile_definitions(neuralmvs_cu PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}") # points to the CMakeList folder of whichever project included easy_pbr


