# Created by the script cgal_create_cmake_script
# This is the CMake script for compiling a CGAL application.

cmake_minimum_required(VERSION 3.1...3.23)
project(Surface_mesh_skeletonization_Examples)

find_package(CGAL REQUIRED)

find_package(Eigen3 3.2.0 QUIET) #(requires 3.2.0 or greater)
include(CGAL_Eigen3_support)
if(TARGET CGAL::Eigen3_support)

  create_single_source_cgal_program("simple_mcfskel_sm_example.cpp")

  foreach(
    target
    simple_mcfskel_sm_example)
    target_link_libraries(${target} PUBLIC CGAL::Eigen3_support)
  endforeach()

else()
  message("NOTICE: These programs require the Eigen library (3.2 or greater), and will not be compiled.")
endif()
