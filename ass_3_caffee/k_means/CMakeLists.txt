cmake_minimum_required(VERSION 2.8.8)

project (k_means)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(k_means k_means.cpp)
target_link_libraries(k_means ${Caffe_LIBRARIES})

