project(log_classifier)
cmake_minimum_required(VERSION 2.8)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(log_classifier main.cpp)
target_link_libraries(log_classifier ${Caffe_LIBRARIES})




