project(nn)
cmake_minimum_required(VERSION 2.8)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(nn main.cpp)
target_link_libraries(nn ${Caffe_LIBRARIES})




