project(mnist_le_net)
cmake_minimum_required(VERSION 2.8)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(mnist_le_net main.cpp)
target_link_libraries(mnist_le_net ${Caffe_LIBRARIES})




