project(mnist)
cmake_minimum_required(VERSION 2.8)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(mnist_layers main.cpp mnist.cpp)
target_link_libraries(mnist_layers ${Caffe_LIBRARIES})




