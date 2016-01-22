cmake_minimum_required(VERSION 2.8.8)

project (mnist_protobuf)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(mnist_train mnist_train.cpp)
target_link_libraries(mnist_train ${Caffe_LIBRARIES})

add_executable(mnist_test mnist_test.cpp)
target_link_libraries(mnist_test ${Caffe_LIBRARIES})
