cmake_minimum_required(VERSION 2.8.8)

project (logreg_binary)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

set(headers
)

set(sources
    main.cpp
)

add_executable(logreg_binary ${headers} ${sources})
target_link_libraries(logreg_binary ${Caffe_LIBRARIES})
