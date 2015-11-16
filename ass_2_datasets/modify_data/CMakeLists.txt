cmake_minimum_required(VERSION 2.4.6)

project (modify_data)

## CUDA
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)

set (SOURCES
        main.cpp
        multiplier.cu
)

set  (HEADERS
        multiplier.h
)

CUDA_ADD_LIBRARY (multiplier
        multiplier.cu
        multiplier.h
)

CUDA_ADD_EXECUTABLE(modify_data multiplier.cu)
ADD_EXECUTABLE  (modify_data ${SOURCES} ${HEADERS})
TARGET_LINK_LIBRARIES (modify_data)
