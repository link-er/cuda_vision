cmake_minimum_required(VERSION 2.4.6)

project (matrices)

## CUDA
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)

set (SOURCES
        main.cpp
        addgpu.cu
        addcpu.cpp
)

set  (HEADERS
        addgpu.h
        addcpu.h
)

CUDA_ADD_LIBRARY (addgpulib
        addgpu.cu
        addgpu.h
)

CUDA_ADD_EXECUTABLE(matrices addgpu.cu)
ADD_EXECUTABLE  (matrices ${SOURCES} ${HEADERS})
TARGET_LINK_LIBRARIES (matrices)
