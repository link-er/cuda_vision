cmake_minimum_required(VERSION 2.4.6)

project (test_cpu)

set (SOURCES
	main.cpp
  axpycpu.cpp
)

set  (HEADERS
  axpycpu.h
)

ADD_EXECUTABLE  (test_cpu ${SOURCES} ${HEADERS})
TARGET_LINK_LIBRARIES (test_cpu)
