# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/stud/adilova/cuda_vision/ass_7_protobuf/mnist_protobuf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/stud/adilova/cuda_vision/ass_7_protobuf/build-mnist_protobuf-Desktop-Default

# Include any dependencies generated for this target.
include CMakeFiles/mnist_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mnist_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mnist_test.dir/flags.make

CMakeFiles/mnist_test.dir/mnist_test.cpp.o: CMakeFiles/mnist_test.dir/flags.make
CMakeFiles/mnist_test.dir/mnist_test.cpp.o: /home/stud/adilova/cuda_vision/ass_7_protobuf/mnist_protobuf/mnist_test.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/stud/adilova/cuda_vision/ass_7_protobuf/build-mnist_protobuf-Desktop-Default/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/mnist_test.dir/mnist_test.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/mnist_test.dir/mnist_test.cpp.o -c /home/stud/adilova/cuda_vision/ass_7_protobuf/mnist_protobuf/mnist_test.cpp

CMakeFiles/mnist_test.dir/mnist_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mnist_test.dir/mnist_test.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/stud/adilova/cuda_vision/ass_7_protobuf/mnist_protobuf/mnist_test.cpp > CMakeFiles/mnist_test.dir/mnist_test.cpp.i

CMakeFiles/mnist_test.dir/mnist_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mnist_test.dir/mnist_test.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/stud/adilova/cuda_vision/ass_7_protobuf/mnist_protobuf/mnist_test.cpp -o CMakeFiles/mnist_test.dir/mnist_test.cpp.s

CMakeFiles/mnist_test.dir/mnist_test.cpp.o.requires:
.PHONY : CMakeFiles/mnist_test.dir/mnist_test.cpp.o.requires

CMakeFiles/mnist_test.dir/mnist_test.cpp.o.provides: CMakeFiles/mnist_test.dir/mnist_test.cpp.o.requires
	$(MAKE) -f CMakeFiles/mnist_test.dir/build.make CMakeFiles/mnist_test.dir/mnist_test.cpp.o.provides.build
.PHONY : CMakeFiles/mnist_test.dir/mnist_test.cpp.o.provides

CMakeFiles/mnist_test.dir/mnist_test.cpp.o.provides.build: CMakeFiles/mnist_test.dir/mnist_test.cpp.o

# Object files for target mnist_test
mnist_test_OBJECTS = \
"CMakeFiles/mnist_test.dir/mnist_test.cpp.o"

# External object files for target mnist_test
mnist_test_EXTERNAL_OBJECTS =

mnist_test: CMakeFiles/mnist_test.dir/mnist_test.cpp.o
mnist_test: CMakeFiles/mnist_test.dir/build.make
mnist_test: /home/stud/adilova/caffe/caffe-rc2/build/lib/libcaffe.so
mnist_test: /home/stud/adilova/caffe/caffe-rc2/build/lib/libproto.a
mnist_test: /usr/lib/x86_64-linux-gnu/libboost_system.so
mnist_test: /usr/lib/x86_64-linux-gnu/libboost_thread.so
mnist_test: /usr/lib/x86_64-linux-gnu/libpthread.so
mnist_test: /usr/lib/x86_64-linux-gnu/libglog.so
mnist_test: /usr/lib/x86_64-linux-gnu/libgflags.so
mnist_test: /usr/lib/x86_64-linux-gnu/libprotobuf.so
mnist_test: /usr/lib/x86_64-linux-gnu/libglog.so
mnist_test: /usr/lib/x86_64-linux-gnu/libgflags.so
mnist_test: /usr/lib/x86_64-linux-gnu/libprotobuf.so
mnist_test: /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
mnist_test: /usr/lib/x86_64-linux-gnu/libhdf5.so
mnist_test: /usr/lib/x86_64-linux-gnu/liblmdb.so
mnist_test: /usr/lib/x86_64-linux-gnu/libleveldb.so
mnist_test: /usr/lib/libsnappy.so
mnist_test: /usr/local/cuda/lib64/libcudart.so
mnist_test: /usr/local/cuda/lib64/libcurand.so
mnist_test: /usr/local/cuda/lib64/libcublas.so
mnist_test: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
mnist_test: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
mnist_test: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
mnist_test: /usr/lib/liblapack_atlas.so
mnist_test: /usr/lib/libcblas.so
mnist_test: /usr/lib/libatlas.so
mnist_test: CMakeFiles/mnist_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable mnist_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mnist_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mnist_test.dir/build: mnist_test
.PHONY : CMakeFiles/mnist_test.dir/build

CMakeFiles/mnist_test.dir/requires: CMakeFiles/mnist_test.dir/mnist_test.cpp.o.requires
.PHONY : CMakeFiles/mnist_test.dir/requires

CMakeFiles/mnist_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mnist_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mnist_test.dir/clean

CMakeFiles/mnist_test.dir/depend:
	cd /home/stud/adilova/cuda_vision/ass_7_protobuf/build-mnist_protobuf-Desktop-Default && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/stud/adilova/cuda_vision/ass_7_protobuf/mnist_protobuf /home/stud/adilova/cuda_vision/ass_7_protobuf/mnist_protobuf /home/stud/adilova/cuda_vision/ass_7_protobuf/build-mnist_protobuf-Desktop-Default /home/stud/adilova/cuda_vision/ass_7_protobuf/build-mnist_protobuf-Desktop-Default /home/stud/adilova/cuda_vision/ass_7_protobuf/build-mnist_protobuf-Desktop-Default/CMakeFiles/mnist_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mnist_test.dir/depend
