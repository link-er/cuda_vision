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
CMAKE_SOURCE_DIR = /home/stud/adilova/cuda_vision/ass_3_caffee/caffetest

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/stud/adilova/cuda_vision/ass_3_caffee/build-caffetest-Desktop-Default

# Include any dependencies generated for this target.
include CMakeFiles/test_math.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_math.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_math.dir/flags.make

CMakeFiles/test_math.dir/test_math.cpp.o: CMakeFiles/test_math.dir/flags.make
CMakeFiles/test_math.dir/test_math.cpp.o: /home/stud/adilova/cuda_vision/ass_3_caffee/caffetest/test_math.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/stud/adilova/cuda_vision/ass_3_caffee/build-caffetest-Desktop-Default/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/test_math.dir/test_math.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/test_math.dir/test_math.cpp.o -c /home/stud/adilova/cuda_vision/ass_3_caffee/caffetest/test_math.cpp

CMakeFiles/test_math.dir/test_math.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_math.dir/test_math.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/stud/adilova/cuda_vision/ass_3_caffee/caffetest/test_math.cpp > CMakeFiles/test_math.dir/test_math.cpp.i

CMakeFiles/test_math.dir/test_math.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_math.dir/test_math.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/stud/adilova/cuda_vision/ass_3_caffee/caffetest/test_math.cpp -o CMakeFiles/test_math.dir/test_math.cpp.s

CMakeFiles/test_math.dir/test_math.cpp.o.requires:
.PHONY : CMakeFiles/test_math.dir/test_math.cpp.o.requires

CMakeFiles/test_math.dir/test_math.cpp.o.provides: CMakeFiles/test_math.dir/test_math.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_math.dir/build.make CMakeFiles/test_math.dir/test_math.cpp.o.provides.build
.PHONY : CMakeFiles/test_math.dir/test_math.cpp.o.provides

CMakeFiles/test_math.dir/test_math.cpp.o.provides.build: CMakeFiles/test_math.dir/test_math.cpp.o

# Object files for target test_math
test_math_OBJECTS = \
"CMakeFiles/test_math.dir/test_math.cpp.o"

# External object files for target test_math
test_math_EXTERNAL_OBJECTS =

test_math: CMakeFiles/test_math.dir/test_math.cpp.o
test_math: CMakeFiles/test_math.dir/build.make
test_math: /home/stud/adilova/caffe/caffe-rc2/build/lib/libcaffe.so
test_math: /home/stud/adilova/caffe/caffe-rc2/build/lib/libproto.a
test_math: /usr/lib/x86_64-linux-gnu/libboost_system.so
test_math: /usr/lib/x86_64-linux-gnu/libboost_thread.so
test_math: /usr/lib/x86_64-linux-gnu/libpthread.so
test_math: /usr/lib/x86_64-linux-gnu/libglog.so
test_math: /usr/lib/x86_64-linux-gnu/libgflags.so
test_math: /usr/lib/x86_64-linux-gnu/libprotobuf.so
test_math: /usr/lib/x86_64-linux-gnu/libglog.so
test_math: /usr/lib/x86_64-linux-gnu/libgflags.so
test_math: /usr/lib/x86_64-linux-gnu/libprotobuf.so
test_math: /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
test_math: /usr/lib/x86_64-linux-gnu/libhdf5.so
test_math: /usr/lib/x86_64-linux-gnu/liblmdb.so
test_math: /usr/lib/x86_64-linux-gnu/libleveldb.so
test_math: /usr/lib/libsnappy.so
test_math: /usr/local/cuda/lib64/libcudart.so
test_math: /usr/local/cuda/lib64/libcurand.so
test_math: /usr/local/cuda/lib64/libcublas.so
test_math: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
test_math: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
test_math: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
test_math: /usr/lib/liblapack_atlas.so
test_math: /usr/lib/libcblas.so
test_math: /usr/lib/libatlas.so
test_math: CMakeFiles/test_math.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable test_math"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_math.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_math.dir/build: test_math
.PHONY : CMakeFiles/test_math.dir/build

CMakeFiles/test_math.dir/requires: CMakeFiles/test_math.dir/test_math.cpp.o.requires
.PHONY : CMakeFiles/test_math.dir/requires

CMakeFiles/test_math.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_math.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_math.dir/clean

CMakeFiles/test_math.dir/depend:
	cd /home/stud/adilova/cuda_vision/ass_3_caffee/build-caffetest-Desktop-Default && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/stud/adilova/cuda_vision/ass_3_caffee/caffetest /home/stud/adilova/cuda_vision/ass_3_caffee/caffetest /home/stud/adilova/cuda_vision/ass_3_caffee/build-caffetest-Desktop-Default /home/stud/adilova/cuda_vision/ass_3_caffee/build-caffetest-Desktop-Default /home/stud/adilova/cuda_vision/ass_3_caffee/build-caffetest-Desktop-Default/CMakeFiles/test_math.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_math.dir/depend

