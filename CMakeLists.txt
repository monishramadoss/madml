# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#

cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

project(CMake)
find_package(pybind11 REQUIRED)  
find_package(Vulkan REQUIRED FATAL_ERROR)

set(${VULKAN_LIB_LIST} "vulkan-1")

# Add source to this project's executable.
include_directories(AFTER ${VULKAN_PATH}/Include)
link_directories(${VULKAN_PATH}/Bin;${VULKAN_PATH}/Lib;)

project ("ml_on_vulkan")

# Include sub-projects.
add_subdirectory ("ext")
add_subdirectory ("CMake")
