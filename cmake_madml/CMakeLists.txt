# CMakeList.txt : CMake project for cmake_madml, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.9 FATAL_ERROR)

find_package(Vulkan REQUIRED FATAL_ERROR)
message(STATUS "Attempting to locate Vulkan SDK using manual path......")
set(VULKAN_SDK "C:/VulkanSDK")
set(VULKAN_VERSION "1.2.154.1")
set(VULKAN_PATH "${VULKAN_SDK}/${VULKAN_VERSION}")
message(STATUS "Using manual specified path: ${VULKAN_PATH}")
include_directories(AFTER ${VULKAN_PATH}/Include)
link_directories(${VULKAN_PATH}/Bin;${VULKAN_PATH}/Lib;)
# Add source to this project's executable.
find_package(pybind11 REQUIRED)
pybind11_add_module(cmake_madml cmake_madml.cpp cmake_madml.h)

# TODO: Add tests and install targets if needed.

target_include_directories (cmake_madml PUBLIC ${VULKAN_INCLUDE_DIRS})

target_link_libraries (cmake_madml ${VULKAN_LIB_LIST})