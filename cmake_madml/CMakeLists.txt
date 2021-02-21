# CMakeList.txt : CMake project for cmake_madml, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.9 FATAL_ERROR)

project(cmake_madml VERSION "0.0.1")

find_package(pybind11 CONFIG REQUIRED)

include_directories(${VULKAN_PATH}/Include)
include_directories(AFTER ${VKNN_INCLUDE})
link_directories(${VULKAN_PATH}/Bin;${VULKAN_PATH}/Lib;)

pybind11_add_module(cmake_madml cmake_madml.cpp cmake_madml.h)

target_link_libraries(cmake_madml PUBLIC vulkan-1)
target_link_libraries(cmake_madml PUBLIC)
target_include_directories (cmake_madml PUBLIC ${VULKAN_INCLUDE_DIRS})

install(TARGETS cmake_madml COMPONENT python DESTINATION .)