cmake_minimum_required (VERSION 3.9 FATAL_ERROR)
project(vknn VERSION "0.0.1")

find_package(pybind11 CONFIG REQUIRED)

file (GLOB_RECURSE VKNN_H ${CMAKE_SOURCE_DIR}/*.h)
file (GLOB_RECURSE VKNN_CPP ${CMAKE_SOURCE_DIR}/*.cpp)

include_directories(${VULKAN_PATH}/Include)
include_directories(${BACKEND_INCLUDE})
include_directories(AFTER ${VKNN_INCLUDE})

link_directories(${VULKAN_PATH}/Bin;${VULKAN_PATH}/Lib;)

pybind11_add_module(vknn MODULE ${VKNN_CPP} ${VKNN_H})

target_link_libraries(vknn PUBLIC vulkan-1)
target_link_libraries(vknn PUBLIC)

target_include_directories (vknn PUBLIC ${VULKAN_INCLUDE_DIRS})

install(TARGETS vknn COMPONENT python DESTINATION .)