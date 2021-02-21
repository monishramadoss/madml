cmake_minimum_required (VERSION 3.9 FATAL_ERROR)
project(backend VERSION "0.0.1")

find_package(pybind11 CONFIG REQUIRED)

file (GLOB_RECURSE BACKEND_H ${CMAKE_SOURCE_DIR}/*.h)
file (GLOB_RECURSE BACKEND_CPP ${CMAKE_SOURCE_DIR}/*.cpp)

include_directories(${VULKAN_PATH}/Include)
include_directories(AFTER ${BACKEND_INCLUDE})


link_directories(${VULKAN_PATH}/Bin; ${VULKAN_PATH}/Lib;)

pybind11_add_module(backend MODULE ${BACKEND_CPP} ${BACKEND_H})
	
target_link_libraries(backend PUBLIC vulkan-1)
target_link_libraries(backend PUBLIC)

target_include_directories (backend PUBLIC ${VULKAN_INCLUDE_DIRS})

install(TARGETS backend COMPONENT python DESTINATION .)