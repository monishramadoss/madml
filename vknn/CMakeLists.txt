cmake_minimum_required (VERSION 3.9 FATAL_ERROR)
project(vknn VERSION "0.0.1")

find_package(pybind11 CONFIG REQUIRED)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(BACKEND_INCLUDE ${INCLUDE_DIR}/backend)
set(BACKEND_SRC ${SRC_DIR}/backend)
set(VKNN_INCLUDE ${INCLUDE_DIR}/vknn)
set(VKNN_SRC ${SRC_DIR}/vknn)

find_package(Vulkan REQUIRED FATAL_ERROR)
message(STATUS "Attempting to locate Vulkan SDK using manual path......")
set(VULKAN_SDK "C:/VulkanSDK")
set(VULKAN_VERSION "1.2.154.1")
set(VULKAN_PATH "${VULKAN_SDK}/${VULKAN_VERSION}")
message(STATUS "Using manual specified path: ${SRC_DIR}")

file (GLOB_RECURSE VKNN_H ${CMAKE_SOURCE_DIR}/*.h)
file (GLOB_RECURSE VKNN_CPP ${CMAKE_SOURCE_DIR}/*.cpp)

include_directories(${VULKAN_PATH}/Include)
include_directories(AFTER ${BACKEND_INCLUDE})
include_directories(AFTER ${VKNN_INCLUDE})

link_directories(${VULKAN_PATH}/Bin;${VULKAN_PATH}/Lib;)

pybind11_add_module(vknn MODULE ${VKNN_CPP} ${VKNN_H})

target_link_libraries(vknn PUBLIC vulkan-1)
target_link_libraries(vknn PUBLIC)

target_include_directories (vknn PUBLIC ${VULKAN_INCLUDE_DIRS})

install(TARGETS vknn COMPONENT python DESTINATION .)