cmake_minimum_required (VERSION 3.9 FATAL_ERROR)
project(backend VERSION "0.0.1")

find_package(pybind11 CONFIG REQUIRED)

find_package(Vulkan REQUIRED FATAL_ERROR)
message(STATUS "Attempting to locate Vulkan SDK using manual path......")
set(VULKAN_SDK "C:/VulkanSDK")
set(VULKAN_VERSION "1.2.154.1")
set(VULKAN_PATH "${VULKAN_SDK}/${VULKAN_VERSION}")
message(STATUS "Using manual specified path: ${SRC_DIR}")
set(BACKEND_SRC ${CMAKE_SOURCE_DIR})

file (GLOB_RECURSE BACKEND_H ${CMAKE_SOURCE_DIR}/*.h)
file (GLOB_RECURSE BACKEND_CPP ${CMAKE_SOURCE_DIR}/*.cpp)

include_directories(${VULKAN_PATH}/Include)
include_directories(AFTER ${CMAKE_SOURCE_DIR})


link_directories(${VULKAN_PATH}/Bin; ${VULKAN_PATH}/Lib;)

pybind11_add_module(backend MODULE ${BACKEND_CPP} ${BACKEND_H})


target_link_libraries(backend PUBLIC vulkan-1)
target_link_libraries(backend PUBLIC)
target_link_libraries(backend PUBLIC kernel)

target_include_directories(backend PUBLIC ${VULKAN_INCLUDE_DIRS})
install(TARGETS backend COMPONENT python DESTINATION .)