cmake_minimum_required (VERSION 3.9 FATAL_ERROR)

find_package(Vulkan REQUIRED FATAL_ERROR)
message(STATUS "Attempting to locate Vulkan SDK using manual path......")
set(VULKAN_SDK "C:/VulkanSDK")
set(VULKAN_VERSION "1.2.154.1")
set(VULKAN_PATH "${VULKAN_SDK}/${VULKAN_VERSION}")
message(STATUS "Using manual specified path: ${SRC_DIR}")
set(ENGINE_SRC ${CMAKE_SOURCE_DIR})

file (GLOB_RECURSE ENGINE_H ${CMAKE_SOURCE_DIR}/*.h)
file (GLOB_RECURSE ENGINE_CPP ${CMAKE_SOURCE_DIR}/*.cpp)

include_directories(${VULKAN_PATH}/Include)
include_directories(${ENGINE_H})
link_directories(${VULKAN_PATH}/Bin; ${VULKAN_PATH}/Lib;)

add_library(engine STATIC ${ENGINE_CPP})

# target_include_directories(engine_lib PUBLIC ${VULKAN_INCLUDE_DIRS})