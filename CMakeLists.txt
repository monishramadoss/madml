# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("ml_on_vulkan")
include_directories(ext)
find_package(Vulkan)
IF (NOT Vulkan_FOUND)
	find_library(Vulkan_LIBRARY NAMES vulkan-1 vulkan PATHS ${CMAKE_SOURCE_DIR}/libs/vulkan)
ENDIF()


# Include sub-projects.
add_subdirectory("ext")
add_subdirectory ("cmake")
