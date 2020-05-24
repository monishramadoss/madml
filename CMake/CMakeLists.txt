# CMakeList.txt : CMake project for CMake, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

project(CMake)
find_package(pybind11 REQUIRED)  
find_package(Vulkan REQUIRED FATAL_ERROR)
if (NOT ${VULKAN_INCLUDE_DIRS} STREQUAL "")
	set(VULKAN_PATH ${Vulkan_INCLUDE_DIRS})
	STRING(REGEX REPLACE "/Include" "" VULKAN_PATH ${VULKAN_PATH})
endif()
set(${VULKAN_LIB_LIST} "vulkan-1")

# Add source to this project's executable.
include_directories(AFTER ${VULKAN_PATH}/Include)
include_directories(
        ${SOLUTION_DIR}/include
        ${SOLUTION_DIR}/src
)
link_directories(${VULKAN_PATH}/Bin;${VULKAN_PATH}/Lib;)

add_executable (CMake "CMake.cpp" "CMake.h")

	# Include Vulkan header files from Vulkan SDK

target_include_directories ( CMake
	PUBLIC ${VULKAN_INCLUDE_DIRS}
)

target_link_libraries (CMake ${VULKAN_LIB_LIST})
target_link_libraries (CMake Vulkan::Vulkan)


# TODO: Add tests and install targets if needed.
