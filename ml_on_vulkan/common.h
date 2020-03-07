#ifndef COMMON_H
#define COMMON_H
#include <math.h>
#include <string.h>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <memory>

#include <vulkan/vulkan.h>
#include "shaders/spv_shader.h"
#include "kernel.h"

namespace kernel {
	extern VkPhysicalDevice kPhysicalDevice;
	extern VkDevice kDevice;
	extern VkQueue kQueue;
	extern VkCommandPool kCmdPool;
	extern std::mutex kContextMtx;

	/*enum ShapeIdx
	{
		kShapeIdxBatch = 0,
		kShapeIdxChannel,
		kShapeIdxHeight,
		kShapeIdxWidth,
	};*/

#define VK_CHECK_RESULT(f) \
{ \
        if (f != VK_SUCCESS) \
        { \
			std::cout << "VULKAN KERNEL ERROR: " << (int)f; \
		} \
}
	   	 
}
#endif