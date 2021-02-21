#pragma once

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


extern VkPhysicalDevice kPhysicalDevice;
extern VkDevice kDevice;
extern VkQueue kQueue;
extern VkCommandPool kCmdPool;
extern std::mutex kContextMtx;

#define VK_CHECK_RESULT(f) \
{ \
        if (f != VK_SUCCESS) \
        { \
			std::cout << "VULKAN KERNEL ERROR: " << f; \
		} \
}
