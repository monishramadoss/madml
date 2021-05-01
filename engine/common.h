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

extern std::vector<VkPhysicalDevice> kPhysicalDevices;
extern std::vector<VkDevice> kDevices;
extern std::vector<VkQueue> kQueues;
extern std::vector<VkCommandPool> kCmdPools;
extern std::mutex kContextMtx;
extern std::mutex kDesciptorMtx;
extern size_t number_devices();
extern size_t avalible_memory(int device_id);


#define VK_CHECK_RESULT(f) \
{ \
		if (f != VK_SUCCESS) \
		{ \
			std::cout << "VULKAN KERNEL ERROR: " << f; \
		} \
}
