#include "common.h"
#include "context.h"

struct tensorParam
{
	int total;
};

std::shared_ptr<context> kCtx;
bool enableValidationLayers = false;
VkInstance kInstance;
VkPhysicalDevice kPhysicalDevice;
VkDevice kDevice;
VkQueue kQueue;
VkCommandPool kCmdPool;
VkDebugReportCallbackEXT kDebugReportCallback;
uint32_t kQueueFamilyIndex;
std::vector<const char*> kEnabledLayers;
std::map<std::string, std::vector<uint32_t>> kShaders;
std::mutex kContextMtx;

static uint32_t getComputeQueueFamilyIndex()
{
	uint32_t queueFamilyCount;
	vkGetPhysicalDeviceQueueFamilyProperties(kPhysicalDevice, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(kPhysicalDevice, &queueFamilyCount, queueFamilies.data());
	uint32_t i = 0;

	for (; i < queueFamilies.size(); ++i)
	{
		const VkQueueFamilyProperties props = queueFamilies[i];
		if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT))
			break;
	}

	if (i == queueFamilies.size())
		throw std::runtime_error(
			"could not find a queue family that supports operations");
	return i;
}

bool checkExtensionAvailability(const char* extension_name,
	const std::vector<VkExtensionProperties>& available_extensions)
{
	for (const auto& available_extension : available_extensions)
	{
		if (strcmp(available_extension.extensionName, extension_name) == 0)
			return true;
	}
	return false;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(VkDebugReportFlagsEXT flags,
	VkDebugReportObjectTypeEXT objectType, uint64_t object,
	size_t location, int32_t messageCode, const char* pLayerPrefix,
	const char* pMessage, void* pUserData)
{
	std::cout << "Debug Report: " << pLayerPrefix << ":" << pMessage << std::endl;
	return VK_FALSE;
}

void createContext()
{
	kContextMtx.lock();
	if (!kCtx)
		kCtx.reset(new context());
	kContextMtx.unlock();
}

bool isAvailable()
{
	try
	{
		createContext();
	}
	catch (std::exception& e)
	{
		std::cout << "FAILED TO INIT VK ENV" << e.what();
		return false;
	}
	return true;
}

context::context()
{
	std::vector<const char*> enabledExtensions;
	if (enableValidationLayers)
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> layerProperties(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());

		bool foundLayer = false;
		for (VkLayerProperties prop : layerProperties)
		{
			if (strcmp("VK_LAYER_LUNARG_standard_validation", prop.layerName) == 0)
			{
				foundLayer = true;
				break;
			}
		}

		if (!foundLayer) throw std::runtime_error("Layer VK_LAYER_LUNARG_standard_validation not supported\n");
		kEnabledLayers.push_back("VK_LAYER_LUNARG_standard_validation");

		uint32_t extensionCount;

		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> extensionProperties(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensionProperties.data());

		bool foundExtension = false;
		for (VkExtensionProperties prop : extensionProperties)
		{
			if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, prop.extensionName) == 0)
			{
				foundExtension = true;
				break;
			}
		}

		if (!foundExtension)
			throw std::runtime_error(
				"Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
		enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
	}

	VkApplicationInfo applicationInfo = {};
	applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	applicationInfo.pApplicationName = "halaml Library";
	applicationInfo.applicationVersion = 0;
	applicationInfo.pEngineName = "halaml";
	applicationInfo.engineVersion = 0;
	applicationInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.flags = 0;
	createInfo.pApplicationInfo = &applicationInfo;

	// Give our desired layers and extensions to vulkan.
	createInfo.enabledLayerCount = static_cast<uint32_t>(kEnabledLayers.size());
	createInfo.ppEnabledLayerNames = kEnabledLayers.data();
	createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
	createInfo.ppEnabledExtensionNames = enabledExtensions.data();

	VK_CHECK_RESULT(vkCreateInstance(&createInfo, nullptr, &kInstance));

	if (enableValidationLayers)
	{
		VkDebugReportCallbackCreateInfoEXT createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
		createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT |
			VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
		createInfo.pfnCallback = &debugReportCallbackFn;

		//VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(kInstance, &createInfo,	nullptr, &kDebugReportCallback));
	}

	uint32_t deviceCount;
	vkEnumeratePhysicalDevices(kInstance, &deviceCount, nullptr);
	if (deviceCount == 0)
	{
		throw std::runtime_error("could not find a device with vulkan support");
	}

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(kInstance, &deviceCount, devices.data());

	for (VkPhysicalDevice device : devices)
	{
		kPhysicalDevice = device;
		break;
	}

	kQueueFamilyIndex = getComputeQueueFamilyIndex();

	VkDeviceQueueCreateInfo queueCreateInfo = {};
	queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.queueFamilyIndex = kQueueFamilyIndex;
	queueCreateInfo.queueCount = 1; // create one queue in this family. We don't need more.
	float queuePriorities = 1.0; // we only have one queue, so this is not that imporant.
	queueCreateInfo.pQueuePriorities = &queuePriorities;

	VkDeviceCreateInfo deviceCreateInfo = {};

	// Specify any desired device features here. We do not need any for this application, though.
	VkPhysicalDeviceFeatures deviceFeatures = {};

	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(kEnabledLayers.size());
	deviceCreateInfo.ppEnabledLayerNames = kEnabledLayers.data();
	deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

	VK_CHECK_RESULT(vkCreateDevice(kPhysicalDevice, &deviceCreateInfo, nullptr, &kDevice));

	vkGetDeviceQueue(kDevice, kQueueFamilyIndex, 0, &kQueue);

	VkCommandPoolCreateInfo commandPoolCreateInfo = {};
	commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	commandPoolCreateInfo.queueFamilyIndex = kQueueFamilyIndex;
	VK_CHECK_RESULT(vkCreateCommandPool(kDevice, &commandPoolCreateInfo, nullptr, &kCmdPool));
}

context::~context()
{
	if (kCmdPool != nullptr)
		vkDestroyCommandPool(kDevice, kCmdPool, nullptr);
	if (kDevice != nullptr)
		vkDestroyDevice(kDevice, nullptr);

	if (enableValidationLayers)
	{
		const auto func = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(vkGetInstanceProcAddr(
			kInstance, "vkDestroyDebugReportCallbackEXT"));
		if (func == nullptr)
		{
			printf("Could not load vkDestroyDebugReportCallbackEXT");
		}
		else
		{
			func(kInstance, kDebugReportCallback, nullptr);
		}
	}

	kShaders.clear();
	if (kInstance != nullptr)
		vkDestroyInstance(kInstance, nullptr);
}