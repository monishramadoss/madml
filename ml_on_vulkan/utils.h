#ifndef UTILS_H
#define UTILS_H
#include <float.h>
#ifdef USE_SHADERC
#include "shaderc/shaderc.h"
#else
typedef int shaderc_shader_kind;
#define shaderc_compute_shader 0
#endif

#include "kernel.h"
#include "context.h"

namespace kernel {
	inline size_t alignSize(size_t sz, int n) {
		return (sz + n - 1) & -n;
	}
	
	std::vector<uint32_t> compile(const std::string& name, shaderc_shader_kind kind, const std::string& data);
	void bindTensor(VkDevice& device, tensor& tensor, int binding, VkDescriptorSet descriptor_set); 
	
	inline bool checkFormat(Format fmt) { return fmt > -1 && fmt < kFormatNum; }
	inline size_t elementSize(Format fmt)
	{
		if (fmt == kFormatFp32 || fmt == kFormatInt32) {
			return 4;
		}
		else if (fmt >= 0 && fmt < kFormatNum) {
			printf("Unsupported format %d", fmt);
		}
		else {
			printf("Invalid format %d", fmt);
		}
		return 0;
	}

	inline int shapeCount(const Shape& shape, int start = -1, int end = -1)
	{
		if (start == -1) start = 0;
		if (end == -1) end = (int)shape.size();
		if (shape.empty())	return 0;

		int elems = 1;
		//assert(start <= (int)shape.size() &&	end <= (int)shape.size() && start <= end);
		for (int i = start; i < end; i++)
			elems *= shape[i];

		return elems;
	}

}

#endif