// ml_on_vulkan.cpp : This file contaxs the 'max' function. Program execution begxs and ends there.
//

#include <iostream>
#include "kernel.h"
//need Vulkan SDK https://vulkan.lunarg.com/sdk/home

int main()
{
	int size = (int)1e6;
	float* x = new float[size];
	float* y = new float[size];
	
	for (int i = 0; i < size; ++i) {
		x[i] = 11.1f;
		y[i] = 5.0f;
	}

	std::vector<int> shape;
	auto b_x = (char*)x;
	auto b_y = (char*)y;
	std::cout << sizeof(x) << " " << sizeof(b_x) << std::endl;
	shape.push_back(size);

	auto i = std::shared_ptr<kernel::tensor>(new kernel::tensor(b_x, shape, kernel::kFormatFp32));
	auto o = std::shared_ptr<kernel::tensor>(new kernel::tensor(b_y, shape, kernel::kFormatFp32));

	
    return 0;
}
