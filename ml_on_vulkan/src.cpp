// ml_on_vulkan.cpp : This file contaxs the 'max' function. Program execution begxs and ends there.
//

#include <iostream>
#include "kernel.h"
//need Vulkan SDK https://vulkan.lunarg.com/sdk/home

int main() {
	int size = (int)1e6;
	float* x = new float[size];
	float* y = new float[size];
	float* z = new float[size];
	char* t = new char[size];

	for (int i = 0; i < size; ++i) {
		x[i] = 10.0f;
		y[i] = 5.0f;
		z[i] = 0;
		t[i] = 0;
	}

	std::vector<int> shape;
	shape.push_back(size);

	auto t1 = new kernel::tensor((char*)x, shape, kernel::kFormatFp32);
	auto t2 = new kernel::tensor((char*)y, shape, kernel::kFormatFp32);
	auto t3 = new kernel::tensor((char*)z, shape, kernel::kFormatFp32);
	auto t4 = new kernel::tensor((char*)t, shape, kernel::kFormatBool);

	auto k1 = new kernel::layers::operators(0);
	auto k2 = new kernel::layers::operators(6);

	k1->forward(*t1, *t2, *t3);
	k1->run();

	k1->forward(*t1, *t3);
	k1->run();

	k2->forward(*t1, *t3, *t4);

	float* tmp1 = (float*)t1->toHost();
	float* tmp = (float*)t3->toHost();
	char* tmp2 = t4->toHost();

	delete[] x;
	delete[] y;
	delete[] z;

	delete t1;
	delete t2;
	delete t3;

	delete k1;

    return 0;
}
