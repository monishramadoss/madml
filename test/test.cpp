// test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <memory.h>
#include <iostream>
#include <madml.h>
#include <vector>
#include <map>

#define X M*K
#define Y K*N
#define Z M*N

void PrintDiffer(float* data, int size) {
	std::map<float, int> diff_freq;

	for (int i = 0; i < size; ++i) {
		diff_freq[data[i]] += 1;
	}
	std::cout << "{";
	for (auto df : diff_freq) {
		std::cout << df.first << ": " << df.second << ", ";
	}

	std::cout << "}";

}


void PrintDiffer(int* data, int size) {
	std::map<int, int> diff_freq;

	for (int i = 0; i < size; ++i) {
		diff_freq[(int)data[i]] += 1;
	}
	std::cout << "{";
	for (auto df : diff_freq) {
		std::cout << df.first << ": " << df.second << ", ";
	}

	std::cout << "}";

}

void PrintMatrix(float* data, std::vector<int> shape) {
	
	for (int i = 0; i < shape[0]; ++i) {
		std::cout << "[ ";
		for (int j = 0; j < shape[1]; ++j) {
			std::cout << " " << data[i * shape[1] + j] << ",";
		}
		std::cout << "]" << std::endl;
	}
}

int main()
{
	
	std::cout << "testing operators" << std::endl;
	{
		int size = (int)2097152;
		float* x = new float[size];
		float* y = new float[size];
		float* w = new float[size];
		int* z = new int[size];

		for (int i = 0; i < size; ++i) {
			x[i] = 10.0f;
			y[i] = 5.5f;
			z[i] = 0;
		}

		std::vector<int> shape;
		shape.push_back(size);

		auto t1 = new kernel::tensor((char*)x, shape, kernel::kFormatFp32);
		auto t2 = new kernel::tensor((char*)y, shape, kernel::kFormatFp32);
		auto t3 = new kernel::tensor((char*)z, shape, kernel::kFormatBool);
		auto t4 = new kernel::tensor((char*)w, shape, kernel::kFormatFp32);

		for (int i = 0; i < 15; ++i) {
			if (i == 5)
				i = 12;
			kernel::layer* k1 = new kernel::layers::operators(i);
			std::vector<kernel::tensor> input { *t1, *t2 };
			std::vector<kernel::tensor> output{ *t4 };
			k1->forward(input, output);
			k1->run();

		    std::cout << i << std::endl;
			PrintDiffer((float*)t1->toHost(), size);
			std::cout << std::endl;
			PrintDiffer((float*)t2->toHost(), size);
			std::cout << std::endl;
			PrintDiffer((float*)t4->toHost(), size);
			std::cout << std::endl;
			delete k1;
		}
		
		for (int i = 5; i < 11; ++i) {			
			kernel::layer* k1 = new kernel::layers::operators(i);
			std::vector<kernel::tensor> input{ *t1, *t2 };
			std::vector<kernel::tensor> output{ *t3 };
			k1->forward(input, output);
			k1->run();
			
			std::cout << i << std::endl;
			PrintDiffer((float*)t1->toHost(), size);
			std::cout << std::endl;
			PrintDiffer((float*)t2->toHost(), size);
			std::cout << std::endl;
			PrintDiffer((int*)t3->toHost(), size);
			std::cout << std::endl;
			delete k1;
		}

		for (int i = 15; i < 35; ++i) {
			kernel::layer* k1 = new kernel::layers::operators(i);
			std::vector<kernel::tensor> input{ *t2 };
			std::vector<kernel::tensor> output{ *t4 };
			k1->forward(input, output);
			k1->run();

			std::cout << i << std::endl;
			PrintDiffer((float*)t2->toHost(), size);
			std::cout << std::endl;
			PrintDiffer((float*)t4->toHost(), size);
			std::cout << std::endl;
			delete k1;
		}

		delete[] x;
		delete[] y;
		delete[] z;

		delete t1;
		delete t2;
		delete t3;
	}

	
}