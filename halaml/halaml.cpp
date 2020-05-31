#include <memory.h>
#include <iostream>
#include <vector>
#include <map>
#include <chrono> 
#include <pybind11/pybind11.h>

#include "madml.h"

namespace py = pybind11;
using namespace std::chrono;

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

void test_fn() {

	std::cout << "testing operators" << std::endl;
	{
		int size = (int)654123;

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
			std::vector<kernel::tensor> input{ *t1, *t2 };
			std::vector<kernel::tensor> output{ *t4 };
			k1->forward(input, output);
			{
				auto start = high_resolution_clock::now();
				k1->run();
				auto stop = high_resolution_clock::now();
				auto duration = duration_cast<microseconds>(stop - start);
				std::cout << i << " " << duration.count() / size << " microseconds" << std::endl;
			}
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
			{
				auto start = high_resolution_clock::now();
				k1->run();
				auto stop = high_resolution_clock::now();
				auto duration = duration_cast<microseconds>(stop - start);
				std::cout << i << " " << duration.count() / size << " microseconds" << std::endl;
			}

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
			{
				auto start = high_resolution_clock::now();
				k1->run();
				auto stop = high_resolution_clock::now();
				auto duration = duration_cast<microseconds>(stop - start);
				std::cout << i << " " << duration.count() / size << " microseconds" << std::endl;
			}
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


	std::cout << "testing gemm" << std::endl;
	{
		int M = 4096;
		int N = 4096;
		int K = 4096;
		float* x = new float[X];
		float* y = new float[Y];
		float* z = new float[Z];

		for (int i = 0; i < X; ++i)
			x[i] = 1.0;
		for (int i = 0; i < Y; ++i)
			y[i] = 2.0;
		for (int i = 0; i < Z; ++i)
			z[i] = 0;

		std::vector<int> shape_x{ M, K };
		std::vector<int> shape_y{ K, N };
		std::vector<int> shape_z{ M, N };

		auto t1 = new kernel::tensor((char*)x, shape_x, kernel::kFormatFp32);
		auto t2 = new kernel::tensor((char*)y, shape_y, kernel::kFormatFp32);
		auto t3 = new kernel::tensor((char*)z, shape_z, kernel::kFormatFp32);

		kernel::layers::matmul* mm = new kernel::layers::matmul();
		mm->forward(*t1, *t2, *t3);

		{
			auto start = high_resolution_clock::now();
			mm->run();
			auto stop = high_resolution_clock::now();
			auto duration = duration_cast<microseconds>(stop - start);
			std::cout << std::endl << duration.count() << " microseconds" << std::endl;
		}

		PrintDiffer((float*)t3->toHost(), Z);
		std::cout << std::endl;
		delete mm;
		delete[] x;
		delete[] y;
		delete[] z;

		delete t1;
		delete t2;
		delete t3;

	}


	std::cout << "teting conv" << std::endl;
	{
		int N = 10;
		int IN = 128;
		int C = 1;
		int OC = 8;
		float* x = new float[N * C * IN];
		float* y = new float[OC * C * 3];
		float* z = new float[32];

		for (int i = 0; i < N * C * IN; ++i)
			x[i] = 1.0;
		for (int i = 0; i < OC * C * 3; ++i)
			y[i] = 2.0;
		for (int i = 0; i < 32; ++i)
			z[i] = 0;

		std::vector<int> shape_x{ N, C, IN };
		std::vector<int> shape_y{ OC, C, 3 };
		std::vector<int> shape_z{ 32 };

		auto t1 = new kernel::tensor((char*)x, shape_x, kernel::kFormatFp32);
		auto t2 = new kernel::tensor((char*)y, shape_y, kernel::kFormatFp32);
		auto t3 = new kernel::tensor((char*)z, shape_z, kernel::kFormatFp32);

		/*kernel::layers::convolution* conv = new kernel::layers::convolution(3, 1, 1, 0);
		conv->forward(*t1, *t2, *t3);
		conv->run();*/

		PrintDiffer((float*)t3->toHost(), 32);
		std::cout << std::endl;

	}
}

PYBIND11_MODULE(halaml, m) {
    m.doc() = "pybind11 testing pipleine"; // optional module docstring
    m.def("test", &test_fn, "A function which test ml pipeline");
#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}