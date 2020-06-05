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

	if( false ){
		std::cout << "testing operators" << std::endl;
		int size = (int)654123;

		std::vector<int> shape;
		shape.push_back(size);

		auto x = fill_memory_shape<float>(shape, 10);
		auto y = fill_memory_shape<float>(shape, 5);
		auto w = fill_memory_shape<float>(shape, 0);
		auto z = fill_memory_shape<int>(shape, 0);

		auto t1 = new kernel::tensor(x, shape, kernel::kFormatFp32);
		auto t2 = new kernel::tensor(y, shape, kernel::kFormatFp32);
		auto t3 = new kernel::tensor(z, shape, kernel::kFormatBool);
		auto t4 = new kernel::tensor(w, shape, kernel::kFormatFp32);

		for (int i = 0; i < 15; ++i) {
			if (i == 5)
				i = 12;
			kernel::layers::operators* k1 = new kernel::layers::operators(i);
			k1->forward(t1, t2, t4);
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
			kernel::layers::operators* k1 = new kernel::layers::operators(i);
			k1->forward(t1, t2, t3);
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
			kernel::layers::operators* k1 = new kernel::layers::operators(i);
			k1->forward(t2, t4);
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
		delete[] w;

		delete t1;
		delete t2;
		delete t3;
		delete t4;
	}

	int M = 2;
	int K = 128;
	std::vector<int> shape_x{ M, K };
	auto x = fill_memory_shape(shape_x, 1);
	kernel::tensor* t1 = new kernel::tensor(x, shape_x, kernel::kFormatFp32);
	kernel::tensor* t2 = new kernel::tensor();
	kernel::tensor* t3 = new kernel::tensor();

	auto dense_layer_1 = new kernel::layers::nn::dense(64, false);
	auto dense_layer_2 = new kernel::layers::nn::dense(32, false);

	dense_layer_1->forward(t1, t2);
	dense_layer_2->forward(t2, t3);

	dense_layer_1->run();
	dense_layer_2->run();

	
	PrintDiffer((float*)t2->toHost(), M * 64);
	std::cout << std::endl << std::endl;
	PrintDiffer((float*)t3->toHost(), M * 32);

	delete dense_layer_1;
	delete dense_layer_2;
	delete t2;
	delete t1;

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