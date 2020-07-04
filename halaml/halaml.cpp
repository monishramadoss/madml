#include <memory.h>
#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <future>
#include <pybind11/pybind11.h>

#include "madml.h"

namespace py = pybind11;
using namespace std::chrono;

#define X M*K
#define Y K*N
#define Z M*N

void PrintDiffer(float* data, int size)
{
	std::map<float, int> diff_freq;

	for (int i = 0; i < size; ++i)
	{
		diff_freq[data[i]] += 1;
	}
	std::cout << "{";
	for (const auto df : diff_freq)
	{
		std::cout << df.first << ": " << df.second << ", ";
	}

	std::cout << "}" << std::endl;
}

void PrintDiffer(int* data, int size)
{
	std::map<int, int> diff_freq;

	for (int i = 0; i < size; ++i)
	{
		diff_freq[static_cast<int>(data[i])] += 1;
	}
	std::cout << "{";
	for (const auto df : diff_freq)
	{
		std::cout << df.first << ": " << df.second << ", ";
	}

	std::cout << "}" << std::endl;
}

void PrintMatrix(float* data, std::vector<int> shape)
{
	for (int i = 0; i < shape[0]; ++i)
	{
		std::cout << "[ ";
		for (int j = 0; j < shape[1]; ++j)
		{
			std::cout << " " << data[i * shape[1] + j] << ",";
		}
		std::cout << "]" << std::endl;
	}
}

//#define TEST_MATH
//#define TEST_NN
#define TEST_CNN
//#define TEST_RNN

void test_fn() {
#ifdef TEST_MATH
	std::cout << "testing add_op" << std::endl;
	{
		std::vector<int> shape_x{ 1000 };
		auto* t1 = new kernel::tensor(1.0, shape_x);
		auto* t2 = new kernel::tensor(1.0, shape_x);
		auto* k1 = new kernel::layers::math::add();
		auto* t3 = k1->forward(t1, t2);
		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), 1000);
		k1->super_run();
		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), 1000);
	}
#endif

#ifdef TEST_NN
	std::cout << "testing dnn" << std::endl;
	{
		int M = 240;
		int K = 240;
		int N = 240;
		std::vector<int> shape_x{ M, K };
		auto* t1 = new kernel::tensor(1.0, shape_x);
		auto* layer = new kernel::layers::nn::dense(N, false);
		auto* t3 = layer->forward(t1);
		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), M * N);
		layer->super_run();
		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), M * N);
	}
#endif
#ifdef TEST_CNN
	std::cout << "testing cnn" << std::endl;
	{
		//cdhw
		std::vector<int> shape_x{ 3, 1, 512, 512 };
		auto* t1 = new kernel::tensor(1.0, shape_x);
		auto* cnn_layer_1 = new kernel::layers::nn::conv(8, { 1,3,3 }, { 1,1,1 }, { 0,0,0 }, { 1,1,1 }, 0, false);
		auto* t3 = cnn_layer_1->forward(t1);
		auto* cnn_layer_2 = new kernel::layers::nn::convTranspose(3, { 1,3,3 }, { 1,1,1 }, { 0,0,0 }, { 1,1,1 }, 0, false);
		auto* t4 = cnn_layer_2->forward(t3);
		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());

		cnn_layer_1->super_run();
		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
	}
#endif
#ifdef TEST_RNN
	std::cout << "testing rnn" << std::endl;
	{
		int length = 1;
		int vocab = 16;
		int num_layers = 1;
		int directions = 1;
		int hidden_size = 128;
		std::vector<int> shape_x{ vocab, length };
		std::vector<int> shape_y{ hidden_size, num_layers * directions };
		auto* t1 = new kernel::tensor(1, shape_x);
		auto* t2 = new kernel::tensor(1, shape_y);
		auto* rnn_layer_1 = new kernel::layers::nn::RNN(vocab, hidden_size, num_layers);
		auto t3 = rnn_layer_1->forward(t1, t2);
	}
#endif
	std::cin.get();
}

PYBIND11_MODULE(halaml, m)
{
	m.doc() = "pybind11 testing pipleine"; // optional module docstring
	m.def("test", &test_fn, "A function which test ml pipeline");
#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}
