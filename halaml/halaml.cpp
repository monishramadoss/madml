#include <memory.h>
#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <future>

#include "halmal.h"

using namespace std::chrono;

void PrintDiffer(float* data, int size)
{
	std::map<float, int> diff_freq;
	for (int i = 0; i < size; ++i)
		diff_freq[data[i]] += 1;
	std::cout << "{";
	for (const auto df : diff_freq)
		std::cout << df.first << ": " << df.second << ", ";
	std::cout << "}" << std::endl;
}

void PrintDiffer(int* data, int size)
{
	std::map<int, int> diff_freq;
	for (int i = 0; i < size; ++i)
		diff_freq[static_cast<int>(data[i])] += 1;
	std::cout << "{";
	for (const auto df : diff_freq)
		std::cout << df.first << ": " << df.second << ", ";
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
//#define TEST_TRANS
#define TEST_MATH
//#define TEST_NN
//#define TEST_CNN
//#define TEST_RNN
//#define TEST_MNIST
void test_fn()
{
#ifdef TEST_TRANS
	std::cout << "testing trans_op" << std::endl;
	{
		const std::vector<int> shape_x{ 2, 3, 4 };
		char* dat = kernel::init::normal_distribution_init(shape_x, 20, 2);
		auto t1 = std::make_shared < kernel::tensor>(kernel::tensor(dat, shape_x));
		auto k1 = kernel::layers::transpose(std::vector<int>{ 1, 2, 0});
		auto t2 = k1.hook(t1);

		PrintDiffer(reinterpret_cast<float*>(t2->toHost()), t2->count());
		k1.execute();
		PrintDiffer(reinterpret_cast<float*>(t1->toHost()), t1->count());
		std::cout << "\n\n\n";
		PrintDiffer(reinterpret_cast<float*>(t2->toHost()), t2->count());
	}
#endif

#ifdef TEST_MATH
	std::cout << "testing add_op" << std::endl;
	{
		const std::vector<int> shape_x{ 2000 };
		auto t1 = std::make_shared<kernel::tensor>(kernel::tensor(6.0, shape_x));
		auto t2 = std::make_shared<kernel::tensor>(kernel::tensor(1.0, shape_x));
		auto k1 = kernel::layers::math::abs();
		auto t3 = k1.hook(t1);
		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		k1.execute();
		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
	}
#endif

#ifdef TEST_NN
	std::cout << "testing dnn" << std::endl;
	{
		const int M = 240;
		const int K = 240;
		const int N = 240;
		const std::vector<int> shape_x{ M, K };
		auto t1 = std::make_shared<kernel::tensor>(kernel::tensor(1.0, shape_x));
		auto layer = new kernel::layers::nn::dense(N, false);
		auto layer2 = new kernel::layers::nn::dense(N, false);
		auto t3 = layer->hook(t1);
		auto t4 = layer2->hook(t3);

		layer->execute();
		PrintDiffer(reinterpret_cast<float*>(t4->toHost()), M * N);
		for (int i = 0; i < 100; ++i)
		{
			layer->execute();
		}
	}
#endif
#ifdef TEST_CNN
	std::cout << "testing cnn" << std::endl;
	{
		//cdhw
		std::vector<int> shape_x{ 3, 1, 128, 128 };
		auto t1 = std::make_shared<kernel::tensor>(kernel::tensor(1.0, shape_x));
		auto cnn_layer_1 = kernel::layers::nn::conv(8, { 1,3,3 }, { 1,1,1 }, { 0,0,0 }, { 1,1,1 }, 0, false);
		auto cnn_layer_2 = kernel::layers::nn::convTranspose(3, { 1,3,3 }, { 1,1,1 }, { 0,0,0 }, { 1,1,1 }, 0, false);

		auto t3 = cnn_layer_1.hook(t1);
		auto t4 = cnn_layer_2.hook(t3);

		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
		cnn_layer_1.execute();
		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
	}
#endif
#ifdef TEST_RNN
	std::cout << "testing rnn" << std::endl;
	{
		int length = 4;
		int vocab = 16;
		int num_layers = 4;
		int hidden_size = 128;
		std::vector<int> shape_x{ length, vocab };
		auto t1 = std::make_shared<kernel::tensor>(kernel::tensor(1, shape_x));
		auto rnn_layer_1 = kernel::layers::nn::RNN(vocab, hidden_size, num_layers, length, false);
		auto tup = rnn_layer_1.hook(t1);

		auto t3 = std::get<0>(tup);
		auto t4 = std::get<1>(tup);

		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
		rnn_layer_1.execute();
		std::cout << std::endl;
		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
		std::cout << std::endl;
	}

	std::cout << "testing lstm" << std::endl;
	{
		int length = 4;
		int vocab = 16;
		int num_layers = 2;
		int hidden_size = 128;
		std::vector<int> shape_x{ length, vocab };
		auto t1 = std::make_shared<kernel::tensor>(kernel::tensor(1, shape_x));
		auto rnn_layer_1 = kernel::layers::nn::LSTM(vocab, hidden_size, num_layers, length, false);
		auto tup = rnn_layer_1.hook(t1);

		auto t3 = std::get<0>(tup);
		auto t4 = std::get<1>(tup);
		auto t5 = std::get<2>(tup);

		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
		PrintDiffer(reinterpret_cast<float*>(t5->toHost()), t5->count());
		rnn_layer_1.execute();
		std::cout << std::endl;
		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
		PrintDiffer(reinterpret_cast<float*>(t5->toHost()), t5->count());
		std::cout << std::endl;
	}

	std::cout << "testing gru" << std::endl;
	{
		int length = 4;
		int vocab = 16;
		int num_layers = 2;
		int hidden_size = 128;
		std::vector<int> shape_x{ length, vocab };
		auto t1 = std::make_shared<kernel::tensor>(kernel::tensor(1, shape_x));
		auto rnn_layer_1 = kernel::layers::nn::GRU(vocab, hidden_size, num_layers, length, true);
		auto tup = rnn_layer_1.hook(t1);

		auto t3 = std::get<0>(tup);
		auto t4 = std::get<1>(tup);

		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
		rnn_layer_1.execute();
		std::cout << std::endl;
		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
		std::cout << std::endl;
	}
#endif
#ifdef TEST_MNIST
	std::cout << "testing mnist" << std::endl;
	{
		auto l1 = kernel::layers::nn::dense(64, true);
		auto l2 = kernel::layers::activation::relu();
		auto l3 = kernel::layers::nn::dense(64, true);
		auto l4 = kernel::layers::activation::relu();
		auto l5 = kernel::layers::nn::dense(10, true);
		auto l6 = kernel::layers::activation::relu();
		auto l7 = kernel::layers::math::add();

		auto t0 = std::make_shared<kernel::tensor>(kernel::tensor(-0.5, std::vector<int>{1, 784}));
		auto t1 = l1.hook(t0);
		auto t2 = l2.hook(t1);
		auto t3 = l3.hook(t2);
		auto t4 = l4.hook(t3);
		auto tx = l7.hook(t2, t4);
		auto t5 = l5.hook(tx);
		auto t6 = l6.hook(t5);
		l6.execute();
		PrintDiffer(reinterpret_cast<float*>(t2->toHost()), t2->count());
	}

#endif
}

PYBIND11_MODULE(halaml, m)
{
	m.doc() = "pybind11 testing pipeline"; // optional module docstring
	m.def("test", &test_fn, "A function which tests ml pipeline");
#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}