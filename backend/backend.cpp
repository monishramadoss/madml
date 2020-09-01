#include <memory.h>
#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <future>

#include "backend.h"

using namespace std::chrono;

//#define TEST_TRANS
//#define TEST_MATH

//#define TEST_NN

#define TEST_CNN

//#define TEST_RNN

//#define TEST_MNIST

void test_fn()
{
#ifdef TEST_TRANS
	std::cout << "testing trans_op" << std::endl;
	{
		const std::vector<int> shape_x{ 2, 3, 4 };
		char* dat = init::normal_distribution_init(shape_x, 20, 2);
		auto t1 = std::make_shared < tensor>(tensor(dat, shape_x));
		auto k1 = layers::transpose(std::vector<int>{ 1, 2, 0});
		auto t2 = k1.operator()(t1);

		test::PrintDiffer(reinterpret_cast<float*>(t1->toHost()), t1->count());
		std::cout << "\n\n\n";
		test::PrintDiffer(reinterpret_cast<float*>(t2->toHost()), t2->count());
	}
#endif

#ifdef TEST_MATH
	std::cout << "testing add_op" << std::endl;
	{
		const std::vector<int> shape_x{ 2000 };
		auto t1 = std::make_shared<tensor>(tensor(-6.0, shape_x));
		auto t2 = std::make_shared<tensor>(tensor(-1.0, shape_x));
		auto k1 = layers::math::add();
		auto k2 = layers::math::abs();
		auto k3 = layers::math::abs();

		auto t4 = k2(t2);
		auto t5 = k3(t1);
		auto t3 = k1(t4, t5);
		test::PrintDiffer(reinterpret_cast<float*>(t1->toHost()), t1->count());
		test::PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
		test::PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
	}
#endif

#ifdef TEST_NN
	std::cout << "testing dnn" << std::endl;
	{
		const int M = 64;
		const int K = 64;
		const int N = 64;
		const std::vector<int> shape_x{ M, K };
		auto t1 = std::make_shared<tensor>(tensor(1.0, shape_x));
		auto layer = layers::nn::dense(N, false);
		auto layer2 = layers::nn::dense(N, false);
		auto loss = loss::MSE();

		auto t3 = layer(t1);
		auto t4 = layer2(t3);
		auto y_true = std::make_shared<tensor>(tensor(1.0, t4->getShape()));
		loss(y_true, t4);
		loss.backward();

		test::PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		test::PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
	}
#endif
#ifdef TEST_CNN
	std::cout << "testing cnn" << std::endl;
	{
		//cdhw
		std::vector<int> shape_x{ 1, 1, 1, 5, 5 };
		std::vector<int> shape_y{ 1, 1, 1, 3, 3 };
		std::vector<int> shape_z{ 1, 1, 1, 7, 5 };
		auto* data1 = init::fill_memory_iter(shape_x);
		auto* data2 = init::fill_memory_iter(shape_y);
		auto* data3 = init::fill_memory_iter(shape_z);
		auto t1 = std::make_shared<tensor>(tensor(data1, shape_x));
		auto t2 = std::make_shared<tensor>(tensor(data2, shape_y));
		auto t11 = std::make_shared<tensor>(tensor(data3, shape_z));
		auto cnn_layer_1 = layers::nn::conv(1, { 1,3,3 }, { 1,1,1 }, { 0,1,1 }, { 1,1,1 }, 0, false);
		auto cnn_layer_1_1 = layers::nn::conv(1, { 1,3,3 }, { 1,2,2 }, { 0,1,1 }, { 1,1,1 }, 0, false);
		auto cnn_layer_2 = layers::nn::convTranspose(2, { 1,3,3 }, { 1,1,1 }, { 0,0,0 }, { 1,1,1 }, 0, false);

		auto t3 = cnn_layer_1(t1);
		std::cout << "input" << std::endl;
		test::PrintMatrix(reinterpret_cast<float*>(t1->toHost()), t1->getShape());
		std::cout << "output" << std::endl;
		test::PrintMatrix(reinterpret_cast<float*>(t3->toHost()), t3->getShape());
		auto t31 = cnn_layer_1_1(t11);
		std::cout << "input" << std::endl;
		test::PrintMatrix(reinterpret_cast<float*>(t11->toHost()), t11->getShape());
		std::cout << "output" << std::endl;
		test::PrintMatrix(reinterpret_cast<float*>(t31->toHost()), t31->getShape());
		auto t4 = cnn_layer_2(t2);
		std::cout << "input" << std::endl;
		test::PrintMatrix(reinterpret_cast<float*>(t2->toHost()), t2->getShape());
		std::cout << "output" << std::endl;
		test::PrintMatrix(reinterpret_cast<float*>(t4->toHost()), t4->getShape());
	}
#endif
#ifdef TEST_RNN
	int length = 4;
	int vocab = 16;
	int num_layers = 4;
	int hidden_size = 128;

	std::cout << "testing rnn" << std::endl;
	{
		std::vector<int> shape_x{ length, vocab };
		auto t1 = std::make_shared<tensor>(tensor(1, shape_x));
		auto rnn_layer_1 = layers::nn::RNN(vocab, hidden_size, num_layers, length, false);
		auto tup = rnn_layer_1(t1);

		auto t3 = std::get<0>(tup);
		auto t4 = std::get<1>(tup);

		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
		std::cout << std::endl;
	}

	std::cout << "testing lstm" << std::endl;
	{
		std::vector<int> shape_x{ length, vocab };
		auto t1 = std::make_shared<tensor>(tensor(1, shape_x));
		auto rnn_layer_1 = layers::nn::LSTM(vocab, hidden_size, num_layers, length, false);
		auto tup = rnn_layer_1(t1);

		auto t3 = std::get<0>(tup);
		auto t4 = std::get<1>(tup);
		auto t5 = std::get<2>(tup);

		test::PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		test::PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
		test::PrintDiffer(reinterpret_cast<float*>(t5->toHost()), t5->count());
		std::cout << std::endl;
	}

	std::cout << "testing gru" << std::endl;
	{
		std::vector<int> shape_x{ length, vocab };
		auto t1 = std::make_shared<tensor>(tensor(1, shape_x));
		auto rnn_layer_1 = layers::nn::GRU(vocab, hidden_size, num_layers, length, true);
		auto tup = rnn_layer_1(t1);

		auto t3 = std::get<0>(tup);
		auto t4 = std::get<1>(tup);

		test::PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		test::PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
		std::cout << std::endl;
	}
#endif
#ifdef TEST_MNIST
	std::cout << "testing mnist" << std::endl;
	{
		auto l1 = layers::nn::dense(64, true);
		auto l2 = layers::activation::relu();
		auto l3 = layers::nn::dense(64, true);
		auto l4 = layers::activation::relu();
		auto l5 = layers::nn::dense(10, true);
		auto l6 = layers::activation::relu();
		auto l7 = layers::math::add();
		auto loss_fn = loss::MSE();

		auto t0 = std::make_shared<tensor>(tensor(-0.5, std::vector<int>{3, 1, 784}));
		auto y_true = std::make_shared<tensor>(tensor(1, std::vector<int>{3, 1, 10}));
		auto t1 = l1(t0);
		auto t2 = l2(t1);
		auto t3 = l3(t2);
		auto t4 = l4(t3);
		auto tx = l7(t2, t4);
		auto t5 = l5(tx);
		auto t6 = l6(t5);

		loss_fn(t6, y_true);
		test::PrintDiffer(reinterpret_cast<float*>(t6->toHost()), t6->count());
	}

#endif
}

PYBIND11_MODULE(backend, m)
{
	m.doc() = "pybind11 testing pipeline"; // optional module docstring
	m.def("test", &test_fn, "A function which tests ml pipeline");
#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}