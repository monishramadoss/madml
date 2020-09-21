#ifndef TEST_H
#define TEST_H
#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <future>
#include <thread>
#include <memory.h>
#include <Python.h>

using namespace std::chrono;

namespace test
{
	PyDoc_STRVAR(backend_test_doc, "madml backend test functions function");

	template <typename T = float>
	void PrintDiffer(T* data, int size)
	{
		std::map<T, int> diff_freq;
		for (int i = 0; i < size; ++i)
			diff_freq[data[i]] += 1;
		std::cout << "{";
		for (const auto df : diff_freq)
			std::cout << df.first << ": " << df.second << ", ";
		std::cout << "}" << std::endl;
	}

	template <typename T = float>
	void PrintMatrix(T* data, std::vector<int> shape)
	{
		for (int i = 0; i < shape.size(); ++i)
			std::cout << shape[i] << ((shape.size() - 1) == i ? "" : ", ");
		std::cout << std::endl;

		int shape_offset = 1;
		std::vector<int> buckets;
		int counter = 0;
		int stage_counter = 0;

		if (shape.size() > 2)
		{
			counter = static_cast<int>(shape.size()) - 3;
			stage_counter = shape[counter];
			for (int i = 0; i < shape.size() - 2; ++i)
			{
				shape_offset *= shape[i];
				buckets.push_back(0);
			}
			buckets.back() = -1;
		}

		int m = shape.size() < 2 ? 1 : shape[shape.size() - 2];
		int n = shape[shape.size() - 1];

		for (int offset = 0; offset < shape_offset; ++offset)
		{
			if (shape.size() > 2)
			{
				buckets[shape.size() - 3]++;
				if (offset >= stage_counter)
				{
					if (counter > 0)
						counter--;
					for (int c = counter; c < buckets.size(); ++c)
						buckets[c] = 0;
					buckets[counter]++;
					stage_counter *= shape[counter];
				}

				for (size_t b = 0; b < buckets.size(); ++b)
					std::cout << buckets[b] << ((b + 1) == buckets.size() ? "\n" : ":");
			}

			for (int i = 0; i < m; ++i)
			{
				std::cout << "[ ";
				for (int j = 0; j < n; ++j)
				{
					std::cout << " " << data[offset * m * n + i * n + j] << ((j + 1) == n ? " ]\n" : ", ");
				}
			}
			std::cout << std::endl;
		}
	}

	void test_memory()
	{
		std::cout << "\ntesting memory" << std::endl;
		const std::vector<int> shape_x{ 512,512,512,4 };
		auto t1 = std::make_shared<tensor>(tensor(1.0, shape_x));
		std::vector<double> toHost, toDevice;
		for (int i = 0; i < 10; ++i)
		{
			std::cout << '\r' << i << " toHost ";
			auto start = std::chrono::system_clock::now();
			char* data = t1->toHost();
			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> seconds = end - start;
			toHost.push_back(seconds.count());
			std::cout << seconds.count() << " toDevice ";
			std::this_thread::sleep_for(5s);
			start = std::chrono::system_clock::now();
			t1->toDevice(data);
			end = std::chrono::system_clock::now();
			seconds = end - start;
			toDevice.push_back(seconds.count());
			std::cout << seconds.count();
			std::this_thread::sleep_for(5s);
		}
		double avg1 = 0;
		double avg2 = 0;
		for (int i = 0; i < toHost.size(); ++i)
		{
			avg1 += toHost[i];
			avg2 += toDevice[i];
		}
		std::cout << "\nAvg toHost: " << avg1 / toHost.size() << " Avg toDevice: " << avg2 / toDevice.size() << std::endl;
	}

	void test_trans()
	{
		std::cout << "\ntesting trans_op" << std::endl;
		const std::vector<int> shape_x{ 2, 3, 4 };
		auto t1 = std::make_shared<tensor>(tensor(1.0, shape_x));
		auto k1 = layers::transpose(std::vector<int>{ 1, 2, 0});
		auto t2 = k1(t1);

		PrintDiffer(reinterpret_cast<float*>(t1->toHost()), t1->count());
		std::cout << "\n";
		PrintDiffer(reinterpret_cast<float*>(t2->toHost()), t2->count());
		std::cout << "\n";
	}

	void test_math()
	{
		std::cout << "\ntesting add_op" << std::endl;
		const std::vector<int> shape_x{ 2000 };
		auto t1 = std::make_shared<tensor>(tensor(-6.0, shape_x));
		auto t2 = std::make_shared<tensor>(tensor(-1.0, shape_x));
		auto k1 = layers::math::add();
		auto k2 = layers::math::abs();
		auto k3 = layers::math::abs();

		auto t4 = k2(t2);
		auto t5 = k3(t1);
		auto t3 = k1(t4, t5);
		PrintDiffer(reinterpret_cast<float*>(t1->toHost()), t1->count());
		std::cout << "\n";
		PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
		std::cout << "\n";
		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		std::cout << "\n";
	}

	void test_dnn()
	{
		std::cout << "\ntesting dnn" << std::endl;
		const int B = 8;
		const int M = 512;
		const int K = 512;
		const int N = 512;
		const std::vector<int> shape_x{ B, M, K };
		auto* data1 = init::fill_memory_iter(shape_x);
		auto t1 = std::make_shared<tensor>(tensor(data1, shape_x));
		auto layer1 = layers::nn::dense(N, false);
		auto layer2 = layers::nn::dense(N, false);
		auto layer3 = layers::nn::dense(N, false);
		auto layer4 = layers::nn::dense(N, false);

		std::vector<double> toHost;
		auto start = std::chrono::system_clock::now();
		auto t3 = layer1(t1); // {36: 25, 117: 25, 198: 25, }
		auto t4 = layer2(t3);
		auto t5 = layer3(t4);
		auto t6 = layer4(t5);
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> seconds = end - start;
		double first_latency = seconds.count();
		for (int i = 0; i < 100; ++i)
		{
			start = std::chrono::system_clock::now();

			t3 = layer1(t1);
			t4 = layer2(t3);
			t5 = layer3(t4);
			t6 = layer4(t5);

			end = std::chrono::system_clock::now();
			seconds = end - start;
			toHost.push_back(seconds.count());
			std::cout << '\r' << i << " compute " << seconds.count();
		}

		std::cout << std::endl << "Fist Latency " << first_latency;
		std::cout << " Avg " << std::accumulate(toHost.begin(), toHost.end(), 0.0) / toHost.size();
		std::cout << " Max " << *std::max_element(toHost.begin(), toHost.end());
		std::cout << " Min " << *std::min_element(toHost.begin(), toHost.end());
		auto const Q1 = toHost.size() / 4;
		auto const Q2 = toHost.size() / 2;
		auto const Q3 = Q1 + Q2;
		std::sort(toHost.begin(), toHost.end());
		std::cout << " 25th-Q " << toHost[Q1];
		std::cout << " 50th-Q " << toHost[Q2];
		std::cout << " 75th-Q " << toHost[Q3];

		PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
		std::cout << "\n";
		delete[] data1;
	}

	void test_conv()
	{
		std::cout << "\ntesting cnn" << std::endl;
		std::vector<int> shape_x{ 1, 1, 1, 5, 5 };
		std::vector<int> shape_y{ 1, 1, 1, 3, 3 };
		std::vector<int> shape_z{ 1, 1, 1, 7, 5 };
		auto* data1 = init::fill_memory_iter(shape_x);
		auto* data2 = init::fill_memory_iter(shape_y);
		auto* data3 = init::fill_memory_iter(shape_z);
		auto t1 = std::make_shared<tensor>(tensor(data1, shape_x));
		auto t2 = std::make_shared<tensor>(tensor(data2, shape_y));
		auto t11 = std::make_shared<tensor>(tensor(data3, shape_z));
		auto cnn_layer_1 = layers::nn::conv(3, { 1,3,3 }, { 1,1,1 }, { 0,1,1 }, { 1,1,1 }, 0, false);
		auto cnn_layer_1_1 = layers::nn::conv(1, { 1,3,3 }, { 1,2,2 }, { 0,1,1 }, { 1,1,1 }, 0, false);
		auto cnn_layer_2 = layers::nn::convTranspose(2, { 1,3,3 }, { 1,1,1 }, { 0,0,0 }, { 1,1,1 }, 0, false);

		auto t3 = cnn_layer_1(t1);
		std::cout << "input" << std::endl;

		//std::cout << *t1;
		std::cout << "output" << std::endl;
		PrintMatrix(reinterpret_cast<float*>(t3->toHost()), t3->getShape());
		auto t31 = cnn_layer_1_1(t11);
		std::cout << "input" << std::endl;
		PrintMatrix(reinterpret_cast<float*>(t11->toHost()), t11->getShape());
		std::cout << "output" << std::endl;
		PrintMatrix(reinterpret_cast<float*>(t31->toHost()), t31->getShape());
		auto t4 = cnn_layer_2(t2);
		std::cout << "input" << std::endl;
		PrintMatrix(reinterpret_cast<float*>(t2->toHost()), t2->getShape());
		std::cout << "output" << std::endl;
		PrintMatrix(reinterpret_cast<float*>(t4->toHost()), t4->getShape());
		std::cout << std::endl;
		delete[] data1;
		delete[] data2;
		delete[] data3;
	}

	void test_norm()
	{
		std::cout << "\ntesting normalization" << std::endl;
		std::vector<int> shape_x{ 2, 3, 1, 5, 5 };
		char* data1 = init::fill_memory_iter(shape_x);
		auto t1 = std::make_shared<tensor>(tensor(data1, shape_x));
		auto l = layers::normalization::BatchNormalization();

		auto t2 = l(t1);

		PrintMatrix(reinterpret_cast<float*>(t1->toHost()), t1->getShape());
		PrintMatrix(reinterpret_cast<float*>(t2->toHost()), t2->getShape());
		std::cout << std::endl;
		delete[] data1;
	}

	void test_rnn()
	{
		int length = 4;
		int vocab = 16;
		int num_layers = 4;
		int hidden_size = 128;

		std::cout << "\ntesting rnn" << std::endl;
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

		std::cout << "\ntesting lstm" << std::endl;
		{
			std::vector<int> shape_x{ length, vocab };
			auto t1 = std::make_shared<tensor>(tensor(1, shape_x));
			auto rnn_layer_1 = layers::nn::LSTM(vocab, hidden_size, num_layers, length, false);
			auto tup = rnn_layer_1(t1);

			auto t3 = std::get<0>(tup);
			auto t4 = std::get<1>(tup);
			auto t5 = std::get<2>(tup);

			PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
			PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
			PrintDiffer(reinterpret_cast<float*>(t5->toHost()), t5->count());
			std::cout << std::endl;
		}

		std::cout << "\ntesting gru" << std::endl;
		{
			std::vector<int> shape_x{ length, vocab };
			auto t1 = std::make_shared<tensor>(tensor(1, shape_x));
			auto rnn_layer_1 = layers::nn::GRU(vocab, hidden_size, num_layers, length, true);
			auto tup = rnn_layer_1(t1);

			auto t3 = std::get<0>(tup);
			auto t4 = std::get<1>(tup);

			PrintDiffer(reinterpret_cast<float*>(t3->toHost()), t3->count());
			PrintDiffer(reinterpret_cast<float*>(t4->toHost()), t4->count());
			std::cout << std::endl;
		}
	}
	void test_mnist()
	{
		std::cout << "\ntesting mnist" << std::endl;
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

		//loss_fn(t6, y_true);
		PrintDiffer(reinterpret_cast<float*>(t6->toHost()), t6->count());
		std::cout << std::endl;
	}

	void test_resnet()
	{
		std::cout << "testing resent" << std::endl;
	}
}

#endif