// test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <memory.h>
#include <iostream>
#include <madml.h>
#include <vector>
#define X M*K
#define Y K*N
#define W M*N
int main()
{
	{
		const int M = 6;
		const int K = 4;
		const int N = 2;
		float* d_x = new float[X];
		float* d_y = new float[Y];
		float* d_z = new float[1];
		float* d_w = new float[W];

		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < K; ++j)
				d_x[i * K + j] = 2.0;
		}
		for (int i = 0; i < K; ++i) {
			for (int j = 0; j < N; ++j)
				d_y[i * N + j] = 3.0;
		}

		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j)
				d_w[i * N + j] = 1;
		}
		d_z[0] = 0;
		std::vector<int> shape_x = { M, K };
		std::vector<int> shape_y = { K, N };
		std::vector<int> shape_z = { 1 };
		std::vector<int> shape_w = { M, N };
		kernel::layers::matmul* mm = new kernel::layers::matmul(false);
		auto t1 = new kernel::tensor((char*)d_x, shape_x, kernel::kFormatFp32);
		auto t2 = new kernel::tensor((char*)d_y, shape_y, kernel::kFormatFp32);
		auto t3 = new kernel::tensor((char*)d_z, shape_z, kernel::kFormatFp32);
		auto t4 = new kernel::tensor((char*)d_w, shape_w, kernel::kFormatFp32);

		mm->forward(*t1, *t2, *t3, *t4);
		mm->run();
		float* tmp1 = (float*)t1->toHost();
		float* tmp2 = (float*)t4->toHost();
		std::cout << tmp1[1] << std::endl << " :: " << tmp2[1] << std::endl;

		delete[] d_x;
		delete[] d_y;
		delete[] d_z;
		delete[] d_w;
		delete t1;
		delete t2;
		delete t3;
		delete t4;
		delete mm;
	}

	{
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
		std::cout << tmp1[0] << " :: " << tmp[0] << std::endl;

		delete[] x;
		delete[] y;
		delete[] z;
		delete t1;
		delete t2;
		delete t3;
		delete k1;
		delete k2;

	}
	std::cin.get();
}