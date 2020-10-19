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
    const std::vector<int> shape_x{ 512, 512, 512, 4 };
    auto t1 = std::make_shared<tensor>(tensor(1.0, shape_x));
    std::vector<double> toHost, toDevice;
    for (int i = 0; i < 1; ++i)
    {
        std::cout << '\r' << i << " toHost ";
        auto start = system_clock::now();
        char* data = t1->toHost();
        auto end = system_clock::now();
        duration<double> seconds = end - start;
        toHost.push_back(seconds.count());
        std::cout << seconds.count() << " toDevice ";
        std::this_thread::sleep_for(5s);
        start = system_clock::now();
        t1->toDevice(data);
        end = system_clock::now();
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

#endif
