#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "../engine/engine.h"

#include "spv_shader.h"
#include "activation.h"
#include "convolution.h"
#include "gemm.h"
#include "loss.h"
#include "math.h"
#include "normalization.h"
#include "pooling.h"
#include "rnn.h"
#include "transform.h"
#include "optimizer.h"

template<typename T>
tensor init_tensor(py::array_t<T, py::array::c_style | py::array::forcecast> a)
{
    std::vector<int> shape;
    for (size_t i = 0; i < a.ndim(); ++i)
        shape.push_back((int)a.shape()[i]);
    const T* data_ptr = a.data();

    if (std::is_same<T, float>::value)
        return tensor((char*)data_ptr, shape, Format::kFormatFp32);
    else if (std::is_same<T, double>::value)
        return tensor((char*)data_ptr, shape, Format::kFormatFp64);
    else if (std::is_same<T, int>::value)
        return tensor((char*)data_ptr, shape, Format::kFormatInt32);
    else if (std::is_same<T, size_t>::value)
        return tensor((char*)data_ptr, shape, Format::kFormatInt64);
    else if (std::is_same<T, char>::value)
        return tensor((char*)data_ptr, shape, Format::kFormatInt8);
    else if (std::is_same<T, bool>::value)
        return tensor((char*)data_ptr, shape, Format::kFormatBool);
    else if (std::is_same<T, uint32_t>::value)
        return tensor((char*)data_ptr, shape, Format::kFormatInt32);
    else
        return tensor(Format::kFormatInvalid);
}

template<typename T>
void np_to_tensor(tensor& t, const py::array_t<T, py::array::c_style | py::array::forcecast>& a)
{
    //py::gil_scoped_release release;
    std::vector<int> shape;
    for (size_t i = 0; i < a.ndim(); ++i)
        shape.push_back((int)a.shape()[i]);
    t.reshape((char*)a.data(), t.getShape());
}

template<typename T>
void tensor_to_np(const tensor& t, py::array_t<T, py::array::c_style | py::array::forcecast>& a)
{
    std::vector<int> shape;
    for (size_t i = 0; i < a.ndim(); ++i)
        shape.push_back((int)a.shape()[i]);
    char* host_ptr = (char*)a.data();
    char* device_ptr = t.toHost();
    memcpy(host_ptr, device_ptr, t.size());
    delete[] device_ptr;
}

template<typename T>
void list_to_tensor(tensor& t, const std::vector<T>& v)
{
    t.reshape((const char*)v.data(), t.getShape());
}

template<typename T>
void tensor_to_list(const tensor& t, std::vector<T> v)
{
    char* host_ptr = (char*)v.data();
    char* device_ptr = t.toHost();
    memcpy(host_ptr, device_ptr, t.size());
    delete[] device_ptr;
}