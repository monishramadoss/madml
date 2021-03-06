#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template<typename T = float>
tensor init_tensor(py::array_t<T, py::array::c_style | py::array::forcecast> a)
{
    std::vector<int> shape;
    for (size_t i = 0; i < a.ndim(); ++i)
        shape.push_back((int)a.shape()[i]);
    const T* data = a.data();

    if (std::is_same<T, float>::value)
        return tensor((char*)data, shape, Format::kFormatFp32);
    else if (std::is_same<T, double>::value)
        return tensor((char*)data, shape, Format::kFormatFp64);
    else if (std::is_same<T, int>::value)
        return tensor((char*)data, shape, Format::kFormatInt32);
    else if (std::is_same<T, size_t>::value)
        return tensor((char*)data, shape, Format::kFormatInt64);
    else if (std::is_same<T, char>::value)
        return tensor((char*)data, shape, Format::kFormatInt8);
    else if (std::is_same<T, bool>::value)
        return tensor((char*)data, shape, Format::kFormatBool);
    else
        return tensor(Format::kFormatInvalid);
}
