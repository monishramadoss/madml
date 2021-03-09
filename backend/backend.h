#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template<typename T = float>
std::shared_ptr<tensor> init_tensor(py::array_t<T, py::array::c_style | py::array::forcecast> a)
{
    std::vector<int> shape;
    for (size_t i = 0; i < a.ndim(); ++i)
        shape.push_back((int)a.shape()[i]);
    const T* data_ptr = a.data();

    if (std::is_same<T, float>::value)
        return std::make_shared<tensor>((char*)data_ptr, shape, Format::kFormatFp32);
    else if (std::is_same<T, double>::value)
        return std::make_shared<tensor>((char*)data_ptr, shape, Format::kFormatFp64);
    else if (std::is_same<T, int>::value)
        return std::make_shared<tensor>((char*)data_ptr, shape, Format::kFormatInt32);
    else if (std::is_same<T, size_t>::value)
        return std::make_shared<tensor>((char*)data_ptr, shape, Format::kFormatInt64);
    else if (std::is_same<T, char>::value)
        return std::make_shared<tensor>((char*)data_ptr, shape, Format::kFormatInt8);
    else if (std::is_same<T, bool>::value)
        return std::make_shared<tensor>((char*)data_ptr, shape, Format::kFormatBool);
    else
        return std::make_shared<tensor>(Format::kFormatInvalid);
}


template<typename T = float>
void np_to_tensor(std::shared_ptr<tensor>& t, const py::array_t<T, py::array::c_style | py::array::forcecast>& a)
{
    std::vector<int> shape;
    for (size_t i = 0; i < a.ndim(); ++i)
        shape.push_back((int)a.shape()[i]);
    if (shape != t->getShape()) printf("SHAPES DON'T MATCH \n");
    t->reshape((char*)a.data(), t->getShape());
}


template<typename T = float>
void tensor_to_np(const std::shared_ptr<tensor>& t, py::array_t<T, py::array::c_style | py::array::forcecast>& a)
{
    std::vector<int> shape;
    for (size_t i = 0; i < a.ndim(); ++i)
        shape.push_back((int)a.shape()[i]);
    if (shape != t->getShape()) printf("SHAPES DON'T MATCH \n");
    
    char* ptr = (char*)a.mutable_data();
    auto vec = t->toHost();
    memcpy(ptr, vec.data(), t->size());
}

template<typename T = float>
void list_to_tensor(std::shared_ptr<tensor>& t, const std::vector<T>& v)
{
    if (t->count() != v.size()) printf("SHAPES DON'T MATCH \n");
    t->reshape((const char*)v.data(), t->getShape());
}

template<typename T = float>
void tensor_to_list(const std::shared_ptr<tensor>& t, std::vector<T> v)
{
    char* ptr = (char*)v.data();
    auto vec = t->toHost();
    memcpy(ptr, vec.data(), t->size());
}