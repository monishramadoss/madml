#include <vector>
#include "vknn.h"

PYBIND11_MODULE(vknn, m)
{
    py::class_<gemm>(m, "gemm")
        .def(py::init<float&, float&, bool&>())
        .def("forward", &gemm::forward);
    
    py::class_<vol2col>(m, "vol2col")
        .def(py::init<std::vector<int>&>())
        .def("forward", &vol2col::forward);
    py::class_<col2vol>(m, "col2vol")
        .def(py::init<std::vector<int>&>())
        .def("forward", &col2vol::forward);
  
    py::class_<relu>(m, "relu")
        .def(py::init<bool&>())
        .def("forward", &relu::forward);
}