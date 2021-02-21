#include <vector>
#include "vknn.h"

PYBIND11_MODULE(vknn, m){
    py::class_<gemm>(m, "gemm")
        .def(py::init<float, float>())
        .def("forward", &gemm::forward);
}