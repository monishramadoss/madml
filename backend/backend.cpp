#include "kernel.h"
#include "backend.h"

PYBIND11_MODULE(backend, m)
{ 
    py::class_<tensor, std::shared_ptr<tensor>>(m, "tensor")
        .def(py::init<std::vector<float>&, const std::vector<int>&>())
        .def("reshape", &tensor::reShape)
        .def_readonly("shape", &tensor::m_shape)
        .def("byte_count", &tensor::size)
        .def("size", &tensor::count)
        .def("copy", &tensor::copyTo)
        .def("toHost", &tensor::toHost);
}
