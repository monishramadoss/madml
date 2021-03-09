#include "../kernel/kernel.h"
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
        .def("toHost", &tensor::toHost)
        .def("toDevice", &tensor::toDevice);

    m.def("init_float", &init_tensor<float>);
    m.def("init_int", &init_tensor<int>);
    m.def("init_char", &init_tensor<char>);
    m.def("init_bool", &init_tensor<bool>);
    m.def("init_double", &init_tensor<double>);

    m.def("np_to_tensor_float", &np_to_tensor<float>);
    m.def("np_to_tensor_int", &np_to_tensor<int>);
    m.def("np_to_tensor_char", &np_to_tensor<char>);
    m.def("np_to_tensor_bool", &np_to_tensor<bool>);
    m.def("np_to_tensor_double", &np_to_tensor<double>);

    m.def("tensor_to_np_float", &tensor_to_np<float>);
    m.def("tensor_to_np_int", &tensor_to_np<int>);
    m.def("tensor_to_np_char", &tensor_to_np<char>);
    m.def("tensor_to_np_bool", &tensor_to_np<bool>);
    m.def("tensor_to_np_double", &tensor_to_np<double>);

    m.def("list_to_tensor_float", &list_to_tensor<float>);
    m.def("list_to_tensor_int", &list_to_tensor<int>);
    m.def("list_to_tensor_char", &list_to_tensor<char>);
    // m.def("list_to_tensor_bool", &list_to_tensor<bool>);
    m.def("list_to_tensor_double", &list_to_tensor<double>);

    m.def("tensor_to_list_float", &tensor_to_list<float>);
    m.def("tensor_to_list_int", &tensor_to_list<int>);
    m.def("tensor_to_list_char", &tensor_to_list<char>);
    // m.def("tensor_to_list_bool", &tensor_to_list<bool>);
    m.def("tensor_to_list_double", &tensor_to_list<double>);



}