#include <vector>
#include "vknn.h"

PYBIND11_MODULE(vknn, m)
{
    py::class_<gemm, std::shared_ptr<gemm>>(m, "gemm")
        .def(py::init<float&, float&, bool&, bool&, bool&>())
        .def("forward", &gemm::forward)
        .def("run", &gemm::runCommandBuffer);

    py::class_<vol2col>(m, "vol2col")
        .def(py::init<std::vector<int>&>())
        .def("forward", &vol2col::forward)
        .def("run", &vol2col::runCommandBuffer);

    py::class_<col2vol>(m, "col2vol")
        .def(py::init<std::vector<int>&>())
        .def("forward", &col2vol::forward)
        .def("run", &col2vol::runCommandBuffer);

    py::class_<relu>(m, "relu")
        .def(py::init<bool&, bool&>())
        .def("forward", &relu::forward)
        .def("run", &relu::runCommandBuffer);

    py::class_<transpose>(m, "transpose")
        .def(py::init<std::vector<int>&>())
        .def("forward", &transpose::forward)
        .def("run", &transpose::runCommandBuffer);

    py::class_<max_reduce>(m, "max_reduce")
        .def(py::init< bool&>())
        .def("forward", &max_reduce::forward)
        .def("run", &max_reduce::runCommandBuffer);

    //OPTIMIZERS
    py::class_<sgd>(m, "sgd")
        .def(py::init<float&, float&, float&, float&, bool&>())
        .def("forward", &sgd::forward)
        .def("run", &sgd::runCommandBuffer);

    py::class_<adam>(m, "adam")
        .def(py::init<float&, float&, float&, float&, float&, bool&>())
        .def("forward", &adam::forward)
        .def("run", &adam::runCommandBuffer);

    py::class_<adagrad>(m, "adagrad")
        .def(py::init<float&, float&, float&, float&>())
        .def("forward", &adagrad::forward)
        .def("run", &adagrad::runCommandBuffer);

    py::class_<rmsprop>(m, "rmsprop")
        .def(py::init<float&, float&, float&, float&, float&, bool&>())
        .def("forward", &rmsprop::forward)
        .def("run", &rmsprop::runCommandBuffer);

    py::class_<tensor>(m, "tensor")
        .def(py::init<std::vector<float>&, const std::vector<int>&>())
        .def("reshape", &tensor::reShape)
        .def_readonly("shape", &tensor::m_shape)
        .def("byte_count", &tensor::size)
        .def("size", &tensor::count)
        .def("copy", &tensor::copyTo)
        .def("toHost", &tensor::toHost)
        .def("toDevice", &tensor::toDevice);

    m.def("cpu_vol2col", &cpu_vol2col);
    m.def("cpu_col2vol", &cpu_col2vol);

    m.def("number_physcial_devices", &number_devices);
    m.def("avalible_memory", &avalible_memory);

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