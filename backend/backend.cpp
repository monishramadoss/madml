//#include <Python.h>

#include "backend.h"
#include "test.h"
#include <math.h>

PYBIND11_MODULE(backend, m)
{
	m.def("test_memory", test::test_memory);
	m.def("im2col_cpu", &layers::im2col_cpu);
}