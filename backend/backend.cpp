//#include <Python.h>
#include <pybind11/pybind11.h>
#include "backend.h"
#include "test.h"
#include <math.h>
PYBIND11_MODULE(backend, m)
{
	m.def("test_memory", test::test_memory);
	m.def("test_trans", test::test_trans);

	//	m.def("test_math", test::test_math);
}