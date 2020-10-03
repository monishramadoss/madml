//#include <Python.h>

#include "backend.h"
#include "test.h"
#include <math.h>

PYBIND11_MODULE(backend, m)
{
	m.def("test_memory", test::test_memory);
}