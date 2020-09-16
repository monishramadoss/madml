#include <Python.h>

#include "backend.h"
#include "test.h"
#include <math.h>

PyDoc_STRVAR(backend_test_doc, "madml backend test functions function");

/* int gcd(int, int) */
static PyObject* py_gcd(PyObject* self, PyObject* args)
{
	int x, y, result;
	if (!PyArg_ParseTuple(args, "ii", &x, &y))
	{
		return nullptr;
	}
	result = x / y;
	return Py_BuildValue("i", result);
}

/* int divide(int, int, int *) */
static PyObject* py_divide(PyObject* self, PyObject* args)
{
	int a, b, quotient, remainder;
	if (!PyArg_ParseTuple(args, "ii", &a, &b))
	{
		return nullptr;
	}
	quotient = a / b;
	remainder = a % b;
	return Py_BuildValue("(ii)", quotient, remainder);
}

/* Module method table */
static PyMethodDef methods[] =
{
	{ "test_memory", (PyCFunction)test::test_memory, METH_NOARGS, backend_test_doc },
	{ "test_trans", (PyCFunction)test::test_trans, METH_NOARGS, backend_test_doc },
	{ "test_math", (PyCFunction)test::test_math, METH_NOARGS, backend_test_doc },
	{ "test_dnn", (PyCFunction)test::test_dnn, METH_NOARGS, backend_test_doc },
	{ "test_conv", (PyCFunction)test::test_conv, METH_NOARGS, backend_test_doc },
	{ "test_norm", (PyCFunction)test::test_norm, METH_NOARGS, backend_test_doc },
	{ "test_rnn", (PyCFunction)test::test_rnn, METH_NOARGS, backend_test_doc },
	{ "test_mnist", (PyCFunction)test::test_mnist, METH_NOARGS, backend_test_doc },

	{ nullptr, nullptr, 0, nullptr}
};

/* Module structure */
static struct PyModuleDef backend =
{
	PyModuleDef_HEAD_INIT,
	"backend", /* name of module */
	"A sample module", /* Doc string (may be nullptr) */
	-1, /* Size of per-interpreter state or -1 */
	methods /* Method table */
};

/* Module initialization function */
PyMODINIT_FUNC
PyInit_backend(void)
{
	return PyModule_Create(&backend);
}