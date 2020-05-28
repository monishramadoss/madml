#include <Python.h>
#include "madml.h"
/*
 * Implements an example function.
 */
PyDoc_STRVAR(halalml_example_doc, "example(obj, number)\
\
Example function");

PyObject *halalml_example(PyObject *self, PyObject *args, PyObject *kwargs) {
    /* Shared references that do not need Py_DECREF before returning. */
    PyObject *obj = NULL;
    int number = 0;

    /* Parse positional and keyword arguments */
    static char* keywords[] = { "obj", "number", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", keywords, &obj, &number)) {
        return NULL;
    }

    /* Function implementation starts here */

    if (number < 0) {
        PyErr_SetObject(PyExc_ValueError, obj);
        return NULL;    /* return NULL indicates error */
    }

    Py_RETURN_NONE;
}

/*
 * List of functions to add to halalml in exec_halalml().
 */
static PyMethodDef halalml_functions[] = {
    { "example", (PyCFunction)halalml_example, METH_VARARGS | METH_KEYWORDS, halalml_example_doc },
    { NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Initialize halalml. May be called multiple times, so avoid
 * using static state.
 */
int exec_halalml(PyObject *module) {
    PyModule_AddFunctions(module, halalml_functions);

    PyModule_AddStringConstant(module, "__author__", "MonishRamadoss");
    PyModule_AddStringConstant(module, "__version__", "1.0.0");
    PyModule_AddIntConstant(module, "year", 2020);

    return 0; /* success */
}

/*
 * Documentation for halalml.
 */
PyDoc_STRVAR(halalml_doc, "The halalml module");


static PyModuleDef_Slot halalml_slots[] = {
    { Py_mod_exec, exec_halalml },
    { 0, NULL }
};

static PyModuleDef halalml_def = {
    PyModuleDef_HEAD_INIT,
    "halalml",
    halalml_doc,
    0,              /* m_size */
    NULL,           /* m_methods */
    halalml_slots,
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit_halalml() {
    return PyModuleDef_Init(&halalml_def);
}
