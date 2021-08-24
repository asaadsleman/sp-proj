#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <spkmeans.h>
#include <stdlib.h>
#include <stdio.h>


static PyObject * spkmeans_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}
