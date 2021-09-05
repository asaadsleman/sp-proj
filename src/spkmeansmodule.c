#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include "spkmeans.h"




static PyObject * spkmeans_WAMattrix(PyObject *self, PyObject *args)
{
    PyObject* obj;
    double** data;
    double** adj;
    int dim, i, dims[2];

    if (!PyArg_ParseTuple(args, "Oi", &obj, &dim)){
        printf("can't parse arguments - WAMatrix")
        return NULL;
    }
    data = (double**)PyArray_DATA(obj);
    Py_DECREF(obj);
    adj = (double**)malloc(dim * sizeof(double*));
    WAMatrix(data, dim, adj);
    free(data);
    dims[0] = dims[1] = dim;
    return PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, adj);
    
}

static PyObject * spkmeans_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}
