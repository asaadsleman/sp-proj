#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "spkmeans.h"

#define assert__(x) for ( ; !(x) ; assert(x) )


/* ----------------- UTILITY FUNC - convert py list to c array ---------------------- */
double* parsePyArrayToC(PyObject *input, int dim){
    double *data;
    int i;
    data = (double*)malloc(dim * sizeof(double));
    assert__(data){
        printf("An Error Has Occured");
        return NULL;
    }
    for (i = 0; i < dim; i++)
    {
        data[i] = PyFloat_AsDouble(PyList_GetItem(input, i));
    }
    return data;
}

/* ----------------- UTILITY FUNC - convert py list to c matrix ---------------------- */
double** parsePyMatrixToC(PyObject *input, int dim, int features){
    PyObject *line;
    double **data;
    int i, j;
    data = (double**)malloc(dim * sizeof(double*));
    assert__(data){
        printf("An Error Has Occured");
        return NULL;
    }
    for (i = 0; i < dim; i++)
    {
        line = PyList_GetItem(input, i);
        data[i] = (double*)malloc(features * sizeof(double));
        assert__(data[i]){
            printf("An Error Has Occured");
            return NULL;
        }
        for (j = 0; j < features; j++)
        {
            data[i][j] = PyFloat_AsDouble(PyList_GetItem(line, j));
        }
    }
    return data;
}

/* ----------------- UTILITY FUNC - convert c matrix to py list  ---------------------- */
PyObject * parseCMatrixToPy(double **input, int dim, int features){
    PyObject *listoflists, *line;
    int i, j;
    listoflists = PyList_New(dim);
    for (i = 0; i < dim; i++)
    {
        line = PyList_New(features);
        for (j = 0; j < features; j++)
        {
            PyList_SetItem(line, j, Py_BuildValue("d", input[i][j]));
        }
        PyList_SetItem(listoflists, i, Py_BuildValue("O", line));
    }
    return Py_BuildValue("O", listoflists);
}

/* ----------------- UTILITY FUNC - convert c matrix to py list  ---------------------- */
PyObject * parseCArrayToPy(double *input, int dim){
    PyObject *list;
    int i;
    list = PyList_New(dim);
    for (i = 0; i < dim; i++)
    {
        PyList_SetItem(list, i, Py_BuildValue("d", input[i]));
    }
    return Py_BuildValue("O", list);
}


static PyObject * spkmeans_WAMattrix(PyObject *self, PyObject *args)
{
    PyObject *inputObj, *out;
    double** data;
    double** adj;
    int dim, features, i;

    if (!PyArg_ParseTuple(args, "Oii",&inputObj, &dim, &features)){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    data = parsePyMatrixToC(inputObj, dim, features);
    adj = (double**)malloc(dim * sizeof(double*));
    assert__(adj){
        free(data);
        printf("An Error Has Occured");
    }
    for (i = 0; i < dim; i++)
    {
        adj[i] = (double*) malloc(dim * sizeof(double));
        assert__(adj[i]){
            free(data);
            free(adj);
            printf("An Error Has Occured");
        }
    }
    WAMatrix(data, dim, adj, features);
    out = parseCMatrixToPy(adj, dim, dim);
    free(data);
    free(adj);
    return out;
}

static PyObject * spkmeans_BuildLap(PyObject *self, PyObject *args)
{
    PyObject *ddg, *wam, *lap;
	double **din, **win, **lout;
	int i, dim;
	/* Parse tuples separately since args will differ between C fcns */
	if (!PyArg_ParseTuple(args, "OOi", &ddg, &wam, &dim)) {
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    din=parsePyMatrixToC(ddg, dim, dim);
    win=parsePyMatrixToC(wam, dim, dim);
    lout=(double**)malloc(dim * sizeof(double*));
    assert__(lout){
        free(din);
        free(win);
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
	for ( i=0; i<dim; i++)  {
		lout[i]=malloc(dim * sizeof(double));
        assert__(lout[i]){
            printf("An Error Has Occured");
            Py_INCREF(Py_None);
            return Py_None;
        }
    }
    lout = BuildLap(din, win, dim);
    lap = parseCMatrixToPy(lout, dim, dim);
    free(din);
    free(lout);
    return lap;
}

static PyObject * spkmeans_BuildDDG(PyObject *self, PyObject *args)
{
    PyObject *wam, *ddg;
	double **win, **dout;
	int i, dim;
	/* Parse tuples separately since args will differ between C fcns */
	if (!PyArg_ParseTuple(args, "Oi", &wam, &dim)) {
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    win = parsePyMatrixToC(wam, dim, dim);
    dout = (double**)malloc(dim * sizeof(double*));
    assert__(dout){
        free(win);
        printf("An Error Has Occured");
    }
    for (i = 0; i < dim; i++)
    {
        dout[i] = (double*) malloc(dim * sizeof(double));
        assert__(dout[i]){
            printf("An Error Has Occured");
        }
    }
    BuildDDG(win, dim, dout);
    ddg = parseCMatrixToPy(dout, dim, dim);
    free(win);
    free(dout);
    return ddg;
}

static PyObject * spkmeans_BuildJacobi(PyObject *self, PyObject *args)
{
    PyObject *lapin, *evin, *jacout, *wrapper;
    int i, dim;
    double **jacobi, **lap;

    if (!PyArg_ParseTuple(args, "OOi", &lapin, &evin, &dim)){
        printf("An Error Has Occured");
    }
    jacobi = (double**)malloc(dim * sizeof(double*));
    assert__(jacobi){
        printf("An Error Has Occured");
    }
    for(i = 0; i < dim; i++){
        jacobi[i] = (double*)calloc(dim , sizeof(double));
        assert__(jacobi[i]){
            free(jacobi);
            printf("An Error Has Occured");
        }
        jacobi[i][i] = 1.0;
    }
    lap = parsePyMatrixToC(lapin, dim, dim);
    BuildJacobi(dim, lap, jacobi);
    for (i = 0; i < dim; i++)
    {
        PyList_SetItem(evin, i, Py_BuildValue("d", lap[i][i]));
    }
    wrapper = PyList_New(2);
    PyList_SetItem(wrapper, 0, Py_BuildValue("O", evin));
    jacout = parseCMatrixToPy(jacobi, dim, dim);
    PyList_SetItem(wrapper, 1, Py_BuildValue("O", jacout));
    free(jacobi);
    free(lap);
    return  Py_BuildValue("O", wrapper);
}

static PyObject * spkmeans_BuildU(PyObject *self, PyObject *args)
{
    PyObject *jacin, *evin, *uout;
    double **jacobi, *eigV, **umat;
    int i, dim, k;

    if (!PyArg_ParseTuple(args, "OOii", &jacin, &evin, &dim, &k)) {
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    jacobi=parsePyMatrixToC(jacin, dim, dim);
    eigV=parsePyArrayToC(evin, dim);
    eigsrt(eigV, jacobi, dim);
    if(k == 0){
        k = eigengap(dim, eigV);
    }
    umat = (double**)malloc(dim * sizeof(double*));
    assert__(umat){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    for (i = 0; i < dim; i++)
    {
        umat[i] = (double*)malloc(k * sizeof(double));
        assert__(umat[i]){
            printf("An Error Has Occured");
            Py_INCREF(Py_None);
            return Py_None;
        }
    }
    BuildU(dim, k, jacobi, umat);
    free(jacobi);
    NormalizeU(dim, k, umat);
    uout = parseCMatrixToPy(umat, dim, k);
    free(eigV);
    free(umat);
    return uout;
}

static PyObject * spkmeans_fit(PyObject *self, PyObject *args){
    PyObject *uin, *centin;
    double **umat, **centmat;
    int dim, k;
    if(!PyArg_ParseTuple(args, "OOii", &uin, &centin, &dim, &k)){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    umat = parsePyMatrixToC(uin, dim, k);
    centmat = parsePyMatrixToC(centin, k, k);
    kmeans(umat, centmat, dim, k);
    centin = parseCMatrixToPy(centmat, k, k);
    return centin;
}

static PyMethodDef spkMethods[] ={
    {"WAMatrix",
        (PyCFunction) spkmeans_WAMattrix,
        METH_VARARGS,
        PyDoc_STR("Calculate and build weighted adjacency matrix")
    },
    {"BuildDDG",
        (PyCFunction) spkmeans_BuildDDG,
        METH_VARARGS,
        PyDoc_STR("Diagonal Degree Matrix")
    },
    {"BuildLap",
        (PyCFunction) spkmeans_BuildLap,
        METH_VARARGS,
        PyDoc_STR("Normalized Laplacian")
    },
    {"BuildJacobi",
        (PyCFunction) spkmeans_BuildJacobi,
        METH_VARARGS,
        PyDoc_STR("Jacobi matrix calculated on input matrix")
    },
    {"BuildU",
        (PyCFunction) spkmeans_BuildU,
        METH_VARARGS,
        PyDoc_STR("Calculate and applies the eigengap heuristic")
    },
    {"fit",
        (PyCFunction) spkmeans_fit,
        METH_VARARGS,
        PyDoc_STR("Calculate and applies the eigengap heuristic")
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spkmeans",
    "module doc",
    -1,
    spkMethods
};

PyMODINIT_FUNC PyInit_spkmeans(void){
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if(!m){
        return NULL;
    }
    return m;
}