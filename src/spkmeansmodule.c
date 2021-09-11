#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "spkmeans.h"

#define assert__(x) for ( ; !(x) ; assert(x) )

/* Util prototypes */
double** parsePyMatrixToC(PyObject *input, int dim, int features);
PyObject * parseCMatrixToPy(double **input, int dim, int features);

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
    printf("parsed args\n");
    data = parsePyMatrixToC(inputObj, dim, features);
    printf("converted data to C\n");
    print_mat(data, dim, features);
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
    printf("created adj\n");
    WAMatrix(data, dim, adj, features);
    printf("ran c\n");
    out = parseCMatrixToPy(adj, dim, dim);
    printf("Parsed Back");
    free(data);
    free(adj);
    printf("Returning");
    return out;
}

static PyObject * spkmeans_BuildLap(PyObject *self, PyObject *args)
{
    PyArrayObject *ddg, *wam;
	double **din, **win, **mout, *a;
	int i,n,m, dims[2];
	/* Parse tuples separately since args will differ between C fcns */
	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &ddg, &PyArray_Type, &wam)) {
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
	if (NULL == ddg || NULL == wam)  {
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    n=dims[0]=PyArray_DIM(wam, 0);
	m=dims[1]=PyArray_DIM(wam, 1);
    din=(double**)malloc((size_t) n * sizeof(double));
    assert__(din){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    a=(double *) PyArray_DATA(ddg);  /* pointer to matin data as double */
	for ( i=0; i<n; i++)  {
		din[i]=a+i*m;  
    }
    win=(double**)malloc(n * sizeof(double*));
    assert__(win){
        free(din);
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    a=(double *) PyArray_DATA(wam);  /* pointer to matin data as double */
	for ( i=0; i<n; i++)  {
		win[i]=a+i*m;  
    }
    mout=(double**)malloc(n * sizeof(double*));
    assert__(mout){
        free(din);
        free(win);
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
	for ( i=0; i<n; i++)  {
		mout[i]=malloc(n * sizeof(double));
        assert__(mout){
            free(din);
            free(win);
            printf("An Error Has Occured");
            Py_INCREF(Py_None);
            return Py_None;
        }
    }
    mout = BuildLap(din, win, n);
    free((char*) din);
    free((char*) win);
    return PyArray_SimpleNewFromData(2, (intptr_t*) dims, NPY_DOUBLE, mout);
}

static PyObject * spkmeans_BuildDDG(PyObject *self, PyObject *args)
{
    PyArrayObject *matin, *matout;
	double **cin, **cout, *a;
	int i,n,m, dims[2];
	/* Parse tuples separately since args will differ between C fcns */
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &matin)) {
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
	if (NULL == matin)  {
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
	n = dims[0] = PyArray_DIM(matin, 0);
	m = dims[1] = PyArray_DIM(matin, 1);	
	/* Make a new double matrix of same dims */
	matout=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
	/* Change contiguous arrays into C ** arrays (Memory is Allocated!) */
	cin=(double**)malloc((size_t) n * sizeof(double));
    assert__(cin){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    a=(double *) PyArray_DATA(matin);  /* pointer to matin data as double */
	for ( i=0; i<n; i++)  {
		cin[i]=a+i*m;  
    }
    cout=(double**)malloc((size_t) n * sizeof(double));
    assert__(cout){
        free(cin);
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    a=(double *) PyArray_DATA(matout); 
	for ( i=0; i<n; i++)  {
		cout[i]=a+i*m;  
    }
    BuildDDG(cin, n, cout);
    free((char*) cin);
    free((char*) cout);
    return PyArray_Return(matout);
}

static PyObject * spkmeans_BuildJacobi(PyObject *self, PyObject *args)
{
    int i, n, dims[2];
    PyArrayObject *ndarray;
    double **jacobi, **mat, *ev;

    if (!PyArg_ParseTuple(args, "iO!", &n, &PyArray_Type , &ndarray)){
        printf("An Error Has Occured");
        return Py_BuildValue("");
    }
    jacobi = (double**)malloc(n * sizeof(double*));
    assert__(jacobi){
        printf("An Error Has Occured");
        return Py_BuildValue("");
    }
    for(i = 0; i < n; i++){
        jacobi[i] = (double*)calloc(n , sizeof(double));
        assert__(jacobi[i]){
            free(jacobi);
            printf("An Error Has Occured");
            return Py_BuildValue("");
        }
        jacobi[i][i] = 1.0;
    }
    mat = (double**) PyArray_DATA(ndarray);
    dims[0] = dims[1] = n;
    BuildJacobi(n, mat, jacobi);
    ev = (double*)malloc(n * sizeof(double));
    assert__(ev){
        free(jacobi);
        printf("An Error Has Occured");
        return Py_BuildValue("");
    }
    for (i = 0; i < n; i++)
    {
        ev[i] = mat[i][i];
    }
    return  PyArray_SimpleNewFromData(2, (intptr_t*) dims, NPY_FLOAT64, jacobi);
}

static PyObject * spkmeans_eigengap(PyObject *self, PyObject *args)
{
    int dim, k;
    PyArrayObject *ndarray;
    double *ev;
    if (!PyArg_ParseTuple(args, "iO!", &dim, &ndarray)){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if(NULL == ndarray){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    ev = (double*) PyArray_DATA(ndarray);
    k = eigengap(dim, ev);
    return Py_BuildValue("i", k);
}

static PyObject * spkmeans_BuildU(PyObject *self, PyObject *args)
{
    PyArrayObject *jac, *ev, *u;
    double **jacobi, *eigV, **umat;
    double *a, *a2;
    int i, n, m=0, k;

    if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &jac, &PyArray_Type, &ev, &PyArray_Type, &u, &k)) {
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (NULL == jac || NULL == ev || NULL == u)  {
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    n = PyArray_DIM(jac, 0);
    jacobi=(double**)malloc( n * sizeof(double*));
    assert__(jacobi){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    a=(double *) PyArray_DATA(jac); 
	for ( i=0; i<n; i++)  {
		jacobi[i]=a+i*m;  
    }
    eigV=(double*)malloc(n * sizeof(double));
    assert__(eigV){
        free(jacobi);
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    a2=(double*) PyArray_DATA(ev);  
	for ( i=0; i<n; i++)  {
		eigV[i]= *(a2+i); 
    }
    eigsrt(eigV, jacobi, n);
    if(k == 0){
        k = eigengap(n, eigV);
    }
    umat = (double**)malloc(n * sizeof(double*));
    assert__(umat){
        free(jacobi);
        free(eigV);
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    a=(double *) PyArray_DATA(u);
    m = PyArray_DIM(u, 1);  
	for ( i=0; i<n; i++)  {
		umat[i]=a+i*m; 
    }
    BuildU(n, k, jacobi, umat);
    NormalizeU(n, k, umat);
    free((char*)jacobi);
    free((char*)eigV);
    free((char*)umat);
    Py_RETURN_NONE;
}

static PyObject * spkmeans_fit(PyObject *self, PyObject *args){
    PyArrayObject *u, *cent;
    double **umat, **centmat, *line;
    int dim, k, i;
    if(!PyArg_ParseTuple(args, "O!O!ii",&PyArray_Type, &u, &PyArray_Type, &cent)){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    if(NULL == u || NULL == cent){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    dim = PyArray_DIM(u, 0);
    k = PyArray_DIM(u, 1);;
    umat = (double**)malloc(dim * sizeof(double*));
    assert__(umat){
        printf("An Error Has Occured");
    }
    line = (double*)PyArray_DATA(u);
    for(i = 0; i < dim; i++){
        umat[i] = line+i*k;
    }
    centmat = (double**) malloc(k  * sizeof(double*));
    assert__(centmat){
        free(umat);
        printf("An Error Has Occured");
    }
    line = (double*)PyArray_DATA(cent);
    for(i = 0; i < k; i++){
        centmat[i] = line+i*k;
    }
    kmeans(umat, centmat, dim, k);
    free(umat);
    free(centmat);
    return PyArray_Return(cent);
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
    {"eigengap",
        (PyCFunction) spkmeans_eigengap,
        METH_VARARGS,
        PyDoc_STR("Calculate and applies the eigengap heuristic")
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
    import_array();
    return m;
}