#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "spkmeans.h"

#define assert__(x) for ( ; !(x) ; assert(x) )

static PyObject * spkmeans_WAMattrix(PyObject *self, PyObject *args);
static PyObject * spkmeans_BuildLap(PyObject *self, PyObject *args);
static PyObject * spkmeans_BuildDDG(PyObject *self, PyObject *args);
static PyObject * spkmeans_BuildJacobi(PyObject *self, PyObject *args);
static PyObject * spkmeans_eigengap(PyObject *self, PyObject *args);
static PyObject * spkmeans_BuildU(PyObject *self, PyObject *args);



static PyObject * spkmeans_WAMattrix(PyObject *self, PyObject *args)
{
    PyObject* obj;
    PyArrayObject *array;
    double** data;
    double** adj;
    int dim, i, dims[2];

    if (!PyArg_ParseTuple(args, "Oi", &obj, &dim)){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    data = (double**)PyArray_DATA(obj);
    Py_DECREF(obj);
    adj = (double**)malloc(dim * sizeof(double*));
    WAMatrix(data, dim, adj);
    free(data);
    dims[0] = dims[1] = dim;
    return PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, adj);
    
}

static PyObject * spkmeans_BuildLap(PyObject *self, PyObject *args)
{
    PyArrayObject *ddg, *wam, *lap;
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
    n=dims[0]=wam->dimensions[0];
	m=dims[1]=wam->dimensions[1];
    lap = (PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
    din=(double**)malloc((size_t) n * sizeof(double));
    assert__(din){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    a=(double *) ddg->data;  /* pointer to matin data as double */
	for ( i=0; i<n; i++)  {
		din[i]=a+i*m;  
    }
    win=(double**)malloc((size_t) n * sizeof(double));
    assert__(win){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    a=(double *) wam->data;  /* pointer to matin data as double */
	for ( i=0; i<n; i++)  {
		win[i]=a+i*m;  
    }
    mout=(double**)malloc(n * sizeof(double*));
    assert__(mout){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
	for ( i=0; i<n; i++)  {
		mout[i]=malloc(n * sizeof(double));
        assert__(mout){
            printf("An Error Has Occured");
            Py_INCREF(Py_None);
            return Py_None;
        }
    }
    mout = BuildLap(din, win, n);
    free((char*) din);
    free((char*) win);
    return PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, mout);
    
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
	n=dims[0]=matin->dimensions[0];
	m=dims[1]=matin->dimensions[1];	
	/* Make a new double matrix of same dims */
	matout=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
	/* Change contiguous arrays into C ** arrays (Memory is Allocated!) */
	cin=(double**)malloc((size_t) n * sizeof(double));
    assert__(cin){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    a=(double *) matin->data;  /* pointer to matin data as double */
	for ( i=0; i<n; i++)  {
		cin[i]=a+i*m;  
    }
    cout=(double**)malloc((size_t) n * sizeof(double));
    assert__(cin){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    a=(double *) matout->data; 
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
    int i, j, n, dims[2];
    PyObject* obj;
    PyArrayObject *ndarray;
    double **jacobi, **mat, *ev;

    if (!PyArg_ParseTuple(args, "iO!", &n, &PyArray_Type , &obj)){
        printf("An Error Has Occured");
        return Py_BuildValue("");
    }
    ndarray = PyArray_GETCONTIGUOUS(obj);
    assert__(ndarray){
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
            printf("An Error Has Occured");
            return Py_BuildValue("");
        }
        jacobi[i][i] = 1.0;
    }
    mat = (double**) ndarray->data;
    dims[0] = dim[1] = n;
    BuildJacobi(n,mat, jacobi);
    ev = (double*)malloc(n * sizeof(double));
    assert__(ev){
        printf("An Error Has Occured");
        return Py_BuildValue("");
    }
    for (i = 0; i < n; i++)
    {
        ev[i] = mat[i][i];
    }

    return  PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, jacobi);
}

static PyObject * spkmeans_eigengap(PyObject *self, PyObject *args)
{
    int dim, i, k;
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
    ev = (double*) ndarray->data;
    k = eigengap(dim, ev);
    return Py_BuildValue("i", k);
}

static PyObject * spkmeans_BuildU(PyObject *self, PyObject *args)
{
    PyArrayObject *jac, *ev, *u;
    double **jacobi, *eigV, **umat;
    double *a;
    int i, n, m, k;

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
    n = jac->dimensions[0];
    jacobi=(double**)malloc((size_t)( n * sizeof(double)));
    assert__(jacobi){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    a=(double *) jac->data; 
	for ( i=0; i<n; i++)  {
		jacobi[i]=a+i*m;  
    }
    eigV=(double*)malloc((size_t)( n * sizeof(double)));
    assert__(eigV){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    a=(double *) ev->data;  
	for ( i=0; i<n; i++)  {
		eigV[i]=a+i; 
    }
    eigsrt(eigV, jacobi, n);
    if(k == 0){
        k = eigengap(n, eigV);
    }
    umat = (double**)malloc((size_t)(n * sizeof(double)));
    assert__(umat){
        printf("An Error Has Occured");
        Py_INCREF(Py_None);
        return Py_None;
    }
    a=(double *) u->data;
    m = u->dimensions[1];  
	for ( i=0; i<n; i++)  {
		umat[i]=a+i*m; 
    }
    BuildU(n, k, jacobi, u);
    NormalizeU(n, k, umat);
    free((char*)jacobi);
    free((char*)eigV);
    free((char*)umat);
    Py_RETURN_NONE;


}


static PyMethodDef spkMethods[] ={
    {"WAMatrix",
        (PythonCFunction) spkmeans_WAMattrix,
        METH_VARARGS,
        PyDoc_STR("Calculate and build weighted adjacency matrix")
    },
    {"BuildDDG",
        (PythonCFunction) spkmeans_BuildDDG,
        METH_VARARGS,
        PyDoc_STR("Diagonal Degree Matrix")
    },
    {"BuildLap",
        (PythonCFunction) spkmeans_BuildLap,
        METH_VARARGS,
        PyDoc_STR("Normalized Laplacian")
    },
    {"BuildJacobi",
        (PythonCFunction) spkmeans_BuildJacobi,
        METH_VARARGS,
        PyDoc_STR("Jacobi matrix calculated on input matrix")
    },
    {"eigengap",
        (PythonCFunction) spkmeans_eigengap,
        METH_VARARGS,
        PyDoc_STR("Calculate and applies the eigengap heuristic")
    },
    {"BuildU",
        (PythonCFunction) spkmeans_BuildU,
        METH_VARARGS,
        PyDoc_STR("Calculate and applies the eigengap heuristic")
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef=
{
    PyModuleDef_HEAD_INIT,
    "spkmeans",
    NULL,
    -1,
    spkMethods
};

PyMODINIT_FUNC
PyInit_spkmeans(void){
    import_array();
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if(!m){
        return NULL;
    }
    return m;
}