#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/types.h>
#include "spkmeans.h"

/* some pre processing defs */

#define assert__(x) for ( ; !(x) ; assert(x) )
#define MAXLEN 50
#define MAXWID 100
#define EPS 1e-15



int main(int argc, char **argv){
    int dim, cols, i; /*, k;*/
    char *fname, *goal;
    double **data, **Adjacency, **DiagMat, **laplacian, **jacobi; /*,, **u, **t, */
    if(argc < 4){
        printf("not enough parameters");
        return -1;
    }
    /* Read User Data
    k = atoi(argv[1]);
    */
    goal = argv[2];
    fname = argv[3];
    /*Read File */
    /* get dims of data*/
    cols = get_feature_num(fname);
    dim = get_points_num(fname);
    printf("cols: %d dim: %d\n", cols, dim);
    data = (double**)malloc(dim * sizeof(double*));
    assert__(data){
        printf("An Error Has Occured");
    }
    init_data_rows(data, dim, cols);
    read_csv_file(fname, data);
    if(strcmp(goal, "jacobi") == 0){
        jacobi = (double**)malloc(dim * sizeof(double*));
        assert(jacobi);
        for (i = 0; i < dim; i++) /* initialize as ID n x n */
        {
            jacobi[i] = (double*)calloc(dim , sizeof(double));
            assert__(jacobi[i]){
                printf("An Error Has Occured");
            }
            jacobi[i][i] = 1.0;
        }
        BuildJacobi(dim, data, jacobi);
        for (i = 0; i < dim; i++)
        {
            printf("ev_%d : %f\n", i, data[i][i]);
        }
        return 5;
    }
    Adjacency = (double**)malloc(dim * sizeof(double*));
    assert__(Adjacency){
        printf("An Error Has Occured");
    }
    init_data_rows(Adjacency, dim, dim);
    WAMatrix(data, dim, Adjacency);
    if(strcmp(goal, "wam") == 0){
        print_mat(Adjacency, dim, dim);
        return 2;
    }
    DiagMat = (double**)malloc(dim * sizeof(double*));
    assert__(DiagMat){
        printf("An Error Has Occured");
    }
    init_data_rows(DiagMat, dim, dim);
    BuildDDG(Adjacency, dim, DiagMat);
    if(strcmp(goal, "ddg") == 0){
        print_mat(DiagMat, dim, dim);
        return 3;
    }
    laplacian = BuildLap(DiagMat, Adjacency, dim);
    if(strcmp(goal, "lnorm") == 0){
        print_mat(laplacian, dim, dim);
        return 4;
    }
    jacobi = (double**)malloc(dim * sizeof(double*));
    assert__(jacobi){
        printf("An Error Has Occured");
    }
    for (i = 0; i < dim; i++)
    {
        jacobi[i] = (double*)malloc(dim * sizeof(double));
        assert__(jacobi[i]){
            printf("An Error Has Occured");
        }
        jacobi[i][i] = 1.0;
    }
    BuildJacobi(dim, laplacian, jacobi);
    if(strcmp(goal, "jacobi") == 0){
        return 4;
    }
    /*
    ev = (double**)malloc(dim*sizeof(double*));
    assert(ev != NULL);
    for (i = 0; i < dim; i++)
    {
        ev[i] = (double*)malloc(2 * sizeof(double));
        assert(ev[i] != NULL);
        ev[i][0] = jacobi[i][i];
        ev[i][1] = i;
    }
    qsort(ev, dim, sizeof(double), cmp);
    if (k == 0){
        k = eigengap(dim, ev);
    }
    u = (double**)malloc(dim * sizeof(double*));
    assert(u);
    BuildU(dim, k, jacobi, u);
    t = (double**)malloc(dim * sizeof(double*));
    assert(t);
    BuildT(dim, k, u, t);
    */

	return 0;
}

void init_data_rows(double** data, int rows, int cols){
    int i;
    assert(data);
    for (i = 0; i < rows; i++)
    {
        data[i] = (double*)malloc(cols * sizeof(double));
        assert(data[i]);
    }
}

void print_mat(double** mat, int dim, int cols){
    int i, j;
    for (i = 0; i < dim; i++){
        for (j = 0; j < cols; j++){
        printf("%f ", mat[i][j]);
        }
        printf("\n");
    }
}

int get_feature_num(char* file){
    int col = 0;
    char buffer[MAXWID];
    char *value;
    FILE *fp;

    fp = fopen (file, "r");
    if (!fp) {
        fprintf (stderr, "failed to open file for reading\n");
        exit(-1);
    }
    if(fgets(buffer, MAXWID+1, fp)){
        value = strtok(buffer, ",");
        while (value) {
            value = strtok(NULL, ",");
            col++;
        }
    }
    fclose(fp);
    return col;
}

int get_points_num(char* file){
    int row = 0;
    char buffer[MAXWID];
    FILE *fp;

    fp = fopen (file, "r");
    if (!fp) {
        fprintf (stderr, "failed to open file for reading\n");
        exit(-1);
    }
    while(!feof(fp)) {
        if(fgets(buffer, MAXWID+1, fp)){
            row++;
        }
    }
    fclose(fp);
    return row;
}

void BuildT(int dim, int k, double** u, double** t){
    int i, j;
    double *lineSums;
    double sum = 0.0;
    lineSums = (double*)malloc(dim* sizeof(double));
    assert(lineSums);
    t = (double**) malloc(dim * sizeof(double*));
    assert(t);
    for(i = 0; i < dim ; i++){
        sum = 0.0;
        for (j = 0; j < k; j++)
        {
            sum += pow(u[i][j], 2.0);
        }
        lineSums[i] = sqrt(sum);
    }
    for (i = 0; i < dim; i++)
    {
        t[i] = (double*) malloc(k * sizeof(double));
        assert(t[i] != NULL);
        for (j = 0; j < k; j++)
        {
            t[i][j] = u[i][j]/lineSums[i];
        }
    }
    free(lineSums);
}

void BuildU(int dim, int k, double** jac, double** u){
    int i, j;
    u = (double**) malloc(k * sizeof(double*));
    assert(u);
    transpose(dim, dim, jac);
    for (i = 0; i < k; i++)
    {
        u[i] = (double*) malloc(dim* sizeof(double));
        assert(u[i]);
        for (j = 0; j < dim; j++)
        {
            u[i] = jac[i];
        }
    }
    transpose(k, dim, u);
}

void transpose(int up, int side, double** mat){
    double** temp;
    int k,m;
    temp = (double**) malloc(side * sizeof(double*));
    assert(temp != NULL);
    for(k = 0; k< side; k++){
        temp[k] = (double*) malloc(up*sizeof(double));
        assert(temp[k] != NULL);
        for(m = 0; m < up; m++){
            temp[k][m] = mat[m][k];
        }
    }
    mat = temp;
}

int eigengap(int dim, double** eigenVals){
    double *diffs, maxAbs;
    int i, max_i;
    diffs = (double*)malloc((dim-1)*sizeof(double));
    assert(diffs);
    for (i = 0; i < dim-1; i++)
    {
        diffs[i] = fabs(eigenVals[i][0] - eigenVals[i+1][0]);
        if (diffs[i] > maxAbs)
        {
            maxAbs = diffs[i];
            max_i = i;
        }
    }
    free(diffs);
    return max_i;
}

int cmp(const void *x, const void *y)
{
  double dx = *(double*)x, dy = *(double*)y;
  if (dx < dy) return -1;
  if (dx > dy) return  1;
  return 0;
}

/* apply jacobi algorithm as detailed in assignment*/
void BuildJacobi(int dim, double** mat, double** jacobi){
    int iterations = 0, maxiter = 100;
    double *maxnondiag;
    maxnondiag = offdiag(mat, dim);
    while ( maxnondiag[0] > EPS && iterations < maxiter)
    {
        maxnondiag = offdiag(mat, dim);
        Jacobi_rotate(mat, jacobi, (int)maxnondiag[1], (int)maxnondiag[2], dim);
        iterations++;
    }
}

double* offdiag(double** A, int n)
{
    int i, j;
    double *max, aij;
    max = (double*) calloc(3, sizeof(double));
    assert__(max){
        printf("An Error Has Occured");
    }
    for (i = 0; i < n; ++i)
    {
        for (j = i+1; j < n; ++j)
        {
            aij = fabs(A[i][j]);
            if ( aij > max[0])
            {
                max[0] = aij;
                max[1] = (double)i;
                max[2] = (double)j;
            }
        }
    }
    return max;
}

void Jacobi_rotate ( double **A, double **R, int k, int l, int n )
{
    int i;
    double s, c, t, tau;
    double a_kk, a_ll, a_ik, a_il, r_ik, r_il;
    if ( A[k][l] != 0.0 ) {
        tau = (A[l][l] - A[k][k])/(2.0*A[k][l]);
        if ( tau >= 0.0 ) {
            t = 1.0/(tau + sqrt(1.0 + tau*tau));
        } else {
            t = -1.0/(-tau +sqrt(1.0 + tau*tau));
        }
        c = 1/sqrt(1+t*t);
        s = c*t;
    }
     else {
        c = 1.0;
        s = 0.0;
    }
    a_kk = A[k][k];
    a_ll = A[l][l];
    A[k][k] = c*c*a_kk - 2.0*c*s*A[k][l] + s*s*a_ll;
    A[l][l] = s*s*a_kk + 2.0*c*s*A[k][l] + c*c*a_ll;
    A[k][l]= 0.0; 
    A[l][k] = 0.0;
    for (i = 0; i < n; i++ ) {
        if ( i != k && i != l ) {
            a_ik = A[i][k];
            a_il = A[i][l];
            A[i][k] = c*a_ik - s*a_il;
            A[k][i] = A[i][k];
            A[i][l] = c*a_il + s*a_ik;
            A[l][i] = A[i][l];
        }
        /* the eigenvecs */
        r_ik = R[i][k];
        r_il = R[i][l];
        R[i][k] = c*r_ik - s*r_il;
        R[i][l] = c*r_il + s*r_ik;
    }
    return;
}

/* build laplacian normalized matrix */
double** BuildLap(double** Diag, double** Adj, int dim){
    int i;
    /*sqrt of diag*/
    for (i = 0; i < dim; i++)
    {
        Diag[i][i] = 1/ sqrt(Diag[i][i]);
    }
    multiply_diag_by(dim, Diag, Adj);
    multiply_by_Diag(dim, Adj, Diag);
    eye_minus(dim, Adj);
    return Adj;
}

/* calculates (I - mat) where I is identity matrix */
void eye_minus(int dim, double** mat){
    int i, j;
    double tmpVal = 0.0;
    for (i = 0; i < dim; i++)
    {
        for (j = 0; j < dim; j++)
        {
            if (i == j)
            {
                tmpVal = 1 - mat[i][i];
                mat[i][i] = tmpVal;
            } else{
                tmpVal = -mat[i][j];
                mat[i][j] = tmpVal;
            }
        }
    }
}

/* multiply A @ B where A is diagonal*/
void multiply_diag_by(int dim, double** A, double** B){
    int i, j;
    double tmpVal = 0.0;
    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            tmpVal = A[i][i] * B[i][j];
            B[i][j] = tmpVal;
        }
    }
}

/* multiply A @ B where B is diagonal*/
void multiply_by_Diag(int dim, double** A, double** B){
    int i, j;
    double tmpVal = 0.0;
    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            tmpVal = A[i][j] * B[j][j];
            A[i][j] = tmpVal;
        }
    }
}

/* builds the diagonal degree matrix from the weighted adjacency mat*/
void BuildDDG(double** Adj, int dim, double **diag){
    int  i = 0, j;
    double sumline;
    assert(diag);
    for(i=0; i < dim; i++){
        assert(diag[i]);
        sumline = 0;
        for(j = 0; j <= dim; j++){
            sumline += Adj[i][j];
            if(i != j){
                diag[i][j] = 0;
            }
        }
        diag[i][i] = sumline;
    }
}

/* fills weighted adjacency matrix according to points in data*/
void WAMatrix(double** data, int dim, double** adj){
    int i = 0, j;
    assert(adj);
    for(i=0; i < dim; i++){
        assert(adj[i]);
        for(j = 0; j <= i; j++){
            if (i == j)
            {
                adj[i][i] = 0;
                continue;
            }
            adj[i][j] = CalcWeight(data[i], data[j]);
            adj[j][i] = adj[i][j];
        }
    }

}

/* calculates weight between 2 given datapoints*/
double CalcWeight(double* point1, double* point2){
    double dis_sq = 0;
    int i = 0, len;
    len = sizeof(point1)/sizeof(double);
    for(i = 0; i < len; i++){
        dis_sq += sqrt(pow(point1[i] - point2[i], 2));
    }
    return exp(-dis_sq/2);
}

/* returns a double matrix of data points from filename*/
void read_csv_file(char *filename, double** data){
    int row = 0;
    int col = 0;
    double elem = 0.0;
    char buffer[MAXWID];
    char *value;
    FILE *fp;

    fp = fopen (filename, "r");
    if (!fp) {
        fprintf (stderr, "failed to open file for reading\n");
        exit(-1);
    }

    while (fgets(buffer, MAXWID + 1, fp)) {
            col = 0;
            row++;
            /* Splitting the data */
            value = strtok(buffer, ",");
            while (value) {
                elem = atof(value);
                data[row-1][col] = elem;
                value = strtok(NULL, ",");
                col++;
            }
        }
        fclose(fp);
}
