#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/types.h>
#include "spkmeans.h"

/* some pre processing defs */

#define MAXLEN 1000
#define MAXWID 100
#define EPS 1e-15



int main(int argc, char **argv){
    int dim, cols, i; /*, k;*/
    char *fname, *goal;
    double **data, **Adjacency, **DiagMat, **laplacian, **jacobi; /*,, **u, **t, **ev; */
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
    assert(data);
    init_data_rows(data, dim, cols);
    read_csv_file(fname, data);
    Adjacency = (double**)malloc(dim * sizeof(double*));
    assert(Adjacency);
    init_data_rows(Adjacency, dim, dim);
    WAMatrix(data, dim, Adjacency);
    if(strcmp(goal, "wam") == 0){
        print_mat(Adjacency, dim, dim);
        return 2;
    }
    DiagMat = (double**)malloc(dim * sizeof(double*));
    assert(DiagMat);
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
    assert(jacobi);
    for (i = 0; i < dim; i++)
    {
        jacobi[i] = (double*)malloc(dim * sizeof(double));
        assert(jacobi[i]);
        jacobi[i][i] = 1.0;
    }
    BuildJacobi(dim, laplacian, jacobi);
    if(strcmp(goal, "jacobi") == 0){
        print_jac(jacobi, dim, dim);
        printf("fin jac");
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

void print_jac(double** mat, int dim, int cols){
    int i, j;
    for (i = 0; i < dim; i++){
        break;
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

void correctMat(int dim, double* arr, double** mat, double **final){
    int i, j;
    for (i = 0; i < dim; i++)
    {
        for (j = 0; j < i; j++)
        {
            final[i][j] = mat[i][j];
            final[j][i] = final[i][j];
        }
        final[i][i] = arr[i];
    }
}

double* diagonal(double** mat, int dim){
    int i;
    double* res;
    res = (double*)malloc(dim * sizeof(double));
    assert(res);
    for (i = 0; i < dim; i++)
    {
        res[i] = mat[i][i];   
    }
    return res;
    
}

/* apply jacobi algorithm as detailed in assignment*/
void BuildJacobi(int dim, double** lap, double** jacobi){
    double sum, sumNew = 0.0;
    double *max;
    int i, j, k, max_it;
    i = 0;
    j = 0;
    max_it = 300;
    sum = offDiagSum(lap, dim);
    /*repeat unntil max_it*/
    for (k = 0; k < max_it; k++)
    {
        max = offElem(dim, lap);
        printf("sum : %f\n", sum);
        printf("max : %f\n", *max);
        i = (int)max[1];
        j = (int)max[2];
        printf("i : %d   j : %d \n", i, j);
        /* in case of convergence*/
        if(fabs(sumNew-sum)<EPS){
            printf("reached conv.\n");
            return;
        }
        /*Jacobi rotation movement*/
        rotate(dim, jacobi, lap, i, j);
        sum = sumNew;
        sumNew = offDiagSum(lap, dim);
        printf("rotation num: %d\n",k);
    }
    printf("no conversion - JACOBI ERROR\n");
}

/* calculate off-diagonal elements' sum */
double offDiagSum(double** mat, int dim){
    double sum = 0.0, curr = 0.0;
    int i,j;
    for ( i = 0; i < dim; i++)
    {
        for ( j = 0; j < i; j++)
        {
            curr = pow(fabs(mat[i][j]), 2.0);
            sum += curr;
        }
    }
    return sum;
}

/* find and return maximum (absolute value) non-diagonal element in mat and it's coordinates (i,j)*/
double* offElem(int dim, double** mat){
    double max, max_i, max_j;
    double* res;
    int i, j;
    max = 0.0;
    for (i = 0; i < dim; i++)
    {
        for (j = 0; j < i; j++)
        {
            if (fabs(mat[i][j]) > max)
            {
                max = fabs(mat[i][j]);
                max_i = (double)i;
                max_j = (double)j;
            }
        }
    }
    res = (double*)malloc(3 * sizeof(double));
    assert(res);
    res[0] = max;
    res[1] = max_i;
    res[2] = max_j;
    return  res;
}

/* return sign of x  --- sign(0) = 1.0 */
double sign(double x){
    if(x >= 0.0){
        return 1.0;
    }
    return -1.0;
}

/* applies one rotation to mat in jacobi algorithm*/
void rotate(int dim, double** p, double** mat, int i, int j){
    double phi, t, c, s, tau, temp, temp1, temp2;
    int r;
    phi = (mat[j][j] - mat[i][i])/(2.0 * mat[i][j]);
    t = sign(phi)/(fabs(phi) + sqrt(pow(phi, 2.0) + 1.0));
    c = 1.0 / (sqrt(pow(t, 2) + 1.0));
    s = t*c;
    tau = s/(1.0 + c);
    temp = mat[i][j];
    mat[i][j] = 0.0;
    mat[j][i] = 0.0;
    for (r=0; r<dim ; r++)   /* A'[r][i] */
    {
        if(r != i && r != j){
            temp1 = mat[r][i];
            temp2 = mat[r][j];
            mat[r][i] = (c * temp1) -(s * mat[r][j]);
            mat[r][j] = (c * temp2) + (s * temp1);
            mat[i][r] = mat[r][i];
            mat[j][r] = mat[r][j];
        }
    }
    temp1 = mat[i][i];
    temp2 = mat[j][j];
    mat[i][i] = (c * c * temp1) + (s * s * temp2) - (2.0 * s * c * temp);
    mat[j][j] = (s * s * temp1) + (c * c * temp2) + (2.0 * s * c * temp);
    for (r = 0; r < dim; r++) /*update p */
    {
        temp = p[r][i];
        p[r][i] = temp - (s*(p[r][j] + tau * p[r][i]));
        p[r][j] = p[r][j] + (s*(temp - tau*p[r][j]));
        p[i][r] = p[r][i];
        p[j][r] = p[r][j];
    }
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
