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
#define MAXITER 300
#define MAXWID 1000
#define EPS 1e-15



int main(int argc, char **argv){
    int dim, cols, k;
    char *goal;
    double **data;
    if(argc < 4){
        printf("not enough parameters");
        return -1;
    }
    /* read user args */
    k = atoi(argv[1]);
    goal = argv[2];
    /*Read File */
    /* get dims of data*/
    cols = get_feature_num(argv[3]);
    dim = get_points_num(argv[3]);
    data = (double**)malloc(dim * sizeof(double*));
    assert__(data){
        printf("An Error Has Occured");
    }
    init_data_rows(data, dim, cols);
    read_csv_file(argv[3], data);
    perform(k, goal, data, dim, cols);
    free(data);
	return 0;
}

/* main Interface for the program,
does the required goal and is responsible for 
main operations of memory allocation / deallocation */
void perform(int k, char *goal, double **data, int dim, int features){
    int i, j;
    double **Adjacency, **DiagMat, **laplacian;
    double **jacobi, *ev, **cent, **u;
    if(strcmp(goal, "jacobi") == 0){  /*jacobi on input matrix only! */
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
        ev = (double*)malloc(dim * sizeof(double));
        assert__(ev){
            printf("An Error Has Occured");
        }
        for (i = 0; i < dim; i++)
        {
            ev[i] = data[i][i];
        }
        transpose(dim, jacobi);
        for (i = 0; i < dim; i++)
        {
            if(i != dim-1) printf("%.4f,", ev[i]);
            else printf("%.4f", ev[i]);
        }
        printf("\n");
        print_mat(jacobi, dim, dim);
        free(jacobi);
        free(ev);
        return;
    }
    Adjacency = (double**)calloc(dim , sizeof(double*));
    assert__(Adjacency){
        printf("An Error Has Occured");
    }
    init_data_rows(Adjacency, dim, dim);
    WAMatrix(data, dim, Adjacency, features);
    if(strcmp(goal, "wam") == 0){
        print_mat(Adjacency, dim, dim);
        free(Adjacency);
        return;
    }
    DiagMat = (double**)malloc(dim * sizeof(double*));
    assert__(DiagMat){
        printf("An Error Has Occured");
    }
    init_data_rows(DiagMat, dim, dim);
    BuildDDG(Adjacency, dim, DiagMat);
    if(strcmp(goal, "ddg") == 0){
        print_mat(DiagMat, dim, dim);
        free(DiagMat);
        free(Adjacency);
        return;
    }
    laplacian = BuildLap(DiagMat, Adjacency, dim);
    if(strcmp(goal, "lnorm") == 0){
        print_mat(laplacian, dim, dim);
        free(Adjacency);
        free(DiagMat);
        free(laplacian);
        return;
    }
    if (strcmp(goal, "spk") == 0)
    {
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
            free(Adjacency);
            free(DiagMat);
            BuildJacobi(dim, laplacian, jacobi);
            ev = (double*)malloc(dim * sizeof(double)); /* eigen values */
            assert__(ev){
                printf("An Error Has Occured");
            }
            for (i = 0; i < dim; i++)
            {
                ev[i] = laplacian[i][i];
            }
            eigsrt(ev, jacobi, dim);
            free(laplacian);
            if (k == 0){
                k = eigengap(dim, ev);
            }
            free(ev);
            u = (double**)malloc(dim * sizeof(double*));
            assert__(u){
                printf("An Error Has Occured");
            }
            BuildU(dim, k, jacobi, u);
            free(jacobi);
            NormalizeU(dim, k, u);
            cent = (double**)calloc(k, sizeof(double*));
            assert__(cent){
                printf("An Error Has Occured");
            }
            for (i = 0; i < k; i++)
            {
                cent[i] = calloc(k, sizeof(double));
                assert__(cent[i]){
                    printf("An Error Has Occured");
                }
                for (j = 0; j < k; j++)
                {
                    cent[i][j] = u[i][j];
                }
            }
            kmeans(u, cent, dim, k);
            free(u);
            print_mat(cent, k,k);
            free(cent);
    }
    else{
        printf("Invalid Input!");
        return;
    }
}

/* sort ev (eigenvalues) in ascending order, along with the 
corresponding eigenvectors in vec columns */
void eigsrt(double *ev, double **vec, int n)
{
    int k,j,i;
    double p;
    for (i=0;i<n;i++) {
        p = ev[i];
        k = i;
        for (j=i+1 ; j<n ; j++)
        if (ev[j] <= p){ p=ev[j];k = j;}
        if (k != i) {
            ev[k] = ev[i];
            ev[i] = p;
            for (j=0 ; j<n ; j++) {
                p = vec[j][i];
                vec[j][i] = vec[j][k];
                vec[j][k] = p;
            }
        }
    }
}

/* -convenience method- allocates memory for rows */
void init_data_rows(double** data, int rows, int cols){
    int i;
    assert(data);
    for (i = 0; i < rows; i++)
    {
        data[i] = (double*)malloc(cols * sizeof(double));
        assert(data[i]);
    }
}

/* -convenience method- prints matrix in proper form required */
void print_mat(double** mat, int dim, int cols){
    int i, j;
    for (i = 0; i < dim; i++){
        for (j = 0; j < cols; j++){
            if(j < cols - 1){
                printf("%.4f,", mat[i][j]);
            } else {
                printf("%.4f", mat[i][j]);
            }
        }
        printf("\n");
    }
}

/* calculate num of data columns in csv formatted file */
int get_feature_num(char* file){
    int col = 0;
    char buffer[MAXWID];
    char *value;
    FILE *fp;

    fp = fopen (file, "r");
    if (!fp) {
        printf ("An Error Has Occured");
        exit(-1);
    }
    if(fgets(buffer, MAXWID, fp)){
        value = strtok(buffer, ",");
        while (value) {
            value = strtok(NULL, ",");
            col++;
        }
    }
    fclose(fp);
    return col;
}

/* get num of data points in file */
int get_points_num(char* file){
    int row = 0;
    char buffer[MAXWID];
    FILE *fp;

    fp = fopen (file, "r");
    if (!fp) {
        printf("An Error Has Occured");
        exit(-1);
    }
    while(!feof(fp)) {
        if(fgets(buffer, MAXWID, fp)){
            row++;
        }
    }
    fclose(fp);
    return row;
}

/* Normalizes U elements (nxk) by row */
void NormalizeU(int dim, int k, double** u){
    int i, j;
    double linesum = 0.0, elem = 0.0;
    for (i = 0; i < dim; i++)
    {
        linesum = 0.0;
        for (j = 0; j < k; j++) /* find line sqrt of square sum */
        {
            linesum += pow(u[i][j], 2.0);
        }
        linesum = sqrt(linesum);
        for (j = 0; j < k; j++) /* normalize elements in row */
        {
            elem = u[i][j];
            u[i][j] = elem/linesum;
        }
    }
    
}

/* take first k vectors of jacobi into u */
void BuildU(int dim, int k, double** jac, double** u){
    int i, j;
    for (i = 0; i < dim; i++)
    {
        u[i] = (double*) malloc(k * sizeof(double));
        assert__(u[i]){
            printf("An Error Has Occured");
        }
        for (j = 0; j < k; j++)
        {
            u[i][j] = jac[i][j];
        }
    }
}

/* transpose an nxn matrix */ 
void transpose(int n, double** mat){
    double matij;
    int i,j;
    for(i = 0; i < n; i++){
        for(j = 0; j < i; j++){
            matij = mat[i][j];
            mat[i][j] = mat[j][i];
            mat[j][i] = matij;
        }
    }
}

/* performs eigengap heuristic on eigen-vals and returns k*/
int eigengap(int dim, double* eigenVals){
    double maxAbs = 0.0;
    int i, k=0;
    for (i = 0; i < dim-1; i++)
    {
        if (fabs(eigenVals[i] - eigenVals[i+1]) > maxAbs)
        {
            maxAbs = fabs(eigenVals[i] - eigenVals[i+1]);
            k = i;
        }
    }
    return k+1; /* indexing in heuristic starts at 1 */
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

/* returns array of doubles (for convenience) [max, i, j]
 where max = A[i][j] is largest off diagonal elem */
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

/* for chosen pivot A[i][j], perform rotation as described in algorithm */
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
    for(i=0; i < dim; i++){
        sumline = 0;
        for(j = 0; j <= dim; j++){
            sumline += Adj[i][j];
            if(i != j){
                diag[i][j] = 0.0;
            }
        }
        diag[i][i] = sumline;
    }
}

/* fills weighted adjacency matrix according to points in data*/
void WAMatrix(double** data, int dim, double** adj, int features){
    int i = 0, j;
    for(i=0; i < dim; i++){
        for(j = 0; j <= i; j++){
            if (i == j)
            {
                adj[i][i] = 0;
                continue;
            }
            adj[i][j] = CalcWeight(data[i], data[j], features);
            adj[j][i] = adj[i][j];
        }
    }

}

/* calculates weight between 2 given datapoints*/
double CalcWeight(double* point1, double* point2, int features){
    double dis_sq = 0;
    int i = 0;
    for(i = 0; i < features; i++){
        dis_sq += pow(point1[i] - point2[i], 2.0);
    }
    dis_sq = sqrt(dis_sq);
    return exp(-dis_sq/2.0);
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
        printf("An Error Has Ocurred");
        exit(-1);
    }

    while (fgets(buffer, MAXWID, fp)) {
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

/* distance between 2 points in R^dim */
double calcDistance(double *point1, double *point2, int dim){
    double dis = 0.0;
    int i;
    for(i = 0; i < dim; i++){
        dis+= pow(point1[i] - point2[i], 2.0);
    }
    return sqrt(dis);

}

/* index of minimum element in array of size dim */
int minIndx(double *arr, int dim){
    int i, j = 0;
    double min;
    min = *arr;
    for (i = 0; i < dim; i++)
    {
        if(arr[i] < min){
            j = i;
            min = arr[i];
        }
    }
    return j;
}

/* make zeroes matrix dim x dim */
void zeroes(double **mat, int dim){
    int i, j;
    for (i = 0; i < dim; i++)
    {
        for(j = 0; j < dim; j++){
            mat[i][j] = 0.0;
        }
    }
}

/* calculate mean for specified cluster */
void new_mean(double *new_cent, double **data, int dim, int k, int *classif, int m){
    int cnt = 0, i, j;
    double tmp = 0.0;
    for (i = 0; i < dim; i++)
    {
        if(classif[i] == m){
            cnt++;
            for (j = 0; j < k; j++)
            {
                new_cent[j] += data[i][j];
                
            }
        }
    }
    for(j = 0; j < k; j++){
        tmp = new_cent[j];
        new_cent[j] = tmp/cnt;
    }
}

/* update values of oldcents to match newcents (points of length k) */
void update_cents(double **newcents, double **oldcents, int k){
    int i, j;
    for (i = 0; i < k; i++)
    {
        for(j = 0; j < k; j++){
            oldcents[i][j] = newcents[i][j];
        }
    }
}

/* calculate difference between two centroids of length k */
double diff(double *p1, double *p2, int k){
    int i;
    int diff = 0.0;
    for (i = 0; i < k; i++)
    {
        diff += fabs(p1[i] - p2[i]);
    }
    return diff;
}

/* Given Data Points, calculate and return final centroids after
 clustering them with KMeans as in HW1 */
void kmeans(double **data, double **centroids, int dim, int k){
    double *distances, **new_cents;
    double dist = 0.0, delta = 0.0; 
    int *classification;
    int i, j,n;

    classification = (int*)calloc(dim, sizeof(int));
    assert__(classification){
        printf("An Error Has Occured");
    }
    new_cents = (double**)calloc(k, sizeof(double*));
    assert__(new_cents){
        printf("An Error Has Occured");
    }
    for(i = 0; i < k; i++){
        new_cents[i] = (double*)calloc(k, sizeof(double));
        assert__(new_cents[i]){
            printf("An Error Has Occured");
        }
    }
    distances = (double*)calloc(k, sizeof(double));
    assert__(distances){
        printf("An Error Has Occured");
    }

    
    for(i = 0; i < MAXITER; i++){ /* iterate until MAXITER or convergence */
        for(j = 0; j < dim; j++){ /* for all elements in data - classify */
            for(n = 0; n < k; n++){  /* distance from point j to each cluster */
                dist = calcDistance(centroids[n], data[j], k);
                distances[n] = dist;
            }
            classification[j] = minIndx(distances, k); /* index of cluster point j is closest to */
        }
        zeroes(new_cents, k);
        delta = 0.0;
        for (j = 0; j < k; j++)
        {
            new_mean(new_cents[j], data, dim, k, classification, j);
            delta += diff(new_cents[j], centroids[j], k);
        }
        update_cents(new_cents, centroids, k);
        if(delta == 0.0){
            free(new_cents);
            free(classification);
            free(distances);
            return;
        }
    }
}