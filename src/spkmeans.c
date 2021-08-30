#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/types.h>
#include "spkmeans.h"

#define MAXLEN 1000
#define EPS 1e-15



int main(int argc, char **argv){
    int k, dim, i;
    char *fname, *goal;
    double **data, **Adjacency, **DiagMat, **laplacian, **jacobi, **u, **t, **ev;
    double *data[MAXLEN];
    if(argc < 4){
        printf("not enough parameters");
        return -1;
    }
    /* Read User Data */
    k = atoi(argv[1]);
    goal = argv[2];
    fname = argv[3];
    /* Read File */
    read_csv_file(fname, data);
    dim = sizeof(data) / sizeof(double *);
    Adjacency = WAMatrix(data, dim);
    if (strcmp(goal, "wam"))
    {
        return 1;
    }
    DiagMat = BuildDDG(Adjacency, dim);
    if(strcmp(goal, "ddg")){
        return 2;
    }
    laplacian = BuildLap(DiagMat, Adjacency, dim);
    if(strcmp(goal, "lnorm")){
        return 3;
    }
    jacobi = BuildJacobi(dim, laplacian);
    assert(jacobi != NULL);
    if(strcmp(goal, "jacobi")){
        return 4;
    }
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
    BuildU(dim, k, jacobi, u, ev);
    BuildT(dim, k, u, t);

	return 0;
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

void BuildU(int dim, int k, double** jac, double** u, double** eigVSrtd){
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
        diffs[i] = abs(eigenVals[i][0] - eigenVals[i+1][0]);
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

void correctMat(int dim,double** mat1, double** mat2, double **final){
    int i, j;
    for (i = 0; i < dim; i++)
    {
        for (j = 0; j < i; j++)
        {
            final[i][j] = mat2[i][j];
            final[j][i] = final[i][j];
        }
        final[i][i] = mat1[i][i];
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

double** BuildJacobi(int dim, double** mat){
    double* max;
    double** p, **diag, **jacobi;
    int limiter, i, j, k;
    p = buildID(dim);
    limiter = 5 * pow(dim,2);
    jacobi = (double**) malloc(dim*sizeof(double*));
    assert(jacobi != NULL);
    for (k = 0; k < dim; k++)
    {
        jacobi[k] = (double*) malloc(dim*sizeof(double));
        assert(jacobi[k] != NULL);
    }
    for (k = 0; k < limiter; k++)
    {
        max = offElem(dim, mat);
        if(max[0] < EPS){
            *diag = diagonal(mat, dim);
            correctMat(dim, diag, p, jacobi);
            return jacobi;
        }
        rotate(dim, p, mat, i, j);
    }
    printf("Jacobi did not converge!");
    return NULL;
}

double** buildID(int N){
    int i;
    double** ID;
    ID = (double**)malloc(N*sizeof(double*));
    assert(ID);
    for(i = 0; i < N; i++){
        ID[i] = (double*)calloc(N, sizeof(double));
        assert(ID[i] != NULL);
        ID[i][i] = 1.0;
    }
    return ID;
}

double* offElem(int dim, double** mat){
    double max, max_i, max_j;
    double* res;
    int i, j;
    max = mat[0][1];
    for (i = 0; i < dim; i++)
    {
        for (j = 0; j < dim; j++)
        {
            if (mat[i][j] > max && i != j)
            {
                max = mat[i][j];
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

int checkDiag(double **mat, int dim){
    int i,j;
    for (i = 0; i < dim; i++)
    {
        for (j = 0; j < i; j++)
        {
            if (mat[i][j] != 0 || mat[j][i] != 0)
            {
                return 0;
            } 
        }
    }
    return 1;
}

void rotate(int dim, double** p, double** mat, int i, int j){
    double matDiff, phi, t, c, s, tau, temp;
    int q,m,r;
    matDiff = mat[j][j] - mat[i][i];
    if (abs(mat[i][j]) < (abs(matDiff)*EPS))
    {
        t = mat[i][j] / matDiff;
    }
    else{
        phi = matDiff/(2.0 * mat[i][j]);
        t = 1.0/(abs(phi) + sqrt(2.0*phi + 1.0));
        if (phi < 0.0)
        {
            t = -t;
        }
    }
    c = 1.0 / (sqrt(pow(phi, 2) + 1.0));
    s = t*c;
    tau = s/(1.0 + c);
    temp = mat[i][j];
    mat[i][j] = 0.0;
    mat[i][i] = mat[i][i] - t*temp;
    mat[j][j] = mat[j][j] + t*temp;
    for (q = 0; q < i; q++)
    {
        temp = mat[q][i];
        mat[q][i] = temp - s*(mat[q][j] + tau*temp);
        mat[q][j] = mat[q][j] + s*(temp - tau*mat[q][j]);
    }
    for (m = i+1; m < j; m++)
    {
        temp = mat[i][m];
        mat[i][m] = temp - s*(mat[m][j] + tau*mat[i][m]);
        mat[m][j] = mat[m][j] + s*(temp - tau*mat[m][j]);
    }
    for (r = j+1; r < dim; r++)
    {
        temp = mat[i][r];
        mat[i][r] = temp - s*(mat[j][r] + tau*temp);
        mat[j][r] = mat[j][r] + s*(temp - tau*mat[j][r]);
    }
    for (q = 0; q < dim; q++)
    {
        temp = p[q][i];
        p[q][i] = temp - s*(p[q][j] + tau*p[q][i]);
        p[q][j] = p[q][j] + s*(temp - tau*p[q][j]);
    }    
}

double** BuildLap(double** Diag, double** Adj, int dim){
    double **Laplacian, **Id;
    double **Mid, **Mid2;
    Id = buildID(dim);
    Mid = multiply(dim, Diag, Adj);
    Mid2 = multiply(dim, Mid, Diag);
    subtract(dim, Id, Mid2, Laplacian);
    return Laplacian;

}

void subtract(int dim, double** A, double** B, double** res){
    int i, j;
    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            res[i][j] = 0;
            res[i][j] = A[i][j] - B[i][j];
    }
}
}

double** multiply(int dim, double** A, double** B){
    double **res;
    int i, j, k;
    res = (double**)malloc(dim * sizeof(double*));
    assert(res);
    for (i = 0; i < dim; i++) {
        res[i] = (double*)malloc(dim * sizeof(double));
        assert(res[i]);
        for (j = 0; j < dim; j++) {
            res[i][j] = 0;
            for (k = 0; k < dim; k++){
                res[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

double** BuildDDG(double** Adj, int dim){
    int  i = 0, j;
    double **diag;
    double sumline;
    diag = (double**)malloc(dim * sizeof(double*));
    assert(diag != NULL);
    for(i=0; i < dim; i++){
        diag[i] = (double*)malloc(dim * sizeof(double));
        assert(diag[i] != NULL);
        sumline = 0;
        for(j = 0; j <= dim; j++){
            sumline += Adj[i][j];
            if(i != j){
                diag[i][j] = 0;
            }
        }
        diag[i][i] = 1/ sqrt(sumline);
    }
    return diag;
}

double** WAMatrix(double** data, int dim){
    int i = 0, j;
    double **AdMat;
    AdMat = (double**)malloc(dim * sizeof(double *));
    assert(AdMat != NULL);
    for(i=0; i < dim; i++){
        AdMat[i] = (double*)malloc(dim * sizeof(double));
        assert(AdMat[i] != NULL);
        for(j = 0; j <= i; j++){
            if (i == j)
            {
                AdMat[i][i] = 0;
                continue;
            }
            AdMat[i][j] = CalcWeight(data[i], data[j]);
            AdMat[j][i] = AdMat[i][j];
        }
    }
    return AdMat;

}

double CalcWeight(double* point1, double* point2){
    double dis_sq = 0;
    int i = 0, len;
    len = sizeof(point1)/sizeof(double);
    for(i = 0; i < len; i++){
        dis_sq += sqrt(pow(point1[i] - point2[i], 2));
    }
    return exp(-dis_sq/2);
}

void read_csv_file(char *filename, double** array){
    int idx = 0;
    int j = 0;
    char *buffer = NULL;
    size_t len = 0;
    ssize_t read;
    char *ptr = NULL;

    FILE *fp;
    fp = fopen (filename, "r");
    if (!fp) {
        fprintf (stderr, "failed to open file for reading\n");
        exit(-1);
    }

    while ((read = getline(&buffer, &len, fp)) != -1) {
        array[idx] = (double*)malloc (sizeof (array));
        assert(array[idx]);
        for (j = 0, ptr = buffer; j < MAXLEN; j++, ptr++){
            array [idx][j] = (double)strtol(ptr, &ptr, 10);
        }
        idx++;
    }

    fclose (fp);
}
