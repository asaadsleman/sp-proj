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
    int k, dim, i;
    char *fname, *goal;
    double **Adjacency, **DiagMat, **laplacian, **jacobi, **u, **t, **ev;
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
    data = (double**)malloc(MAXLEN * sizeof(double*));
    assert(data);
    dim = read_csv_file(fname, &data);
    Adjacency = WAMatrix(&data, dim);
    if (strcmp(goal, "wam"))
    {
        print_mat(&Adjacency, dim);
        return 1;
    }
    /* DiagMat = BuildDDG(Adjacency, dim);
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
    u = (double**)malloc(dim * sizeof(double*));
    assert(u);
    BuildU(dim, k, jacobi, u);
    t = (double**)malloc(dim * sizeof(double*));
    assert(t);
    BuildT(dim, k, u, t);
    */

	return 0;
}

void print_mat(double** mat, int dim){
    int i, j;
    for (i = 0; i < m; i++){
        for (j = 0; j < N; j++)
        printf("%d ", arr[i][j]);
        printf("\n");
    }
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

double** BuildJacobi(int dim, double** mat){
    double* max;
    double** p, *diag, **jacobi;
    int i, j, k;
    i = 0;
    j = 0;
    p = (double**)malloc(dim * sizeof(double*));
    assert(p);
    buildID(dim, p);
    jacobi = (double**) malloc(dim*sizeof(double*));
    assert(jacobi);
    diag = (double*) malloc(dim * sizeof(double));
    assert(diag);
    for (k = 0; k < dim; k++)
    {
        jacobi[k] = (double*) malloc(dim*sizeof(double));
        assert(jacobi[k] != NULL);
    }
    for (k = 0; k < 100; k++)
    {
        max = offElem(dim, mat);
        if(max[0] < EPS){
            diag = diagonal(mat, dim);
            correctMat(dim, diag, p, jacobi);
            return jacobi;
        }
        rotate(dim, p, mat, i, j);
    }
    printf("Jacobi did not converge!");
    return NULL;
}

void buildID(int N, double** res){
    int i;
    res = (double**)malloc(N*sizeof(double*));
    assert(res);
    for(i = 0; i < N; i++){
        res[i] = (double*)calloc(N, sizeof(double));
        assert(res[i] != NULL);
        res[i][i] = 1.0;
    }
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
    Laplacian = (double**)malloc(dim * sizeof(double*));
    assert(Laplacian);
    Id = (double**)malloc(dim * sizeof(double*));
    assert(Id);
    Mid = (double**)malloc(dim * sizeof(double*));
    assert(Mid);
    Mid2 = (double**)malloc(dim * sizeof(double*));
    assert(Mid2);
    buildID(dim, Id);
    multiply(dim, Diag, Adj, Mid);
    multiply(dim, Mid, Diag, Mid2);
    subtract(dim, Id, Mid2, Laplacian);
    free(Id);
    free(Mid2);
    free(Mid);
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

void multiply(int dim, double** A, double** B, double** res){
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

int read_csv_file(char *filename, double** data){
    int row = 0;
    int col = 0;
    int colCount = 0;
    double elem = 0.0;
    char buffer[MAXWID];
    FILE *fp;

    fp = fopen (filename, "r");
    if (!fp) {
        fprintf (stderr, "failed to open file for reading\n");
        exit(-1);
    }

    while (fgets(buffer, MAXWID, fp)) {
            /* first line */
            if(row == 0){
                char* cntr = strtok(buffer, ", ");
                while (cntr) {
                value = strtok(NULL, ", ");
                colCount++;
                }
            }
            col = 0;
            row++;
            data = (double**)realloc(data, row * sizeof(double*));
            assert(data);
            data[row-1] = (double*)malloc(colCount * sizeof(double));
            assert(data[row-1]);
            /* Splitting the data */
            char* value = strtok(buffer, ", ");
            while (value) {
                elem = atof(value);
                data[row-1][col] = elem;
                value = strtok(NULL, ", ");
                col++;
            }
        }
        fclose(fp);
        return row;
}
