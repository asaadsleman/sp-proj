#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>

#define MAXLEN 1000
#define EPS 0.001

int main(int argc, char **argv){
    int k, dim, i;
    char *fname, *goal, buffer[MAXLEN], *ev;
    FILE *file;
    double **data, **Adjacency, **DiagMat, **laplacian, **jacobi, **u, **t;
    // Read User Data
    k = atoi(argv[1]);
    goal = argv[2];
    fname = argv[3];
    // Read File
    data = read_csv_file(fname);
    dim = sizeof(data) / sizeof(double *);
    // Build WAM
    Adjacency = WAMatrix(data, dim);
    if (strcmp(goal, 'wam')) // If Adj. matrix requested, print and stop
    {
        // Print Adjacency
        return 1;
    }
    // Build Diagonal
    DiagMat = BuildDDG(Adjacency, dim);
    if(strcmp(goal, 'ddg')){
        // print
        return 2;
    }
    // Create Laplacian
    laplacian = BuildLap(DiagMat, Adjacency, dim);
    if(strcmp(goal, 'lnorm')){
        // print
        return 3;
    }
    // Find eigenvalues
    jacobi = BuildJacobi(dim, laplacian);
    assert(jacobi != NULL);
    if(strcmp(goal, 'jacobi')){
        // print jacobi
        return 4;
    }
    // Determine K
    ev = (double**)malloc(dim*sizeof(double*));
    assert(ev != NULL);
    for (i = 0; i < dim; i++)
    {
        ev[i] = (double*)malloc(2 * sizeof(double));
        assert(ev[i] != NULL);
        ev[i] = {jacobi[i][i], i};
    }
    qsort(ev, dim, sizeof(double), cmp);
    if (k == 0){
        k = eigengap(ev);
    }
    BuildU(dim, k, jacobi, u, ev);
    // Build T (normalized by line)
    BuildT(dim, k, u, t);
    
	return 0;
}

void BuildT(int dim, int k, double** u, double** t){
    int i, j;
    double lineSums[dim];
    double sum = 0.0;
    t = (double**) malloc(dim * sizeof(double*));
    assert(t != NULL);
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
}

void BuildU(int dim, int k, double** jac, double** u, double** eigenVals){
    int i, j;
    u = (double**) malloc(k * sizeof(double*));
    assert(u != NULL);
    transpose(dim, dim, jac);
    for (i = 0; i < k; i++)
    {
        u[i] = (double*) malloc(dim* sizeof(double));
        assert(u[i] != NULL);
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
    double diffs[dim-1], maxAbs;
    int i, max_i;
    for (i = 0; i < dim-1; i++)
    {
        diffs[i] = abs(eigenVals[i][0] - eigenVals[i+1][0]);
        if (diffs[i] > maxAbs)
        {
            maxAbs = diffs[i];
            max_i = i;
        }
    }
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
    int i, j;
    double* res;
    res = (double*)malloc(dim * sizeof(double));
    assert(res !- NULL);
    for (i = 0; i < dim; i++)
    {
        res[i] = mat[i][i];   
    }
    return res;
    
}

double*** BuildJacobi(int dim, double** mat){
    double* max;
    double** p, **diag, **jacobi;
    double maxV;
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
            diag = &(diagonal(mat));
            correctMat(dim, diag, p, jacobi);
            return jacobi;
        }
        rotate(dim, p, mat, i, j);
    }
    printf("Jacobi did not converge!");
    return NULL;
}

double** buildID(int N){
    double** ID;
    ID = (double**)malloc(dim*sizeof(double*));
    assert(ID !- NULL);
    for(i = 0; i < dim; i++){
        ID[i] = (double*)calloc(dim, sizeof(double));
        assert(ID[i] != NULL);
        ID[i][i] = 1.0;
    }
    return ID;
}

double* offElem(int dim, double** mat){
    double max, max_i = 0.0, max_j = 0.0;
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
    return  {max, max_i, max_j};
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
    int i,j;
    Id = buildID(dim);
    Mid = multiply(dim, Diag, Adj;
    Mid2 = multiply(dim, Mid, Diag);
    subtract(dim, Id, Mid2, Laplacian);
    free(Id);
    free(Mid);
    free(Mid2);
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
    double res[dim][dim];
    int i, j, k;
    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            res[i][j] = 0;
            for (k = 0; k < dim; k++){
                res[i][j] += mat1[i][k] * mat2[k][j];
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

double** read_csv_file(char *filename){
    double *array[MAXLEN];
    int idx = 0;
    int j = 0;
    char *buffer = NULL;
    size_t len = 0;
    ssize_t read;
    char *ptr = NULL;

    FILE *fp;
    fp = fopen ("test.txt", "r");      //open file , read only
    if (!fp) {
        fprintf (stderr, "failed to open file for reading\n");
        return 1;
    }

    while ((read = getline (&buffer, &len, fp)) != -1) {
        array[idx] = (double*)malloc (sizeof (array));
        assert(array[idx] != NULL);
        for (j = 0, ptr = buffer; j < MAXLEN; j++, ptr++)
            array [idx][j] = (double)strtol(ptr, &ptr, 10);
        idx++;
    }

    fclose (fp);
    return array;
}
