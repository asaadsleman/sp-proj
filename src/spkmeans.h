void BuildT(int dim, int k, double** u, double** t);
void BuildU(int dim, int k, double** jac, double** u, double** eigenVals);
void transpose(int up, int side, double** mat);
int eigengap(int dim, double** eigenVals);
int cmp(const void *x, const void *y);
void correctMat(int dim,double** mat1, double** mat2, double **final);
double* diagonal(double** mat, int dim);
double*** BuildJacobi(int dim, double** mat);
double** buildID(int N);
double* offElem(int dim, double** mat);
int checkDiag(double **mat, int dim);
void rotate(int dim, double** p, double** mat, int i, int j);
double** BuildLap(double** Diag, double** Adj, int dim);
void subtract(int dim, double** A, double** B, double** res);
double** multiply(int dim, double** A, double** B);
double** BuildDDG(double** Adj, int dim);
double** WAMatrix(double** data, int dim);
double CalcWeight(double* point1, double* point2);
double** read_csv_file(char *filename)