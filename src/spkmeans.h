void BuildT(int dim, int k, double** u, double** t);
void BuildU(int dim, int k, double** jac, double** u);
void transpose(int up, int side, double** mat);
int eigengap(int dim, double** eigenVals);
int cmp(const void *x, const void *y);
void correctMat(int dim,double* arr, double** mat, double **final);
double* diagonal(double** mat, int dim);
double** BuildJacobi(int dim, double** mat);
void buildID(int N, double** mat);
double* offElem(int dim, double** mat);
int checkDiag(double **mat, int dim);
void rotate(int dim, double** p, double** mat, int i, int j);
double** BuildLap(double** Diag, double** Adj, int dim);
void subtract(int dim, double** A, double** B, double** res);
void multiply(int dim, double** A, double** B, double** res);
double** BuildDDG(double** Adj, int dim);
double** WAMatrix(double** data, int dim);
double CalcWeight(double* point1, double* point);
int read_csv_file(char *filename, double** data);
void print_mat(double** mat, int dim);