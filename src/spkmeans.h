void BuildT(int dim, int k, double** u, double** t);
void BuildU(int dim, int k, double** jac, double** u);
void transpose(int up, int side, double** mat);
int eigengap(int dim, double** eigenVals);
int cmp(const void *x, const void *y);
void correctMat(int dim,double* arr, double** mat, double **final);
double* diagonal(double** mat, int dim);
double** BuildJacobi(int dim, double** mat);
double* offElem(int dim, double** mat);
int checkDiag(double **mat, int dim);
void rotate(int dim, double** p, double** mat, int i, int j);
double** BuildLap(double** Diag, double** Adj, int dim);
void eye_minus(int dim, double** mat);
void BuildDDG(double** Adj, int dim, double **diag);
void WAMatrix(double** data, int dim, double** adj);
double CalcWeight(double* point1, double* point);
void read_csv_file(char *filename, double** data);
void print_mat(double** mat, int dim, int cols);
double atof(const char *str);
char *strtok(char *str, const char *delim);
int get_feature_num(char* file);
int get_points_num(char* file);
void init_data_rows(double** data, int rows, int cols);
void multiply_by_Diag(int dim, double** A, double** B);
void multiply_diag_by(int dim, double** A, double** B);
double offDiagSum(double** mat, int dim);
void print_jac(double** mat, int dim, int cols);