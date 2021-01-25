/**
 * This file is a test file for taco conversion
 * routines
 * **/

#include <assert.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

/**
 * CSR Struct with routines for
 * converting to and from taco_tensor_t.
 * */
typedef struct csr {
  double *vals; // values
  int *cols;    // column index
  int *rptr;    // row pointer
  int nnz;      // number of non zeros
  int *order;   // storage mapping order 0,1 for row first
  int nr;       // number of rows
  int nc;

  // function pointer to be overloaded
  // for converting csr to taco. if not
  // overloaded default is used.
  taco_tensor_t *(*csr2taco)(struct csr *);

  // function pointer for
  // converting back
  // to csr
  void (*taco2csr)(taco_tensor_t *, struct csr *);

} csr;

/**
 *  Dense Vector struct with routines for converting
 *  to and from taco_tensor_t.
 */
typedef struct vector {
  double *vals; // values of vector
  int length;   // length of vector
  taco_tensor_t *(*vector2taco)(struct vector *);
  void (*taco2vector)(taco_tensor_t *, struct vector *);

} vector;

/**
 * Dense Matrix struct with routines for converting
 * to and from taco_tensor_t.
 *
 *  */
typedef struct matrix {
  double *vals; // values of matrix
  int nr;       // number of rows
  int nc;       // number of columns
  int *order;   // data layout order of matrix
  taco_tensor_t *(*matrix2taco)(struct matrix *);
  void (*taco2matrix)(taco_tensor_t *, struct matrix *);
} matrix;

/**
 * CSR Struct with routines for
 * converting to and from taco_tensor_t.
 * */
typedef struct coo_d {
  double *vals; // values
  int *cols;    // column index array
  int *rows;    // row index array
  int nnz;      // number of non zeros
  int nr;       // rows
  int nc;       // columns

  // convert from coo to taco_tensor_t
  taco_tensor_t *(*coo_d2taco)(struct coo_d *);

  // convert from taco to coo
  void (*taco2coo_d)(taco_tensor_t *, struct coo_d *);
} coo_d;

static inline void taco2csrd(taco_tensor_t *, csr *);
static inline taco_tensor_t *csr2taco(csr *);
static inline taco_tensor_t *vector2taco(vector *);
static inline void taco2vector(taco_tensor_t *, vector *);
static inline taco_tensor_t *matrix2taco(matrix *);
static inline void taco2matrix(taco_tensor_t *, matrix *);
/**
 * Matrix Matrix sum
 * */
[[clang::syntax(taco)]] void
mat_mat_sum(csr *a, csr *b, csr *c,
            std::string format = "-f=b:ds:0,1 -f=c:ds:0,1 -f=a:ds:0,1") {
  a(i, j) = b(i, j) + c(i, j)
}

/**
 * Matrix vector multiplication.
 */
[[clang::syntax(taco)]] void
mat_vec_mul(vector *y, csr *A, vector *x,
            std::string format = "-f=A:ds:0,1 -f=y:d -f=x:d") {
  y(i) = A(i, j) * x(j)
}

/**
 *Sparse  Matrix Matrix multiplication
 * */
[[clang::syntax(taco)]] void
mat_mat_mul(matrix *a, csr *b, csr *c,
            std::string format = "-f=b:ds:0,1 -f=c:ds:0,1 -f=a:dd:0,1") {
  a(i, k) = b(i, j) * c(j, k)
}

/**
 *Sparse  Matrix Matrix multiplication with CSR output
 * */
[[clang::syntax(taco)]] void
compute_01(csr *a, csr *b, csr *c,
           std::string format = "-f=b:ds:0,1 -f=c:ds:0,1 -f=a:ds:0,1") {
  a(i, k) = b(i, j) * c(j, k)
}

/**
 * B =  (A^TA)
 */
[[clang::syntax(taco)]] void
mat_transpose(csr *B, csr *A, std::string format = "-f=B:ds:0,1 -f=A:ds:0,1 ") {
  B(i, j) = A(k, i) * A(k, j)
}

/**
 * a = (B^TB) * c
 */
[[clang::syntax(taco)]] void
compute_02(vector *a, csr *B, vector *c,
           std::string format = "-f=B:ds:0,1 -f=a:d -f=c:d") {
  a(i) = B(k, i) * B(k, j) * c(j)
}

/**
 * Compare two arrays
 */
bool compare_array_d(double *, double *, int);
bool compare_array(int *, int *, int);

int main(int argc, char **argv) {
  // simple 8 by 8 matrix addition
  csr B;
  B.csr2taco = &csr2taco;
  double vals1[] = {4, 8, 3, 4};
  int cols1[] = {1, 6, 4, 6};
  int rptr1[] = {0, 0, 2, 3, 3, 3, 4, 4, 4};
  int order[] = {0, 1};
  B.vals = &vals1[0];
  B.cols = &cols1[0];
  B.rptr = &rptr1[0];
  B.order = &order[0];
  B.nr = 8;
  B.nc = 8;
  B.nnz = 4;
  taco_tensor_t *__taco_a = csr2taco(&B);
  csr C;
  C.csr2taco = &csr2taco;
  double vals2[] = {10, 12, 3, 4};
  int cols2[] = {0, 6, 4, 6};
  int rptr2[] = {0, 0, 2, 3, 3, 3, 4, 4, 4};

  C.vals = &vals2[0];
  C.cols = &cols2[0];
  C.rptr = &rptr2[0];
  C.order = &order[0];
  C.nr = 8;
  C.nc = 8;
  C.nnz = 4;

  csr A;
  A.nr = 8;
  A.nc = 8;
  A.taco2csr = &taco2csrd;
  A.csr2taco = &csr2taco;
  A.order = &order[0];

  mat_mat_sum(&A, &B, &C);

  // expected result
  double result_1_vals[] = {10, 4, 20, 6, 8};
  int result_1_cols[] = {0, 1, 6, 4, 6};
  int result_1_rptr[] = {0, 0, 3, 4, 4, 4, 5, 5, 5};
  assert(compare_array_d(result_1_vals, A.vals, 5));
  assert(compare_array(result_1_cols, A.cols, 5));
  assert(compare_array(result_1_rptr, A.rptr, 9));

  printf("Successfully passed 8 X 8 matrix sum test [PASSED]\n");

  // Simple matrix vector multiplication.
  // Dense Vector b(8) , sparse matrix A (8 x 8)
  //
  vector vector_b;
  double vector_val[] = {1, 2, 3, 4, 5, 6, 7, 8};
  vector_b.vals = &vector_val[0];
  vector_b.length = 8;
  vector_b.vector2taco = &vector2taco;
  vector_b.taco2vector = &taco2vector;

  vector vector_c;
  vector_c.length = 8;
  vector_c.vector2taco = &vector2taco;
  vector_c.taco2vector = &taco2vector;

  csr matrix_csr_A;
  matrix_csr_A.csr2taco = &csr2taco;
  double matrix_A_val[] = {10, 12, 3, 4};
  int matrix_A_cols[] = {0, 6, 4, 6};
  int matrix_A_rptr[] = {0, 0, 2, 3, 3, 3, 4, 4, 4};

  matrix_csr_A.vals = &matrix_A_val[0];
  matrix_csr_A.cols = &matrix_A_cols[0];
  matrix_csr_A.rptr = &matrix_A_rptr[0];
  matrix_csr_A.order = &order[0];
  matrix_csr_A.nr = 8;
  matrix_csr_A.nc = 8;
  matrix_csr_A.nnz = 4;

  mat_vec_mul(&vector_c, &matrix_csr_A, &vector_b);

  double result_2_vals[] = {0, 94, 15, 0, 0, 28, 0, 0};
  assert(compare_array_d(result_2_vals, vector_c.vals, 8));
  printf("Successfully Dense Vector b(8)"
         " , sparse matrix A (8 x 8) [PASSED]  \n");

  // Sparse matrix  matrix multplication
  // A(5 X 3)-DENSE = B(5 X 4)-CSR  * C(4 X 3)-CSR

  double matrix_B0_val[] = {4, 3, 2, 1, 6, 4, 6, 3, 7, 1, 8};
  int matrix_B0_cols[] = {0, 1, 2, 3, 0, 1, 0, 2, 0, 3, 0};
  int matrix_B0_rptr[] = {0, 4, 6, 8, 10, 11};

  double matrix_C0_val[] = {1, 2, 4, 3};
  int matrix_C0_cols[] = {2, 1, 0, 2};
  int matrix_C0_rptr[] = {0, 1, 2, 2, 4};

  double result_A0_val[] = {4, 6, 7, 0, 8, 6, 0, 0, 6, 4, 0, 10, 0, 0, 8};

  csr matrix_B0;
  matrix_B0.vals = &matrix_B0_val[0];
  matrix_B0.cols = &matrix_B0_cols[0];
  matrix_B0.rptr = &matrix_B0_rptr[0];
  matrix_B0.order = &order[0];
  matrix_B0.nr = 5;
  matrix_B0.nc = 4;
  matrix_B0.nnz = 11;
  matrix_B0.taco2csr = &taco2csrd;
  matrix_B0.csr2taco = &csr2taco;

  csr matrix_C0;
  matrix_C0.vals = &matrix_C0_val[0];
  matrix_C0.cols = &matrix_C0_cols[0];
  matrix_C0.rptr = &matrix_C0_rptr[0];
  matrix_C0.order = &order[0];
  matrix_C0.nr = 4;
  matrix_C0.nc = 3;
  matrix_C0.nnz = 4;
  matrix_C0.taco2csr = &taco2csrd;
  matrix_C0.csr2taco = &csr2taco;

  matrix matrix_A0;
  matrix_A0.order = &order[0];
  matrix_A0.nr = 5;
  matrix_A0.nc = 3;
  matrix_A0.taco2matrix = &taco2matrix;
  matrix_A0.matrix2taco = &matrix2taco;
  mat_mat_mul(&matrix_A0, &matrix_B0, &matrix_C0);
  assert(compare_array_d(result_A0_val, matrix_A0.vals, 15));

  printf(" Sparse matrix  matrix multplication "
         " A(5 X 3) = B(5 X 4)  * C(4 X 3) [PASSED]\n");

  //

  // Sparse matrix  matrix multplication
  // A(5 X 3)-CSR = B(5 X 4)-CSR  * C(4 X 3)-CSR

  return 0;
}

bool compare_array_d(double *ar1, double *ar2, int length) {
  for (int i = 0; i < length; i++) {
    if (ar1[i] != ar2[i]) {
      return false;
    }
  }
  return true;
}
bool compare_array(int *ar1, int *ar2, int length) {
  for (int i = 0; i < length; i++) {
    if (ar1[i] != ar2[i]) {
      return false;
    }
  }
  return true;
}

static inline taco_tensor_t *matrix2taco(matrix *m) {
  taco_tensor_t *tensor = (taco_tensor_t *)malloc(sizeof(taco_tensor_t));
  tensor->order = 2;
  tensor->dimensions = (int32_t *)malloc(sizeof(int32_t) * 2);
  tensor->dimensions[0] = m->nr;
  tensor->dimensions[1] = m->nc;
  tensor->csize = sizeof(*m->vals);
  tensor->mode_ordering = (int32_t *)malloc(sizeof(int32_t) * 2);
  // storage data layout order
  if (m->order) {
    tensor->mode_ordering[0] = m->order[0];
    tensor->mode_ordering[1] = m->order[1];
  } else {
    tensor->mode_ordering[0] = 0;
    tensor->mode_ordering[1] = 1;
  }
  tensor->mode_types = (taco_mode_t *)malloc(2 * sizeof(taco_mode_t));
  tensor->indices = (uint8_t ***)malloc(2 * sizeof(uint8_t ***));

  // allocate memory for dense indices and store information
  // on dense indices for row
  tensor->indices[0] = (uint8_t **)malloc(1 * sizeof(uint8_t **));
  (tensor->indices[0])[0] = (uint8_t *)malloc(sizeof(uint8_t *));
  *(tensor->indices[0][0]) = m->nr;

  // allocate memory for compressed indices and store information
  // on compressed indices rptr, colidx
  tensor->indices[1] = (uint8_t **)malloc(1 * sizeof(uint8_t **));
  (tensor->indices[1])[0] = (uint8_t *)malloc(sizeof(uint8_t *));
  *(tensor->indices[1][0]) = m->nr;

  // m is dense sparse
  tensor->mode_types[0] = taco_mode_dense;
  tensor->mode_types[1] = taco_mode_dense;

  tensor->vals_size = m->nr * m->nc;
  tensor->vals = (uint8_t *)m->vals;
  return tensor;
}

static inline void taco2matrix(taco_tensor_t *t, matrix *m) {
  m->order[0] = t->mode_ordering[0];
  m->order[1] = t->mode_ordering[1];
  m->nr = t->dimensions[0];
  m->nc = t->dimensions[1];
  m->vals = (double *)t->vals;
}

static inline void taco2csrd(taco_tensor_t *t, csr *csr) {
  csr->order[0] = t->mode_ordering[0];
  csr->order[1] = t->mode_ordering[1];
  csr->nr = t->dimensions[0];
  csr->rptr = (int *)t->indices[1][0];
  csr->cols = (int *)t->indices[1][1];
  csr->vals = (double *)t->vals;
}

static inline taco_tensor_t *csr2taco(csr *csr) {
  taco_tensor_t *tensor = (taco_tensor_t *)malloc(sizeof(taco_tensor_t));
  tensor->order = 2;
  tensor->dimensions = (int32_t *)malloc(sizeof(int32_t) * 2);
  tensor->dimensions[0] = csr->nr;
  tensor->dimensions[1] = csr->nc;
  tensor->csize = sizeof(*csr->vals);
  tensor->mode_ordering = (int32_t *)malloc(sizeof(int32_t) * 2);
  // storage data layout order
  if (csr->order) {
    tensor->mode_ordering[0] = csr->order[0];
    tensor->mode_ordering[1] = csr->order[1];
  } else {
    tensor->mode_ordering[0] = 0;
    tensor->mode_ordering[1] = 1;
  }
  tensor->mode_types = (taco_mode_t *)malloc(2 * sizeof(taco_mode_t));
  tensor->indices = (uint8_t ***)malloc(2 * sizeof(uint8_t ***));

  // allocate memory for dense indices and store information
  // on dense indices for row
  tensor->indices[0] = (uint8_t **)malloc(1 * sizeof(uint8_t **));
  (tensor->indices[0])[0] = (uint8_t *)malloc(sizeof(uint8_t *));
  *(tensor->indices[0][0]) = csr->nr;

  // allocate memory for compressed indices and store information
  // on compressed indices rptr, colidx
  tensor->indices[1] = (uint8_t **)malloc(2 * sizeof(uint8_t **));
  tensor->indices[1][0] = (uint8_t *)csr->rptr;
  tensor->indices[1][1] = (uint8_t *)csr->cols;

  // csr is dense sparse
  tensor->mode_types[0] = taco_mode_dense;
  tensor->mode_types[1] = taco_mode_sparse;

  tensor->vals_size = csr->nnz;
  tensor->vals = (uint8_t *)csr->vals;
  return tensor;
}

static inline taco_tensor_t *vector2taco(vector *v) {
  taco_tensor_t *tensor = (taco_tensor_t *)malloc(sizeof(taco_tensor_t));
  tensor->order = 1;
  tensor->dimensions = (int32_t *)malloc(sizeof(int32_t));
  tensor->dimensions[0] = v->length;
  tensor->csize = sizeof(*v->vals);
  tensor->mode_ordering = (int32_t *)malloc(sizeof(int32_t));
  tensor->mode_types = (taco_mode_t *)malloc(sizeof(taco_mode_t));
  tensor->indices = (uint8_t ***)malloc(sizeof(uint8_t ***));

  // allocate memory for dense indices and store information
  // on dense indices for row
  tensor->indices[0] = (uint8_t **)malloc(1 * sizeof(uint8_t **));
  (tensor->indices[0])[0] = (uint8_t *)malloc(sizeof(uint8_t *));
  *(tensor->indices[0][0]) = v->length;

  // v is dense sparse
  tensor->mode_types[0] = taco_mode_dense;

  tensor->vals_size = v->length;
  tensor->vals = (uint8_t *)v->vals;
  return tensor;
}
static inline void taco2vector(taco_tensor_t *t, vector *v) {
  v->length = t->dimensions[0];
  v->vals = (double *)t->vals;
}
