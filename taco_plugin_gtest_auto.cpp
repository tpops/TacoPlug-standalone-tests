/**
 * This file is a test file for taco conversion
 * routines
 * **/
#include "gtest/gtest.h"
#include <assert.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

struct scalar{
  double val;
  
  taco_tensor_t *__2taco() {
    taco_tensor_t *tensor = (taco_tensor_t *)malloc(sizeof(taco_tensor_t));  
    tensor->order = 1;
    tensor->dimensions = (int32_t *)malloc(sizeof(int32_t)); 
    tensor->mode_ordering = (int32_t *)malloc(sizeof(int32_t));
    tensor->mode_types = (taco_mode_t *)malloc(sizeof(taco_mode_t)); 
    tensor->vals = (uint8_t *) &this->val;  
    return tensor;
  }
  void taco2__(taco_tensor_t *t) {
    this->val = *((double *)t->vals);
  }
};

/**
 * CSR Struct with routines for
 * converting to and from taco_tensor_t.
 * */
struct csr {
  double *vals; // values
  int *cols;    // column index
  int *rptr;    // row pointer
  int nnz;      // number of non zeros
  int *order;   // storage mapping order 0,1 for row first
  int nr;       // number of rows
  int nc;

  taco_tensor_t *__2taco() {
    taco_tensor_t *tensor = (taco_tensor_t *)malloc(sizeof(taco_tensor_t));
    tensor->order = 2;
    tensor->dimensions = (int32_t *)malloc(sizeof(int32_t) * 2);
    tensor->dimensions[0] = this->nr;
    tensor->dimensions[1] = this->nc;
    tensor->csize = sizeof(*this->vals);
    tensor->mode_ordering = (int32_t *)malloc(sizeof(int32_t) * 2);
    // storage data layout order
    if (this->order) {
      tensor->mode_ordering[0] = this->order[0];
      tensor->mode_ordering[1] = this->order[1];
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
    *(tensor->indices[0][0]) = this->nr;

    // allocate memory for compressed indices and store information
    // on compressed indices rptr, colidx
    tensor->indices[1] = (uint8_t **)malloc(2 * sizeof(uint8_t **));
    tensor->indices[1][0] = (uint8_t *)this->rptr;
    tensor->indices[1][1] = (uint8_t *)this->cols;

    // csr is dense sparse
    tensor->mode_types[0] = taco_mode_dense;
    tensor->mode_types[1] = taco_mode_sparse;

    tensor->vals_size = this->nnz;
    tensor->vals = (uint8_t *)this->vals;
    return tensor;
  }

  void taco2__(taco_tensor_t *t) {

    this->order[0] = t->mode_ordering[0];
    this->order[1] = t->mode_ordering[1];
    this->nr = t->dimensions[0];
    this->rptr = (int *)t->indices[1][0];
    this->cols = (int *)t->indices[1][1];
    this->vals = (double *)t->vals;
  }
};

/**
 *  Dense Vector struct with routines for converting
 *  to and from taco_tensor_t.
 */
struct vector {
  double *vals; // values of vector
  int length;   // length of vector

  taco_tensor_t *__2taco() {
    taco_tensor_t *tensor = (taco_tensor_t *)malloc(sizeof(taco_tensor_t));
    tensor->order = 1;
    tensor->dimensions = (int32_t *)malloc(sizeof(int32_t));
    tensor->dimensions[0] = this->length;
    tensor->csize = sizeof(*this->vals);
    tensor->mode_ordering = (int32_t *)malloc(sizeof(int32_t));
    tensor->mode_types = (taco_mode_t *)malloc(sizeof(taco_mode_t));
    tensor->indices = (uint8_t ***)malloc(sizeof(uint8_t ***));

    // allocate memory for dense indices and store information
    // on dense indices for row
    tensor->indices[0] = (uint8_t **)malloc(1 * sizeof(uint8_t **));
    (tensor->indices[0])[0] = (uint8_t *)malloc(sizeof(uint8_t *));
    *(tensor->indices[0][0]) = this->length;

    // v is dense sparse
    tensor->mode_types[0] = taco_mode_dense;

    tensor->vals_size = this->length;
    tensor->vals = (uint8_t *)this->vals;
    return tensor;
  }

  void taco2__(taco_tensor_t *t) {
    this->length = t->dimensions[0];
    this->vals = (double *)t->vals;
  }
};

/**
 * Dense Matrix struct with routines for converting
 * to and from taco_tensor_t.
 *
 *  */
struct matrix {
  double *vals; // values of matrix
  int nr;       // number of rows
  int nc;       // number of columns
  int *order;   // data layout order of matrix

  taco_tensor_t *__2taco() {

    taco_tensor_t *tensor = (taco_tensor_t *)malloc(sizeof(taco_tensor_t));
    tensor->order = 2;
    tensor->dimensions = (int32_t *)malloc(sizeof(int32_t) * 2);
    tensor->dimensions[0] = this->nr;
    tensor->dimensions[1] = this->nc;
    tensor->csize = sizeof(*this->vals);
    tensor->mode_ordering = (int32_t *)malloc(sizeof(int32_t) * 2);
    // storage data layout order
    if (this->order) {
      tensor->mode_ordering[0] = this->order[0];
      tensor->mode_ordering[1] = this->order[1];
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
    *(tensor->indices[0][0]) = this->nr;

    // allocate memory for compressed indices and store information
    // on compressed indices rptr, colidx
    tensor->indices[1] = (uint8_t **)malloc(1 * sizeof(uint8_t **));
    (tensor->indices[1])[0] = (uint8_t *)malloc(sizeof(uint8_t *));
    *(tensor->indices[1][0]) = this->nr;

    // m is dense sparse
    tensor->mode_types[0] = taco_mode_dense;
    tensor->mode_types[1] = taco_mode_dense;

    tensor->vals_size = this->nr * this->nc;
    tensor->vals = (uint8_t *)this->vals;
    return tensor;
  }

  void taco2__(taco_tensor_t *t) {
    this->order[0] = t->mode_ordering[0];
    this->order[1] = t->mode_ordering[1];
    this->nr = t->dimensions[0];
    this->nc = t->dimensions[1];
    this->vals = (double *)t->vals;
  }
};

/**
 * CSR Struct with routines for
 * converting to and from taco_tensor_t.
 * */
struct coo_d {
  double *vals; // values
  int *cols;    // column index array
  int *rows;    // row index array
  int nnz;      // number of non zeros
  int nr;       // rows
  int nc;       // columns

  taco_tensor_t *__2taco() { return NULL; }

  void taco2__(taco_tensor_t *t) {}
};

/**
 * Matrix Matrix sum
 * */
template< typename MatrixA, typename MatrixB, typename MatrixC>
[[clang::syntax(taco)]] void
mat_mat_sum(MatrixA* a, MatrixB* b,  MatrixC* c,
            std::string format = "-f=b:ds:0,1 -f=c:ds:0,1 -f=a:ds:0,1") {
  a(i, j) = b(i, j) + c(i, j)
}

/**
 * Matrix vector multiplication.
 */
template <typename Vector, typename Matrix>
[[clang::syntax(taco)]] void
mat_vec_mul(Vector *a, Matrix *b, Vector *c,
            std::string format = "-f=b:ds:0,1 -f=c:d -f=a:d") {
  a(i) = b(i, j) * c(j)
}

/**
 *Sparse  Matrix Matrix multiplication
 * */
template <typename DenseMatrix,typename SparseMatrix>
[[clang::syntax(taco)]] void
mat_mat_mul(DenseMatrix *a, SparseMatrix *b, SparseMatrix *c,
            std::string format = "-f=b:ds:0,1 -f=c:ds:0,1 -f=a:dd:0,1") {
  a(i, k) = b(i, j) * c(j, k)
}

/**
 *Sparse  Matrix Matrix multiplication with CSR output
 * */
template <typename MatrixA, typename MatrixB, typename MatrixC>
[[clang::syntax(taco)]] void
compute_01(MatrixA *a, MatrixB *b, MatrixC *c,
           std::string format = "-f=b:ds:0,1 -f=c:ds:0,1 -f=a:ds:0,1") {
  a(i, k) = b(i, j) * c(j, k)
}

/**
 * B =  (A^TA)
 */
template <typename MatrixType>
[[clang::syntax(taco)]] void
mat_transpose_matrix_mul(MatrixType *B, MatrixType *A,
                         std::string format = "-f=B:dd:0,1 -f=A:dd:0,1 ") {
  B(i, j) = A(k, i) * A(k, j)
}

/**
 * a = (B^TB) * c
 */
template <typename Vector, typename Matrix>
[[clang::syntax(taco)]] void
compute_02(Vector *a, Matrix *B, Vector *c,
           std::string format = "-f=B:ds:0,1 -f=a:d -f=c:d") {
  a(i) = B(k, i) * B(k, j) * c(j)
}

/**
 * Vector inner product
 * x = u . v
 * */
template <typename Scalar, typename Vector>
[[clang::syntax(taco)]] void vector_inner_product(Scalar *x, Vector *u,
                                                  Vector *v,
                                                  std::string format = "") {
  x = u(j) * v(j)
}

/**
 * Compare two arrays
 */
bool compare_array_d(double *, double *, int);
bool compare_array(int *, int *, int);

TEST(TACOPlugUseTest, MatrixMatrixSum) {
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
  csr C;
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
  A.order = &order[0];

  mat_mat_sum<csr,csr,csr>(&A, &B, &C);

  double result_1_vals[] = {10, 4, 20, 6, 8};
  int result_1_cols[] = {0, 1, 6, 4, 6};
  int result_1_rptr[] = {0, 0, 3, 4, 4, 4, 5, 5, 5};
  EXPECT_TRUE(compare_array_d(result_1_vals, A.vals, 5));
  EXPECT_TRUE(compare_array(result_1_cols, A.cols, 5));
  EXPECT_TRUE(compare_array(result_1_rptr, A.rptr, 9));
}

TEST(TACOPlugUseTest, SparseMatrixVectorMultiplication) {
  // Simple matrix vector multiplication.
  // Dense Vector b(8) , sparse matrix A (8 x 8)
  //
  int order[] = {0, 1};
  vector vector_b;
  double vector_val[] = {1, 2, 3, 4, 5, 6, 7, 8};
  vector_b.vals = &vector_val[0];
  vector_b.length = 8;

  vector vector_c;
  vector_c.length = 8;

  csr matrix_csr_A;
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

  mat_vec_mul<vector,csr>(&vector_c, &matrix_csr_A, &vector_b);

  double result_2_vals[] = {0, 94, 15, 0, 0, 28, 0, 0};
  EXPECT_TRUE(compare_array_d(result_2_vals, vector_c.vals, 8));
}

TEST(TACOPlugUseTest, SparseMatrixSparseMatrixMultiplication) {
  // Sparse matrix  matrix multplication
  // A(5 X 3)-DENSE = B(5 X 4)-CSR  * C(4 X 3)-CSR
  int order[] = {0, 1};

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

  csr matrix_C0;
  matrix_C0.vals = &matrix_C0_val[0];
  matrix_C0.cols = &matrix_C0_cols[0];
  matrix_C0.rptr = &matrix_C0_rptr[0];
  matrix_C0.order = &order[0];
  matrix_C0.nr = 4;
  matrix_C0.nc = 3;
  matrix_C0.nnz = 4;

  matrix matrix_A0;
  matrix_A0.order = &order[0];
  matrix_A0.nr = 5;
  matrix_A0.nc = 3;
  mat_mat_mul<matrix,csr> (&matrix_A0, &matrix_B0, &matrix_C0);
  EXPECT_TRUE(compare_array_d(result_A0_val, matrix_A0.vals, 15));
}

TEST(TACOPlugUseTest, ATransposeACompute) {

  // A is orthogonal.
  // B = A^TA
  int order[] = {0, 1};
  double matrix_A_val[] = {0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0};
  double expected_result_val[] = {1, 0, 0, 0, 0, 1, 0, 0,
                                  0, 0, 1, 0, 0, 0, 0, 1};

  matrix matrix_A;
  matrix_A.order = &order[0];
  matrix_A.nr = 4;
  matrix_A.nc = 4;
  matrix_A.vals = &matrix_A_val[0];

  matrix matrix_B;
  matrix_B.order = &order[0];
  matrix_B.nr = 4;
  matrix_B.nc = 4;
  mat_transpose_matrix_mul<matrix>(&matrix_B, &matrix_A);

  EXPECT_TRUE(compare_array_d(expected_result_val, matrix_B.vals, 16));
}

TEST(TACOPlugUseTest, VectorInnerProduct) {
  double u_val[] = {1, 2, 3, 4, 5, 6, 7, 8};
  double v_val[] = {2, 2, 2, 2, 2, 2, 2, 2};

  vector u;
  u.vals = &u_val[0];
  u.length = 8;

  vector v;
  v.length = 8;
  v.vals = &v_val[0];

  // Scalar result.
  scalar x;

  vector_inner_product<scalar,vector>(&x, &u, &v);
  EXPECT_EQ(72, x.val);
}

bool compare_array_d(double *ar1, double *ar2, int length) {
  for (int i = 0; i < length; i++) {
    // printf("%f - %f\n",ar1[i],ar2[i]);
    if (ar1[i] != ar2[i]) {
      // printf("failed\n");
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
