/**
 * This file is a test file for taco conversion
 * routines
 * **/

#include <assert.h>
#include <mkl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;

typedef struct taco_tensor_t {
  int32_t order;           // tensor order (number of modes)
  int32_t *dimensions;     // tensor dimensions
  int32_t csize;           // component size
  int32_t *mode_ordering;  // mode storage ordering
  taco_mode_t *mode_types; // mode storage types
  uint8_t ***indices;      // tensor index data (per mode)
  uint8_t *vals;           // tensor values
  int32_t vals_size;       // values array size
} taco_tensor_t;

typedef struct csr_d {
  double *vals; // values
  int *cols;    // column index
  int *rptr;    // row pointer
  int nnz;      // number of non zeros
  int *order;   // storage mapping order 0,1 for row first
  int nr;       // number of rows

  // function pointer to be overloaded
  // for converting csr to taco. if not
  // overloaded default is used.
  taco_tensor_t *(*csr2taco)(struct csr_d *);

  // function pointer for
  // converting back
  // to csr
  void (*taco2csr)(taco_tensor_t *, struct csr_d *);

} csr_d;

typedef struct coo_d {
  double *vals; // values
  int *cols;    // column index array
  int *rows;    // row index array
  int nnz;      // number of non zeros
  int nr;       // rows
  int nc;       // columns

  // convert from coo to taco_tensor_t
  taco_tensor_t *(*coo2taco)(struct coo_d *);

  // convert from taco to coo
  void (*taco2coo)(taco_tensor_t *, struct coo_d *);
} coo_d;

void read_sparse_coo(const char *, coo_d *);

void split(char *, char *, char *[3]);

/**
 * This routine cleans up auxiliary memory allocated to taco
 */
static inline void cleanup_taco(taco_tensor_t *t) {
  if (t) {
    for (int i = 0; i < t->order; i++) {
      free(t->indices[i]);
    }
    free(t->indices);

    free(t->dimensions);
    free(t->mode_ordering);
    free(t->mode_types);
    free(t);
  }
}

static inline void taco2csr(taco_tensor_t *t, csr_d *csr) {
  csr->order[0] = t->mode_ordering[0];
  csr->order[1] = t->mode_ordering[1];
  csr->nr = t->dimensions[0];
  csr->rptr = (int *)t->indices[1][0];
  csr->cols = (int *)t->indices[1][1];
  csr->vals = (double *)t->vals;
  cleanup_taco(t); // clean up taco auxiliary allocations
}

static inline taco_tensor_t *csr2Taco(csr_d *csr) {
  taco_tensor_t *tensor = (taco_tensor_t *)malloc(sizeof(taco_tensor_t));
  tensor->order = 2;
  tensor->dimensions = (int32_t *)malloc(sizeof(int32_t) * 2);
  tensor->dimensions[0] = csr->nr;
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
}

int main(int argc, char **argv) {
  char *file = argv[1];
  int count = atoi(argv[2]);
  double elapsed = 0;
  MKL_INT job[6] = {
      1 /*coo.csr*/,
      0 /* zero based indexing CSR*/,
      0 /*zero based indexing COO*/,
      0,
      0, /*max number of non zero elements allowed*/
      1, /*fill up acsr, ja and ia when converting to CSR*/
  };
  coo_d coo;
  fprintf(stderr, "reading data to memory:\n\n");
  read_sparse_coo(file, &coo);
  MKL_INT dimension[2] = {coo.nr /*row count*/, coo.nc /*col count*/};
  csr_d csr;
  csr.cols = (int *)malloc(sizeof(*csr.cols) * coo.nnz);
  csr.vals = (double *)malloc(sizeof(*csr.vals) * coo.nnz);
  csr.rptr = (int *)malloc(sizeof(*csr.rptr) * (coo.nr + 1));
  csr.nnz = coo.nnz;
  fprintf(stderr, "finished reading data to memory:\n\n");
  MKL_INT res;
  mkl_dcsrcoo(
      job
      /*job instructions check documentation for
                    specification
     https://software.intel.com/en-us/mkl-developer-reference-c-mkl-csrcoo */
      ,
      dimension
      /*dimension pointer for the matrix, seems to require
       * square matrix. any way i placed both the row and col
       * in an array structure*/
      ,
      csr.vals /*output: result of the nnz csr vector*/,
      (MKL_INT *)csr.cols, /*output: col component of csr.sorted*/
      (MKL_INT *)csr.rptr, /*output: row ptr component of csr*/

      (MKL_INT *)&coo.nnz /*
      *specifies number of non zero important for conversion from
      coo.csr but not necessarry for csr.coo
      * */
      ,
      coo.vals, /*input: non zero coo values of matrix*/
      coo.rows, /*input: row indices*/
      coo.cols, /*input: col indices*/
      &res);
  assert(res == 0 && "something happened during conversion aborting");
  taco_tensor_t *t = csr2Taco(&csr);
}

void read_sparse_coo(const char *filename, coo_d *coo_data) {
  int past_comments = 0;
  int i = 0;
  char buffer[1024];
  FILE *mat_d = fopen(filename, "r");
  if (mat_d) {
    while (fgets(buffer, sizeof(buffer), mat_d)) {
      // takes out comments from the
      // file
      if (buffer[0] != '%') {
        char *elems[3];
        split(buffer, " ", elems);
        if (!past_comments) {
          coo_data->nr = atoi(elems[0]);
          coo_data->nc = atoi(elems[1]);
          coo_data->nnz = atoi(elems[2]);
          coo_data->rows = (int *)malloc(coo_data->nnz * sizeof(int));
          coo_data->cols = (int *)malloc(coo_data->nnz * sizeof(int));
          coo_data->vals = (double *)malloc(coo_data->nnz * sizeof(double));
          past_comments = 1;
        } else {
          coo_data->rows[i] = atoi(elems[0]);
          coo_data->cols[i] = atoi(elems[1]);
          coo_data->vals[i] = atoi(elems[2]);
          i++;
        }
      }
    }
  }
}

void split(char *s, char *delim, char *result[3]) {
  char *tok = strtok(s, delim);
  int i = 0;

  while (tok != NULL) {
    result[i] = tok;
    tok = strtok(NULL, delim);
    i++;
  }
}
