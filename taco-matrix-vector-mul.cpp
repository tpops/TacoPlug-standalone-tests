
[[clang::syntax(taco)]] void
mat_vec_mul(taco_tensor_t *a, taco_tensor_t *b, taco_tensor_t *c,
            char *format = "-f=a:d:0 -f=b:uq:0,1 -f=c:d:0") {
  a(i) = b(i, j) * c(i)
}

[[clang::syntax(taco)]] void mat_mat_mul(taco_tensor_t *a, taco_tensor_t *b,
                                         taco_tensor_t *c, char *format = "") {
  a(i) = b(i, j) * c(j, k)
}

int main(int argc, char **argv) {
  taco_tensor_t a;
  taco_tensor_t b;
  taco_tensor_t c;
  mat_vec_mul(&a, &b, &c);
  mat_vec_mul(&a, &b, &c);
}
