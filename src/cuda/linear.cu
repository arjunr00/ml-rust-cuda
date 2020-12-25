#include <iostream>
#include <limits>

#include "kernels/kernels_linear.cu"

extern "C"
bool eq_mats(float *lhs, float *rhs, size_t len) {
  size_t matrix_size = len * sizeof(float);

  float *d_lhs, *d_rhs;
  cudaMalloc(&d_lhs, matrix_size);
  cudaMalloc(&d_rhs, matrix_size);

  cudaMemcpy(d_lhs, lhs, matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_rhs, rhs, matrix_size, cudaMemcpyHostToDevice);

  bool *equal_flag;
  cudaMallocManaged(&equal_flag, sizeof(bool));
  *equal_flag = true;

  int threads_per_block = 256;
  int blocks_per_grid = 1 + ((len - 1) / threads_per_block);
  kernel_eq_mats<<<blocks_per_grid, threads_per_block>>>
    (d_lhs, d_rhs, len, std::numeric_limits<float>::epsilon(), equal_flag);

  cudaDeviceSynchronize();

  bool equal = *equal_flag;

  cudaFree(equal_flag);
  cudaFree(d_lhs);
  cudaFree(d_rhs);

  return equal;
}

extern "C"
void add_mats(float *lhs1, float *lhs2, float *rhs, size_t len) {
  size_t matrix_size = len * sizeof(float);

  float *d_lhs1, *d_lhs2, *d_rhs;
  cudaMalloc(&d_lhs1, matrix_size);
  cudaMalloc(&d_lhs2, matrix_size);
  cudaMalloc(& d_rhs, matrix_size);

  cudaMemcpy(d_lhs1, lhs1, matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lhs2, lhs2, matrix_size, cudaMemcpyHostToDevice);

  int threads_per_block = 256;
  int blocks_per_grid = 1 + ((len - 1) / threads_per_block);
  kernel_add_mats<<<blocks_per_grid, threads_per_block>>>
    (d_lhs1, d_lhs2, d_rhs, len);

  cudaDeviceSynchronize();

  cudaMemcpy(rhs, d_rhs, matrix_size, cudaMemcpyDeviceToHost);

  cudaFree(d_lhs1);
  cudaFree(d_lhs2);
  cudaFree(d_rhs);
}

extern "C"
void mul_scalar_mat(float scalar, float *lhs, float *rhs, size_t len) {
  size_t matrix_size = len * sizeof(float);

  float *d_lhs, *d_rhs;
  cudaMalloc(&d_lhs, matrix_size);
  cudaMalloc(&d_rhs, matrix_size);

  cudaMemcpy(d_lhs, lhs, matrix_size, cudaMemcpyHostToDevice);

  int threads_per_block = 256;
  int blocks_per_grid = 1 + ((len - 1) / threads_per_block);
  kernel_mul_scalar_mat<<<blocks_per_grid, threads_per_block>>>
    (scalar, d_lhs, d_rhs, len);

  cudaDeviceSynchronize();

  cudaMemcpy(rhs, d_rhs, matrix_size, cudaMemcpyDeviceToHost);

  cudaFree(d_lhs);
  cudaFree(d_rhs);
}

extern "C"
void mul_mats
  (float *lhs1, size_t lhs1_rows, size_t lhs1_cols,
   float *lhs2, size_t lhs2_rows, size_t lhs2_cols,
   float *rhs)
{
  size_t lhs1_size = lhs1_cols * lhs1_rows * sizeof(float);
  size_t lhs2_size = lhs2_cols * lhs2_rows * sizeof(float);
  size_t rhs_size = lhs1_rows * lhs2_cols * sizeof(float);

  float *d_lhs1, *d_lhs2, *d_rhs;
  cudaMalloc(&d_lhs1, lhs1_size);
  cudaMalloc(&d_lhs2, lhs2_size);
  cudaMalloc(&d_rhs, rhs_size);

  cudaMemcpy(d_lhs1, lhs1, lhs1_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lhs2, lhs2, lhs2_size, cudaMemcpyHostToDevice);

  Matrix lhs1_mat = { lhs1_rows, lhs1_cols, d_lhs1 };
  Matrix lhs2_mat = { lhs2_rows, lhs2_cols, d_lhs2 };
  Matrix rhs_mat = { lhs1_rows, lhs2_cols, d_rhs };

  dim3 threads_per_block(SUB_MATRIX_DIM, SUB_MATRIX_DIM);
  dim3 blocks_per_grid
    ((lhs2_cols / SUB_MATRIX_DIM) + (lhs2_cols % SUB_MATRIX_DIM != 0), 
     (lhs1_rows / SUB_MATRIX_DIM) + (lhs1_rows % SUB_MATRIX_DIM != 0));
  kernel_mul_mats<<<blocks_per_grid, threads_per_block>>>
    (lhs1_mat, lhs2_mat, rhs_mat);

  cudaDeviceSynchronize();

  cudaMemcpy(rhs, d_rhs, rhs_size, cudaMemcpyDeviceToHost);

  cudaFree(d_lhs1);
  cudaFree(d_lhs2);
  cudaFree(d_rhs);
}

extern "C"
void transpose_mat(float *lhs, size_t lhs_rows, size_t lhs_cols, float *rhs) {
  size_t lhs_size = lhs_rows * lhs_cols * sizeof(float);
  size_t rhs_size = lhs_size;

  float *d_lhs, *d_rhs;
  cudaMalloc(&d_lhs, lhs_size);
  cudaMalloc(&d_rhs, rhs_size);

  cudaMemcpy(d_lhs, lhs, lhs_size, cudaMemcpyHostToDevice);

  Matrix lhs_mat = { lhs_rows, lhs_cols, d_lhs };
  Matrix rhs_mat = { lhs_cols, lhs_rows, d_rhs };

  dim3 threads_per_block(SUB_MATRIX_DIM, SUB_MATRIX_DIM);
  dim3 blocks_per_grid
    ((lhs_cols / SUB_MATRIX_DIM) + (lhs_cols % SUB_MATRIX_DIM != 0),
     (lhs_rows / SUB_MATRIX_DIM) + (lhs_rows % SUB_MATRIX_DIM != 0));
  kernel_transpose_mat<<<blocks_per_grid, threads_per_block>>>
    (lhs_mat, rhs_mat);

  cudaDeviceSynchronize();

  cudaMemcpy(rhs, d_rhs, rhs_size, cudaMemcpyDeviceToHost);

  cudaFree(d_lhs);
  cudaFree(d_rhs);
}

extern "C"
void dot_vecs(float *lhs1, float *lhs2, float *rhs, size_t len) {
  size_t vec_size = len * sizeof(float);

  float *d_lhs1, *d_lhs2, *d_rhs;
  cudaMalloc(&d_lhs1, vec_size);
  cudaMalloc(&d_lhs2, vec_size);
  cudaMalloc(& d_rhs, sizeof(float));

  cudaMemcpy(d_lhs1, lhs1, vec_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lhs2, lhs2, vec_size, cudaMemcpyHostToDevice);

  int threads_per_block = SUB_VECTOR_LEN;
  int blocks_per_grid = (len / SUB_VECTOR_LEN) + (len % SUB_VECTOR_LEN != 0);
  kernel_dot_vecs<<<blocks_per_grid, threads_per_block>>>
    (d_lhs1, d_lhs2, d_rhs);

  cudaDeviceSynchronize();

  cudaMemcpy(rhs, d_rhs, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_lhs1);
  cudaFree(d_lhs2);
  cudaFree(d_rhs);
}

extern "C"
void p_norm_vec(float *lhs, float p, float *rhs, size_t len) {
  size_t vec_size = len * sizeof(float);

  float *d_lhs, *d_rhs;
  cudaMalloc(&d_lhs, vec_size);
  cudaMalloc(&d_rhs, sizeof(float));

  cudaMemcpy(d_lhs, lhs, vec_size, cudaMemcpyHostToDevice);

  int threads_per_block = SUB_VECTOR_LEN;
  int blocks_per_grid = (len / SUB_VECTOR_LEN) + (len % SUB_VECTOR_LEN != 0);
  if (isinf(p))
    kernel_inf_norm_vec<<<blocks_per_grid, threads_per_block>>>
      (d_lhs, d_rhs);
  else
    kernel_p_norm_vec<<<blocks_per_grid, threads_per_block>>>
      (d_lhs, p, d_rhs);

  cudaDeviceSynchronize();

  cudaMemcpy(rhs, d_rhs, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_lhs);
  cudaFree(d_rhs);
}
