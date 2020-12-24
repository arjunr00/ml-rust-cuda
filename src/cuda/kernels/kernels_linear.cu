#include "math.h"

extern "C"
const size_t BLOCK_SIZE = 32;

typedef struct {
  size_t rows;
  size_t cols;
  float *elements;
} Matrix;

__device__ float *sub_block(Matrix m, int block_row, int block_col) {
  return m.elements + (block_row * BLOCK_SIZE * m.cols) + (block_col * BLOCK_SIZE);
}

__global__
void kernel_eq_mats(float *lhs, float *rhs, size_t len, float epsilon, bool *equal) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; *equal && i < len; i += stride)
    if (fabs(lhs[i] - rhs[i]) >= epsilon)
      *equal = false;
}

__global__
void kernel_add_mats(float *lhs1, float *lhs2, float *rhs, size_t len) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < len; i += stride)
    rhs[i] = lhs1[i] + lhs2[i];
}

__global__
void kernel_mul_scalar_mat(float scalar, float *lhs, float *rhs, size_t len) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < len; i += stride)
    rhs[i] = scalar * lhs[i];
}

__global__
void kernel_mul_mats(Matrix lhs1, Matrix lhs2, Matrix rhs) {
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;

  float *rhs_sub = sub_block(rhs, block_row, block_col);

  float rhs_elem_val = 0;

  int block_cols = (lhs1.cols / BLOCK_SIZE) + (lhs1.cols % BLOCK_SIZE != 0);
  for (int sub_i = 0; sub_i < block_cols; ++sub_i) {
    float *lhs1_sub = sub_block(lhs1, block_row, sub_i);
    float *lhs2_sub = sub_block(lhs2, sub_i, block_col);

    __shared__ float shared_lhs1_sub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_lhs2_sub[BLOCK_SIZE][BLOCK_SIZE];

    shared_lhs1_sub[thread_row][thread_col]
      = lhs1_sub[thread_row * lhs1.cols + thread_col];
    shared_lhs2_sub[thread_row][thread_col]
      = lhs2_sub[thread_row * lhs2.cols + thread_col];

    __syncthreads();

    for (int rhs_i = 0; rhs_i < BLOCK_SIZE; ++rhs_i) {
      rhs_elem_val +=
        shared_lhs1_sub[thread_row][rhs_i] * shared_lhs2_sub[rhs_i][thread_col];
    }

    __syncthreads();
  }

  rhs_sub[thread_row * rhs.cols + thread_col] = rhs_elem_val;
}

__global__
void kernel_transpose_mat(Matrix lhs, Matrix rhs) {
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;

  float *lhs_sub = sub_block(lhs, block_row, block_col);
  float *rhs_sub = sub_block(rhs, block_col, block_row);

  __shared__ float shared_lhs_sub[BLOCK_SIZE][BLOCK_SIZE];

  shared_lhs_sub[thread_row][thread_col]
    = lhs_sub[thread_row * lhs.cols + thread_col];

  __syncthreads();

  rhs_sub[thread_col * rhs.cols + thread_row]
    = shared_lhs_sub[thread_row][thread_col];
}

__global__
void kernel_dot_vecs(float *lhs1, float *lhs2, float *rhs) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ float sub_vec[8 * BLOCK_SIZE];
  sub_vec[threadIdx.x] = lhs1[index] * lhs2[index];

  __syncthreads();

  if (threadIdx.x == 0) {
    float sub_vec_sum = 0.0f;
    for (int i = 0; i < 8 * BLOCK_SIZE; ++i)
      sub_vec_sum += sub_vec[i];

    atomicAdd(rhs, sub_vec_sum);
  }
}
