//This file is to be used for profiling CUDA kernels

#include <algorithm>
#include <ctime>
#include <iostream>
#include <time.h>

#include "cuda/linear.cu"

void naive_mul_mats
  (float *lhs1, size_t lhs1_rows, size_t lhs1_cols,
   float *lhs2, size_t lhs2_rows, size_t lhs2_cols,
   float *rhs)
{
  for (int i = 0; i < lhs1_rows; ++i) {
    for (int j = 0; j < lhs1_cols; ++j) {
      float val = 0;
      for (int k = 0; k < lhs1_cols; ++k) {
        val += lhs1[i * lhs1_cols + k] * lhs2[k * lhs2_cols + j];
      }

      rhs[i * lhs2_cols + j] = val;
    }
  }
}

void naive_transpose_mat(float *lhs, size_t lhs_rows, size_t lhs_cols, float *rhs)
{
  for (int i = 0; i < lhs_rows; ++i) {
    for (int j = 0; j < lhs_cols; ++j) {
      rhs[j * lhs_rows + i] = lhs[i * lhs_cols + j];
    }
  }
}

void naive_dot_vecs(float *lhs1, float *lhs2, float *rhs, size_t len) {
  for (int i = 0; i < len; ++i) {
    *rhs += lhs1[i] * lhs2[i];
  }
}

void profile_mul_mats(bool out) {
  size_t dim_1 = 1 << 10, dim_2 = (1 << 10) + 123, dim_3 = (1 << 10) - 231;
  size_t pad_1 = BLOCK_SIZE * ((dim_1 / BLOCK_SIZE) + (dim_1 % BLOCK_SIZE != 0)),
         pad_2 = BLOCK_SIZE * ((dim_2 / BLOCK_SIZE) + (dim_2 % BLOCK_SIZE != 0)),
         pad_3 = BLOCK_SIZE * ((dim_3 / BLOCK_SIZE) + (dim_3 % BLOCK_SIZE != 0));
  float *m1 = new float[pad_1 * pad_3]; // dim_1 x dim_3
  std::fill_n(m1, pad_1 * pad_3, 0.f);
  float *m2 = new float[pad_3 * pad_2]; // dim_3 x dim_2
  std::fill_n(m2, pad_3 * pad_2, 0.f);
  float *m3 = new float[pad_1 * pad_2]; // dim_1 x dim_2

  for (size_t i = 0; i < dim_1; ++i)
    for (size_t j = 0; j < dim_3; ++j)
      m1[i * pad_3 + j] = 2.f;

  for (size_t i = 0; i < dim_3; ++i)
    for (size_t j = 0; j < dim_2; ++j)
      m2[i * pad_2 + j] = 2.f;

  if (out) {
    std::cout << "m1 = " << std::endl;
    for (int i = 0; i < dim_1; ++i) {
      for (int j = 0; j < dim_3; ++j)
        std::cout << m1[i * pad_3 + j] << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << " m2 = " << std::endl;
    for (int i = 0; i < dim_3; ++i) {
      for (int j = 0; j < dim_2; ++j)
        std::cout << m2[i * pad_2 + j] << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout
      << "Multiplying a "
      << dim_1 << "x" << dim_3
      << " matrix with a "
      << dim_3 << "x" << dim_2 << " matrix"
      << std::endl;
  }

  mul_mats(m1, pad_1, pad_3, m2, pad_3, pad_2, m3);

  if (out) {
    std::cout << "m3 = " << std::endl;
    for (int i = 0; i < dim_1; ++i) {
      for (int j = 0; j < dim_2; ++j)
        std::cout << m3[i * pad_2 + j] << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  clock_t start = clock();
  naive_mul_mats(m1, pad_1, pad_3, m2, pad_3, pad_2, m3);
  clock_t end = clock();

  if (out) {
    for (int i = 0; i < dim_1; ++i) {
      for (int j = 0; j < dim_2; ++j)
        std::cout << m3[i * pad_2 + j] << " ";
      std::cout << std::endl;
    }
  }

  std::cerr << "Naive matrix multiplication implementation took "
    << ((double) 1000) * ((double) (end - start)) / CLOCKS_PER_SEC << " ms"
    << std::endl << std::endl;

  delete [] m1;
  delete [] m2;
  delete [] m3;
}

void profile_transpose_mat(bool out) {
  size_t dim_1 = (1 << 10) - 123, dim_2 = (1 << 10) + 13;
  size_t pad_1 = BLOCK_SIZE * ((dim_1 / BLOCK_SIZE) + (dim_1 % BLOCK_SIZE != 0));
  size_t pad_2 = BLOCK_SIZE * ((dim_2 / BLOCK_SIZE) + (dim_2 % BLOCK_SIZE != 0));

  float *m = new float[pad_1 * pad_2];
  std::fill_n(m, pad_1 * pad_2, 0.f);
  float *m_T = new float[pad_2 * pad_1];

  for (size_t i = 0; i < dim_1; ++i)
    for (size_t j = 0; j < dim_2; ++j)
        m[i * pad_2 + j] = 2.f;

  if (out) {
    std::cout << "m = " << std::endl;
    for (int i = 0; i < dim_1; ++i) {
      for (int j = 0; j < dim_2; ++j)
        std::cout << m[i * pad_2 + j] << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  transpose_mat(m, pad_1, pad_2, m_T);

  if (out) {
    std::cout << std::endl;
    std::cout << "m_T = " << std::endl;
    for (int i = 0; i < dim_2; ++i) {
      for (int j = 0; j < dim_1; ++j)
        std::cout << m_T[i * pad_1 + j] << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  clock_t start = clock();
  naive_transpose_mat(m, pad_1, pad_2, m_T);
  clock_t end = clock();

  if (out) {
    for (int i = 0; i < dim_2; ++i) {
      for (int j = 0; j < dim_1; ++j)
        std::cout << m_T[i * pad_1 + j] << " ";
      std::cout << std::endl;
    }
  }

  std::cerr << "Naive matrix transpose implementation took "
    << ((double) 1000) * ((double) (end - start)) / CLOCKS_PER_SEC << " ms"
    << std::endl << std::endl;

  delete [] m;
  delete [] m_T;
}

void profile_dot_vecs(bool out) {
  size_t dim = 1 << 20;
  float *v1 = new float[dim];
  std::fill_n(v1, dim, 2.f);
  float *v2 = new float[dim];
  std::fill_n(v2, dim, 3.f);
  float res;

  if (out) {
    std::cout << "v1 = " << std::endl;
    for (size_t i = 0; i < dim; ++i)
      std::cout << v1[i] << " ";
    std::cout << std::endl << std::endl;

    std::cout << "v2 = " << std::endl;
    for (size_t i = 0; i < dim; ++i)
      std::cout << v2[i] << " ";
    std::cout << std::endl << std::endl;
  }

  dot_vecs(v1, v2, &res, dim);

  if (out)
    std::cout << std::endl << "res = " << res << std::endl;

  res = 0.f;

  clock_t start = clock();
  naive_dot_vecs(v1, v2, &res, dim);
  clock_t end = clock();

  if (out)
    std::cout << std::endl << "res = " << res << std::endl;

  std::cerr << "Naive vector dot product implementation took "
    << ((double) 1000) * ((double) (end - start)) / CLOCKS_PER_SEC << " ms"
    << std::endl << std::endl;

  delete [] v1;
  delete [] v2;
}

int main() {
  profile_transpose_mat(false);
  profile_mul_mats(false);
  profile_dot_vecs(false);
}
