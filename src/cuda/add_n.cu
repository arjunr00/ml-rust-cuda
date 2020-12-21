#include <iostream>
#include <math.h>

#include "kernels/kernel_add_n.cu"

// to match Rust type names
#ifndef RUST_TYPES
#define RUST_TYPES
typedef unsigned long int u32;
typedef float f32;
#endif

extern "C"
f32 add_n(f32 x_val, f32 y_val, u32 n) {
  f32 *x, *y;
  cudaMallocManaged(&x, n * sizeof(f32));
  cudaMallocManaged(&y, n * sizeof(f32));

  for (u32 i = 0; i < n; ++i) {
    x[i] = x_val;
    y[i] = y_val;
  }

  u32 block_size = 256;
  u32 n_blocks = ((n - 1) / block_size) + 1;
  kernel_add_n<<<n_blocks, block_size>>>(n, x, y);

  cudaDeviceSynchronize();

  f32 max_error = 0.0;
  for (u32 i = 0; i < n; ++i)
    max_error = fmax(max_error, fabs(y[i] - 3.0));

  f32 ret_val = max_error + y[0];

  cudaFree(x);
  cudaFree(y);

  return ret_val;
}
