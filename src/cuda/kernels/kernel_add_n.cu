// to match Rust type names
#ifndef RUST_TYPES
#define RUST_TYPES
typedef unsigned long int u32;
typedef float f32;
#endif

__global__
void kernel_add_n(u32 n, f32 *x, f32 *y) {
  u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  u32 stride = blockDim.x * gridDim.x;
  for (u32 i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

