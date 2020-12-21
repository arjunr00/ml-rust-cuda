//This file is to be used for profiling CUDA kernels

#include <iostream>

#include "cuda/add.cu"

int main() {
  std::cout << "y[i] = " << add_n(1.0f, 2.0f, 1 << 20) << " for all i" << std::endl;
}
