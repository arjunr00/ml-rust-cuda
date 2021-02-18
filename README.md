# GPU-Accelerated Machine Learning with Rust and CUDA C++ [WIP]

I took a class on Machine Learning this past quarter at college, and figured the best way to supplement and strengthen my understanding of the underlying mathematics and algorithms of the field (or at least the subset thereof which we learned) would be to implement them myself.
I've been enjoying using Rust (I made a [raytracer](https://github.com/arjunr00/raytracer-rust)), hence the language choice.
I'm also considering this project to be my way of simultaneously learning [CUDA](https://developer.nvidia.com/cuda-toolkit) programming (for GPU acceleration on my NVIDIA card) and [foreign function interfaces in Rust](https://doc.rust-lang.org/nomicon/ffi.html) (since CUDA code is written in C/C++).

My aim is to use as few dependencies as is reasonable, so that I can focus on implementing the mathematics and ML stuff by myself.

## Current Functionality

* Efficient matrix and vector operations using CUDA kernels. This includes:
  * Matrix addition; subtraction; scalar, matrix, and vector multiplication; and transpose.
  * Vector addition, subtraction, scalar multiplication, and dot product.

### Known Bugs

(Bugs marked with **[*]** are considered high priority.)

* **[*]** There appears to be race conditions which cause `math::linear::vector::tests::test_dot_smaller_vecs` and `learning::instance::tests::test_normalize_instance` to fail inconsistently.

## Dependencies

* [libc 0.2](https://crates.io/crates/libc)
* [rand 0.8](https://crates.io/crates/rand)
* [Build] [cc 1.0](https://crates.io/crates/cc)

## Usage

Rustdocs for this library are available [here](https://arjunr00.github.io/ml-rust-cuda).

If you want to build from source, first, of course, clone the repository.
```
$ git clone https://github.com/arjunr00/ml-rust-cuda.git
$ cd ml-rust-cuda
```

To build, simply type:
```
$ cargo build
```

To run the built-in unit and documentation tests, type:
```
$ cargo test -- --test-threads=1
```

## Implementation Details

[TODO]
