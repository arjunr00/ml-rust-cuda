# GPU-Accelerated Machine Learning with Rust and CUDA C

I took a class on Machine Learning this past quarter at college, and figured the best way to supplement and strengthen my understanding of the underlying mathematics and algorithms of the field (or at least the subset thereof which we learned) would be to implement them myself.
I've been enjoying using Rust (I made a [raytracer](https://github.com/arjunr00/raytracer-rust)), hence the language choice.
I'm also considering this project to be my way of simultaneously learning [CUDA](https://developer.nvidia.com/cuda-toolkit) programming (for GPU acceleration on my NVIDIA card) and [foreign function interfaces in Rust](https://doc.rust-lang.org/nomicon/ffi.html) (since CUDA code is written in C/C++).

My aim is to use as few dependencies as is reasonable, so that I can focus on implementing the mathematics and ML stuff by myself.
