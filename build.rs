fn main() {
  println!("cargo:rerun-if-changed=src/cuda/");
  println!("cargo:rerun-if-changed=src/cuda/kernels/");

  cc::Build::new()
    .cpp(true)
    .cuda(true) // implicitly adds C++ support, but I like being explicit
    .flag("-gencode").flag("arch=compute_61,code=sm_61") // compute capability of GTX 1050 = 6.1
    .file("src/cuda/add_n.cu")
    .compile("kernels");

  // replace /usr/local/cuda/lib64 with path to your CUDA installation if necessary
  println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
  println!("cargo:rustc-link-lib=cudart");
}
