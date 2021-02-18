use super::linear::{ Vector };

/// Returns the arithmetic mean of the values contained in a mathematical
/// vector.
/// 
/// # Examples
/// ```
/// use ml_rust_cuda::math::{
///   f32_eq,
///   linear::Vector,
///   stats::mean
/// };
///
/// let vec = Vector::new(vec![9_f32, 5_f32, 8_f32, 3_f32, 4_f32, 0_f32]);
///
/// let mean = mean(&vec);
/// assert!(f32_eq(mean, 29_f32 / 6_f32));
/// ```
pub fn mean(vec: &Vector) -> f32 {
  let ones = Vector::new(vec![1_f32; vec.dim()]);
  (1_f32 / (vec.dim() as f32)) * vec.dot(&ones)
}

/// Returns the standard deviation of the values contained in a mathematical
/// vector.
/// 
/// # Examples
/// ```
/// use ml_rust_cuda::math::{
///   f32_eq,
///   linear::Vector,
///   stats::std_dev
/// };
///
/// let vec = Vector::new(vec![9_f32, 5_f32, 8_f32, 3_f32, 4_f32, 0_f32]);
///
/// let std_dev = std_dev(&vec);
/// println!("{}", std_dev);
/// assert!(f32_eq(std_dev, 3.3116));
/// ```
pub fn std_dev(vec: &Vector) -> f32 {
  (1_f32 / ((vec.dim() - 1) as f32).sqrt()) *
    (vec - &Vector::new(vec![mean(vec); vec.dim()])).p_norm(2_f32)
}
