use std::{ cmp, fmt, ops };
use libc::{ c_float, size_t };

use super::Matrix;

extern "C" {
  fn dot_vecs(lhs1: *const c_float, lhs2: *const c_float, rhs: *mut c_float, len: size_t);
  fn p_norm_vec(lhs: *const c_float, p: c_float, rhs: *mut c_float, len: size_t);
}

/// A representation of a 32-bit float mathematical vector.
///
/// A vector is simply a wrapper around a [Matrix] with additional and/or
/// vector-specific operations.
/// Don't confuse this with Rust's [Vec] data type.
///
/// ```
/// use ml_rust_cuda::math::linear::Vector;
///
/// let elements = vec![1_f32, 3_f32, 4_f32];
/// let vector = Vector::new(elements);
///
/// // "Vector { matrix: Matrix { elements: [ 1.0, 3.0, 4.0 ], dims: (3, 1) } }
/// println!("{:?}", vector);
/// ```
#[derive(Clone, Debug)]
pub struct Vector {
  matrix: Matrix
}

impl Vector {
  /// Returns a mathematical vector representation from a supplied [f32] vector.
  ///
  /// # Arguments
  ///
  /// * `elements` - A vector of `f32`s for the mathematical vector.
  ///
  /// # Examples
  /// ```
  /// use ml_rust_cuda::math::linear::Vector;
  ///
  /// let vec = vec![9_f32, 5_f32, 8_f32, 3_f32, 4_f32, 0_f32];
  /// let vector = Vector::new(vec);
  ///
  /// println!("{}", vector);
  /// ```
  pub fn new(elements: Vec<f32>) -> Self {
    let n_rows = elements.len();
    Self {
      matrix: Matrix::from_flat(elements, (n_rows, 1))
    }
  }

  /// Returns a mathematical vector around a given [Matrix].
  ///
  /// # Arguments
  ///
  /// * `matrix` - The matrix to contain in the resulting vector.
  ///
  /// # Examples
  /// ```
  /// use ml_rust_cuda::math::linear::{ Matrix, Vector };
  ///
  /// let matrix = Matrix::new(vec![vec![1_f32, 2_f32, 5_f32, 3_f32]]);
  /// let vector = Vector::from_matrix(matrix);
  ///
  /// println!("{}", vector);
  /// ```
  pub fn from_matrix(matrix: Matrix) -> Self {
    Self { matrix }
  }

  /// Returns a mathematical vector of specified dimension filled with zeros.
  ///
  /// # Arguments
  ///
  /// * `n_elements` - The dimension of the resulting vector.
  ///
  /// # Examples
  /// ```
  /// use ml_rust_cuda::math::linear::Vector;
  ///
  /// let vector = Vector::zero(6);
  ///
  /// println!("{}", vector);
  /// ```
  pub fn zero(n_elements: usize) -> Self {
    Self {
      matrix: Matrix::zero((n_elements, 1))
    }
  }

  /// Returns the element of the vector at a given position, or `None` if the
  /// index is out of bounds of the vector.
  ///
  /// # Arguments
  ///
  /// * `i` - The target element's index.
  ///
  /// # Examples
  ///
  /// ```
  /// use ml_rust_cuda::math::linear::Vector;
  ///
  /// let vec = vec![9_f32, 5_f32, 8_f32, 3_f32, 4_f32, 0_f32];
  /// let vector = Vector::new(vec);
  ///
  /// assert_eq!(vector.get(2), Some(8_f32));
  /// assert!(vector.get(8).is_none());
  /// ```
  pub fn get(&self, i: usize) -> Option<f32> {
    self.matrix.get((i, 0))
  }

  /// Replaces the element of the vector at a given position with a given value.
  /// Does nothing if the position is out of bounds of the vector.
  ///
  /// # Arguments
  ///
  /// * `i` - The target element's position.
  /// * `val` - The value with which to replace the target element.
  ///
  /// # Examples
  ///
  /// ```
  /// use ml_rust_cuda::math::linear::Vector;
  ///
  /// let vec = vec![9_f32, 5_f32, 8_f32, 3_f32, 4_f32, 0_f32];
  /// let mut vector = Vector::new(vec);
  ///
  /// vector.set(2, 5_f32);
  ///
  /// assert_eq!(vector.get(2), Some(5_f32));
  /// ```
  pub fn set(&mut self, i: usize, val: f32) {
    self.matrix.set((i, 0), val);
  }

  /// Returns a reference to the interal matrix ofthe vector.
  pub fn matrix(&self) -> &Matrix {
    &self.matrix
  }

  /// Returns the dimension of the vector.
  pub fn dim(&self) -> usize {
    self.matrix.dims().0
  }

  /// Returns the dot product of the vector with another vector.
  ///
  /// Uses a CUDA kernel under the hood.
  ///
  /// # Arguments
  ///
  /// * `other` - A reference to another vector operand.
  ///
  /// # Examples
  /// ```
  /// use ml_rust_cuda::math::{
  ///   f32_eq,
  ///   linear::Vector
  /// };
  ///
  /// let v1 = Vector::new(vec![9_f32, 5_f32, 8_f32, 3_f32, 4_f32, 0_f32]);
  /// let v2 = Vector::new(vec![8_f32, 1_f32, 2_f32, 4_f32, 0_f32, 1_f32]);
  ///
  /// let dot = v1.dot(&v2);
  /// assert!(f32_eq(dot, 105_f32));
  /// ```
  ///
  /// # Panics
  ///
  /// Panics if the dimensions of both vectors are not equal.
  pub fn dot(&self, other: &Self) -> f32 {
    if self.dim() != other.dim() {
      panic!("Vector dot product dimension mismatch: \
              first operand is {}-dimensional but \
              second operand is {}-dimensional",
              self.dim(), other.dim());
    }

    let lhs1 = self.matrix.elements().as_ptr();
    let lhs2 = other.matrix.elements().as_ptr();
    let mut rhs = 0_f32;

    unsafe {
      dot_vecs(lhs1, lhs2, &mut rhs, self.dim())
    };

    rhs
  }

  /// Returns the p-norm of the vector.
  ///
  /// The p-norm of a vector `v` of dimension `n` is defined as
  ///
  /// `||v||p = (|v_1|^p + ... |v_n|^p)^(1/p)`
  ///
  /// for any real number `p >= 0`.
  ///
  /// The infinity norm (i.e. the limit of `||v||p` as `p` tends to infinity)
  /// is the same as `max(v_i)` for each element `v_i` of `v`.
  ///
  /// Uses a CUDA kernel under the hood.
  ///
  /// # Arguments
  ///
  /// * `p` - A positive 32-bit float.
  ///
  /// # Examples
  /// ```
  /// use ml_rust_cuda::math::{
  ///   f32_eq,
  ///   linear::Vector
  /// };
  ///
  /// let v = Vector::new(vec![9_f32, 5_f32, 8_f32, 3_f32, 4_f32, 0_f32]);
  /// let p = 2_f32;
  ///
  /// let p_norm = v.p_norm(p);
  /// let inf_norm = v.p_norm(f32::INFINITY);
  /// assert!(f32_eq(p_norm, 195_f32.sqrt()));
  /// assert!(f32_eq(inf_norm, 9_f32));
  /// ```
  ///
  /// # Panics
  ///
  /// Panics if `p` is less than or equal to 0.
  pub fn p_norm(&self, p: f32) -> f32 {
    if p <= 0_f32 || p.is_nan() {
      panic!("Cannot calculate the p-norm for p = {}", p);
    }

    let lhs = self.matrix.elements().as_ptr();
    let mut rhs = 0_f32;

    unsafe {
      p_norm_vec(lhs, p, &mut rhs, self.dim())
    };

    if p.is_infinite() { rhs } else { rhs.powf(1_f32 / p) }
  }

  /// Returns the vector transposed. See [Matrix::transposed].
  ///
  /// Uses a CUDA kernel under the hood.
  pub fn transposed(&self) -> Self {
    Self {
      matrix: self.matrix.transposed()
    }
  }
}

impl cmp::PartialEq for Vector {
  /// See [Matrix].
  fn eq(&self, other: &Self) -> bool {
    self.matrix == other.matrix
  }
}

impl ops::Add for &Vector {
  type Output = Vector;

  /// See [Matrix].
  fn add(self, other: Self) -> Self::Output {
    Self::Output {
      matrix: &self.matrix + &other.matrix
    }
  }
}

impl ops::Sub for &Vector {
  type Output = Vector;

  /// See [Matrix].
  fn sub(self, other: Self) -> Self::Output {
    Self::Output {
      matrix: &self.matrix - &other.matrix
    }
  }
}

impl ops::Mul<&Vector> for f32 {
  type Output = Vector;

  /// See [Matrix].
  fn mul(self, other: &Vector) -> Self::Output {
    Self::Output {
      matrix: self * &other.matrix
    }
  }
}

impl fmt::Display for Vector {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.matrix.fmt(f)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::math::f32_eq;

  use rand::seq::SliceRandom;

  #[test]
  fn test_vecs_equal() {
    let v1 = Vector::new(vec![1.618_f32; 1 << 20]);
    let v2 = Vector::new(vec![1.618_f32; 1 << 20]);

    assert_eq!(v1, v2);
  }

  #[test]
  fn test_vecs_not_equal() {
    let v1 = Vector::new(vec![1.618_f32; 1 << 20]);
    let mut v2 = Vector::new(vec![1.618_f32; 1 << 20]);
    v2.set(1 << 16, 1.619_f32);
    v2.set(1 << 13, 2.71_f32);
    v2.set(1 << 19, 4.20_f32);

    assert_ne!(v1, v2);
  }

  #[test]
  fn test_add_vecs() {
    let v1 = Vector::new(vec![1_f32; 1 << 20]);
    let v2 = Vector::new(vec![2_f32; 1 << 20]);

    let expected = Vector::new(vec![3_f32; 1 << 20]);
    assert_eq!(&v1 + &v2, expected);
  }

  #[test]
  fn test_dot_vecs() {
    let v1 = Vector::new(vec![3_f32; (1 << 20) + 123]);
    let v2 = Vector::new(vec![5_f32; (1 << 20) + 123]);

    let expected = (15 * ((1 << 20) + 123)) as f32;
    assert!(f32_eq(v1.dot(&v2), expected));
  }

  #[test]
  fn test_p_norm_vec() {
    let p = 3_f32;
    let v = Vector::new(vec![3_f32; (1 << 20) + 123]);

    let expected = // (3^p * (2^20 + 123)
      ((3_f32.powf(p) * ((1 << 20) + 123) as f32)).powf(1_f32/p);
    assert!(f32_eq(v.p_norm(p), expected));
  }

  #[test]
  fn test_inf_norm_vec() {
    let vals = vec![3.14_f32, 6.67_f32, 6.02_f32, 1.61_f32];
    let mut vec = Vec::<f32>::with_capacity((1 << 20) + 123);
    for _ in 0..vec.capacity() {
      vec.push(vals.choose(&mut rand::thread_rng()).unwrap().clone());
    }
    let v = Vector::new(vec);

    let expected = 6.67_f32;
    assert!(f32_eq(v.p_norm(f32::INFINITY), expected));
  }

  #[test]
  #[should_panic(expected = "Vector dot product dimension mismatch")]
  fn test_dot_dim_mismatch() {
    let v1 = Vector::new(vec![3_f32; (1 << 20) + 123]);
    let v2 = Vector::new(vec![5_f32; (1 << 20) + 122]);

    v1.dot(&v2);
  }

  #[test]
  #[should_panic(expected = "Cannot calculate the p-norm for p = 0")]
  fn test_p_norm_zero() {
    let v = Vector::new(vec![3_f32; (1 << 20) + 123]);

    v.p_norm(0_f32);
  }
}
