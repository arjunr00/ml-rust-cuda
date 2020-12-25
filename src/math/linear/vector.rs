use std::{ cmp, fmt, ops };
use libc::{ c_float, size_t };

use super::Matrix;

extern "C" {
  fn dot_vecs(lhs1: *const c_float, lhs2: *const c_float, rhs: *const c_float, len: size_t);
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
  /// # Arguments
  ///
  /// * `other` - A reference to another vector operand.
  ///
  /// # Examples
  /// ```
  /// use ml_rust_cuda::math::linear::Vector;
  ///
  /// let v1 = Vector::new(vec![9_f32, 5_f32, 8_f32, 3_f32, 4_f32, 0_f32]);
  /// let v2 = Vector::new(vec![8_f32, 1_f32, 2_f32, 4_f32, 0_f32, 1_f32]);
  ///
  /// let dot = v1.dot(&v2);
  /// assert_eq!(dot, 105_f32);
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
}

impl cmp::PartialEq for Vector {
  fn eq(&self, other: &Self) -> bool {
    self.matrix == other.matrix
  }
}

impl ops::Add for &Vector {
  type Output = Vector;

  fn add(self, other: Self) -> Self::Output {
    Self::Output {
      matrix: &self.matrix + &other.matrix
    }
  }
}

impl ops::Sub for &Vector {
  type Output = Vector;

  fn sub(self, other: Self) -> Self::Output {
    Self::Output {
      matrix: &self.matrix - &other.matrix
    }
  }
}

impl ops::Mul<&Vector> for f32 {
  type Output = Vector;

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
    assert_eq!(v1.dot(&v2), expected);
  }

  #[test]
  #[should_panic(expected = "Vector dot product dimension mismatch")]
  fn test_dot_dim_mismatch() {
    let v1 = Vector::new(vec![3_f32; (1 << 20) + 123]);
    let v2 = Vector::new(vec![5_f32; (1 << 20) + 122]);

    v1.dot(&v2);
  }
}
