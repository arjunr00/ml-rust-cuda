use std::{ cmp, fmt, ops };
use libc::{ c_float, size_t };

use super::Vector;

extern "C" {
  static SUB_MATRIX_DIM: size_t;
  fn eq_mats(lhs: *const c_float, rhs: *const c_float, len: size_t) -> bool;
  fn add_mats(lhs1: *const c_float, lhs2: *const c_float, rhs: *mut c_float, len: size_t);
  fn mul_scalar_mat(scalar: c_float, lhs: *const c_float, rhs: *mut c_float, len: size_t);
  fn mul_mats(
    lhs1: *const c_float, lhs1_rows: size_t, lhs1_cols: size_t,
    lhs2: *const c_float, lhs2_rows: size_t, lhs2_cols: size_t,
    rhs: *mut c_float
  );
  fn transpose_mat
    (lhs: *const c_float, lhs_rows: size_t, lhs_cols: size_t, rhs: *mut c_float);
}

/// A representation of a 32-bit float matrix.
///
/// A matrix's elements are stored as a flat, 1-dimensional vector, with an
/// internal tuple which defines the dimensions of the matrix, i.e. how the
/// matrix's elements are interpreted, in (rows, columns) format.
///
/// ```
/// use ml_rust_cuda::math::linear::Matrix;
///
/// let elements =
///   vec![vec![1_f32, 3_f32, 4_f32],
///        vec![2_f32, 5_f32, 8_f32]];
/// let matrix = Matrix::new(elements);
///
/// // "Matrix { elements: [ 1.0, 3.0, 4.0, 2.0, 5.0, 8.0 ], dims: (2, 3) }"
/// println!("{:?}", matrix);
/// ```
#[derive(Clone, Debug)]
pub struct Matrix {
  elements: Vec<f32>,
  dims: (usize, usize)
}

impl Matrix {
  /// Returns a matrix representation from a supplied 2-dimensional [f32] vector.
  ///
  /// # Arguments
  ///
  /// * `elements` - A [Vec] of rows of [f32]s for the matrix.
  ///
  /// # Examples
  /// ```
  /// use ml_rust_cuda::math::linear::Matrix;
  ///
  /// let vec_of_vecs =
  ///   vec![vec![1_f32, 3_f32, 4_f32],
  ///        vec![2_f32, 5_f32, 8_f32],
  ///        vec![9_f32, 5_f32, 8_f32],
  ///        vec![3_f32, 4_f32, 0_f32]];
  /// let matrix = Matrix::new(vec_of_vecs);
  ///
  /// println!("{}", matrix);
  /// ```
  ///
  /// # Panics
  ///
  /// Panics if all [Vec]s within `elements` are not of equal length.
  pub fn new(elements: Vec<Vec<f32>>) -> Self {
    let n_cols: usize =
      if let Some(first_row) = elements.get(0) {
        first_row.len()
      } else {
        0
      };

    if let Some(mismatch) = elements.iter().find(|row| row.len() != n_cols) {
      panic!("Matrix row length mismatch: expected {} but got {}", n_cols, mismatch.len());
    }

    let n_rows = elements.len();
    let elements: Vec<f32> = elements.into_iter().flatten().collect();

    Self {
      elements,
      dims: (n_rows, n_cols)
    }
  }

  /// Returns a matrix representation from a flat 1-dimensional [f32] vector
  /// of specified dimensions.
  ///
  /// # Arguments
  ///
  /// * `elements` - A flat vector of [f32]s **stored in row-major form** (i.e.
  ///                elements in the same row are adjacent to each other in
  ///                `elements`).
  /// * `dims` - The dimensions of the matrix in (rows, columns).
  ///
  /// # Examples
  /// ```
  /// use ml_rust_cuda::math::linear::Matrix;
  ///
  /// let flat_vec =
  ///   vec![1_f32, 3_f32, 4_f32,
  ///        2_f32, 5_f32, 8_f32,
  ///        9_f32, 5_f32, 8_f32,
  ///        3_f32, 4_f32, 0_f32];
  /// let matrix = Matrix::from_flat(flat_vec, (4, 3));
  ///
  /// println!("{}", matrix);
  /// ```
  pub fn from_flat(elements: Vec<f32>, dims: (usize, usize)) -> Self {
    Self { elements, dims }
  }

  /// Returns a matrix of specified dimensions filled with zeros.
  ///
  /// # Arguments
  ///
  /// * `dims` - The dimensions of the matrix in (rows, columns).
  ///
  /// # Examples
  /// ```
  /// use ml_rust_cuda::math::linear::Matrix;
  ///
  /// let matrix = Matrix::zero((4, 3));
  ///
  /// println!("{}", matrix);
  /// ```
  pub fn zero(dims: (usize, usize)) -> Self {
    Self {
      elements: vec![0_f32; dims.0 * dims.1],
      dims: dims
    }
  }

  /// Returns the element of the matrix at a given position, or `None` if the index
  /// is out of bounds of the matrix.
  ///
  /// # Arguments
  ///
  /// * `pos` - The target element's position in (row, column) form, zero-indexed.
  ///
  /// # Examples
  ///
  /// ```
  /// use ml_rust_cuda::math::linear::Matrix;
  ///
  /// let vec_of_vecs =
  ///   vec![vec![1_f32, 3_f32, 4_f32],
  ///        vec![2_f32, 5_f32, 8_f32]];
  /// let matrix = Matrix::new(vec_of_vecs);
  ///
  /// assert_eq!(matrix.get((1, 2)), Some(8_f32));
  /// assert!(matrix.get((2, 0)).is_none());
  /// ```
  pub fn get(&self, pos: (usize, usize)) -> Option<f32> {
    Some(self.elements.get(pos.0 * self.dims.1 + pos.1)?.clone())
  }

  /// Replaces the element of the matrix at a given position with a given value.
  /// Does nothing if the position is out of bounds of the matrix.
  ///
  /// # Arguments
  ///
  /// * `pos` - The target element's position in (row, column) form, zero-indexed.
  /// * `val` - The value with which to replace the target element.
  ///
  /// # Examples
  ///
  /// ```
  /// use ml_rust_cuda::math::linear::Matrix;
  ///
  /// let vec_of_vecs =
  ///   vec![vec![1_f32, 3_f32, 4_f32],
  ///        vec![2_f32, 5_f32, 8_f32]];
  /// let mut matrix = Matrix::new(vec_of_vecs);
  ///
  /// matrix.set((1, 2), 5_f32);
  ///
  /// assert_eq!(matrix.get((1, 2)), Some(5_f32));
  /// ```
  pub fn set(&mut self, pos: (usize, usize), val: f32) {
    if let Some(elem) = self.elements.get_mut(pos.0 * self.dims.1 + pos.1) {
      *elem = val;
    }
  }

  /// Returns a reference to the matrix's internal flat element vector.
  pub fn elements(&self) -> &Vec<f32> {
    &self.elements
  }

  /// Returns a reference to the matrix's dimension tuple in the form (rows,
  /// columns), zero-indexed.
  pub fn dims(&self) -> &(usize, usize) {
    &self.dims
  }

  /// Returns a copy of the matrix padded with zeros such that each dimension
  /// of the resulting matrix is a multiple of a specified value.
  ///
  /// # Arguments
  ///
  /// * `target` - The number by which the resulting matrix's dimensions should
  ///              be divisible by.
  ///
  /// # Examples
  /// ```
  /// use ml_rust_cuda::math::linear::Matrix;
  ///
  /// let vec_of_vecs =
  ///   vec![vec![1_f32, 3_f32],
  ///        vec![2_f32, 5_f32],
  ///        vec![0_f32, 7_f32],
  ///        vec![8_f32, 4_f32]];
  /// let matrix = Matrix::new(vec_of_vecs);
  ///
  /// println!("{}", matrix.zero_padded(3));
  /// ```
  pub fn zero_padded(&self, target: usize) -> Self {
    let n_extra_rows
      = target * ((self.dims.0 / target) + ((self.dims.0 % target != 0) as usize))
        - self.dims.0;
    let n_extra_cols
      = target * ((self.dims.1 / target) + ((self.dims.1 % target != 0) as usize))
        - self.dims.1;

    let mut new_elements: Vec<f32>
      = Vec::with_capacity((self.dims.0 + n_extra_rows) * (self.dims.1 + n_extra_cols));

    for i in 0..self.dims.0 {
      new_elements.extend(&self.elements[(i * self.dims.1)..((i+1) * self.dims.1)]);
      new_elements.extend(vec![0_f32; n_extra_cols]);
    }

    for _ in 0..n_extra_rows {
      new_elements.extend(vec![0_f32; self.dims.1 + n_extra_cols]);
    }

    let new_dims = (self.dims.0 + n_extra_rows, self.dims.1 + n_extra_cols);

    Matrix::from_flat(new_elements, new_dims) 
  }

  /// Returns a copy of the matrix truncated such that the resulting matrix's
  /// dimensions match the provided dimensions.
  ///
  /// # Arguments
  ///
  /// * `new_dims` - The dimensions of the new matrix in (rows, columns).
  ///
  /// # Examples
  /// ```
  /// use ml_rust_cuda::math::linear::Matrix;
  ///
  /// let vec_of_vecs =
  ///   vec![vec![1_f32, 3_f32, 0_f32, 0_f32],
  ///        vec![2_f32, 5_f32, 0_f32, 0_f32],
  ///        vec![0_f32, 7_f32, 0_f32, 0_f32],
  ///        vec![8_f32, 4_f32, 0_f32, 0_f32],
  ///        vec![0_f32, 0_f32, 0_f32, 0_f32],
  ///        vec![0_f32, 0_f32, 0_f32, 0_f32]];
  /// let matrix = Matrix::new(vec_of_vecs);
  ///
  /// println!("{}", matrix.truncated((4, 2)));
  /// ```
  pub fn truncated(&self, new_dims: (usize, usize)) -> Self {
    let new_dims =
      (if new_dims.0 > self.dims.0 { self.dims.0 } else { new_dims.0 },
       if new_dims.1 > self.dims.1 { self.dims.1 } else { new_dims.1 });

    let mut new_elements: Vec<f32>
      = Vec::with_capacity(new_dims.0 * new_dims.1);

    for i in 0..new_dims.0 {
      let adjusted_i = i * self.dims.1;
      new_elements.extend(&self.elements[adjusted_i..(adjusted_i + new_dims.1)]);
    }

    Matrix::from_flat(new_elements, new_dims)
  }

  /// Returns a transposed copy of the matrix.
  ///
  /// Uses a CUDA kernel under the hood.
  ///
  /// # Examples
  /// ```
  /// use ml_rust_cuda::math::linear::Matrix;
  ///
  /// let vec_of_vecs =
  ///   vec![vec![1_f32, 3_f32],
  ///        vec![2_f32, 5_f32],
  ///        vec![0_f32, 7_f32],
  ///        vec![8_f32, 4_f32]];
  /// let matrix = Matrix::new(vec_of_vecs);
  ///
  /// println!("{}", matrix.transposed());
  /// ```
  pub fn transposed(&self) -> Self {
    let block_size = unsafe { SUB_MATRIX_DIM };

    // Pad the matrices to BLOCK_SIZE since kernel operates on sub-matrices
    // of size BLOCK_SIZE x BLOCK_SIZE
    let mut result = Matrix::zero((self.dims.1, self.dims.0)).zero_padded(block_size);
    let lhs_padded = self.zero_padded(block_size);

    let lhs = lhs_padded.elements.as_ptr();
    let rhs = result.elements.as_mut_ptr();

    unsafe {
      transpose_mat(lhs, lhs_padded.dims.0, lhs_padded.dims.1, rhs);
    }

    result.truncated((self.dims.1, self.dims.0))
  }
}

impl cmp::PartialEq for Matrix {
  /// Uses a CUDA kernel under the hood.
  fn eq(&self, other: &Self) -> bool {
    if self.elements.len() != other.elements.len() {
      return false;
    }

    let len = self.elements.len();
    let lhs = self.elements.as_ptr();
    let rhs = other.elements.as_ptr();

    self.dims == other.dims && unsafe {
      eq_mats(lhs, rhs, len)
    }
  }
}

impl ops::Add for &Matrix {
  type Output = Matrix;

  /// Uses a CUDA kernel under the hood.
  /// 
  /// # Panics
  ///
  /// Panics if both operand matrices are not of the same dimension.
  fn add(self, other: Self) -> Self::Output {
    if self.dims != other.dims {
      panic!("Matrix addition dimension mismatch: \
              first operand is {}x{} \
              but second operand is {}x{}",
              self.dims.0, self.dims.1,
              other.dims.0, other.dims.1);
    }

    let dims = self.dims;
    let mut result = Matrix::zero(dims);
    let len = dims.0 * dims.1;

    let lhs1 = self.elements.as_ptr();
    let lhs2 = other.elements.as_ptr();
    let rhs  = result.elements.as_mut_ptr();

    unsafe {
      add_mats(lhs1, lhs2, rhs, len);
    }

    result
  }
}

impl ops::Sub for &Matrix {
  type Output = Matrix;

  /// Uses two CUDA kernels under the hood (one for scalar multiplication by -1,
  /// the other for addition).
  /// 
  /// # Panics
  ///
  /// Panics if both operand matrices are not of the same dimension.
  fn sub(self, other: Self) -> Self::Output {
    self + &(-1_f32 * other)
  }
}

impl ops::Mul<&Matrix> for f32 {
  type Output = Matrix;

  /// Uses a CUDA kernel under the hood.
  fn mul(self, other: &Matrix) -> Self::Output {
    let dims = other.dims;
    let mut result = Matrix::zero(dims);
    let len = dims.0 * dims.1;

    let lhs = other.elements.as_ptr();
    let rhs = result.elements.as_mut_ptr();

    unsafe {
      mul_scalar_mat(self, lhs, rhs, len)
    };

    result
  }
}

impl ops::Mul for &Matrix {
  type Output = Matrix;

  /// Uses a CUDA kernel under the hood.
  /// 
  /// # Panics
  ///
  /// Panics if the number of columns in the first operand is not equal to the
  /// number of rows in the second operand.
  fn mul(self, other: Self) -> Self::Output {
    if self.dims.1 != other.dims.0 {
      panic!("Matrix multiplication dimension mismatch: \
              first operand is {}x{} \
              but second operand is {}x{}",
              self.dims.0, self.dims.1,
              other.dims.0, other.dims.1);
    }

    let block_size = unsafe { SUB_MATRIX_DIM };

    // Pad the matrices to BLOCK_SIZE since kernel operates on sub-matrices
    // of size BLOCK_SIZE x BLOCK_SIZE
    let mut result
      = Matrix::zero((self.dims.0, other.dims.1)).zero_padded(block_size);
    let lhs1_padded = self.zero_padded(block_size);
    let lhs2_padded = other.zero_padded(block_size);

    let lhs1 = lhs1_padded.elements.as_ptr();
    let lhs2 = lhs2_padded.elements.as_ptr();
    let rhs  = result.elements.as_mut_ptr();

    unsafe {
      mul_mats(
        lhs1, lhs1_padded.dims.0, lhs1_padded.dims.1,
        lhs2, lhs2_padded.dims.0, lhs2_padded.dims.1,
        rhs
      );
    }

    result.truncated((self.dims.0, other.dims.1))
  }
}

impl ops::Mul<&Vector> for &Matrix {
  type Output = Vector;

  /// Uses the same CUDA kernel as matrix multiplication.
  ///
  /// # Panics
  ///
  /// Panics if the number of columns in the matrix is not equal to the
  /// dimension of the vector.
  fn mul(self, other: &Vector) -> Self::Output {
    Vector::from_matrix(self * other.matrix())
  }
}

impl fmt::Display for Matrix {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let mut matrix_str: String = "[".to_owned();

    for i in 0..self.dims.0 {
      if i != 0 {
        matrix_str += " ";
      }
      matrix_str += "[";
      for j in 0..self.dims.1 {
        matrix_str +=
          &format!("{:>4}", self.elements[i * self.dims.1 + j]);
        if j != self.dims.1 - 1 {
          matrix_str += ",\t";
        }
      }
      matrix_str += "]";
      if i != self.dims.0 - 1 {
        matrix_str += ", \n";
      }
    }
    matrix_str += "]";

    f.write_str(&matrix_str)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  use crate::math::f32_eq;

  // Happy path tests

  #[test]
  fn test_pad_mat() {
    let pad = 32;
    let dims = ((1 << 10) - 123, (1 << 8) + 42);
    let m = Matrix::new(vec![vec![2_f32; dims.1]; dims.0]);

    let pad_dims = (pad * ((dims.0 / pad) + 1), pad * ((dims.1 / pad) + 1));
    let mut row = vec![2_f32; dims.1];
    row.append(&mut vec![0_f32; pad_dims.1 - dims.1]);
    let mut rows = vec![row; dims.0];
    rows.append(&mut vec![vec![0_f32; pad_dims.1]; pad_dims.0 - dims.0]);
    let expected = Matrix::new(rows);
    assert_eq!(m.zero_padded(pad), expected);
  }

  #[test]
  fn test_trunc_mat() {
    let pad = 32;
    let dims = ((1 << 10) - 123, (1 << 8) + 42);
    let m = Matrix::new(vec![vec![3.14_f32; dims.1]; dims.0]);

    assert_eq!(m.zero_padded(pad).truncated(dims), m);
  }

  #[test]
  fn test_transpose_mat() {
    let dims = ((1 << 10) - 123, (1 << 7) + 13);
    let m = Matrix::new(vec![vec![2.71_f32; dims.1]; dims.0]);

    let expected = Matrix::new(vec![vec![2.71_f32; dims.0]; dims.1]);
    assert_eq!(m.transposed(), expected);
  }

  #[test]
  fn test_mats_equal() {
    let m1 = Matrix::new(vec![vec![1.618_f32; 1 << 10]; (1 << 9) + 10]);
    let m2 = Matrix::new(vec![vec![1.618_f32; 1 << 10]; (1 << 9) + 10]);

    let mut equal = true;
    for i in 0..m1.elements().len() {
      if !f32_eq(m1.elements()[i], m2.elements()[i]) {
        equal = false;
        break;
      }
    }
    assert_eq!(m1 == m2, equal);
  }

  #[test]
  fn test_mats_not_equal() {
    let m1 = Matrix::new(vec![vec![1.618_f32; 1 << 10]; (1 << 9) + 10]);
    let mut m2 = Matrix::new(vec![vec![1.618_f32; 1 << 10]; (1 << 9) + 10]);
    m2.set((1 << 7, 1 << 9), 1.619_f32);
    m2.set((1 << 8, 1 << 5), 2.71_f32);
    m2.set((1 << 9, 1 << 9), 4.20_f32);


    let mut equal = true;
    for i in 0..m1.elements().len() {
      if !f32_eq(m1.elements()[i], m2.elements()[i]) {
        equal = false;
        break;
      }
    }
    assert_eq!(m1 == m2, equal);
  }

  #[test]
  fn test_add_mats() {
    let m1 = Matrix::new(vec![vec![1_f32; 1 << 10]; (1 << 7) + 10]);
    let m2 = Matrix::new(vec![vec![2_f32; 1 << 10]; (1 << 7) + 10]);

    let expected = Matrix::new(vec![vec![3_f32; 1 << 10]; (1 << 7) + 10]);
    assert_eq!(&m1 + &m2, expected);
  }

  #[test]
  fn test_sub_mats() {
    let m1 = Matrix::new(vec![vec![1_f32; 1 << 10]; (1 << 7) + 10]);
    let m2 = Matrix::new(vec![vec![2_f32; 1 << 10]; (1 << 7) + 10]);

    let expected = Matrix::new(vec![vec![-1_f32; 1 << 10]; (1 << 7) + 10]);
    assert_eq!(&m1 - &m2, expected);
  }

  #[test]
  fn test_mul_scalar_mat() {
    let k = 6.022_f32;
    let m = Matrix::new(vec![vec![1_f32; 1 << 10]; (1 << 8) + 10]);

    let expected = Matrix::new(vec![vec![6.022_f32; 1 << 10]; (1 << 8) + 10]);
    assert_eq!(k * &m, expected);
  }

  #[test]
  fn test_mul_mats() {
    let dims = ((1 << 10) + 10, (1 << 10) - 123, 1 << 5);
    let m1 = Matrix::new(vec![vec![2_f32; dims.0]; dims.1]);
    let m2 = Matrix::new(vec![vec![3_f32; dims.2]; dims.0]);

    let expected
      = Matrix::new(vec![vec![2_f32 * 3_f32 * dims.0 as f32; dims.2]; dims.1]);
    assert_eq!(&m1 * &m2, expected);
  }

  #[test]
  fn test_mul_mat_vec() {
    let dims = ((1 << 10) + 10, (1 << 10) - 123, 1 << 5);
    let m = Matrix::new(vec![vec![2_f32; dims.0]; dims.1]);
    let v = Vector::new(vec![3_f32; dims.0]);

    let expected
      = Vector::new(vec![2_f32 * 3_f32 * dims.0 as f32; dims.1]);
    assert_eq!(&m * &v, expected);
  }

  // Failure tests

  #[test]
  #[should_panic(expected = "Matrix row length mismatch")]
  fn test_new_row_mismatch() {
    let mut vecs = vec![vec![1_f32; 1 << 8]; 1 << 9];
    vecs.push(vec![1_f32; 1 << 10]);
    vecs.extend(vec![vec![1_f32; 1 << 8]; 1 << 9]);

    Matrix::new(vecs);
  }

  #[test]
  #[should_panic(expected = "Matrix addition dimension mismatch")]
  #[allow(unused_must_use)]
  fn test_add_dim_mismatch() {
    let m1 = Matrix::new(vec![vec![3_f32; 1 << 10]; 1 << 10]);
    let m2 = Matrix::new(vec![vec![3_f32; (1 << 10) + 12]; 1 << 10]);

    &m1 + &m2;
  }

  #[test]
  #[should_panic(expected = "Matrix multiplication dimension mismatch")]
  #[allow(unused_must_use)]
  fn test_mul_dim_mismatch() {
    let m1 = Matrix::new(vec![vec![3_f32; 1 << 9]; 1 << 10]);
    let m2 = Matrix::new(vec![vec![3_f32; 1 << 9]; 1 << 10]);

    &m1 * &m2;
  }
}
