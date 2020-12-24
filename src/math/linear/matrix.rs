use std::{ cmp, ops };
use libc::{ c_float, size_t };

extern "C" {
  static BLOCK_SIZE: size_t;
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

#[derive(Clone, Debug)]
pub struct Matrix {
  elements: Vec<f32>,
  dims: (usize, usize)
}

impl Matrix {
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

  pub fn from_flat(elements: Vec<f32>, dims: (usize, usize)) -> Self {
    Self { elements, dims }
  }

  pub fn zero(n_rows: usize, n_cols: usize) -> Self {
    Self {
      elements: vec![0_f32; n_rows * n_cols],
      dims: (n_rows, n_cols)
    }
  }

  pub fn get(&self, i: usize, j: usize) -> Option<f32> {
    Some(self.elements.get(i * self.dims.1 + j)?.clone())
  }

  pub fn set(&mut self, i: usize, j: usize, val: f32) {
    if let Some(elem) = self.elements.get_mut(i * self.dims.1 + j) {
      *elem = val;
    }
  }

  pub fn elements(&self) -> &Vec<f32> {
    &self.elements
  }

  pub fn dims(&self) -> (usize, usize) {
    self.dims
  }

  pub fn zero_padded(&self, amount: usize) -> Self {
    let n_extra_rows
      = amount * ((self.dims.0 / amount) + ((self.dims.0 % amount != 0) as usize))
        - self.dims.0;
    let n_extra_cols
      = amount * ((self.dims.1 / amount) + ((self.dims.1 % amount != 0) as usize))
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

  pub fn transposed(&self) -> Self {
    let block_size = unsafe { BLOCK_SIZE };
    let mut result = Matrix::zero(self.dims.1, self.dims.0).zero_padded(block_size);

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

  fn add(self, other: Self) -> Self::Output {
    if self.dims != other.dims {
      panic!("Matrix addition dimension mismatch: \
              first operand is {}x{} \
              but second operand is {}x{}",
              self.dims.0, self.dims.1,
              other.dims.0, other.dims.1);
    }

    let dims = self.dims;
    let mut result = Matrix::zero(dims.0, dims.1);
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

  fn sub(self, other: Self) -> Self::Output {
    self + &(-1_f32 * other)
  }
}

impl ops::Mul<&Matrix> for f32 {
  type Output = Matrix;

  fn mul(self, other: &Matrix) -> Self::Output {
    let dims = other.dims;
    let mut result = Matrix::zero(dims.0, dims.1);
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

  fn mul(self, other: Self) -> Self::Output {
    if self.dims.1 != other.dims.0 {
      panic!("Matrix multiplication dimension mismatch: \
              first operand is {}x{} \
              but second operand is {}x{}",
              self.dims.0, self.dims.1,
              other.dims.0, other.dims.1);
    }

    let block_size = unsafe { BLOCK_SIZE };
    let mut result = Matrix::zero(self.dims.0, other.dims.1).zero_padded(block_size);

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
    m2.set(1 << 7, 1 << 9, 1.619_f32);
    m2.set(1 << 8, 1 << 5, 2.71_f32);
    m2.set(1 << 9, 1 << 9, 4.20_f32);


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

    let expected = Matrix::new(vec![vec![(2 * 3 * dims.0) as f32; dims.2]; dims.1]);
    assert_eq!(&m1 * &m2, expected);
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
