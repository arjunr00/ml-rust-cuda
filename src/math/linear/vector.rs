use std::{ cmp, ops };
use libc::{ c_float, size_t };

use super::matrix::Matrix;

extern "C" {
  fn dot_vecs(lhs1: *const c_float, lhs2: *const c_float, rhs: *const c_float, len: size_t);
}

#[derive(Clone, Debug)]
pub struct Vector {
  matrix: Matrix
}

impl Vector {
  pub fn new(elements: Vec<f32>) -> Self {
    let n_rows = elements.len();
    Self {
      matrix: Matrix::from_flat(elements, (n_rows, 1))
    }
  }

  pub fn zero(n_elements: usize) -> Self {
    Self {
      matrix: Matrix::zero(n_elements, 1)
    }
  }

  pub fn set(&mut self, i: usize, val: f32) {
    self.matrix.set(i, 0, val);
  }

  pub fn dim(&self) -> usize {
    self.matrix.dims().0
  }
  
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
