use std::{ cmp, ops };
use libc::{ c_float, size_t };

use super::matrix;

#[derive(Clone, Debug)]
pub struct Vector {
  elements: Vec<f32>
}

impl Vector {
  pub fn new(elements: Vec<f32>) -> Self {
    Self { elements }
  }

  pub fn zero(n_elements: usize) -> Self {
    Self {
      elements: vec![0_f32; n_elements]
    }
  }

  pub fn set(&mut self, i: usize, val: f32) {
    self.elements[i] = val;
  }

  pub fn dim(&self) -> usize {
    self.elements.len()
  }
}

extern "C" {
  fn eq_vecs(lhs: *const c_float, rhs: *const c_float, len: size_t) -> bool;
  fn add_vecs(lhs1: *const c_float, lhs2: *const c_float, rhs: *mut c_float, len: size_t);
}

impl cmp::PartialEq for Vector {
  fn eq(&self, other: &Self) -> bool {
    if self.elements.len() != other.elements.len() {
      return false;
    }

    let len = self.elements.len();
    let lhs = self.elements.as_ptr();
    let rhs = other.elements.as_ptr();

    unsafe {
      eq_vecs(lhs, rhs, len)
    }
  }
}

impl ops::Add for &Vector {
  type Output = Vector;

  fn add(self, other: Self) -> Self::Output {
    if self.elements.len() != other.elements.len() {
      panic!("Vector dimension mismatch: first operand is {}-dimensional \
              but second operand is {}-dimensional.",
              self.elements.len(), other.elements.len());
    }

    let len = self.elements.len();
    let mut result = Vector::zero(len);

    let lhs1 = self.elements.as_ptr();
    let lhs2 = other.elements.as_ptr();
    let rhs  = result.elements.as_mut_ptr();

    unsafe {
      add_vecs(lhs1, lhs2, rhs, len);
      result.elements.set_len(len);
      result
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
}
