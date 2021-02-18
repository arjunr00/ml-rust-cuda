use std::{ cmp, fmt };

use crate::math::{
  linear::{ Vector },
  stats::{ mean, std_dev }
};

#[derive(Clone, Debug)]
pub struct Instance {
  feature_vec: Vector
}

impl Instance {
  pub fn new(instances: Vec<f32>) -> Self {
    Self {
      feature_vec: Vector::new(instances)
    }
  }

  pub fn from_vector(feature_vec: Vector) -> Self {
    Self { feature_vec }
  }

  /// Mutates the feature vector of the instance such that all features are
  /// normalized to have a zero mean and unit standard deviation.
  /// 
  /// # Examples
  /// ```
  /// use ml_rust_cuda::{
  ///   learning::Instance,
  ///   math::f32_eq
  /// };
  ///
  /// let mut x = Instance::new(vec![9_f32, 5_f32, 8_f32, 3_f32, 4_f32, 0_f32]);
  /// x.normalize();
  ///
  /// println!("{}", x);
  /// ```
  pub fn normalize(&mut self) {
    let mean = mean(&self.feature_vec);
    let mean_vec = Vector::new(vec![mean; self.feature_vec.dim()]);
    let std_dev = std_dev(&self.feature_vec);

    let normalized_vec = (1_f32 / std_dev) * &(&self.feature_vec - &mean_vec);
    self.feature_vec = normalized_vec;
  }

  /// Returns the Lp-distance between two instances (i.e. the p-norm of the
  /// difference between the two vectors representing those instances).
  /// 
  /// # Arguments
  ///
  /// * `p` - A positive 32-bit float.
  ///
  /// # Examples
  /// ```
  /// use ml_rust_cuda::{
  ///   learning::Instance,
  ///   math::f32_eq
  /// };
  ///
  /// let x1 = Instance::new(vec![9_f32, 5_f32, 8_f32, 3_f32, 4_f32, 0_f32]);
  /// let x2 = Instance::new(vec![3_f32, 8_f32, 1_f32, 2_f32, 3_f32, 1_f32]);
  ///
  /// let dist_1 = Instance::distance(&x1, &x2, 1_f32);
  /// let dist_2 = Instance::distance(&x1, &x2, 2_f32);
  /// let dist_3 = Instance::distance(&x1, &x2, 3_f32);
  ///
  /// assert!(f32_eq(dist_1, 19_f32));
  /// assert!(f32_eq(dist_2, 97_f32.sqrt()));
  /// assert!(f32_eq(dist_3, 589_f32.cbrt()));
  /// ```
  ///
  /// # Panics
  ///
  /// Panics if `p` is less than or equal to 0, or if the number of attributes
  /// in both instances is not equal.
  pub fn distance(lhs1: &Instance, lhs2: &Instance, p: f32) -> f32 {
    (&lhs1.feature_vec - &lhs2.feature_vec).p_norm(p)
  }
}

impl cmp::PartialEq for Instance {
  /// See [Vector].
  fn eq(&self, other: &Self) -> bool {
    self.feature_vec == other.feature_vec
  }
}

impl fmt::Display for Instance {
  /// See [Vector].
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.feature_vec.fmt(f)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_normalize_instance() {
    let features =
      vec![9_f32, 5_f32, 8_f32, 3_f32, 4_f32, 0_f32,
           3_f32, 8_f32, 1_f32, 2_f32, 3_f32, 1_f32];
    let n: f32 = features.len() as f32;
    let mean: f32 = features.iter().sum::<f32>() / n;
    let mut variance = 0_f32;
    for feat in features.iter() {
      variance += (feat - mean).powi(2);
    }
    variance /= n - 1_f32;
    let std_dev = variance.sqrt();
    let mean_vec = Vector::new(vec![mean; features.len()]);

    let mut x = Instance::new(features);

    let expected
      = Instance::from_vector((1_f32 / std_dev) * &(&x.feature_vec - &mean_vec));
    x.normalize();
    assert_eq!(x, expected);
  }
}
