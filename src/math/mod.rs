pub mod linear;

const EPSILON: f32 = 0.00001;

pub fn f32_eq(lhs: f32, rhs: f32) -> bool {
  // TODO: improve correctness
  (lhs - rhs).abs() <= EPSILON
}
