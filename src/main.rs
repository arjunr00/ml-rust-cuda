extern "C" {
  fn add_n(x_val: f32, y_val: f32, n: u32) -> f32;
}

fn safe_add_n(x_val: f32, y_val: f32, n: u32) -> f32 {
  unsafe {
    add_n(x_val, y_val, n)
  }
}

fn main() {
  let n = 1_u32 << 20;
  println!("y[i] = {} for all i from 0 to {}",
    safe_add_n(1.0_f32, 2.0_f32, 1_u32 << 20), n - 1);
}
