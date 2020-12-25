use ml_rust_cuda::math::linear::{ Matrix, Vector };

fn main() {
  let m1 = Matrix::new(vec![vec![10., 2.], vec![3., 4.4], vec![5., 6.]]);
  let m2 = Matrix::new(vec![vec![1., 22.], vec![3.1, 4.2], vec![2., 7.1]]);
  let m3 = Matrix::new(vec![vec![20., 3.2, 4.], vec![4., 5., 6.]]);
  println!("m1:\n{}\n\nm2:\n{}\n\nm3:\n{}\n", &m1, &m2, &m3);
  println!("m1 + m2:\n{}\n", &m1 + &m2);
  println!("m1 * m3:\n{}\n", &m1 * &m3);
  println!("m1 padded 4x:\n{}\n", &m1.zero_padded(4));

  let m1 = Vector::new(vec![1., 4., 2., 5., 6.]);
  println!("{:?}", &m1);
  println!("{}", &m1);
}
