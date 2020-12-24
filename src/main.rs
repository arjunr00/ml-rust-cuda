use ml_rust::math::linear::Matrix;

fn main() {
  let v1 = Matrix::new(vec![vec![1., 2.], vec![3., 4.], vec![5., 6.]]);
  let v2 = Matrix::new(vec![vec![2., 3.], vec![4., 5.], vec![6., 7.]]);
  if &v1 + &v2 == Matrix::new(vec![vec![3., 5.], vec![7., 9.], vec![11., 13.]]) {
    println!("Correct: {:?}", &v1 + &v2);
  } else {
    println!("Incorrect: {:?}", &v1 + &v2);
  }
}
