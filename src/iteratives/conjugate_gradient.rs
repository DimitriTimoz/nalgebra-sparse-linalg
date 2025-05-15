//! Conjugate Gradient iterative solver for sparse linear systems.
//!
//! This module provides functions to solve symmetric positive-definite linear systems
//! using the Conjugate Gradient (CG) method for matrices in CSR and CSC formats.
//!
//! # Examples
//!
//! ```
//! use nalgebra_sparse::{na::DVector, CsrMatrix};
//! use nalgebra_sparse_linalg::iteratives::conjugate_gradient::solve;
//!
//! let a = CsrMatrix::identity(3);
//! let b = DVector::from_vec(vec![2.0; 3]);
//! let result = solve(&a, &b, 100, 1e-10);
//! assert!(result.is_some());
//! ```
//!
//! For CSC matrices:
//!
//! ```
//! use nalgebra_sparse::{na::DVector, CscMatrix};
//! use nalgebra_sparse_linalg::iteratives::conjugate_gradient::solve;
//!
//! let a = CscMatrix::identity(3);
//! let b = DVector::from_vec(vec![2.0; 3]);
//! let result = solve(&a, &b, 100, 1e-10);
//! assert!(result.is_some());
//! ```

use nalgebra_sparse::{na::{DVector, SimdRealField}, CscMatrix, CsrMatrix};

pub trait SpMatVecMul<T: SimdRealField> {
    fn nrows(&self) -> usize;
    fn mul_vec(&self, v: &DVector<T>) -> DVector<T>;
}
impl<T: SimdRealField> SpMatVecMul<T> for CsrMatrix<T> {
    #[inline] fn nrows(&self) -> usize { self.nrows() }
    #[inline] fn mul_vec(&self, v: &DVector<T>) -> DVector<T> { self * v }
}

impl<T: SimdRealField> SpMatVecMul<T> for CscMatrix<T> {
    #[inline] fn nrows(&self) -> usize { self.nrows() }
    #[inline] fn mul_vec(&self, v: &DVector<T>) -> DVector<T> { self * v }
}

/// Solve a symmetric positive-definite linear system using the Conjugate Gradient method.
/// Generic over any matrix type implementing `SpMatVecMul`.
pub fn solve<M, T>(a: &M, b: &DVector<T>, max_iter: usize, tol: T) -> Option<DVector<T>> 
where 
    M: SpMatVecMul<T>,
    T: SimdRealField + PartialOrd
{
    let mut x = DVector::<T>::zeros(a.nrows());

    let mut residual = b - &a.mul_vec(&x);
    let mut residual_dot = residual.dot(&residual);
    // Check if the inital guess is already a solution
    let norm = residual.magnitude();
    if norm <= tol {
        return Some(x);
    }
    let mut p = residual.clone();
    for _ in 0..max_iter {
        let ap = a.mul_vec(&p);
        let alpha = residual_dot.clone() / p.dot(&ap);
        x.axpy(alpha.clone(), &p, T::one());
        let new_residual = &residual - &ap * alpha;
        
        // Check for convergence
        let norm = new_residual.magnitude();
        if norm <= tol {
            return Some(x);
        }
        let new_residual_dot = new_residual.dot(&new_residual);
        let beta = new_residual_dot.clone() / residual_dot;
        residual_dot = new_residual_dot;
        p = &new_residual + &p * beta;
        residual = new_residual;
    }
    None
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::{na::DVector, CooMatrix, CsrMatrix};

    #[test]
    fn test_conjugate_grad() {
        let a = CsrMatrix::identity(10);
        let b = DVector::from_vec(vec![3.0;10]);
        let max_iter = 2500;
        let tol = 1e-10;

        let result = solve::<CsrMatrix<f64>, f64>(&a, &b, max_iter, tol);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.len(), 10);
        for i in 0..result.len() {
            assert!((result[i] - 3.0).abs() < tol);
        }
        
        let m = [[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0.0, 3., -1., 8.]];
        let b = [6., 25., -11., 15.];
        
        // Create a CooMatrix and fill it with values
        let mut coo = CooMatrix::new(4, 4);
        for (i, row) in m.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0.0 {
                    coo.push(i, j, val);
                }
            }
        }
        // Convert CooMatrix to CsrMatrix
        let a = CsrMatrix::from(&coo);
        let b = DVector::from_vec(b.to_vec());
        let result = solve::<CsrMatrix<f64>, f64>(&a, &b, max_iter, tol);
        assert!(result.is_some());
        let result = result.unwrap();
        let prod = &a * &result;
        assert_eq!(prod.len(), 4);
        for i in 0..prod.len() {
            assert!((prod[i] - b[i]).abs() < tol);
        }

        // Null test
        let a = CsrMatrix::identity(10);
        let b = DVector::from_vec(vec![0.0;10]);
        let result = solve::<CsrMatrix<f64>, f64>(&a, &b, max_iter, tol);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.len(), 10);
        for i in 0..result.len() {
            assert!((result[i]).abs() < tol);
        }
    }
}
