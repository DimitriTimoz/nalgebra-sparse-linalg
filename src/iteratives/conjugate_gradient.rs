//! Conjugate Gradient iterative solver for sparse linear systems.
//!
//! This module provides functions to solve symmetric positive-definite linear systems
//! using the Conjugate Gradient (CG) method for matrices in CSR and CSC formats.
//!
//! # Examples
//!
//! ```
//! use nalgebra_sparse::{na::DVector, CsrMatrix};
//! use nalgebra_sparse_linalg::iteratives::conjugate_gradient::solve_csr;
//!
//! let a = CsrMatrix::identity(3);
//! let b = DVector::from_vec(vec![2.0; 3]);
//! let result = solve_csr(&a, &b, 100, 1e-10);
//! assert!(result.is_some());
//! ```
//!
//! For CSC matrices:
//!
//! ```
//! use nalgebra_sparse::{na::DVector, CscMatrix};
//! use nalgebra_sparse_linalg::iteratives::conjugate_gradient::solve_csc;
//!
//! let a = CscMatrix::identity(3);
//! let b = DVector::from_vec(vec![2.0; 3]);
//! let result = solve_csc(&a, &b, 100, 1e-10);
//! assert!(result.is_some());
//! ```

use nalgebra_sparse::{na::{DVector, SimdComplexField, SimdRealField}, CscMatrix, CsrMatrix};


pub fn solve_csr<T>(a: &CsrMatrix<T>, b: &DVector<T>, max_iter: usize, tol: T) -> Option<DVector<T>> 
where 
    T: SimdRealField + PartialOrd
{
    let mut x = DVector::<T>::zeros(a.nrows());

    let mut residual = b - a * &x;
    let mut residual_dot = residual.dot(&residual);
    // Check if the inital guess is already a solution
    let norm = residual.magnitude();
    if norm <= tol {
        return Some(x);
    }
    let mut p = residual.clone();
    for _ in 0..max_iter {
        let ap = a * &p;
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

pub fn solve_csc<T>(a: &CscMatrix<T>, b: &DVector<T>, max_iter: usize, tol: T) -> Option<DVector<T>> 
where 
    T: SimdRealField + PartialOrd
{
    let mut x = DVector::<T>::zeros(a.nrows());

    let mut residual = b - a * &x;
    let mut residual_dot = residual.dot(&residual);
    // Check if the inital guess is already a solution
    let norm = residual.magnitude();
    if norm <= tol {
        return Some(x);
    }
    let mut p = residual.clone();
    for _ in 0..max_iter {
        let ap = a * &p;
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
    }    None
}

pub fn solve<T>(a: &CsrMatrix<T>, b: &DVector<T>, max_iter: usize, tol: T) -> Option<DVector<T>> 
where 
    T: SimdRealField + PartialOrd
{
    solve_csr(a, b, max_iter, tol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::{na::DVector, CooMatrix};

    #[test]
    fn test_conjugate_grad_csr() {
        let a = CsrMatrix::identity(10);
        let b = DVector::from_vec(vec![3.0;10]);
        let max_iter = 2500;

        let result = solve_csr(&a, &b, max_iter,1e-10);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.len(), 10);
        for i in 0..result.len() {
            assert_eq!(result[i], 3.0);
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
        let max_iter = 2500;
        let result = solve_csr(&a, &b, max_iter, 1e-10);
        assert!(result.is_some());
        let result = result.unwrap();
        let prod = &a * &result;
        assert_eq!(prod.len(), 4);
        for i in 0..prod.len() {
            assert!((prod[i] - b[i]).simd_abs() < 1e-10);
        }

        // Null test
        let a = CsrMatrix::identity(10);
        let b = DVector::from_vec(vec![0.0;10]);
        let max_iter = 2500;
        let result = solve_csr(&a, &b, max_iter, 1e-10);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.len(), 10);
        for i in 0..result.len() {
            assert_eq!(result[i], 0.0);
        }
    }

    #[test]
    fn test_conjugate_grad_csc() {
        use nalgebra_sparse::{CooMatrix, CscMatrix};
        // Identity test
        let a = CscMatrix::identity(10);
        let b = DVector::from_vec(vec![3.0;10]);
        let max_iter = 2500;

        let result = super::solve_csc(&a, &b, max_iter, 1e-10);
        let result = result.unwrap();
        assert_eq!(result.len(), 10);
        for i in 0..result.len() {
            assert_eq!(result[i], 3.0);
        }

        // 4x4 matrix test
        let m = [[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0.0, 3., -1., 8.]];
        let b = [6., 25., -11., 15.];

        let mut coo = CooMatrix::new(4, 4);
        for (i, row) in m.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0.0 {
                    coo.push(i, j, val);
                }
            }
        }
        let a = CscMatrix::from(&coo);
        let b = DVector::from_vec(b.to_vec());
        let max_iter = 2500;
        let result = super::solve_csc(&a, &b, max_iter, 1e-10);
        assert!(result.is_some());
        let result = result.unwrap();
        let prod = &a * &result;
        assert_eq!(prod.len(), 4);
        for i in 0..prod.len() {
            assert!((prod[i] - b[i]).simd_abs() < 1e-10);
        }

        // Null test
        let a = CscMatrix::identity(10);
        let b = DVector::from_vec(vec![0.0;10]);
        let max_iter = 2500;
        let result = super::solve_csc(&a, &b, max_iter, 1e-10);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.len(), 10);
        for i in 0..result.len() {
            assert_eq!(result[i], 0.0);
        }
    }
}
