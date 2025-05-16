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

use super::*;

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
