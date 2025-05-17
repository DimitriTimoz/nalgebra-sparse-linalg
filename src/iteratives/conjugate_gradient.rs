//! Conjugate Gradient iterative solver for sparse linear systems.
//!
//! This module provides functions to solve symmetric positive-definite linear systems Ax = b
//! using the Conjugate Gradient (CG) method. The method is suitable for large, sparse
//! matrices and is generic over any matrix type implementing the `SpMatVecMul` trait,
//! allowing compatibility with both CSR and CSC matrix formats.
//!
//! The CG method iteratively refines a solution, typically converging faster than simpler
//! methods like Jacobi or Gauss-Seidel for well-conditioned symmetric positive-definite systems.
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

/// Solves the symmetric positive-definite linear system Ax = b using the Conjugate Gradient method.
///
/// This function initializes the solution vector `x` to zeros.
/// It is generic over any matrix type `M` that implements the `SpMatVecMul<T>` trait,
/// allowing it to work with different sparse matrix formats (e.g., CSR, CSC).
///
/// # Arguments
/// * `a` - A reference to the sparse matrix `A`. Must be symmetric and positive-definite.
/// * `b` - A reference to the right-hand side vector `b`.
/// * `max_iter` - The maximum number of iterations to perform.
/// * `tol` - The tolerance for convergence. The iteration stops if the norm of the residual
///   is less than or equal to `tol`.
///
/// # Returns
/// * `Some(DVector<T>)` - The solution vector `x` if convergence is achieved within `max_iter`.
/// * `None` - If the method does not converge within `max_iter` iterations.
pub fn solve<M, T>(a: &M, b: &DVector<T>, max_iter: usize, tol: T) -> Option<DVector<T>> 
where 
    M: SpMatVecMul<T>,
    T: SimdRealField + PartialOrd + Copy
{
    let mut x = DVector::<T>::zeros(a.nrows());
    if solve_with_initial_guess(a, b, &mut x, max_iter, tol) {
        Some(x)
    } else {
        None
    }
}
/// Solves the symmetric positive-definite linear system Ax = b using the Conjugate Gradient method,
/// starting with an initial guess for `x`.
///
/// This function modifies `x` in place. It is generic over any matrix type `M` that
/// implements the `SpMatVecMul<T>` trait.
///
/// # Arguments
/// * `a` - A reference to the sparse matrix `A`. Must be symmetric and positive-definite.
/// * `b` - A reference to the right-hand side vector `b`.
/// * `x` - A mutable reference to the initial guess for the solution vector. This vector
///   will be updated in place with the refined solution.
/// * `max_iter` - The maximum number of iterations to perform.
/// * `tol` - The tolerance for convergence. The iteration stops if the norm of the residual
///   is less than or equal to `tol`.
///
/// # Returns
/// * `true` - If the method converges to a solution within `max_iter` iterations.
/// * `false` - If the method does not converge within `max_iter` iterations.
pub fn solve_with_initial_guess<M, T>(a: &M, b: &DVector<T>,  x: &mut DVector<T>, max_iter: usize, tol: T) -> bool
where 
    M: SpMatVecMul<T>,
    T: SimdRealField + PartialOrd + Copy
{
    let mut residual = b - &a.mul_vec(x);
    let mut residual_dot = residual.dot(&residual);
    // Check if the inital guess is already a solution
    let norm = residual.magnitude();
    if norm <= tol {
        return true
    }
    let mut p = residual.clone();
    for _ in 0..max_iter {
        let ap = a.mul_vec(&p);
        let alpha = residual_dot / p.dot(&ap);
        x.axpy(alpha, &p, T::one());
        let new_residual = &residual - &ap * alpha;
        
        // Check for convergence
        let norm = new_residual.magnitude();
        if norm <= tol {
            return true
        }
        let new_residual_dot = new_residual.dot(&new_residual);
        let beta = new_residual_dot / residual_dot;
        residual_dot = new_residual_dot;
        p = &new_residual + &p * beta;
        residual = new_residual;
    }
    false
}
