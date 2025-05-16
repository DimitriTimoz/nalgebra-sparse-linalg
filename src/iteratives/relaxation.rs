//! Successive Over-Relaxation (SOR) and related iterative solvers for sparse linear systems.
//!
//! This module provides implementations of relaxation methods, including the
//! Successive Over-Relaxation (SOR) method, for solving linear systems of the
//! form `Ax = b`. These methods are iterative and particularly useful for large,
//! sparse matrices. The SOR method is a refinement of the Gauss-Seidel method
//! and can converge faster for a suitable choice of the relaxation factor ω (omega).
//!
//! - When ω = 1, the method is equivalent to the Gauss-Seidel method.
//! - For 0 < ω < 1, the method is under-relaxed and can be used to stabilize convergence.
//! - For 1 < ω < 2, the method is over-relaxed and can accelerate convergence for
//!   matrices that satisfy certain properties (e.g., symmetric positive-definite).
//!
//! Convergence is not guaranteed for all matrices or all choices of ω. The matrix
//! should ideally be strictly or irreducibly diagonally dominant for Gauss-Seidel (ω=1)
//! to converge.

use super::*;

/// Solves the linear system `Ax = b` using the Successive Over-Relaxation (SOR) iterative method.
///
/// This function initializes the solution vector `x` to zeros.
///
/// # Arguments
/// - `a`: The coefficient matrix `A` in CSR (Compressed Sparse Row) format.
/// - `b`: The right-hand side vector `b`.
/// - `max_iter`: The maximum number of iterations to perform.
/// - `weight`: The relaxation parameter ω (omega).
///   - For ω = 1, this is the Gauss-Seidel method.
///   - For 0 < ω < 2, SOR. Convergence is typically only for ω in (0, 2).
///     Typical values are 1.0 (Gauss-Seidel) or e.g., 1.5 for over-relaxation.
/// - `tol`: The convergence tolerance. Iteration stops when the L1-norm of the
///   difference between successive iterates is less than or equal to `tol`.
///
/// # Returns
/// - `Some(DVector<T>)` with the solution vector if the method converges within `max_iter` iterations.
/// - `None` if the method does not converge or if a diagonal entry is found to be less than `tol`
///   (which can lead to division by a small or zero number).
///
/// # Type Parameters
///
/// * `T` - The scalar type, which must implement `SimdRealField`, `PartialOrd`, `Send`, and `Sync`.
///
/// # Example
/// ```rust
/// use nalgebra_sparse::{na::DVector, CsrMatrix};
/// use nalgebra_sparse_linalg::iteratives::relaxation::solve;
///
/// // Create a 3x3 matrix:
/// // 4 1 0
/// // 1 4 1
/// // 0 1 4
/// // This matrix is diagonally dominant.
/// let coo = nalgebra_sparse::CooMatrix::try_from_triplets(
///     3, 3,
///     vec![0, 0, 1, 1, 1, 2, 2],
///     vec![0, 1, 0, 1, 2, 1, 2],
///     vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0]
/// ).unwrap();
/// let a = CsrMatrix::from(&coo);
/// let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);
/// // Using omega = 1.0 (Gauss-Seidel)
/// let result = solve(&a, &b, 100, 1.0f64, 1e-10);
/// assert!(result.is_some());
/// if let Some(x_sol) = result {
///    // A known approximate solution for this system
///    assert!((x_sol[0] - 0.1160714f64).abs() < 1e-5f64);
///    assert!((x_sol[1] - 0.3392857f64).abs() < 1e-5f64);
///    assert!((x_sol[2] - 0.6651785f64).abs() < 1e-5f64);
/// }
/// ```
pub fn solve<T>(a: &CsrMatrix<T>, b: &DVector<T>, max_iter: usize, weight: T, tol: T) -> Option<DVector<T>> 
where 
    T: SimdRealField + PartialOrd + Send + Sync 
{
    let mut x = DVector::<T>::zeros(a.nrows());
    if solve_with_initial_guess(a, b, &mut x,  max_iter, weight, tol) {
        Some(x)
    } else {
        None
    }
}
/// Solves the linear system `Ax = b` using the Successive Over-Relaxation (SOR) iterative method,
/// starting with an initial guess for `x`.
///
/// This function modifies `x` in place.
///
/// # Arguments
/// - `a`: The coefficient matrix `A` in CSR (Compressed Sparse Row) format.
/// - `b`: The right-hand side vector `b`.
/// - `x`: A mutable reference to the initial guess for the solution vector. This vector
///   will be updated in place with the refined solution.
/// - `max_iter`: The maximum number of iterations to perform.
/// - `weight`: The relaxation parameter ω (omega).
///   - For ω = 1, this is the Gauss-Seidel method.
///   - For 0 < ω < 2, SOR. Convergence is typically only for ω in (0, 2).
/// - `tol`: The convergence tolerance. Iteration stops when the L1-norm of the
///   difference between successive iterates is less than or equal to `tol`.
///
/// # Returns
/// - `true` if the method converges to a solution within `max_iter` iterations.
/// - `false` if the method does not converge or if a diagonal entry is found to be
///   less than `tol`.
///
/// # Type Parameters
///
/// * `T` - The scalar type, which must implement `SimdRealField` and `PartialOrd`.
///
/// # Example
/// ```rust
/// use nalgebra_sparse::{na::DVector, CsrMatrix};
/// use nalgebra_sparse_linalg::iteratives::relaxation::solve_with_initial_guess;
///
/// let coo = nalgebra_sparse::CooMatrix::try_from_triplets(
///     3, 3,
///     vec![0, 0, 1, 1, 1, 2, 2],
///     vec![0, 1, 0, 1, 2, 1, 2],
///     vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0]
/// ).unwrap();
/// let a = CsrMatrix::from(&coo);
/// let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);
/// let mut x = DVector::from_vec(vec![0.0, 0.0, 0.0]); // Initial guess
/// // Using omega = 1.2 (over-relaxation)
/// let converged = solve_with_initial_guess(&a, &b, &mut x, 100, 1.2f64, 1e-10);
/// assert!(converged);
/// // Check against a known approximate solution
/// assert!((x[0] - 0.1160714f64).abs() < 1e-5f64);
/// assert!((x[1] - 0.3392857f64).abs() < 1e-5f64);
/// assert!((x[2] - 0.6651785f64).abs() < 1e-5f64);
/// ```
pub fn solve_with_initial_guess<T>(a: &CsrMatrix<T>, b: &DVector<T>, x: &mut DVector<T>, max_iter: usize, weight: T, tol: T) -> bool
where 
    T: SimdRealField + PartialOrd
{
    let mut new_x = (*x).clone();
    for _ in 0..max_iter {
        for row_i in 0..a.nrows() {
            if let Some(row) = &a.get_row(row_i) {
                let mut sigma = b[row_i].clone();

                let col_indices = row.col_indices();
                let values = row.values();
                for (col_i, value) in col_indices.iter().zip(values.iter()) {
                    if *col_i != row_i {
                        sigma -= value.clone() * new_x[*col_i].clone();
                    }
                }
                let diag = a.get_entry(row_i, row_i);
                let diag = match diag {
                    Some(diag) => diag.into_value(),
                    None => return false,
                };
                if diag < tol {
                    return false
                }
                new_x[row_i] = (sigma / diag) * weight.clone() + (T::one() - weight.clone()) * x[row_i].clone();
            }
        }
        // Check for convergence
        let norm = x.iter().zip(new_x.iter()).fold(T::zero(), |m, (x_i, new_x_i)| {
            m + (x_i.clone() - new_x_i.clone()).simd_norm1()
        });
        std::mem::swap(x, &mut new_x);
        if norm <= tol {
            return true;
        }
    }
    false
}
