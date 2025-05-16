//! Jacobi iterative solver for sparse linear systems.
//!
//! This module provides an implementation of the Jacobi iterative method
//! for solving linear systems of the form `Ax = b`, where `A` is a sparse matrix.
//! The Jacobi method is suitable for matrices that are strictly or irreducibly
//! diagonally dominant. It is implemented generically for types that implement
//! the required numeric traits and works with CSR (Compressed Sparse Row) matrices.
//! The method can optionally leverage parallel computation for large matrices using Rayon.

use super::*;

/// Solves the linear system `Ax = b` using the Jacobi iterative method.
///
/// This function initializes the solution vector `x` to zeros.
///
/// # Arguments
///
/// * `a` - A reference to a sparse matrix `A` in CSR format.
/// * `b` - A reference to the right-hand side vector `b`.
/// * `max_iter` - The maximum number of iterations to perform.
/// * `tol` - The tolerance for convergence. The iteration stops if the norm of the
///   difference between successive iterates is less than or equal to `tol`.
///
/// # Returns
///
/// Returns `Some(DVector<T>)` containing the solution vector if convergence is reached
/// within `max_iter` iterations, or `None` if the method fails to converge or if a
/// diagonal element is too close to zero (less than `tol`).
///
/// # Type Parameters
///
/// * `T` - The scalar type, which must implement `SimdRealField` and `PartialOrd`.
///   For parallel execution in `solve_with_initial_guess`, `T` must also
///   implement `Send` and `Sync`.
///
/// # Example
///
/// ```
/// use nalgebra_sparse::{na::DVector, CsrMatrix};
/// use nalgebra_sparse_linalg::iteratives::jacobi::solve;
///
/// // Create a 3x3 matrix:
/// // 4 0 0
/// // 0 4 0
/// // 0 0 4
/// let coo_matrix = nalgebra_sparse::CooMatrix::try_from_triplets(
///     3, 3,
///     vec![0, 1, 2], // row indices
///     vec![0, 1, 2], // col indices
///     vec![4.0, 4.0, 4.0] // values
/// ).unwrap();
/// let a = CsrMatrix::from(&coo_matrix);
/// let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);
/// let result = solve(&a, &b, 100, 1e-10);
/// assert!(result.is_some());
/// if let Some(x) = result {
///     // For A = 4*I, x should be b/4
///     assert!((x[0] - 0.25f64).abs() < 1e-9f64);
///     assert!((x[1] - 0.50f64).abs() < 1e-9f64);
///     assert!((x[2] - 0.75f64).abs() < 1e-9f64);
/// }
/// ```
pub fn solve<T>(a: &CsrMatrix<T>, b: &DVector<T>, max_iter: usize, tol: T) -> Option<DVector<T>> 
where 
    T: SimdRealField + PartialOrd + Send + Sync
{
    let mut x = DVector::<T>::zeros(a.nrows());
    if solve_with_initial_guess(a, b, &mut x, max_iter, tol) {
        Some(x)
    } else {
        None
    }
}
/// Solves the linear system `Ax = b` using the Jacobi iterative method,
/// starting with an initial guess for `x`.
///
/// This function modifies `x` in place.
///
/// # Arguments
///
/// * `a` - A reference to a sparse matrix `A` in CSR format.
/// * `b` - A reference to the right-hand side vector `b`.
/// * `x` - A mutable reference to the initial guess for the solution vector. This vector
///   will be updated in place with the refined solution.
/// * `max_iter` - The maximum number of iterations to perform.
/// * `tol` - The tolerance for convergence. The iteration stops if the norm of the
///   difference between successive iterates is less than or equal to `tol`.
///
/// # Returns
///
/// Returns `true` if convergence is reached within `max_iter` iterations,
/// or `false` if the method fails to converge or if a diagonal element
/// is too close to zero (less than `tol`).
///
/// # Type Parameters
///
/// * `T` - The scalar type, which must implement `SimdRealField`, `PartialOrd`, `Send`, and `Sync`.
///
/// # Note
/// Jacobi is parallelized for large matrices (>= 10,000 rows) using Rayon.
/// For smaller matrices, it runs sequentially.
/// The method may not converge if the matrix `A` is not strictly or irreducibly diagonally dominant.
/// A diagonal entry is considered too small if its absolute value is less than `tol`.
/// 
/// # Example
///
/// ```
/// use nalgebra_sparse::{na::DVector, CsrMatrix};
/// use nalgebra_sparse_linalg::iteratives::jacobi::solve_with_initial_guess;
///
/// // Create a 3x3 matrix:
/// // 4 0 0
/// // 0 4 0
/// // 0 0 4
/// let coo_matrix = nalgebra_sparse::CooMatrix::try_from_triplets(
///     3, 3,
///     vec![0, 1, 2], // row indices
///     vec![0, 1, 2], // col indices
///     vec![4.0, 4.0, 4.0] // values
/// ).unwrap();
/// let a = CsrMatrix::from(&coo_matrix);
/// let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);
/// let mut x = DVector::from_vec(vec![0.0, 0.0, 0.0]); // Initial guess
/// let converged = solve_with_initial_guess(&a, &b, &mut x, 100, 1e-10);
/// assert!(converged);
/// // For A = 4*I, x should be b/4
/// assert!((x[0] - 0.25f64).abs() < 1e-9f64);
/// assert!((x[1] - 0.50f64).abs() < 1e-9f64);
/// assert!((x[2] - 0.75f64).abs() < 1e-9f64);
/// ```
pub fn solve_with_initial_guess<T>(a: &CsrMatrix<T>, b: &DVector<T>, x: &mut DVector<T>, max_iter: usize, tol: T) -> bool
where 
    T: SimdRealField + PartialOrd + Send + Sync
{
    let mut new_x = x.clone();
    let n = a.nrows();
    let use_parallel = n >= 10_000;
    for _ in 0..max_iter {
        if use_parallel {
            new_x.as_mut_slice()
                .par_iter_mut()
                .enumerate()
                .for_each(|(row_i, new_x_i)| {
                    if let Some(row) = a.get_row(row_i) {
                        let mut sigma = b[row_i].clone();
                        let col_indices = row.col_indices();
                        let values = row.values();
                        for (col_i, value) in col_indices.iter().zip(values.iter()) {
                            if *col_i != row_i {
                                sigma -= value.clone() * x[*col_i].clone();
                            }
                        }
                        let diag = a.get_entry(row_i, row_i).map(|v| v.into_value());
                        if let Some(diag) = diag {
                            if diag < tol {
                                *new_x_i = T::zero();
                            } else {
                                *new_x_i = sigma / diag;
                            }
                        } else {
                            *new_x_i = T::zero();
                        }
                    }
                });
        } else {
            for row_i in 0..a.nrows() {
                if let Some(row) = &a.get_row(row_i) {
                    let mut sigma = b[row_i].clone();
                    let col_indices = row.col_indices();
                    let values = row.values();
                    for (col_i, value) in col_indices.iter().zip(values.iter()) {
                        if *col_i != row_i {
                            sigma -= value.clone() * x[*col_i].clone();
                        }
                    }
                    let diag = a.get_entry(row_i, row_i).map(|v| v.into_value());
                    let diag = match diag {
                        Some(diag) => diag,
                        None => return false,
                    };
                    if diag < tol {
                        return false;
                    }
                    new_x[row_i] = sigma / diag;
                }
            }
        }
        let norm = if use_parallel {
            (0..n).into_par_iter()
                .map(|i| (x[i].clone() - new_x[i].clone()).simd_norm1())
                .reduce(|| T::zero(), |a, b| a + b)
        } else {
            x.iter().zip(new_x.iter()).fold(T::zero(), |m, (x_i, new_x_i)| {
                m + (x_i.clone() - new_x_i.clone()).simd_norm1()
            })
        };
        std::mem::swap(x, &mut new_x);
        if norm <= tol {
            return true;
        }
    }
    false
}
