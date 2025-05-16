use super::*;

/// Successive Over-Relaxation (SOR) iterative solver for sparse linear systems.
///
/// This function attempts to solve the linear system `Ax = b` using the relaxation (SOR) method.
///
/// # Parameters
/// - `a`: The coefficient matrix `A` in CSR (Compressed Sparse Row) format.
/// - `b`: The right-hand side vector `b`.
/// - `max_iter`: The maximum number of iterations to perform.
/// - `weight`: The relaxation parameter ω (omega). For ω=1, this is the Gauss-Seidel method; for 0<ω<2, SOR. Typical values are 1.0 (Gauss-Seidel) or 1.5 (over-relaxation).
/// - `tol`: The convergence tolerance. Iteration stops when the L1-norm of the difference between successive iterates is less than or equal to `tol`.
///
/// # Returns
/// - `Some(DVector<T>)` with the solution vector if the method converges within `max_iter` iterations.
/// - `None` if the method does not converge or if a zero/near-zero diagonal entry is encountered.
///
/// # Example
/// ```rust
/// use nalgebra_sparse::{na::DVector, CsrMatrix};
/// use nalgebra_sparse_linalg::iteratives::relaxation::solve;
/// let a = CsrMatrix::identity(3);
/// let b = DVector::from_vec(vec![1.0; 3]);
/// let result = solve(&a, &b, 100, 1.0, 1e-10);
/// assert!(result.is_some());
/// ```
///
pub fn solve<T>(a: &CsrMatrix<T>, b: &DVector<T>, max_iter: usize, weight: T, tol: T) -> Option<DVector<T>> 
where 
    T: SimdRealField + PartialOrd
{
    let mut x = DVector::<T>::zeros(a.nrows());
    let mut new_x = DVector::<T>::zeros(a.nrows());
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
                let diag = a.get_entry(row_i, row_i)?.into_value();
                if diag < tol {
                    return None;
                }
                new_x[row_i] = (sigma / diag) * weight.clone() + (T::one() - weight.clone()) * x[row_i].clone();
            }
        }
        // Check for convergence
        let norm = x.iter().zip(new_x.iter()).fold(T::zero(), |m, (x_i, new_x_i)| {
            m + (x_i.clone() - new_x_i.clone()).simd_norm1()
        });
        std::mem::swap(&mut x, &mut new_x);
        if norm <= tol {
            return Some(new_x);
        }
    }
    None
}
