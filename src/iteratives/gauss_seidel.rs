use super::*;

/// Solves the linear system Ax = b using the Gauss-Seidel iterative method.
///
/// The Gauss-Seidel method improves upon Jacobi by immediately using newly computed values
/// within each iteration, which accelerates convergence for symmetric positive-definite or strictly
/// diagonally dominant matrices. Returns `Some(x)` if the solution converges within `max_iter`
/// iterations to the given tolerance `tol`, otherwise returns `None`.
///
/// # Arguments
/// * `a` - Sparse matrix in CSR format
/// * `b` - Right-hand side vector
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
pub fn solve<T>(a: &CsrMatrix<T>, b: &DVector<T>, max_iter: usize, tol: T) -> Option<DVector<T>> 
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
                new_x[row_i] = sigma / diag;
            }
        }
        // Check for convergence
        let norm = x.iter().zip(new_x.iter()).fold(T::zero(), |m, (x_i, new_x_i)| {
            m + (x_i.clone() - new_x_i.clone()).simd_norm1()
        });
        x = new_x.clone();
        if norm <= tol {
            return Some(new_x);
        }
    }
    None
}
