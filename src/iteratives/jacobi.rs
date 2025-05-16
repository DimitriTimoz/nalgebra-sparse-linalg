//! Jacobi iterative solver for sparse linear systems.
//!
//! This module provides an implementation of the Jacobi iterative method
//! for solving linear systems of the form `Ax = b`, where `A` is a sparse matrix
//! in CSR format. The Jacobi method is suitable for diagonally dominant matrices
//! and is implemented generically for types that implement the required numeric traits.

use super::*;

/// Solves the linear system `Ax = b` using the Jacobi iterative method.
///
/// # Arguments
///
/// * `a` - A reference to a sparse matrix `A` in CSR format.
/// * `b` - A reference to the right-hand side vector `b`.
/// * `max_iter` - The maximum number of iterations to perform.
///
/// # Returns
///
/// Returns `Some(DVector<T>)` containing the solution vector if convergence is reached,
/// or `None` if the method fails to converge or if the matrix is singular.
///
/// # Type Parameters
///
/// * `T` - The scalar type, which must implement the required numeric traits.
///
/// # Example
///
/// ```
/// use nalgebra_sparse::{na::DVector, CsrMatrix};
/// use nalgebra_sparse_linalg::iteratives::jacobi::solve;
///
/// let a = CsrMatrix::identity(3);
/// let b = DVector::from_vec(vec![1.0; 3]);
/// let result = solve(&a, &b, 100, 1e-10);
/// assert!(result.is_some());
/// ```
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
                        sigma -= value.clone() * x[*col_i].clone();
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
            return Some(x);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::{na::{ComplexField, DVector}, CooMatrix};

    
    #[test]
    fn test_jacobi() {
        let a = CsrMatrix::identity(10);
        let b = DVector::from_vec(vec![3.0;10]);
        let max_iter = 2500;

        let result = solve(&a, &b, max_iter, 1e-10);
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
        let result = solve(&a, &b, max_iter, 1e-10);
        assert!(result.is_some());
        let result = result.unwrap();
        let result = a * result;
        assert_eq!(result.len(), 4);
        for i in 0..result.len() {
            assert!((result[i] - b[i]).norm1() < 1e-9);
        }

        // Null test
        let a = CsrMatrix::identity(10);
        let b = DVector::from_vec(vec![0.0;10]);
        let max_iter = 2500;
        let result = solve(&a, &b, max_iter, 1e-10);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.len(), 10);
        for i in 0..result.len() {
            assert_eq!(result[i], 0.0);
        }

        // Non converging test
        let m = [[1., 2., 3., 4.],
              [5., 6., 7., 8.],
              [9., 10., 11., 12.],
              [13., 14., 15., 16.]];
        let b = [1., 2., 3., 4.];
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
        let result = solve(&a, &b, max_iter, 1e-10);
        assert!(result.is_none());
    }
}
