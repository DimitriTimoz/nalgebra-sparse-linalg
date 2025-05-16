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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::{na::{ComplexField, DVector}, CooMatrix};

    #[test]
    fn test_gauss_seidel() {
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
