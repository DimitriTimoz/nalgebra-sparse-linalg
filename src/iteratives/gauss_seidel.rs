use super::*;

struct AlternatingRange {
    len: usize,
    reversed: bool,
    current: usize,
}

impl AlternatingRange {
    fn new(len: usize) -> Self {
        Self {
            len,
            reversed: true,
            current: 0,
        }
    }
    fn is_reversed(&self) -> bool {
        self.reversed
    }
    fn alternate(&mut self) {
        self.reversed = !self.reversed;
        self.current = 0;
    }
}

impl Iterator for AlternatingRange {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.len {
            return None;
        }

        let index = if self.reversed {
            self.len - 1 - self.current
        } else {
            self.current
        };

        self.current += 1;
        Some(index)
    }
}
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
    if solve_with_initial_guess(a, b, &mut x, max_iter, tol) {
        Some(x)
    } else {
        None
    }
}

/// Solves the linear system Ax = b using the Gauss-Seidel iterative method,
/// starting with an initial guess for x.
///
/// This function modifies `x` in place. The Gauss-Seidel method improves upon Jacobi
/// by immediately using newly computed values within each iteration, which accelerates
/// convergence for symmetric positive-definite or strictly diagonally dominant matrices.
/// Returns `true` if the solution converges within `max_iter` iterations to the given
/// tolerance `tol`, otherwise returns `false`.
///
/// # Arguments
/// * `a` - Sparse matrix in CSR format
/// * `b` - Right-hand side vector
/// * `x` - Initial guess for the solution vector, modified in place
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
pub fn solve_with_initial_guess<T>(a: &CsrMatrix<T>, b: &DVector<T>,  x: &mut DVector<T>, max_iter: usize, tol: T) -> bool
where 
    T: SimdRealField + PartialOrd
{
    let mut new_x = (*x).clone();

    let mut range = AlternatingRange::new(a.nrows());
    for _ in 0..max_iter {
        range.alternate();
        let is_reversed = range.is_reversed();
        for row_i in range.by_ref() {
            if let Some(row) = &a.get_row(row_i) {
                let mut sigma = b[row_i].clone();

                let col_indices = row.col_indices();
                let values = row.values();
                for (col_i, value) in col_indices.iter().zip(values.iter()) {
                    let x_val = if *col_i < row_i && !is_reversed {
                        new_x[*col_i].clone()
                    } else {
                        x[*col_i].clone()
                    };
                    if *col_i != row_i {
                        sigma -= value.clone() * x_val;
                    }
                }
                let diag = a.get_entry(row_i, row_i).map(|v| v.into_value());
                let diag = match diag {
                    Some(diag) => diag,
                    None => return false,
                };
                // Check for zero or near-zero diagonal entry
                if diag < tol {
                    return false;
                }
                new_x[row_i] = sigma / diag;
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
