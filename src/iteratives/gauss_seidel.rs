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
    T: SimdRealField + PartialOrd + Copy
{
    let mut solver = GaussSeidel {
        x: DVector::<T>::zeros(a.nrows()),
        new_x: DVector::<T>::zeros(a.nrows()),
        tol,
        max_iter,
        iter: 0,
        converged: false,
    };
    solver.init(a, b, None);
    if solver.solve_iterations(a, b, max_iter) {
        Some(solver.x.clone())
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
pub fn solve_with_initial_guess<T>(a: &CsrMatrix<T>, b: &DVector<T>, x: &mut DVector<T>, max_iter: usize, tol: T) -> bool
where 
    T: SimdRealField + PartialOrd + Copy
{
    let mut solver = GaussSeidel {
        x: x.clone(),
        new_x: DVector::<T>::zeros(a.nrows()),
        tol,
        max_iter,
        iter: 0,
        converged: false,
    };
    solver.init(a, b, Some(x));
    let converged = solver.solve_iterations(a, b, max_iter);
    *x = solver.x.clone();
    converged
}

pub struct GaussSeidel<T> {
    pub x: DVector<T>,
    pub new_x: DVector<T>,
    pub tol: T,
    pub max_iter: usize,
    pub iter: usize,
    pub converged: bool,
}

impl<T> IterativeSolver<CsrMatrix<T>, DVector<T>, T> for GaussSeidel<T>
where
    T: SimdRealField + PartialOrd + Copy,
{
    fn init(&mut self, a: &CsrMatrix<T>, _b: &DVector<T>, x0: Option<&DVector<T>>) {
        let n = a.nrows();
        self.x = match x0 {
            Some(x0) => x0.clone(),
            None => DVector::<T>::zeros(n),
        };
        self.new_x = self.x.clone();
        self.iter = 0;
        self.converged = false;
    }

    fn step(&mut self, a: &CsrMatrix<T>, b: &DVector<T>) -> bool {
        let n = a.nrows();
        let mut range = AlternatingRange::new(n);
        range.alternate();
        let is_reversed = range.is_reversed();
        for row_i in range.by_ref() {
            if let Some(row) = &a.get_row(row_i) {
                let mut sigma = b[row_i];
                let col_indices = row.col_indices();
                let values = row.values();
                for (col_i, value) in col_indices.iter().zip(values.iter()) {
                    let x_val = if *col_i < row_i && !is_reversed {
                        self.new_x[*col_i]
                    } else {
                        self.x[*col_i]
                    };
                    if *col_i != row_i {
                        sigma -= *value * x_val;
                    }
                }
                let diag = a.get_entry(row_i, row_i).map(|v| v.into_value());
                let diag = match diag {
                    Some(diag) => diag,
                    None => return false,
                };
                if diag < self.tol {
                    return false;
                }
                self.new_x[row_i] = sigma / diag;
            }
        }
        let norm = self.x.iter().zip(self.new_x.iter()).fold(T::zero(), |m, (x_i, new_x_i)| {
            m + (*x_i - *new_x_i).simd_norm1()
        });
        std::mem::swap(&mut self.x, &mut self.new_x);
        self.iter += 1;
        if norm <= self.tol {
            self.converged = true;
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.x.fill(T::zero());
        self.new_x.fill(T::zero());
        self.iter = 0;
        self.converged = false;
    }

    fn hard_reset(&mut self) {
        self.x = DVector::<T>::zeros(0);
        self.new_x = DVector::<T>::zeros(0);
        self.iter = 0;
        self.converged = false;
    }

    fn soft_reset(&mut self) {
        self.x.fill(T::zero());
        self.new_x.fill(T::zero());
        self.iter = 0;
        self.converged = false;
    }

    fn solution(&self) -> &DVector<T> {
        &self.x
    }

    fn iterations(&self) -> usize {
        self.iter
    }
}
