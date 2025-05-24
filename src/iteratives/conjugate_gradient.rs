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
    let mut solver = ConjugateGradient {
        x: DVector::<T>::zeros(a.nrows()),
        r: DVector::<T>::zeros(a.nrows()),
        p: DVector::<T>::zeros(a.nrows()),
        ap: DVector::<T>::zeros(a.nrows()),
        residual_dot: T::zero(),
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
pub fn solve_with_initial_guess<M, T>(a: &M, b: &DVector<T>, x: &mut DVector<T>, max_iter: usize, tol: T) -> bool
where 
    M: SpMatVecMul<T>,
    T: SimdRealField + PartialOrd + Copy
{
    let mut solver = ConjugateGradient {
        x: x.clone(),
        r: DVector::<T>::zeros(a.nrows()),
        p: DVector::<T>::zeros(a.nrows()),
        ap: DVector::<T>::zeros(a.nrows()),
        residual_dot: T::zero(),
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

/// Conjugate Gradient iterative solver structure
pub struct ConjugateGradient<T> {
    pub x: DVector<T>,
    pub r: DVector<T>,
    pub p: DVector<T>,
    pub ap: DVector<T>,
    pub residual_dot: T,
    pub tol: T,
    pub max_iter: usize,
    pub iter: usize,
    pub converged: bool,
}

impl<M, T> IterativeSolver<M, DVector<T>, T> for ConjugateGradient<T>
where
    M: SpMatVecMul<T>,
    T: SimdRealField + PartialOrd + Copy,
{
    fn init(&mut self, a: &M, b: &DVector<T>, x0: Option<&DVector<T>>) {
        let n = a.nrows();
        self.x = match x0 {
            Some(x0) => x0.clone(),
            None => DVector::<T>::zeros(n),
        };
        self.r = b - &a.mul_vec(&self.x);
        self.p = self.r.clone();
        self.residual_dot = self.r.dot(&self.r);
        self.ap = DVector::<T>::zeros(n);
        self.iter = 0;
        self.converged = false;
    }

    fn step(&mut self, a: &M, _b: &DVector<T>) -> bool {
        if self.converged {
            return true;
        }
        let norm = self.r.magnitude();
        if norm <= self.tol {
            self.converged = true;
            return true;
        }
        self.ap = a.mul_vec(&self.p);
        let alpha = self.residual_dot / self.p.dot(&self.ap);
        self.x.axpy(alpha, &self.p, T::one());
        let new_r = &self.r - &self.ap * alpha;
        let new_norm = new_r.magnitude();
        if new_norm <= self.tol {
            self.r = new_r;
            self.converged = true;
            return true;
        }
        let new_residual_dot = new_r.dot(&new_r);
        let beta = new_residual_dot / self.residual_dot;
        self.p = &new_r + &self.p * beta;
        self.r = new_r;
        self.residual_dot = new_residual_dot;
        self.iter += 1;
        false
    }

    fn reset(&mut self) {
        self.x.fill(T::zero());
        self.r.fill(T::zero());
        self.p.fill(T::zero());
        self.ap.fill(T::zero());
        self.residual_dot = T::zero();
        self.iter = 0;
        self.converged = false;
    }

    fn hard_reset(&mut self) {
        self.x = DVector::<T>::zeros(0);
        self.r = DVector::<T>::zeros(0);
        self.p = DVector::<T>::zeros(0);
        self.ap = DVector::<T>::zeros(0);
        self.residual_dot = T::zero();
        self.iter = 0;
        self.converged = false;
    }

    fn soft_reset(&mut self) {
        self.r.fill(T::zero());
        self.p.fill(T::zero());
        self.ap.fill(T::zero());
        self.residual_dot = T::zero();
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
