//! BiConjugate Gradient iterative solver for sparse linear systems.
//!
//! This module provides functions to solve general (possibly non-symmetric) linear systems Ax = b
//! using the BiConjugate Gradient (BiCG) method. The method is suitable for large, sparse
//! matrices and is generic over any matrix type implementing the `SpMatVecMul` trait.
//! BiCG is an extension of the Conjugate Gradient method for non-symmetric systems,
//! but it may suffer from irregular convergence behavior or breakdown.
//!
//! # Example
//!
//! ```
//! use nalgebra_sparse::{na::DVector, CsrMatrix};
//! use nalgebra_sparse_linalg::iteratives::biconjugate_gradient::solve;
//!
//! let a = CsrMatrix::identity(3);
//! let b = DVector::from_vec(vec![2.0; 3]);
//! let result = solve(&a, &b, 100, 1e-10);
//! assert!(result.is_some());
//! ```

use super::*;

/// Solves the general linear system Ax = b using the BiConjugate Gradient (BiCG) method.
///
/// This function initializes the solution vector `x` to zeros and uses the BiConjugateGradient struct.
pub fn solve<M, T>(a: &M, b: &DVector<T>, max_iter: usize, tol: T) -> Option<DVector<T>> 
where 
    M: SpMatVecMul<T>,
    T: SimdRealField + PartialOrd + Copy
{
    let mut solver = BiConjugateGradient {
        x: DVector::<T>::zeros(a.nrows()),
        r: DVector::<T>::zeros(a.nrows()),
        r_hat: DVector::<T>::zeros(a.nrows()),
        p: DVector::<T>::zeros(a.nrows()),
        v: DVector::<T>::zeros(a.nrows()),
        iter: 0,
        tol,
        max_iter,
        converged: false,
    };
    solver.init(a, b, None);
    if solver.solve_iterations(a, b, max_iter) {
        Some(solver.x.clone())
    } else {
        None
    }
}

/// Solves the general linear system Ax = b using the BiConjugate Gradient (BiCG) method,
/// starting with an initial guess for `x`.
///
/// This function modifies `x` in place and uses the BiConjugateGradient struct.
pub fn solve_with_initial_guess<M, T>(a: &M, b: &DVector<T>, x: &mut DVector<T>, max_iter: usize, tol: T) -> bool
where 
    M: SpMatVecMul<T>,
    T: SimdRealField + PartialOrd + Copy
{
    let mut solver = BiConjugateGradient {
        x: x.clone(),
        r: DVector::<T>::zeros(a.nrows()),
        r_hat: DVector::<T>::zeros(a.nrows()),
        p: DVector::<T>::zeros(a.nrows()),
        v: DVector::<T>::zeros(a.nrows()),
        iter: 0,
        tol,
        max_iter,
        converged: false,
    };
    solver.init(a, b, Some(x));
    let converged = solver.solve_iterations(a, b, max_iter);
    *x = solver.x.clone();
    converged
}

/// BiConjugate Gradient iterative solver structure
pub struct BiConjugateGradient<T> {
    pub x: DVector<T>,
    pub r: DVector<T>,
    pub r_hat: DVector<T>,
    pub p: DVector<T>,
    pub v: DVector<T>,
    pub iter: usize,
    pub tol: T,
    pub max_iter: usize,
    pub converged: bool,
}

impl<M, T> IterativeSolver<M, DVector<T>, T> for BiConjugateGradient<T>
where
    M: SpMatVecMul<T>,
    T: SimdRealField + PartialOrd + Copy,
{
    /// Initialize the solver state with the system and right-hand side vector.
    fn init(&mut self, a: &M, b: &DVector<T>, x0: Option<&DVector<T>>) {
        let n = a.nrows();
        self.x = match x0 {
            Some(x0) => x0.clone(),
            None => DVector::<T>::zeros(n),
        };
        self.r = b - &a.mul_vec(&self.x);
        self.r_hat = self.r.clone(); // In practice, r_hat should be a fixed or random vector
        self.p = self.r.clone();
        self.v = DVector::<T>::zeros(n);
        self.iter = 0;
        self.converged = false;
    }

    /// Perform one iteration of the solver. Returns true if converged.
    fn step(&mut self, a: &M, _b: &DVector<T>) -> bool {
        if self.converged { return true; }
        let r_dot = self.r.dot(&self.r_hat);
        if self.r.magnitude() <= self.tol {
            self.converged = true;
            return true;
        }
        self.v = a.mul_vec(&self.p);
        let alpha = r_dot / self.r_hat.dot(&self.v);
        self.x.axpy(alpha, &self.p, T::one());
        let s = &self.r - &self.v * alpha;
        if s.magnitude() <= self.tol {
            self.r = s;
            self.converged = true;
            return true;
        }
        let t = a.mul_vec(&s);
        let omega = t.dot(&s) / t.dot(&t);
        self.x.axpy(omega, &s, T::one());
        let new_r = &s - &t * omega;
        if new_r.magnitude() <= self.tol {
            self.r = new_r;
            self.converged = true;
            return true;
        }
        let new_r_dot = self.r_hat.dot(&new_r);
        let beta = (new_r_dot / r_dot) * (alpha / omega);
        self.p = &new_r + (&self.p - &self.v * omega) * beta;
        self.r = new_r;
        self.iter += 1;
        false
    }

    /// Reset the internal state (soft reset, keeps allocated memory).
    fn reset(&mut self) {
        self.x.fill(T::zero());
        self.r.fill(T::zero());
        self.r_hat.fill(T::zero());
        self.p.fill(T::zero());
        self.v.fill(T::zero());
        self.iter = 0;
        self.converged = false;
    }

    /// Completely reset the internal state (hard reset, releases memory if needed).
    fn hard_reset(&mut self) {
        self.x = DVector::<T>::zeros(0);
        self.r = DVector::<T>::zeros(0);
        self.r_hat = DVector::<T>::zeros(0);
        self.p = DVector::<T>::zeros(0);
        self.v = DVector::<T>::zeros(0);
        self.iter = 0;
        self.converged = false;
    }

    /// Partial reset (e.g. to restart with a new b but same matrix).
    fn soft_reset(&mut self) {
        self.r.fill(T::zero());
        self.r_hat.fill(T::zero());
        self.p.fill(T::zero());
        self.v.fill(T::zero());
        self.iter = 0;
        self.converged = false;
    }

    /// Get the current solution vector.
    fn solution(&self) -> &DVector<T> {
        &self.x
    }

    /// Get the number of performed iterations.
    fn iterations(&self) -> usize {
        self.iter
    }
}
