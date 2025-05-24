pub mod jacobi;
pub mod biconjugate_gradient;
pub mod conjugate_gradient;
pub mod gauss_seidel;
pub mod relaxation;

pub use biconjugate_gradient::solve as solve_biconjugate_gradient;

pub use nalgebra_sparse::{CscMatrix, CsrMatrix};
pub(crate) use nalgebra_sparse::{na::{DVector, SimdRealField}};
pub(crate) use rayon::prelude::*;
pub(crate) use crate::preconditioners::{IdentityPreconditioner, Preconditioner};

pub trait SpMatVecMul<T: SimdRealField + Copy> {
    fn nrows(&self) -> usize;
    fn mul_vec(&self, v: &DVector<T>) -> DVector<T>;
    fn inv_diagonal(&self) -> Self;
}
impl<T: SimdRealField + Copy> SpMatVecMul<T> for CsrMatrix<T> {
    #[inline] fn nrows(&self) -> usize { self.nrows() }
    #[inline] fn mul_vec(&self, v: &DVector<T>) -> DVector<T> { self * v }
    #[inline] fn inv_diagonal(&self) -> Self {
        let mut a =self.diagonal_as_csr();
        a.values_mut().iter_mut().for_each(|x| *x = T::one() / *x);
        a
    }
}

impl<T: SimdRealField + Copy> SpMatVecMul<T> for CscMatrix<T> {
    #[inline] fn nrows(&self) -> usize { self.nrows() }
    #[inline] fn mul_vec(&self, v: &DVector<T>) -> DVector<T> { self * v }
    #[inline] fn inv_diagonal(&self) -> Self {
        let mut a =self.diagonal_as_csc();
        a.values_mut().iter_mut().for_each(|x| *x = T::one() / *x);
        a
    }
}

pub trait IterativeSolver<M, V, T> {
    /// Initializes the solver's internal state with the system and vector b.
    fn init(&mut self, a: &M, b: &V, x0: Option<&V>);
    /// Performs one iteration of the solver. Returns true if converged.
    fn step(&mut self, a: &M, b: &V) -> bool;
    /// Resets the internal state (soft reset, preserves allocated memory).
    fn reset(&mut self);
    /// Completely resets the internal state (hard reset, frees memory if needed).
    fn hard_reset(&mut self);
    /// Partially resets the state (e.g., to restart with a new b but same matrix).
    fn soft_reset(&mut self);
    /// Gets the current solution.
    fn solution(&self) -> &V;
    /// Gets the number of iterations performed.
    fn iterations(&self) -> usize;

    /// Runs the iterative solver loop. Returns true if converged, false otherwise.
    fn solve_iterations(&mut self, a: &M, b: &V, max_iter: usize) -> bool {
        for _ in 0..max_iter {
            if self.step(a, b) {
                return true;
            }
        }
        self.step(a, b)
    }
}
