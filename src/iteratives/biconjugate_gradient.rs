//! BiConjugate Gradient iterative solver for sparse linear systems.
//!
//! This module provides a function to solve general (possibly non-symmetric) linear systems
//! using the BiConjugate Gradient (BiCG) method for matrices in CSR format.
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

pub fn solve<M, T>(a: &M, b: &DVector<T>, max_iter: usize, tol: T) -> Option<DVector<T>> 
where 
    M: SpMatVecMul<T>,
    T: SimdRealField + PartialOrd + Copy
{
    let n = a.nrows();
    let mut x = DVector::<T>::zeros(n);

    // Initial residual: r0 = b - A * x0, but x0 = 0 => r0 = b
    let mut residual = b.clone();
    let residual_hat_0 = residual.clone(); // Should be a random or fixed vector for BiCG

    let mut residual_dot = residual.dot(&residual_hat_0);
    if residual.clone().magnitude() <= tol {
        return Some(x);
    }
    let mut p = residual.clone();

    for _ in 0..max_iter {
        let mut v = a.mul_vec(&p);
        let alpha = residual_dot / residual_hat_0.dot(&v);
        x.axpy(alpha, &p, T::one());
        let s = &residual - &v * alpha;

        // Check for convergence
        if s.magnitude() <= tol {
            return Some(x);
        }
        
        let t = a.mul_vec(&s);
        let omega = t.dot(&s)/t.dot(&t);
        x.axpy(omega, &s, T::one());
        let new_residual = &s - &t * omega;
        // Check for convergence
        if new_residual.magnitude() <= tol {
            return Some(x);
        }
        let new_residual_dot = residual_hat_0.dot(&new_residual);
        let beta = (new_residual_dot/residual_dot)*(alpha/omega);
        v.scale_mut(omega);
        p -= &v;
        p.scale_mut(beta);
        p += &new_residual;
        residual_dot = new_residual_dot;
        residual = new_residual;
    }
    None
}
