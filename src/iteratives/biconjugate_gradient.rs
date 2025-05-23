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
/// This function initializes the solution vector `x` to zeros.
/// It is generic over any matrix type `M` that implements the `SpMatVecMul<T>` trait.
///
/// # Arguments
/// * `a` - A reference to the sparse matrix `A`.
/// * `b` - A reference to the right-hand side vector `b`.
/// * `max_iter` - The maximum number of iterations to perform.
/// * `tol` - The tolerance for convergence. The iteration stops if the norm of the residual
///   is less than or equal to `tol`.
///
/// # Returns
/// * `Some(DVector<T>)` - The solution vector `x` if convergence is achieved within `max_iter`.
/// * `None` - If the method does not converge within `max_iter` iterations or encounters a breakdown.
pub fn solve<M, T>(a: &M, b: &DVector<T>, max_iter: usize, tol: T) -> Option<DVector<T>> 
where 
    M: SpMatVecMul<T>,
    T: SimdRealField + PartialOrd + Copy
{
    let mut x = DVector::<T>::zeros(a.nrows());
    if solve_with_initial_guess::<M, T, IdentityPreconditioner<M, M>>(a, b, &mut x, max_iter, tol).is_ok() {
        Some(x)
    } else {
        None
    }
}

/// Solves the general linear system Ax = b using the BiConjugate Gradient (BiCG) method,
/// starting with an initial guess for `x`.
///
/// This function modifies `x` in place. It is generic over any matrix type `M` that
/// implements the `SpMatVecMul<T>` trait.
///
/// # Arguments
/// * `a` - A reference to the sparse matrix `A`.
/// * `b` - A reference to the right-hand side vector `b`.
/// * `x` - A mutable reference to the initial guess for the solution vector. This vector
///   will be updated in place with the refined solution.
/// * `max_iter` - The maximum number of iterations to perform.
/// * `tol` - The tolerance for convergence. The iteration stops if the norm of the residual
///   is less than or equal to `tol`.
///
/// # Returns
/// * `true` - If the method converges to a solution within `max_iter` iterations.
/// * `false` - If the method does not converge within `max_iter` iterations or encounters a breakdown.
pub fn solve_with_initial_guess<M, T, P>(a: &M, b: &DVector<T>,  x: &mut DVector<T>, max_iter: usize, tol: T) -> ConvergedResult
where 
    M: SpMatVecMul<T>,
    T: SimdRealField + PartialOrd + Copy,
    P: Preconditioner<M, M>
{
    // Initial residual: r0 = b - A * x0, but x0 = 0 => r0 = b
    let mut residual = b.clone();
    let residual_hat_0 = residual.clone(); // Should be a random or fixed vector for BiCG

    let mut residual_dot = residual.dot(&residual_hat_0);
    if residual.clone().magnitude() <= tol {
        return Ok(Converged)
    }
    let preconditionner = P::build(a).map_err(|_| NotConverged)?;
    let mut p = residual.clone();
    let mut s = DVector::<T>::zeros(a.nrows());
    for _ in 0..max_iter {
        let mut v: nalgebra_sparse::na::Matrix<T, nalgebra_sparse::na::Dyn, nalgebra_sparse::na::Const<1>, nalgebra_sparse::na::VecStorage<T, nalgebra_sparse::na::Dyn, nalgebra_sparse::na::Const<1>>> = a.mul_vec(&p);
        let alpha = residual_dot / residual_hat_0.dot(&v);
        x.axpy(alpha, &p, T::one());
        s.copy_from(&residual); 
        s.axpy(-alpha, &v, T::one());

        // Check for convergence
        if s.magnitude() <= tol {
            return Ok(Converged)
        }
        
        let t = a.mul_vec(&s);
        let omega = t.dot(&s)/t.dot(&t);
        x.axpy(omega, &s, T::one());
        let new_residual = &s - &t * omega;
        // Check for convergence
        if new_residual.magnitude() <= tol {
            return Ok(Converged)
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
    Err(NotConverged)
}
