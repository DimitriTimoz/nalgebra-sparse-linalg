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

