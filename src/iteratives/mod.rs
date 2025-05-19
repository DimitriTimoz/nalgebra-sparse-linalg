pub mod jacobi;
pub mod biconjugate_gradient;
pub mod conjugate_gradient;
pub mod gauss_seidel;
pub mod relaxation;
pub mod amg;

pub use biconjugate_gradient::solve as solve_biconjugate_gradient;

pub use nalgebra_sparse::{CscMatrix, CsrMatrix};
pub(crate) use nalgebra_sparse::{na::{DVector, RealField, SimdRealField}};
pub(crate) use rayon::prelude::*;

pub trait SpMatVecMul<T: SimdRealField> {
    fn nrows(&self) -> usize;
    fn mul_vec(&self, v: &DVector<T>) -> DVector<T>;
}
impl<T: SimdRealField> SpMatVecMul<T> for CsrMatrix<T> {
    #[inline] fn nrows(&self) -> usize { self.nrows() }
    #[inline] fn mul_vec(&self, v: &DVector<T>) -> DVector<T> { self * v }
}

impl<T: SimdRealField> SpMatVecMul<T> for CscMatrix<T> {
    #[inline] fn nrows(&self) -> usize { self.nrows() }
    #[inline] fn mul_vec(&self, v: &DVector<T>) -> DVector<T> { self * v }
}
