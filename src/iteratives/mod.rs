pub mod jacobi;
pub mod conjugate_gradient;

pub use jacobi::solve;
pub use nalgebra_sparse::CsrMatrix;
