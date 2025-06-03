pub mod iteratives;
pub mod preconditioners;
pub mod svd;

/// Re-exporting nalgebra_sparse for convenience
pub use nalgebra_sparse as na_sparse;
pub use na_sparse::{CscMatrix, CsrMatrix};