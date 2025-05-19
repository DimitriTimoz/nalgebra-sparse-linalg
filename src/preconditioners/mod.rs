/// Preconditioners for linear algebra solvers.
pub type LinAlgResult<T> = Result<T, LinAlgError>;

#[derive(Debug)]
pub enum LinAlgError {
    Singular,
    NotSPD,
    DimensionMismatch,
    Internal(String),
}

pub trait Preconditioner<Mat, Vec> {
    type Info: Default;

    fn build(matrix: &Mat) -> LinAlgResult<Self>
    where
        Self: Sized;

    /// Apply \(M^{-1} v\) (right preconditioning).
    fn apply(&self, rhs: &mut Vec);

    fn apply_left(&self, rhs: &mut Vec) -> LinAlgResult<()>
    where
        Self: Sized,
    {
        Err(LinAlgError::Internal(
            "apply_left not implemented for this preconditioner".to_string(),
        ))
    }
}
