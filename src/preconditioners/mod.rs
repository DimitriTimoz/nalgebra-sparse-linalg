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

    /// Apply \(v M^{-1}\) (left preconditioning).
    fn apply_left(&self, _rhs: &mut Vec) -> LinAlgResult<()>
    where
        Self: Sized,
    {
        Err(LinAlgError::Internal(
            "apply_left not implemented for this preconditioner".to_string(),
        ))
    }
}

pub struct IdentityPreconditioner<Mat, Vec> {
    _marker: std::marker::PhantomData<(Mat, Vec)>,
}

impl<Mat, Vec> Preconditioner<Mat, Vec> for IdentityPreconditioner<Mat, Vec> {
    type Info = ();

    fn build(_: &Mat) -> LinAlgResult<Self> {
        Ok(Self {
            _marker: std::marker::PhantomData,
        })
    }

    fn apply(&self, _rhs: &mut Vec) {}
}
