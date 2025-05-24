/// Preconditioners for linear algebra solvers.
pub type LinAlgResult<T> = Result<T, LinAlgError>;

#[derive(Debug)]
pub enum LinAlgError {
    Singular,
    NotSPD,
    DimensionMismatch,
    Internal(String),
}

pub trait Preconditioner<M, V> {
    fn build(a: &M) -> Result<Self, &'static str>
    where
        Self: Sized;

    fn apply_left(&self, v: &mut V);     // Approx. solves M⁻¹ * v
    fn apply_right(&self, v: &mut V);    // Approx. solves M⁻¹ applied on the right (optional)
}


pub struct IdentityPreconditioner;

impl<M, V> Preconditioner<M, V> for IdentityPreconditioner {
    fn build(_a: &M) -> Result<Self, &'static str> {
        Ok(IdentityPreconditioner)
    }

    fn apply_left(&self, _v: &mut V) {}
    fn apply_right(&self, _v: &mut V) {}
}
