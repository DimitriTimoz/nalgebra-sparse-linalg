use super::*;

pub(crate) fn build_r<N>(_a: &CsrMatrix<N>, p: &CsrMatrix<N>) -> CsrMatrix<N>
where N: RealField + Copy
{
    // Define R as the transpose of P (Galerkin approach)
    // The matrix 'a' is not needed for this definition but kept for API consistency
    p.transpose()
}
