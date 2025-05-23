use super::super::*;

pub(crate) fn build_r<N>(a: &CsrMatrix<N>, p: &CsrMatrix<N>) -> CsrMatrix<N>
where N: RealField + Copy
{
    // Simply define R as the transpose of P.
    // The matrix 'a' (coarse grid operator from previous level) is not needed for this definition.
    p.transpose()
}