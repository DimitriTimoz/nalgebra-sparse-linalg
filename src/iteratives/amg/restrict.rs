use std::{fmt::Debug, ops::{AddAssign, MulAssign, SubAssign}};

use num_traits::Float;

use super::super::*;

fn build_r<N>(a: &CsrMatrix<N>, p: &CsrMatrix<N>) -> CsrMatrix<N>
where N: Float + Debug + 'static + AddAssign + SubAssign + MulAssign
{
    let n = a.nrows();
    let mut d_inv = Vec::with_capacity(n);
    for i in 0..n {
        let diag = a.get_entry(i, i).unwrap_or(nalgebra_sparse::SparseEntry::NonZero(&N::one())).into_value();
        if diag.is_zero() {
            panic!("Matrix is singular");
        }
        d_inv.push(N::one() / diag);
    }
    // helper: Pᵀ × diag(d_inv)
    let mut d_diag = CsrMatrix::zeros(n, n);
    for i in 0..n {
        let entry = d_diag.get_entry_mut(i, i);
        if let Some(nalgebra_sparse::SparseEntryMut::NonZero(value)) = entry {
            *value = d_inv[i];
        } else {
            panic!("Matrix is singular");
        }
    }
    p.transpose() * d_diag
}