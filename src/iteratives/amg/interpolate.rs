use nalgebra_sparse::{na::RealField, CooMatrix};
use super::{super::*, coarsen::Mark};

pub(crate) fn build_p<N>(
    a: &CsrMatrix<N>,
    marks: &[Mark],
    coarse_of: &[usize],
    s: &[Vec<usize>],
) -> CsrMatrix<N>
where
    N: RealField + Copy
{
    let n = a.nrows();
    let n_coarse = coarse_of.iter().filter(|&&c| c != usize::MAX).count();

    let mut trip = CooMatrix::new(n, n_coarse);

    for i in 0..n {
        match marks[i] {
            Mark::C => {
                let j = coarse_of[i];
                trip.push(i, j, N::one());
            }
            Mark::F => {
                // C strength graph
                let c_neighbors: Vec<usize> = s[i]
                    .iter()
                    .copied()
                    .filter(|&nbr| matches!(marks[nbr], Mark::C))
                    .collect();

                if c_neighbors.is_empty() {
                    // If an F-point has no C-neighbors in its strong connections,
                    // connect it to the first coarse point (index 0) as a fallback.
                    // This requires that there is at least one coarse point (trip.ncols() > 0).
                    if trip.ncols() > 0 {
                        trip.push(i, 0, N::one());
                    }
                    continue;
                }

                // Compute weights w_{ij} = -a_ij / a_ii
                let diag = a.get_entry(i, i).unwrap_or(nalgebra_sparse::SparseEntry::NonZero(&N::one())).into_value();
                for &nbr in &c_neighbors {
                    if let Some(a_ij) = a.get_entry(i, nbr) {
                        let w = -a_ij.into_value() / diag;
                        let col = coarse_of[nbr];
                        trip.push(i, col, w);
                    }
                }
                // TODO: normalize weights
            }
            Mark::Unmarked => unreachable!(),
        }
    }
    CsrMatrix::from(&trip)
}
