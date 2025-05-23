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
    // Pre-estimate capacity to reduce reallocations
    let estimated_nnz = n + (n - n_coarse) * 3; // Rough estimate
    trip.reserve(estimated_nnz);

    for i in 0..n {
        match marks[i] {
            Mark::C => {
                let j = coarse_of[i];
                trip.push(i, j, N::one());
            }
            Mark::F => {
                // Find C-point neighbors in the strength graph
                let c_neighbors: Vec<usize> = s[i]
                    .iter()
                    .copied()
                    .filter(|&nbr| matches!(marks[nbr], Mark::C))
                    .collect();

                if c_neighbors.is_empty() {
                    // Fallback: connect to first coarse point if available
                    if n_coarse > 0 {
                        trip.push(i, 0, N::one());
                    }
                    continue;
                }

                // Compute interpolation weights
                if let Some(diag_entry) = a.get_entry(i, i) {
                    let diag = diag_entry.into_value();
                    let mut weight_sum = N::zero();
                    
                    // First pass: compute weights
                    let mut weights = Vec::with_capacity(c_neighbors.len());
                    for &nbr in &c_neighbors {
                        if let Some(a_ij) = a.get_entry(i, nbr) {
                            let w = -a_ij.into_value() / diag;
                            weights.push((coarse_of[nbr], w));
                            weight_sum += w;
                        }
                    }
                    
                    // Normalize weights to sum to 1 for better stability
                    if weight_sum != N::zero() {
                        for (col, w) in weights {
                            trip.push(i, col, w / weight_sum);
                        }
                    } else {
                        // Fallback: equal weights
                        let equal_weight = N::one() / N::from_usize(c_neighbors.len()).unwrap();
                        for &nbr in &c_neighbors {
                            trip.push(i, coarse_of[nbr], equal_weight);
                        }
                    }
                }
            }
            Mark::Unmarked => unreachable!("All points should be marked during coarsening"),
        }
    }
    
    CsrMatrix::from(&trip)
}
