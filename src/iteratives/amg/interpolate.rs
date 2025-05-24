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

    // Better capacity estimation based on connectivity
    let f_points = marks.iter().filter(|&&m| matches!(m, Mark::F)).count();
    let estimated_nnz = n_coarse + f_points * 4; // More accurate estimate
    
    let mut trip = CooMatrix::new(n, n_coarse);
    trip.reserve(estimated_nnz);

    // Pre-allocate reusable vectors
    let mut c_neighbors = Vec::with_capacity(8);
    let mut weights = Vec::with_capacity(8);

    for i in 0..n {
        match marks[i] {
            Mark::C => {
                trip.push(i, coarse_of[i], N::one());
            }
            Mark::F => {
                c_neighbors.clear();
                // Collect coarse neighbors
                for &nbr in &s[i] {
                    if matches!(marks[nbr], Mark::C) {
                        c_neighbors.push(nbr);
                    }
                }

                // If no coarse neighbors, add a zero entry
                if c_neighbors.is_empty() {
                    if n_coarse > 0 {
                        trip.push(i, 0, N::one());
                    }
                    continue;
                }

                // Early exit if no diagonal entry
                let Some(diag_entry) = a.get_entry(i, i) else {
                    let equal_weight = N::one() / N::from_usize(c_neighbors.len()).unwrap();
                    for &nbr in &c_neighbors {
                        trip.push(i, coarse_of[nbr], equal_weight);
                    }
                    continue;
                };

                let diag = diag_entry.into_value();
                let mut weight_sum = N::zero();
                
                weights.clear();
                weights.reserve(c_neighbors.len());
                
                // Compute weights in single pass
                for &nbr in &c_neighbors {
                    if let Some(a_ij) = a.get_entry(i, nbr) {
                        let w = -a_ij.into_value() / diag;
                        weights.push((coarse_of[nbr], w));
                        weight_sum += w;
                    }
                }
                
                // Add entries with normalized weights
                if weight_sum != N::zero() {
                    let inv_sum = N::one() / weight_sum;
                    for (col, w) in weights.drain(..) {
                        trip.push(i, col, w * inv_sum);
                    }
                } else {
                    let equal_weight = N::one() / N::from_usize(c_neighbors.len()).unwrap();
                    for &nbr in &c_neighbors {
                        trip.push(i, coarse_of[nbr], equal_weight);
                    }
                }
            }
            Mark::Unmarked => unreachable!("All points should be marked during coarsening"),
        }
    }
    
    CsrMatrix::from(&trip)
}
