// https://edoc.unibas.ch/server/api/core/bitstreams/d56e8bdd-9b91-49ec-a8f5-04eff6db51ca/content
use super::*;

pub mod cycle;
pub mod graph;
pub mod coarsen;
pub mod interpolate;
pub mod restrict;
pub mod smooth;
pub mod level;

pub(crate) fn rap<N>(
    r: &CsrMatrix<N>,
    a: &CsrMatrix<N>,
    p: &CsrMatrix<N>,
) -> CsrMatrix<N>
where
    N: RealField + Copy,
{
    // 1. C = A * P   (CSR × CSR)
    let c = a * p;

    // 2. A_c = R * C
    r * &c
}

#[cfg(test)]
mod tests {
    use crate::iteratives::amg::{coarsen::Mark, graph::strength_graph, interpolate::build_p};

    use super::*;
    use approx::assert_relative_eq;
    use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};

    /// 1-D Poisson: 2 on the diagonal, −1 on neighbours.
    fn poisson_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0);
            if i + 1 < n {
                coo.push(i, i + 1, -1.0);
                coo.push(i + 1, i, -1.0);
            }
        }
        CsrMatrix::from(&coo)
    }

    #[test]
    fn graph_neighbours() {
        let a = poisson_1d(5);
        let g = strength_graph(&a, 0.0);           // θ = 0 → full graph
        assert_eq!(g[0], [1]);
        let mut v = g[2].clone(); v.sort();
        assert_eq!(v, [1, 3]);
    }

    #[test]
    fn coarsen_basic() {
        let a = poisson_1d(6);
        let (marks, coarse_of) = coarsen::coarsen(&a, 0.0);
        let n_c = marks.iter().filter(|m| matches!(m, Mark::C)).count();
        let n_f = marks.iter().filter(|m| matches!(m, Mark::F)).count();
        assert!(n_c >= 1 && n_f >= 1);
        for (i, m) in marks.iter().enumerate() {
            match m {
                Mark::C => assert!(coarse_of[i] < n_c),
                Mark::F => assert_eq!(coarse_of[i], usize::MAX),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn p_shape_and_identity() {
        let a = poisson_1d(8);
        let (marks, coarse_of) = coarsen::coarsen(&a, 0.0);
        let s = strength_graph(&a, 0.0);
        let p = build_p(&a, &marks, &coarse_of, &s);

        assert_eq!(p.nrows(), a.nrows());
        let n_c = marks.iter().filter(|m| matches!(m, Mark::C)).count();
        assert_eq!(p.ncols(), n_c);

        for (i, m) in marks.iter().enumerate() {
            if matches!(m, Mark::C) {
                let j = coarse_of[i];
                let row = p.row(i);
                assert_eq!(row.nnz(), 1);
                assert_eq!(row.col_indices()[0], j);
                assert_relative_eq!(row.values()[0], 1.0);
            }
        }
    }
}
