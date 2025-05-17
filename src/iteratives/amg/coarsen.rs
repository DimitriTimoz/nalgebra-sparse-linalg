use nalgebra_sparse::CooMatrix;
use num_traits::Float;

use super::{super::*, graph::strength_graph};

pub fn coarsen<T>(a: &CsrMatrix<T>, theta: T) -> CooMatrix<T> 
where 
    T: Float,
{
    let f = strength_graph(a, theta);
    let mut coarsened = CooMatrix::new(a.nrows(), a.ncols());
    // Compute the degrees of the rows
    let mut nodes = vec![(0, 0usize, false); a.nrows()];
    for (row_i, row) in a.row_iter().enumerate() {
        nodes[row_i] = (row_i, row.nnz(), false);
    }

    nodes.sort_by(|a, b| b.1.cmp(&a.1)); //TODO If failed be sure to check the sort order
    // Select the coarsening strategy
    let mut current_index = 0;
    while true {
        let node = nodes[current_index];

    }
    coarsened
}
