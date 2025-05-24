use super::*;

pub(crate) fn strength_graph<T>(a: &CsrMatrix<T>, theta: T) -> Vec<Vec<usize>>
where
    T: RealField + Copy
{
    let n = a.nrows();
    let mut result = Vec::with_capacity(n);
    
    for i in 0..n {
        let row_view = a.get_row(i);
        match row_view {
            Some(row) => {
                let col_indices = row.col_indices();
                let values = row.values();
                
                // Find the maximum absolute off-diagonal value in the current row i
                let mut max_abs_off_diag_val = T::zero();
                for (col_idx, val) in col_indices.iter().zip(values.iter()) {
                    if *col_idx != i { // Exclude diagonal
                        let abs_val = val.abs();
                        if abs_val > max_abs_off_diag_val {
                            max_abs_off_diag_val = abs_val;
                        }
                    }
                }

                // If there are no off-diagonal elements, no strong connections can be formed.
                if max_abs_off_diag_val == T::zero() {
                    result.push(Vec::new());
                    continue;
                }

                let threshold_val = max_abs_off_diag_val * theta;

                // Pre-allocate vector for strong neighbors (estimate based on typical sparsity)
                let mut strong_neighbors = Vec::with_capacity(col_indices.len().min(10));
                
                // Filter connections based on the threshold
                for (j_idx, val) in col_indices.iter().zip(values.iter()) {
                    // Point j is a strong neighbor of i if |a_ij| >= threshold_val
                    // And j must not be i itself.
                    if *j_idx != i && val.abs() >= threshold_val {
                        strong_neighbors.push(*j_idx);
                    }
                }
                
                result.push(strong_neighbors);
            },
            None => result.push(Vec::new()),
        }
    }
    
    result
}
