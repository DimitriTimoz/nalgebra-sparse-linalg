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
                let max_abs_off_diag_val = col_indices.iter()
                    .zip(values.iter())
                    .filter(|(col_idx, _)| **col_idx != i)
                    .map(|(_, val)| val.abs())
                    .fold(T::zero(), |acc, val| if val > acc { val } else { acc });

                // If there are no off-diagonal elements, no strong connections can be formed.
                if max_abs_off_diag_val == T::zero() {
                    result.push(Vec::new());
                    continue;
                }

                let threshold_val = max_abs_off_diag_val * theta;

                // Collect strong neighbors directly using iterator chains
                let strong_neighbors: Vec<usize> = col_indices.iter()
                    .zip(values.iter())
                    .filter_map(|(j_idx, val)| {
                        (*j_idx != i && val.abs() >= threshold_val).then_some(*j_idx)
                    })
                    .collect();
                
                result.push(strong_neighbors);
            },
            None => result.push(Vec::new()),
        }
    }
    
    result
}
