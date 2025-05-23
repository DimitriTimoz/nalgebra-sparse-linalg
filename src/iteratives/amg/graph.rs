use super::super::*;

pub(crate) fn strength_graph<T>(a: &CsrMatrix<T>, theta: T) -> Vec<Vec<usize>>
where
    T: RealField + Copy
{
    (0..a.nrows()).map(|i| {
        let row_view = a.get_row(i);
        match row_view {
            Some(row) => {
                // Find the maximum absolute off-diagonal value in the current row i
                let mut max_abs_off_diag_val = T::zero();
                for (col_idx, val) in row.col_indices().iter().zip(row.values().iter()) {
                    if *col_idx != i { // Exclude diagonal
                        let abs_val = val.abs();
                        if abs_val > max_abs_off_diag_val {
                            max_abs_off_diag_val = abs_val;
                        }
                    }
                }

                // If there are no off-diagonal elements, no strong connections can be formed.
                if max_abs_off_diag_val == T::zero() {
                    return vec![];
                }

                let threshold_val = max_abs_off_diag_val * theta;

                // Filter connections based on the new threshold
                row.col_indices()
                    .iter()
                    .zip(row.values().iter())
                    .filter_map(|(j_idx, val)| {
                        // Point j is a strong neighbor of i if |a_ij| >= threshold_val
                        // And j must not be i itself.
                        if *j_idx != i && val.abs() >= threshold_val {
                            Some(*j_idx)
                        } else {
                            None
                        }
                    })
                    .collect()
            },
            None => vec![],
        }
    }).collect()
}
