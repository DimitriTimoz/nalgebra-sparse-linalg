use num_traits::Float;

use super::super::*;

pub(crate) fn strength_graph<T>(a: &CsrMatrix<T>, theta: T) -> Vec<Vec<usize>>
where
    T: Float
{
    (0..a.nrows()).map(|i| {
        let row = a.get_row(i);
        match row {
            Some(row) => {
                let threashold = row.values().iter().map(|v| v.abs()).fold(T::zero(), T::max)*theta;
                // Filter out the diagonal entry and entries below the threshold
                let r = row.col_indices()
                    .iter()
                    .cloned()
                    .filter(|j| *j != i && unsafe {row.get_entry(*j).unwrap_unchecked().into_value().abs() >= threashold})
                    .collect();
                r
            },
            None => vec![],
            
        }
    }).collect()
}
