/*
Input: initial guess x(0) to the solution, (diagonal dominant) matrix A, right-hand side vector b, convergence criterion
Output: solution when convergence is reached
Comments: pseudocode based on the element-based formula above

k = 0
while convergence not reached do
    for i := 1 step until n do
        σ = 0
        for j := 1 step until n do
            if j ≠ i then
                σ = σ + aij xj(k)
            end
        end
        xi(k+1) = (bi − σ) / aii
    end
    increment k
end
*/
use nalgebra_sparse::{na::{DVector, Scalar}, CsrMatrix};
use num_traits::{Num, NumAssign, NumAssignOps, NumOps, Signed};

pub fn solve<T>(a: &CsrMatrix<T>, b: &DVector<T>, max_iter: usize) -> Option<DVector<T>> 
where 
    T: Scalar + Num + NumOps + NumAssign + NumAssignOps + Signed + Copy
{
    let mut x = DVector::<T>::zeros(a.nrows());
    let mut new_x = DVector::<T>::zeros(a.nrows());
    for _ in 0..max_iter {
        for row_i in 0..a.nrows() {
            if let Some(row) = &a.get_row(row_i) {
                let mut sigma = b[row_i];

                let col_indices = row.col_indices();
                let values = row.values();
                for (col_i, value) in col_indices.iter().zip(values.iter()) {
                    if *col_i != row_i {
                        sigma -= *value * x[*col_i];
                    }
                }
                let diag = a.get_entry(row_i, row_i)?.into_value();
                if diag.is_zero() {
                    return None;
                }
                new_x[row_i] = sigma / diag;
            }
        }
        // Check for convergence
        let norm = x.iter().zip(new_x.iter()).fold(T::zero(), |m, (x_i, new_x_i)| {
            m + (*x_i - *new_x_i).abs()
        });
        x = new_x.clone();
        if norm.is_zero() {
            return Some(new_x);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::{na::DVector, CooMatrix};

    
    #[test]
    fn test_jacobi() {
        let a = CsrMatrix::identity(10);
        let b = DVector::from_vec(vec![3.0;10]);
        let max_iter = 2500;

        let result = solve(&a, &b, max_iter);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.len(), 10);
        for i in 0..result.len() {
            assert_eq!(result[i], 3.0);
        }
        
        let m = [[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0.0, 3., -1., 8.]];
        let b = [6., 25., -11., 15.];
        
        // Create a CooMatrix and fill it with values
        let mut coo = CooMatrix::new(4, 4);
        for (i, row) in m.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0.0 {
                    coo.push(i, j, val);
                }
            }
        }
        // Convert CooMatrix to CsrMatrix
        let a = CsrMatrix::from(&coo);
        let b = DVector::from_vec(b.to_vec());
        let max_iter = 2500;
        let result = solve(&a, &b, max_iter);
        assert!(result.is_some());
        let result = result.unwrap();
        let result = a * result;
        assert_eq!(result.len(), 4);
        for i in 0..result.len() {
            assert_eq!(result[i], b[i]);
        }

        // Null test
        let a = CsrMatrix::identity(10);
        let b = DVector::from_vec(vec![0.0;10]);
        let max_iter = 2500;
        let result = solve(&a, &b, max_iter);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.len(), 10);
        for i in 0..result.len() {
            assert_eq!(result[i], 0.0);
        }

        // Non converging test
        let m = [[1., 2., 3., 4.],
              [5., 6., 7., 8.],
              [9., 10., 11., 12.],
              [13., 14., 15., 16.]];
        let b = [1., 2., 3., 4.];
        // Create a CooMatrix and fill it with values
        let mut coo = CooMatrix::new(4, 4);
        for (i, row) in m.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0.0 {
                    coo.push(i, j, val);
                }
            }
        }
        // Convert CooMatrix to CsrMatrix
        let a = CsrMatrix::from(&coo);
        let b = DVector::from_vec(b.to_vec());
        let max_iter = 2500;
        let result = solve(&a, &b, max_iter);
        assert!(result.is_none());
    }
}
