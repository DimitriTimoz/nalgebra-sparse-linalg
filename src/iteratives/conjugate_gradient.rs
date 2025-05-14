use nalgebra_sparse::{na::{DVector, Scalar, SimdPartialOrd}, CsrMatrix};
use num_traits::{real::Real, Num, NumAssign, NumAssignOps, NumOps, Signed};


pub fn solve<T>(a: &CsrMatrix<T>, b: &DVector<T>, max_iter: usize) -> Option<DVector<T>> 
where 
    T: Scalar + Num + Real + NumOps + NumAssign + NumAssignOps + Copy + SimdPartialOrd + Signed
{
    let mut x = DVector::<T>::zeros(a.nrows());

    let mut residual = b - a * x.clone();
    let mut residual_dot = residual.dot(&residual);
    // Check if the inital guess is already a solution
    let tol = T::from(1e-10).unwrap();
    let norm = residual.abs().max();
    if norm <= tol {
        return Some(x);
    }
    
    let mut p = residual.clone();
    for _ in 0..max_iter {
        let ap = a * p.clone();
        let alpha = residual_dot / (p.clone().dot(&ap));
        x += &p * alpha;
        
        let new_residual = residual.clone() - &ap * alpha;
        
        // Check for convergence
        let norm = new_residual.abs().max();
        if norm <= tol {
            return Some(x);
        }
        let new_residual_dot = new_residual.dot(&new_residual);
        let beta = new_residual_dot / residual_dot;
        residual_dot = new_residual_dot;
        p = &new_residual + &p * beta;
        residual = new_residual;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::{na::DVector, CooMatrix};

    #[test]
    fn test_conjugate_grad() {
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
        let prod = &a * &result;
        assert_eq!(prod.len(), 4);
        for i in 0..prod.len() {
            assert!((prod[i] - b[i]).abs() < 1e-10);
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
    }
}
