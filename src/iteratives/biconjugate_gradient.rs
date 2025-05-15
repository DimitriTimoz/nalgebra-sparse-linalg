use super::*;

pub fn solve<M, T>(a: &M, b: &DVector<T>, max_iter: usize, tol: T) -> Option<DVector<T>> 
where 
    M: SpMatVecMul<T>,
    T: SimdRealField + PartialOrd
{
    let mut x = DVector::<T>::zeros(a.nrows());

    let mut residual = b - &a.mul_vec(&DVector::<T>::zeros(a.nrows()));
    let residual_hat_0 = residual.clone();

    let mut residual_dot = residual.dot(&residual);
    let mut p = residual.clone();

    for _ in 0..max_iter {
        let v = a.mul_vec(&p);
        let alpha = residual_dot.clone() / residual_hat_0.dot(&v);
        x.axpy(alpha.clone(), &p, T::one());
        let s = &residual - &v * alpha.clone();

        // Check for convergence
        if s.magnitude() <= tol {
            return Some(x);
        }
        
        let t = a.mul_vec(&s);
        let omega = t.clone().dot(&s)/t.dot(&t);
        x.axpy(omega.clone(), &s, T::one());
        let new_residual = &s - &t * omega.clone();
        // Check for convergence
        if new_residual.magnitude() <= tol {
            return Some(x);
        }
        let new_residual_dot = residual_hat_0.dot(&new_residual);
        let beta = (new_residual_dot.clone()/residual_dot.clone())*(alpha.clone()/omega.clone());
        p = &new_residual + &((&p - &v * omega.clone()) * beta);
        residual_dot = new_residual_dot;
        residual = new_residual;
    }
    None
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::{na::DVector, CooMatrix, CsrMatrix};

    #[test]
    fn test_biconjugate_grad() {
        let a = CsrMatrix::identity(10);
        let b = DVector::from_vec(vec![3.0;10]);
        let max_iter = 2500;
        let tol = 1e-10;

        let result = solve::<CsrMatrix<f64>, f64>(&a, &b, max_iter, tol);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.len(), 10);
        for i in 0..result.len() {
            assert!((result[i] - 3.0).abs() < tol);
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
        let result = solve::<CsrMatrix<f64>, f64>(&a, &b, max_iter, tol);
        assert!(result.is_some());
        let result = result.unwrap();
        let prod = &a * &result;
        assert_eq!(prod.len(), 4);
        for i in 0..prod.len() {
            assert!((prod[i] - b[i]).abs() < tol);
        }
    }

    #[test]
    fn test_biconjugate_grad_non_symmetric() {
        // Matrice non symétrique 3x3
        // | 4  1  2 |
        // | 0  3 -1 |
        // | 2  0  1 |
        let m = [
            [4.0, 1.0, 2.0],
            [0.0, 3.0, -1.0],
            [2.0, 0.0, 1.0],
        ];
        let b = [7.0, 2.0, 5.0];

        let mut coo = CooMatrix::new(3, 3);
        for (i, row) in m.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0.0 {
                    coo.push(i, j, val);
                }
            }
        }
        let a = CsrMatrix::from(&coo);
        let b = DVector::from_vec(b.to_vec());
        let max_iter = 2500;
        let tol = 1e-10;
        let result = solve::<CsrMatrix<f64>, f64>(&a, &b, max_iter, tol);
        assert!(result.is_some());
        let result = result.unwrap();
        let prod = &a * &result;
        assert_eq!(prod.len(), 3);
        for i in 0..prod.len() {
            assert!((prod[i] - b[i]).abs() < tol);
        }
    }
}
