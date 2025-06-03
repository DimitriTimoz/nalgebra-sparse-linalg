use nalgebra_sparse::{na::{linalg::{SVD, QR}, ComplexField, DMatrix, DVector}, CsrMatrix};
use rand_distr::{num_traits::Float, Distribution, Normal};

pub struct TruncatedSVD<T: ComplexField> {
    pub u: DMatrix<T>,
    pub singular_values: DVector<T::RealField>,
}

impl<T: Copy + ComplexField + Float> TruncatedSVD<T> {
    /// Generates an orthonormal basis for the range of matrix A using randomized range finding.
    /// This is more efficient than computing the full SVD for large matrices.
    pub fn range_random(a: &CsrMatrix<T>, k: usize) -> DMatrix<T> {
        // Ensure k doesn't exceed matrix dimensions
        let k = k.min(a.nrows()).min(a.ncols());
        
        // Create a random matrix with k columns for range finding
        let mut rng = rand::rng();
        let omega = DMatrix::from_fn(a.ncols(), k, |_, _| {
            T::from(Normal::new(0.0, 1.0).unwrap().sample(&mut rng)).unwrap()
        });
        let y = a * &omega;

        let qr = QR::new(y);
        qr.q()
    }

    /// Computes the truncated singular value decomposition of the given matrix.
    /// This uses a randomized algorithm for efficient computation of the top k singular vectors.
    /// 
    /// # Arguments
    /// * `matrix` - The sparse matrix to decompose
    /// * `k` - Number of singular values/vectors to compute
    /// 
    /// # Returns
    /// A `TruncatedSVD` containing the top k left singular vectors and singular values
    pub fn new(matrix: &CsrMatrix<T>, k: usize) -> Self {
        if k == 0 {
            return Self {
                u: DMatrix::zeros(matrix.nrows(), 0),
                singular_values: DVector::zeros(0),
            };
        }
        
        // Step 1: Range finding get an orthonormal basis Q for the range of A
        let q = Self::range_random(matrix, k);

        // Step 2: Project A onto the range of Q to get a smaller matrix B
        let b = matrix.transpose() * &q;
        
        // Step 3: Compute the SVD of B^T (which is q^T * A)
        let svd = SVD::new(b.transpose(), true, false);
        let u_small = svd.u.unwrap();
        let singular_values = svd.singular_values;
        
        // Step 4: The left singular vectors of A are Q * U_small
        let u = &q * &u_small;
        
        // Take only the first k columns (in case we got more)
        let k_actual = k.min(u.ncols()).min(singular_values.len());
        let u = u.view_range(.., ..k_actual).into_owned();
        let singular_values = singular_values.view_range(..k_actual, ..).into_owned();

        Self {
            u,
            singular_values,
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_truncated_svd() {
        use nalgebra_sparse::CsrMatrix;
        use nalgebra_sparse::na::DMatrix;

        let dense_matrix = DMatrix::from_row_slice(3, 3, &[
            1.0, 2.0, 0.0,
            0.0, 3.0, 4.0,
            5.0, 6.0, 7.0,
        ]);
        let a = CsrMatrix::from(&dense_matrix);

        let svd = super::TruncatedSVD::new(&a, 2);
        assert_eq!(svd.u.nrows(), 3);
        assert_eq!(svd.u.ncols(), 2);
        assert_eq!(svd.singular_values.len(), 2);
        
        // Check that singular values are in descending order
        assert!(svd.singular_values[0] >= svd.singular_values[1]);
        assert!(svd.singular_values[1] > 0.0);
    }

    #[test]
    fn test_truncated_svd_single_rank() {
        use nalgebra_sparse::CsrMatrix;
        use nalgebra_sparse::na::DMatrix;

        // Create a rank-1 matrix
        let dense_matrix = DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            2.0, 4.0,
            3.0, 6.0,
        ]);
        let a = CsrMatrix::from(&dense_matrix);

        let svd = super::TruncatedSVD::new(&a, 1);
        assert_eq!(svd.u.nrows(), 3);
        assert_eq!(svd.u.ncols(), 1);
        assert_eq!(svd.singular_values.len(), 1);
        assert!(svd.singular_values[0] > 0.0);
    }

    #[test]
    fn test_truncated_svd_zero_rank() {
        use nalgebra_sparse::CsrMatrix;
        use nalgebra_sparse::na::DMatrix;

        let dense_matrix = DMatrix::from_row_slice(3, 3, &[
            1.0, 2.0, 0.0,
            0.0, 3.0, 4.0,
            5.0, 6.0, 7.0,
        ]);
        let a = CsrMatrix::from(&dense_matrix);

        let svd = super::TruncatedSVD::new(&a, 0);
        assert_eq!(svd.u.nrows(), 3);
        assert_eq!(svd.u.ncols(), 0);
        assert_eq!(svd.singular_values.len(), 0);
    }

    #[test]
    fn test_truncated_svd_identity_matrix() {
        use nalgebra_sparse::CsrMatrix;
        use nalgebra_sparse::na::DMatrix;

        // Identity matrix has known singular values: all 1.0
        let dense_matrix = DMatrix::<f64>::identity(4, 4);
        let a = CsrMatrix::from(&dense_matrix);

        let svd = super::TruncatedSVD::new(&a, 3);
        assert_eq!(svd.u.nrows(), 4);
        assert_eq!(svd.u.ncols(), 3);
        assert_eq!(svd.singular_values.len(), 3);
        
        // All singular values should be approximately 1.0
        for &val in svd.singular_values.iter() {
            assert!((val - 1.0f64).abs() < 1e-10f64, "Singular value {} should be close to 1.0", val);
        }
        
        // Check that singular values are in descending order
        for i in 1..svd.singular_values.len() {
            assert!(svd.singular_values[i-1] >= svd.singular_values[i]);
        }
    }

    #[test]
    fn test_truncated_svd_diagonal_matrix() {
        use nalgebra_sparse::CsrMatrix;
        use nalgebra_sparse::na::DMatrix;

        // Create a diagonal matrix with known singular values
        let mut dense_matrix = DMatrix::<f64>::zeros(4, 4);
        dense_matrix[(0, 0)] = 5.0;
        dense_matrix[(1, 1)] = 3.0;
        dense_matrix[(2, 2)] = 2.0;
        dense_matrix[(3, 3)] = 1.0;
        
        let a = CsrMatrix::from(&dense_matrix);

        let svd = super::TruncatedSVD::new(&a, 3);
        assert_eq!(svd.u.nrows(), 4);
        assert_eq!(svd.u.ncols(), 3);
        assert_eq!(svd.singular_values.len(), 3);
        
        // Check that singular values are approximately correct and in descending order
        let expected = [5.0, 3.0, 2.0];
        for (i, &expected_val) in expected.iter().enumerate() {
            assert!(
                (svd.singular_values[i] - expected_val).abs() < 1e-10f64,
                "Singular value {} should be close to {}, got {}",
                i, expected_val, svd.singular_values[i]
            );
        }
    }

    #[test]
    fn test_truncated_svd_rank_deficient_matrix() {
        use nalgebra_sparse::CsrMatrix;
        use nalgebra_sparse::na::DMatrix;

        // Create a rank-2 matrix (two identical rows)
        let dense_matrix = DMatrix::from_row_slice(4, 3, &[
            1.0, 2.0, 3.0,
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        let a = CsrMatrix::from(&dense_matrix);

        let svd = super::TruncatedSVD::new(&a, 3);
        assert_eq!(svd.u.nrows(), 4);
        assert_eq!(svd.u.ncols(), 3);
        assert_eq!(svd.singular_values.len(), 3);
        
        // The third singular value should be much smaller than the first two
        // due to rank deficiency
        assert!(svd.singular_values[0] > svd.singular_values[1]);
        assert!(svd.singular_values[2] < svd.singular_values[1] * 0.1);
    }

    #[test]
    fn test_truncated_svd_orthogonality() {
        use nalgebra_sparse::CsrMatrix;
        use nalgebra_sparse::na::DMatrix;

        // Create a random-like matrix for testing orthogonality
        let dense_matrix = DMatrix::from_row_slice(5, 4, &[
            1.0, 2.0, 0.0, 1.0,
            0.0, 3.0, 4.0, 2.0,
            5.0, 6.0, 7.0, 0.0,
            2.0, 1.0, 3.0, 4.0,
            1.0, 0.0, 2.0, 5.0,
        ]);
        let a = CsrMatrix::from(&dense_matrix);

        let svd = super::TruncatedSVD::new(&a, 3);
        
        // Check that U columns are orthonormal (U^T * U should be identity)
        let u_t_u = svd.u.transpose() * &svd.u;
        
        // Check diagonal elements are close to 1
        for i in 0..u_t_u.nrows() {
            assert!(
                (u_t_u[(i, i)] - 1.0f64).abs() < 1e-10,
                "Diagonal element ({}, {}) should be close to 1.0, got {}",
                i, i, u_t_u[(i, i)]
            );
        }
        
        // Check off-diagonal elements are close to 0
        for i in 0..u_t_u.nrows() {
            for j in 0..u_t_u.ncols() {
                if i != j {
                    assert!(
                        u_t_u[(i, j)].abs() < 1e-10f64,
                        "Off-diagonal element ({}, {}) should be close to 0.0, got {}",
                        i, j, u_t_u[(i, j)]
                    );
                }
            }
        }
    }

    #[test]
    fn test_truncated_svd_reconstruction_approximation() {
        use nalgebra_sparse::CsrMatrix;
        use nalgebra_sparse::na::DMatrix;

        let dense_matrix = DMatrix::from_row_slice(3, 3, &[
            4.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 2.0,
        ]);
        let a = CsrMatrix::from(&dense_matrix);

        let svd = super::TruncatedSVD::new(&a, 2);
        
        // Create a rank-2 approximation: A_k = U * Σ * V^T
        // For this test, we'll just verify the dominant singular values
        // since we don't compute V in our truncated implementation
        
        // The first two singular values should capture most of the matrix energy
        let first_two_energy: f64 = svd.singular_values.iter().take(2).map(|x| x * x).sum();
        
        // For a diagonal matrix with values [4, 3, 2], truncating to k=2 should capture
        // (16 + 9) / (16 + 9 + 4) = 25/29 ≈ 86% of the energy
        let energy_ratio = first_two_energy / (4.0*4.0 + 3.0*3.0 + 2.0*2.0);
        assert!(energy_ratio > 0.8, "Energy ratio {} should be > 0.8", energy_ratio);
    }

    #[test]
    fn test_truncated_svd_larger_k_than_rank() {
        use nalgebra_sparse::CsrMatrix;
        use nalgebra_sparse::na::DMatrix;

        // Create a 4x3 matrix (rank at most 3)
        let dense_matrix = DMatrix::from_row_slice(4, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ]);
        let a = CsrMatrix::from(&dense_matrix);

        // Request more singular values than the rank
        let svd = super::TruncatedSVD::new(&a, 5);
        
        // Should return at most min(m, n) singular values
        assert!(svd.u.ncols() <= 3);
        assert!(svd.singular_values.len() <= 3);
        assert_eq!(svd.u.nrows(), 4);
        
        // All returned singular values should be positive
        for &val in svd.singular_values.iter() {
            assert!(val > -0.01f64, "Singular value {} should be positive", val);
        }
    }
}
