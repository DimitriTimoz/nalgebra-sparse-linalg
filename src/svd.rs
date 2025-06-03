use nalgebra_sparse::{na::{linalg::{SVD, QR}, ComplexField, DMatrix}, CsrMatrix};
use rand_distr::{num_traits::Float, Distribution, Normal};

pub struct TruncatedSVD<T> {
    pub u: DMatrix<T>,
}

impl<T: Copy + ComplexField + Float> TruncatedSVD<T> {
    pub fn range_random(a: &CsrMatrix<T> ) -> (DMatrix<T>, DMatrix<T>) {
        let omega = DMatrix::from_fn(a.nrows(), a.ncols(), |_, _| {
            T::from(Normal::new(0.0, 1.0).unwrap().sample(&mut rand::rng())).unwrap()
        });
        let y = a * &omega;

        let qr = QR::new(y);
        let q = qr.q();
        let r = qr.r();
        (q, r)
    }

    /// Computes the singular value decomposition of the given matrix.
    pub fn new(matrix: &CsrMatrix<T>, k: usize) -> Self {
        let (q, r) = Self::range_random(matrix);

        let b = (matrix.transpose() * q.clone()).transpose();
        
        // Solve SVD on b;
        let svd = SVD::new(b, true, false);
        let u = svd.u.unwrap();
        let u = u.transpose()*q.transpose();

        Self {
            u
        }
    }
}
