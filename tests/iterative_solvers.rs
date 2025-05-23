//! Tests for all iterative solvers in nalgebra-sparse-linalg

use nalgebra_sparse::{na::DVector, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::amg::solve_with_initial_guess;
use nalgebra_sparse_linalg::iteratives::{
    biconjugate_gradient, conjugate_gradient, gauss_seidel, jacobi, relaxation
};
use nalgebra_sparse_linalg::iteratives::amg::level::setup;

fn get_identity_and_rhs(n: usize, rhs_val: f64) -> (CsrMatrix<f64>, DVector<f64>) {
    (CsrMatrix::identity(n), DVector::from_vec(vec![rhs_val; n]))
}

fn get_simple_csr() -> (CsrMatrix<f64>, DVector<f64>, DVector<f64>) {
    // Matrix: [4 1 0; 1 3 1; 0 1 2]
    let n = 3;
    let rows = vec![0, 2, 5, 7];
    let cols = vec![0, 1, 0, 1, 2, 1, 2];
    let vals = vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];
    let a = CsrMatrix::try_from_csr_data(n, n, rows, cols, vals).unwrap();
    let x_true = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let b = &a * &x_true;
    (a, b, x_true)
}

fn get_diagonal_dominant() -> (CsrMatrix<f64>, DVector<f64>, DVector<f64>) {
    // Matrix: [10 2 0; 3 15 4; 0 1 8]
    let n = 3;
    let rows = vec![0, 2, 5, 7];
    let cols = vec![0, 1, 0, 1, 2, 1, 2];
    let vals = vec![10.0, 2.0, 3.0, 15.0, 4.0, 1.0, 8.0];
    let a = CsrMatrix::try_from_csr_data(n, n, rows, cols, vals).unwrap();
    let x_true = DVector::from_vec(vec![2.0, -1.0, 3.0]);
    let b = &a * &x_true;
    (a, b, x_true)
}

fn get_sparse_large() -> (CsrMatrix<f64>, DVector<f64>, DVector<f64>) {
    // 5x5 sparse matrix with known solution
    let n = 5;
    let rows = vec![0, 2, 4, 7, 9, 11];
    let cols = vec![0, 1, 0, 2, 1, 2, 3, 2, 4, 3, 4];
    let vals = vec![5.0, 1.0, 1.0, 4.0, 2.0, 7.0, 1.0, 3.0, 8.0, 2.0, 6.0];
    let a = CsrMatrix::try_from_csr_data(n, n, rows, cols, vals).unwrap();
    let x_true = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = &a * &x_true;
    (a, b, x_true)
}

fn get_sdp_matrix() -> (CsrMatrix<f64>, DVector<f64>, DVector<f64>) {
    // Symmetric positive definite: [6 2 1; 2 5 2; 1 2 4]
    let n = 3;
    let rows = vec![0, 3, 6, 9];
    let cols = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let vals = vec![6.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
    
    let a = CsrMatrix::try_from_csr_data(n, n, rows, cols, vals).unwrap();
    let x_true = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let b = &a * &x_true;
    (a, b, x_true)
}

fn get_very_large_sparse_matrix() -> (CsrMatrix<f64>, DVector<f64>, DVector<f64>) {
    let n = 100; // Size of the matrix
    let mut row_offsets = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_offsets.push(0);
    let mut nnz = 0;

    for i in 0..n {
        // Add elements for row i
        if i > 0 {
            // Element (i, i-1)
            col_indices.push(i - 1);
            values.push(-1.0);
            nnz += 1;
        }

        // Element (i, i) - Diagonal
        col_indices.push(i);
        values.push(3.0); // Ensure diagonal dominance for stability
        nnz += 1;

        if i < n - 1 {
            // Element (i, i+1)
            col_indices.push(i + 1);
            values.push(-1.0);
            nnz += 1;
        }
        row_offsets.push(nnz);
    }

    let a = CsrMatrix::try_from_csr_data(n, n, row_offsets, col_indices, values).unwrap();
    let x_true = DVector::from_element(n, 1.0); // True solution vector of ones
    let b = &a * &x_true; // Calculate RHS vector b = A * x_true
    (a, b, x_true)
}

#[test]
fn test_jacobi() {
    let (a, b) = get_identity_and_rhs(3, 1.0);
    let result = jacobi::solve(&a, &b, 100, 1e-10);
    assert!(result.is_some());
    assert!((result.unwrap() - b).amax() < 1e-8);
}

#[test]
fn test_gauss_seidel() {
    let (a, b) = get_identity_and_rhs(3, 1.0);
    let result = gauss_seidel::solve(&a, &b, 100, 1e-10);
    assert!(result.is_some());
    assert!((result.unwrap() - b).amax() < 1e-8);
}

#[test]
fn test_relaxation() {
    let (a, b) = get_identity_and_rhs(3, 1.0);
    let result = relaxation::solve(&a, &b, 100, 0.8, 1e-10);
    assert!(result.is_some());
    assert!((result.unwrap() - b).amax() < 1e-8);
}

#[test]
fn test_conjugate_gradient() {
    let (a, b) = get_identity_and_rhs(3, 2.0);
    let result = conjugate_gradient::solve(&a, &b, 100, 1e-10);
    assert!(result.is_some());
    assert!((result.unwrap() - b).amax() < 1e-8);
}

#[test]
fn test_biconjugate_gradient() {
    let (a, b) = get_identity_and_rhs(3, 2.0);
    let result = biconjugate_gradient::solve(&a, &b, 100, 1e-10);
    assert!(result.is_some());
    assert!((result.unwrap() - b).amax() < 1e-8);
}

#[test]
fn test_jacobi_nontrivial() {
    let (a, b, x_true) = get_simple_csr();
    let result = jacobi::solve(&a, &b, 500, 1e-8);
    assert!(result.is_some());
    assert!((result.unwrap() - x_true).amax() < 1e-6);
}

#[test]
fn test_gauss_seidel_nontrivial() {
    let (a, b, x_true) = get_simple_csr();
    let result = gauss_seidel::solve(&a, &b, 500, 1e-8);
    assert!(result.is_some());
    assert!((result.unwrap() - x_true).amax() < 1e-6);
}

#[test]
fn test_relaxation_nontrivial() {
    let (a, b, x_true) = get_simple_csr();
    let result = relaxation::solve(&a, &b, 500, 0.8, 1e-8);
    assert!(result.is_some());
    assert!((result.unwrap() - x_true).amax() < 1e-6);
}

#[test]
fn test_conjugate_gradient_nontrivial() {
    let (a, b, x_true) = get_simple_csr();
    let result = conjugate_gradient::solve(&a, &b, 500, 1e-8);
    assert!(result.is_some());
    assert!((result.unwrap() - x_true).amax() < 1e-6);
}

#[test]
fn test_biconjugate_gradient_nontrivial() {
    let (a, b, x_true) = get_simple_csr();
    let result = biconjugate_gradient::solve(&a, &b, 500, 1e-8);
    assert!(result.is_some());
    assert!((result.unwrap() - x_true).amax() < 1e-6);
}

#[test]
fn test_jacobi_diagonal_dominant() {
    let (a, b, x_true) = get_diagonal_dominant();
    let result = jacobi::solve(&a, &b, 1000, 1e-8);
    assert!(result.is_some());
    assert!((result.unwrap() - x_true).amax() < 1e-6);
}

#[test]
fn test_gauss_seidel_diagonal_dominant() {
    let (a, b, x_true) = get_diagonal_dominant();
    let result = gauss_seidel::solve(&a, &b, 1000, 1e-8);
    assert!(result.is_some());
    assert!((result.unwrap() - x_true).amax() < 1e-6);
}

#[test]
fn test_relaxation_diagonal_dominant() {
    let (a, b, x_true) = get_diagonal_dominant();
    let result = relaxation::solve(&a, &b, 1000, 0.8, 1e-8);
    assert!(result.is_some());
    assert!((result.unwrap() - x_true).amax() < 1e-6);
}

// #[test]
// fn test_conjugate_gradient_diagonal_dominant() {
//     let (a, b, x_true) = get_diagonal_dominant();
//     let result = conjugate_gradient::solve(&a, &b, 1000, 1e-8);
//     assert!(result.is_some());
//     assert!((result.unwrap() - x_true).amax() < 1e-6);
// }

// Les tests suivants sont désactivés car la matrice n'est pas adaptée à l'algorithme :
// #[test]
// fn test_jacobi_sparse_large() {
//     let (a, b, x_true) = get_sparse_large();
//     let result = jacobi::solve(&a, &b, 2000, 1e-8);
//     assert!(result.is_some());
//     assert!((result.unwrap() - x_true).amax() < 1e-6);
// }

// #[test]
// fn test_gauss_seidel_sparse_large() {
//     let (a, b, x_true) = get_sparse_large();
//     let result = gauss_seidel::solve(&a, &b, 2000, 1e-8);
//     assert!(result.is_some());
//     assert!((result.unwrap() - x_true).amax() < 1e-6);
// }

// #[test]
// fn test_relaxation_sparse_large() {
//     let (a, b, x_true) = get_sparse_large();
//     let result = relaxation::solve(&a, &b, 2000, 1.0, 1e-8);
//     assert!(result.is_some());
//     assert!((result.unwrap() - x_true).amax() < 1e-6);
// }

// #[test]
// fn test_conjugate_gradient_sparse_large() {
//     let (a, b, x_true) = get_sparse_large();
//     let result = conjugate_gradient::solve(&a, &b, 2000, 1e-8);
//     assert!(result.is_some());
//     assert!((result.unwrap() - x_true).amax() < 1e-6);
// }

#[test]
fn test_biconjugate_gradient_sparse_large() {
    let (a, b, x_true) = get_sparse_large();
    let result = biconjugate_gradient::solve(&a, &b, 2000, 1e-8);
    assert!(result.is_some());
    assert!((result.unwrap() - x_true).amax() < 1e-6);
}

#[test]
fn test_conjugate_gradient_sdp() {
    let (a, b, x_true) = get_sdp_matrix();
    let result = conjugate_gradient::solve(&a, &b, 1000, 1e-8);
    assert!(result.is_some());
    assert!((result.unwrap() - x_true).amax() < 1e-6);
}

#[test]
fn test_amg() {
    let (a_orig, b, x_true) = get_very_large_sparse_matrix();
    let n = a_orig.nrows();
    let mut x = DVector::zeros(n);
    let tol = 1e-8;

    solve_with_initial_guess(a_orig, &b, &mut x,10000,  tol, 0.8);
    // Final assertion also needs to clone x if it was moved in the loop, 
    // or ensure x is still valid here.
    assert!((x - &x_true).amax() < tol);
}