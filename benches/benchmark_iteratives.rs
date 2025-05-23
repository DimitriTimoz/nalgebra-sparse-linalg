use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use nalgebra_sparse::na::DVector;
use nalgebra_sparse::CsrMatrix;
use nalgebra_sparse_linalg::iteratives::{amg, biconjugate_gradient, conjugate_gradient, gauss_seidel, jacobi, relaxation, SpMatVecMul};
use rand::{rng, Rng};
use rand::seq::SliceRandom;
use std::env;

/// Random sparse triplet (CSR parts) – guarantees **unique, sorted** col indices
fn random_sparse_triplet(size: usize, nnz_per_row: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut rows = Vec::with_capacity(size + 1);
    let mut cols = Vec::with_capacity(size * nnz_per_row);
    let mut vals = Vec::with_capacity(size * nnz_per_row);
    let mut rng = rng();

    rows.push(0);
    for r in 0..size {
        let mut idxs: Vec<usize> = (0..size).collect();
        idxs.shuffle(&mut rng);
        let mut chosen: Vec<usize> = idxs[..nnz_per_row].to_vec();
        // ensure diagonal index `r` is present **exactly once**
        if !chosen.contains(&r) {
            chosen.pop();
            chosen.push(r);
        }
        chosen.sort_unstable();
        for &j in &chosen {
            cols.push(j);
            // diagonal gets positive magnitude to aid convergence
            let v = if j == r {
                rng.random_range(1.1..3.0)
            } else {
                rng.random_range(-1.0..1.0)
            };
            vals.push(v);
        }
        rows.push(cols.len());
    }
    (rows, cols, vals)
}

/// Strictly diagonally dominant nonsymmetric matrix (good for Jacobi & BiCG)
fn generate_diag_dominant(size: usize, nnz_per_row: usize) -> CsrMatrix<f64> {
    let (indptr, indices, mut data) = random_sparse_triplet(size, nnz_per_row);
    for i in 0..size {
        let start = indptr[i];
        let end = indptr[i + 1];
        let row_sum: f64 = indices[start..end]
            .iter()
            .zip(&data[start..end])
            .filter(|(j_ref, _)| **j_ref != i)
            .map(|(_, &v)| v.abs())
            .sum();
        if let Ok(pos) = indices[start..end].binary_search(&i) {
            data[start + pos] = row_sum + 1.0;
        }
    }
    CsrMatrix::try_from_csr_data(size, size, indptr, indices, data).unwrap()
}

/// Symmetric positive‑definite matrix (CG)
fn generate_spd(size: usize, nnz_per_row: usize) -> CsrMatrix<f64> {
    let base = {
        let (rows, cols, vals) = random_sparse_triplet(size, nnz_per_row/2);
        CsrMatrix::try_from_csr_data(size, size, rows, cols, vals).unwrap()
    };
    let at = base.transpose();
    let ata = &at * &base;
    let alpha: f64 = 5.0;
    let identity = CsrMatrix::identity(size) * alpha;
    &ata + &identity
}

/// Generic nonsymmetric but well‑conditioned matrix for BiCG
fn generate_nonsymmetric(size: usize, nnz_per_row: usize) -> CsrMatrix<f64> {
    // Re‑use strictly diagonal dominant generator to ensure convergence while
    // keeping lack of symmetry.
    generate_diag_dominant(size, nnz_per_row)
}

// Benchmarks ---------------------------------------------------------------
fn bench_methods(c: &mut Criterion) {
    // Allow filtering which method to run via BENCH_METHOD env variable
    let bench_method = env::var("BENCH_METHOD").ok();
    let mut group = c.benchmark_group("IterativeSolvers");
    let sizes = [100usize, 500, 1_000,2_000, 10_000];//, 10_000, 50_000, 100_000, 200_000];
    for &n in &sizes {
        let nnz = n.min(50) / 5;
        // Only run Jacobi if BENCH_METHOD is unset or matches
        if bench_method.as_deref().is_none_or(|m| m.eq_ignore_ascii_case("jacobi")) {
            group.bench_with_input(BenchmarkId::new("Jacobi", n), &n, |be, &_n| {
                be.iter_batched(
                    || {
                        let a = generate_spd(n, nnz);
                        let mut rng = rng();
                        let x = DVector::<f64>::from_fn(n, |_, _| rng.random_range(-1.0..1.0));
                        let b = a.mul_vec(&x);
                        (a, b)
                    },
                    |(a, b)| {
                        let r = jacobi::solve(&a, &b, 100_000, 1e-10);
                        assert!(r.is_some());
                    },
                    BatchSize::LargeInput,
                );
            });
        }
        // Only run GaussSeidel if BENCH_METHOD is unset or matches
        if bench_method.as_deref().is_none_or(|m| m.eq_ignore_ascii_case("gaussseidel")) {
            group.bench_with_input(BenchmarkId::new("GaussSeidel", n), &n, |be, &_n| {
                be.iter_batched(
                    || {
                        let a = generate_spd(n, nnz);
                        let mut rng = rng();
                        let x = DVector::<f64>::from_fn(n, |_, _| rng.random_range(-1.0..1.0));
                        let b = a.mul_vec(&x);
                        (a, b)
                    },
                    |(a, b)| {
                        let r = gauss_seidel::solve(&a, &b, 100_000, 1e-10);
                        assert!(r.is_some());
                    },
                    BatchSize::LargeInput,
                );
            });
        }
         // Only run Relaxation if BENCH_METHOD is unset or matches
        if bench_method.as_deref().is_none_or(|m| m.eq_ignore_ascii_case("relaxation")) {
            group.bench_with_input(BenchmarkId::new("Relaxation", n), &n, |be, &_n| {
                be.iter_batched(
                    || {
                        let a = generate_spd(n, nnz);
                        let mut rng = rng();
                        let x = DVector::<f64>::from_fn(n, |_, _| rng.random_range(-1.0..1.0));
                        let b = a.mul_vec(&x);
                        (a, b)
                    },
                    |(a, b)| {
                        let r = relaxation::solve(&a, &b, 100_000, 0.66, 1e-10);
                        assert!(r.is_some());
                    },
                    BatchSize::LargeInput,
                );
            });
        }
        // Only run ConjugateGradient if BENCH_METHOD is unset or matches
        if bench_method.as_deref().is_none_or(|m| m.eq_ignore_ascii_case("conjugategradient")) {
            group.bench_with_input(BenchmarkId::new("ConjugateGradient", n), &n, |be, &_n| {
                be.iter_batched(
                    || {
                        let a = generate_spd(n, nnz);
                        let mut rng = rng();
                        let x = DVector::<f64>::from_fn(n, |_, _| rng.random_range(-1.0..1.0));
                        let b = a.mul_vec(&x);
                        (a, b)
                    },
                    |(a, b)| {
                        let r = conjugate_gradient::solve(&a, &b, 100_000, 1e-10);
                        assert!(r.is_some());
                    },
                    BatchSize::LargeInput,
                );
            });
        }
        // Only run BiConjugateGradient if BENCH_METHOD is unset or matches
        if bench_method.as_deref().is_none_or(|m| m.eq_ignore_ascii_case("biconjugategradient")) {
            group.bench_with_input(BenchmarkId::new("BiConjugateGradient", n), &n, |be, &_n| {
                be.iter_batched(
                    || {
                        let a = generate_spd(n, nnz);
                        let mut rng = rng();
                        let x = DVector::<f64>::from_fn(n, |_, _| rng.random_range(-1.0..1.0));
                        let b = a.mul_vec(&x);
                        (a, b)
                    },
                    |(a, b)| {
                        let r = biconjugate_gradient::solve(&a, &b, 100_000, 1e-10);
                        assert!(r.is_some());
                    },
                    BatchSize::LargeInput,
                );
            });
        }
        // Only run AMG if BENCH_METHOD is unset or matches
        if bench_method.as_deref().is_none_or(|m| m.eq_ignore_ascii_case("amg")) {
            group.bench_with_input(BenchmarkId::new("AMG", n), &n, |be, &_n| {
                be.iter_batched(
                    || {
                        let a = generate_spd(n, nnz);
                        let mut rng = rng();
                        let x = DVector::<f64>::from_fn(n, |_, _| rng.random_range(-1.0..1.0));
                        let b = a.mul_vec(&x);
                        (a, b)
                    },
                    |(a, b)| {
                        let r = amg::solve(a, &b, 100_000, 1e-10, 0.8);
                        assert!(r.is_some());
                    },
                    BatchSize::LargeInput,
                );
            });
        }

    }
    group.finish();
}

criterion_group!(benches, bench_methods);
criterion_main!(benches);
