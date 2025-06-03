# nalgebra-sparse-linalg

[![Crates.io](https://img.shields.io/crates/v/nalgebra-sparse-linalg.svg)](https://crates.io/crates/nalgebra-sparse-linalg)
[![Documentation](https://docs.rs/nalgebra-sparse-linalg/badge.svg)](https://docs.rs/nalgebra-sparse-linalg)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/nalgebra-sparse-linalg.svg)](#license)

**High-performance sparse linear algebra algorithms for Rust** - Iterative solvers, matrix decompositions, and numerical methods for large-scale sparse matrices using [nalgebra_sparse](https://crates.io/crates/nalgebra-sparse).

## Overview

`nalgebra-sparse-linalg` provides efficient numerical algorithms for sparse linear algebra computations in Rust. Built on top of `nalgebra_sparse`, this library offers:

- **Fast iterative solvers** for large sparse linear systems
- **Matrix decompositions** including truncated SVD for dimensionality reduction
- **Memory-efficient algorithms** optimized for sparse matrices
- **Generic implementations** supporting multiple precision types (`f32`, `f64`)
- **Production-ready** with comprehensive test coverage

Perfect for scientific computing, machine learning, data analysis, and numerical simulation applications.

## Features

### Iterative Linear System Solvers
- **Jacobi** - Simple and parallelizable iterative solver
- **Gauss-Seidel** - Faster convergence for well-conditioned systems
- **Successive Over-Relaxation (SOR)** - Accelerated convergence with relaxation parameter
- **Conjugate Gradient (CG)** - Optimal for symmetric positive-definite matrices
- **BiConjugate Gradient (BiCG)** - General solver for non-symmetric systems

### Matrix Decompositions
- **Truncated SVD** - Randomized algorithm for efficient singular value decomposition
- **Dimensionality reduction** for large-scale data analysis
- **Low-rank approximations** for matrix compression

### ⚡ Performance Features
- **CSR and CSC matrix support** - Industry-standard sparse formats
- **Memory-efficient algorithms** - Minimal memory footprint
- **Parallel computation** - Multi-threaded where applicable
- **Generic scalar types** - Support for `f32`, `f64`, and complex numbers
- **Zero-copy operations** - Efficient memory usage

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
nalgebra-sparse-linalg = "0.1"
nalgebra-sparse = "0.9"
```

### Solving Linear Systems

```rust
use nalgebra_sparse::{na::DVector, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::conjugate_gradient::solve;

// Create a sparse matrix and right-hand side
let a = CsrMatrix::identity(1000);
let b = DVector::from_vec(vec![1.0; 1000]);

// Solve Ax = b
let solution = solve(&a, &b, 1000, 1e-10).unwrap();
```

### Truncated SVD for Dimensionality Reduction

```rust
use nalgebra_sparse::CsrMatrix;
use nalgebra_sparse_linalg::svd::TruncatedSVD;

// Create or load your data matrix
let matrix = CsrMatrix::from(/* your dense matrix */);

// Compute top 50 singular vectors and values
let svd = TruncatedSVD::new(&matrix, 50);

// Access results
println!("Singular values: {:?}", svd.singular_values);
println!("Left singular vectors shape: {:?}", svd.u.shape());
```

## Examples

### Linear System Solvers

#### Jacobi Method
```rust
use nalgebra_sparse::{na::DVector, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::jacobi::solve;

let a = CsrMatrix::identity(3);
let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);
let result = solve(&a, &b, 100, 1e-10);
assert!(result.is_some());
```

#### Gauss-Seidel Method
```rust
use nalgebra_sparse::{na::DVector, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::gauss_seidel::solve;

let a = CsrMatrix::identity(3);
let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);
let result = solve(&a, &b, 100, 1e-10);
assert!(result.is_some());
```

#### Successive Over-Relaxation (SOR)
```rust
use nalgebra_sparse::{na::DVector, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::relaxation::solve;

let a = CsrMatrix::identity(3);
let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);
let omega = 1.2; // Relaxation parameter
let result = solve(&a, &b, 100, omega, 1e-10);
assert!(result.is_some());
```

#### Conjugate Gradient
```rust
use nalgebra_sparse::{na::DVector, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::conjugate_gradient::solve;

// Works with both CSR and CSC matrices
let a = CsrMatrix::identity(3);
let b = DVector::from_vec(vec![2.0, 4.0, 6.0]);
let result = solve(&a, &b, 100, 1e-10);
assert!(result.is_some());
```

#### BiConjugate Gradient
```rust
use nalgebra_sparse::{na::DVector, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::biconjugate_gradient::solve;

// Suitable for non-symmetric matrices
let a = CsrMatrix::identity(3);
let b = DVector::from_vec(vec![2.0, 4.0, 6.0]);
let result = solve(&a, &b, 100, 1e-10);
assert!(result.is_some());
```

### Matrix Decompositions

#### Truncated SVD for Large Matrices
```rust
use nalgebra_sparse::{CsrMatrix, na::DMatrix};
use nalgebra_sparse_linalg::svd::TruncatedSVD;

// Create a large data matrix (e.g., document-term matrix)
let dense_matrix = DMatrix::from_row_slice(1000, 500, &[/* your data */]);
let sparse_matrix = CsrMatrix::from(&dense_matrix);

// Compute top 100 components for dimensionality reduction
let svd = TruncatedSVD::new(&sparse_matrix, 100);

// The result contains:
// - svd.u: Left singular vectors (1000 x 100)
// - svd.singular_values: Singular values in descending order (100)
```

#### Principal Component Analysis (PCA) Example
```rust
use nalgebra_sparse::{CsrMatrix, na::DMatrix};
use nalgebra_sparse_linalg::svd::TruncatedSVD;

// Center your data matrix (subtract mean)
let centered_data = /* your centered data matrix */;
let sparse_data = CsrMatrix::from(&centered_data);

// Compute principal components
let n_components = 10;
let svd = TruncatedSVD::new(&sparse_data, n_components);

// svd.u contains the principal component loadings
// svd.singular_values contains the explained variance (after squaring)
```

## Use Cases

### 🔬 Scientific Computing
- Finite element analysis
- Computational fluid dynamics
- Structural analysis
- Physics simulations

### 🤖 Machine Learning & Data Science
- Large-scale linear regression
- Principal component analysis (PCA)
- Recommendation systems
- Natural language processing (LSA/LSI)
- Image processing and computer vision

### 📈 Numerical Analysis
- Solving partial differential equations
- Optimization problems
- Signal processing
- Graph algorithms

## Performance

This library is optimized for:
- **Large sparse matrices** (>10,000 dimensions)
- **Memory efficiency** - Minimal overhead beyond input data
- **Numerical stability** - Robust algorithms with proven convergence
- **Parallel computation** - Multi-threaded operations where beneficial

## Algorithm Selection Guide

| Problem Type | Recommended Solver | Notes |
|--------------|-------------------|-------|
| Symmetric positive-definite | Conjugate Gradient | Fastest convergence |
| General square matrices | BiConjugate Gradient | Good for non-symmetric |
| Diagonally dominant | Jacobi or Gauss-Seidel | Simple and robust |
| Large-scale PCA/SVD | Truncated SVD | Memory efficient |
| Ill-conditioned systems | Relaxation methods | Adjustable convergence |

## Documentation

See [API Documentation](https://docs.rs/nalgebra-sparse-linalg) - Complete API reference

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details on:
- Bug reports and feature requests
- Code contributions and pull requests
- Documentation improvements
- Performance optimizations

## Roadmap

- [ ] Preconditioned iterative methods
- [ ] Additional matrix decompositions (QR, LU)
- [ ] GPU acceleration support
- [ ] Distributed computing capabilities

## License

Licensed under either of:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.

## Related Crates

- [`nalgebra`](https://crates.io/crates/nalgebra) - Dense linear algebra
- [`nalgebra-sparse`](https://crates.io/crates/nalgebra-sparse) - Sparse matrix formats
