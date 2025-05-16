# nalgebra-sparse-linalg

Sparse linear algebra iterative solvers for [nalgebra_sparse](https://crates.io/crates/nalgebra-sparse).

## Overview

`nalgebra-sparse-linalg` provides iterative methods for solving large, sparse linear systems of equations, with a focus on compatibility with the `nalgebra_sparse` crate. The library currently includes Jacobi and Conjugate Gradient iterative solvers for sparse matrices.

## Features

- Jacobi iterative solver for sparse matrices in CSR format.
- Conjugate Gradient solver for symmetric positive-definite matrices in CSR and CSC formats.
- BiConjugate Gradient solver for general (possibly non-symmetric)
- Generic over scalar types (e.g., `f32`, `f64`).
- Simple API compatible with `nalgebra_sparse`'s matrix and vector types.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
nalgebra-sparse-linalg = "0.1"
nalgebra-sparse = "0.9"
```

Example (Jacobi):

```rust
use nalgebra_sparse::{na::DVector, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::jacobi::solve;

let a = CsrMatrix::identity(3);
let b = DVector::from_vec(vec![1.0; 3]);
let result = solve(&a, &b, 100, 1e-10);
assert!(result.is_some());
```

Exxample (Gauss-Seidel):

```rust
use nalgebra_sparse::{na::DVector, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::gauss_seidel::solve;

let a = CsrMatrix::identity(3);
let b = DVector::from_vec(vec![1.0; 3]);
let result = solve(&a, &b, 100, 1e-10);
assert!(result.is_some());
```

Example Relaxation:

```rust
use nalgebra_sparse::{na::DVector, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::relaxation::solve;

let a = CsrMatrix::identity(3);
let b = DVector::from_vec(vec![1.0; 3]);
let result = solve(&a, &b, 100, 0.8, 1e-10);
assert!(result.is_some());
```

Example (Conjugate Gradient, CSC or CSR):

```rust
use nalgebra_sparse::{na::DVector, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::conjugate_gradient::solve;

let a = CsrMatrix::identity(3);
let b = DVector::from_vec(vec![2.0; 3]);
let result = solve(&a, &b, 100, 1e-10);
assert!(result.is_some());
```

Example (BiConjugate Gradient, CSR or CSC):

```rust
use nalgebra_sparse::{na::DVector, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::biconjugate_gradient::solve;

let a = CsrMatrix::identity(3);
let b = DVector::from_vec(vec![2.0; 3]);
let result = solve(&a, &b, 100, 1e-10);
assert!(result.is_some());
```

## Documentation

See [docs.rs](https://docs.rs/nalgebra-sparse-linalg) for full API documentation.

## License

Licensed under either of

- Apache License, Version 2.0
- MIT license

at your option.
