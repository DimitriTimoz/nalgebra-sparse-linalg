# nalgebra-sparse-linalg

Sparse linear algebra iterative solvers for [nalgebra_sparse](https://crates.io/crates/nalgebra-sparse).

## Overview

`nalgebra-sparse-linalg` provides iterative methods for solving large, sparse linear systems of equations, with a focus on compatibility with the `nalgebra_sparse` crate. The initial release includes a Jacobi iterative solver for diagonally dominant matrices.

## Features

- Jacobi iterative solver for sparse matrices in CSR format.
- Generic over scalar types (e.g., `f32`, `f64`).
- Simple API compatible with `nalgebra_sparse`'s matrix and vector types.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
nalgebra-sparse-linalg = "0.1"
nalgebra-sparse = "0.9"
nalgebra = "0.32"
```

Example:

```rust
use nalgebra_sparse::{na::DVector, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::jacobi::solve;

let a = CsrMatrix::identity(3);
let b = DVector::from_vec(vec![1.0; 3]);
let result = solve(&a, &b, 100);
assert!(result.is_some());
```

## Documentation

See [docs.rs](https://docs.rs/nalgebra-sparse-linalg) for full API documentation.

## License

Licensed under either of

- Apache License, Version 2.0
- MIT license

at your option.
