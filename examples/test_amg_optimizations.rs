// Test to verify AMG optimizations work correctly
use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use nalgebra_sparse_linalg::iteratives::amg::{setup, Hierarchy};

fn create_test_matrix(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    
    // Create a simple 5-point stencil matrix (discretized 2D Laplacian)
    let size = (n as f64).sqrt() as usize;
    for i in 0..size {
        for j in 0..size {
            let idx = i * size + j;
            if idx >= n { break; }
            
            // Diagonal element
            coo.push(idx, idx, 4.0);
            
            // Off-diagonal elements
            if i > 0 {
                let neighbor = (i-1) * size + j;
                if neighbor < n {
                    coo.push(idx, neighbor, -1.0);
                }
            }
            if i < size-1 {
                let neighbor = (i+1) * size + j;
                if neighbor < n {
                    coo.push(idx, neighbor, -1.0);
                }
            }
            if j > 0 {
                let neighbor = i * size + (j-1);
                if neighbor < n {
                    coo.push(idx, neighbor, -1.0);
                }
            }
            if j < size-1 {
                let neighbor = i * size + (j+1);
                if neighbor < n {
                    coo.push(idx, neighbor, -1.0);
                }
            }
        }
    }
    
    CsrMatrix::from(&coo)
}

fn main() {
    println!("Testing AMG optimizations...");
    
    // Test small matrix
    let n = 100;
    let a = create_test_matrix(n);
    let b = DVector::from_element(n, 1.0);
    let mut x = DVector::zeros(n);
    
    println!("Matrix size: {}x{}", a.nrows(), a.ncols());
    println!("Non-zeros: {}", a.nnz());
    
    // Setup AMG hierarchy
    let start = std::time::Instant::now();
    let hierarchy = setup(a.clone(), 0.25, 10);
    let setup_time = start.elapsed();
    
    println!("AMG setup completed in {:?}", setup_time);
    println!("Number of levels: {}", hierarchy.levels.len());
    
    for (i, level) in hierarchy.levels.iter().enumerate() {
        println!("Level {}: {}x{} matrix with {} non-zeros", 
                 i, level.a.nrows(), level.a.ncols(), level.a.nnz());
    }
    
    // Test V-cycle
    let mut residual_buffer = DVector::zeros(n);
    let start = std::time::Instant::now();
    for _ in 0..5 {
        hierarchy.vcycle(0, &b, &mut x, &mut residual_buffer, 1e-12, 2, 2);
    }
    let vcycle_time = start.elapsed();
    
    // Check residual
    let residual = &b - &a * &x;
    let residual_norm = residual.norm();
    
    println!("5 V-cycles completed in {:?}", vcycle_time);
    println!("Final residual norm: {:.2e}", residual_norm);
    
    if residual_norm < 1e-6 {
        println!("✅ AMG solver converged successfully!");
    } else {
        println!("⚠️  AMG solver did not reach target tolerance");
    }
    
    // Test larger matrix
    println!("\nTesting larger matrix...");
    let n = 1000;
    let a_large = create_test_matrix(n);
    let b_large = DVector::from_element(n, 1.0);
    
    let start = std::time::Instant::now();
    let hierarchy_large = setup(a_large, 0.25, 10);
    let setup_time_large = start.elapsed();
    
    println!("Large matrix setup: {:?}", setup_time_large);
    println!("Large matrix levels: {}", hierarchy_large.levels.len());
}
