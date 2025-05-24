use log::warn;
use nalgebra_sparse::na::RealField;

use crate::iteratives::amg::restrict::build_r;

use super::{*, coarsen::coarsen, graph::strength_graph, interpolate::build_p, rap};

pub struct Level<N> {
    pub a: CsrMatrix<N>,
    pub p: CsrMatrix<N>,
    pub r: CsrMatrix<N>,
    pub diag: CsrMatrix<N>,
}

pub struct Hierarchy<N> {
    pub levels: Vec<Level<N>>,
}

pub fn setup<N>(a_fin: CsrMatrix<N>, theta: N, n_min: usize) -> Hierarchy<N>
where
    N: RealField + Copy,
{
    let mut levels = Vec::new();
    let mut a = a_fin;
    
    // Pre-allocate vectors to reduce allocation overhead
    levels.reserve(10); // Reasonable estimate for most problems
    
    loop {
        let (marks, coarse_of) = coarsen(&a, theta);
        let s = strength_graph(&a, theta);
        let p_candidate = build_p(&a, &marks, &coarse_of, &s);

        // Determine if this is the coarsest level
        let is_coarsest_level = p_candidate.ncols() >= a.nrows() || a.nrows() <= n_min;

        if is_coarsest_level {
            let r_coarsest = build_r(&a, &p_candidate);
            let diag_coarsest = a.diagonal_as_csr();
            
            levels.push(Level {
                a,
                p: p_candidate,
                r: r_coarsest,
                diag: diag_coarsest,
            });
            break; 
        }

        // This is an intermediate level - avoid unnecessary clones
        let r_intermediate = build_r(&a, &p_candidate);
        let diag_intermediate = a.diagonal_as_csr();

        // Prepare A_coarse for the next iteration first
        let a_coarse = rap(&r_intermediate, &a, &p_candidate);
        
        // Now we can move the current level data without cloning
        levels.push(Level {
            a,
            p: p_candidate,
            r: r_intermediate,
            diag: diag_intermediate,
        });

        // Move to next level
        a = a_coarse;

        if a.nrows() == 0 {
            warn!("Matrix A became empty after RAP. Stopping.");
            break;
        }
        if levels.len() > 20 { // Max levels
            warn!("Too many levels generated ({}). Stopping.", levels.len());
            break;
        }
    }
    
    Hierarchy { levels }
}


impl<N: RealField + Copy> Hierarchy<N> {
    pub fn vcycle(
        &self,
        l: usize,
        b: &DVector<N>,
        x: &mut DVector<N>,
        residual_buffer: &mut DVector<N>,
        tol: N,
        nu_pre: usize,
        nu_post: usize,
    ) {
        let lev = &self.levels[l];

        // 1. Pre-smooth: x_l gets updated by nu_pre smoothing steps for A_l x_l = b_l
        gauss_seidel::solve_with_initial_guess(&lev.a, b, x, nu_pre, tol);

        // 2. Compute fine residual: r_l = b_l - A_l x_l
        // Reuse buffer to avoid allocation
        if residual_buffer.len() != b.len() {
            *residual_buffer = DVector::zeros(b.len());
        }
        residual_buffer.copy_from(b);
        residual_buffer.axpy(-N::one(), &(&lev.a * &*x), N::one());

        // 3. Check if current level 'l' is the coarsest level
        if l + 1 == self.levels.len() {
            // Coarsest level: Solve directly with fewer iterations for efficiency
            jacobi::solve_with_initial_guess(&lev.a, b, x, 50, tol);
            return;
        }

        // 4. If NOT coarsest level:
        //   a. Restrict fine residual to create coarse residual
        let residual_coarse = &lev.r * &*residual_buffer;

        //   b. Initialize coarse grid error vector
        let mut error_coarse = DVector::<N>::zeros(residual_coarse.len());
        let mut residual_buffer_coarse = DVector::<N>::zeros(residual_coarse.len());

        //   c. Recursively call vcycle for level l+1
        self.vcycle(
            l + 1,
            &residual_coarse,
            &mut error_coarse,
            &mut residual_buffer_coarse,
            tol,
            nu_pre,
            nu_post,
        );

        //   d. Prolongate coarse grid error and correct fine grid solution
        let correction = &lev.p * &error_coarse;
        x.axpy(N::one(), &correction, N::one());

        // 5. Post-smooth: x_l gets updated by nu_post smoothing steps
        gauss_seidel::solve_with_initial_guess(&lev.a, b, x, nu_post, tol);
        *residual_buffer = &lev.a * &*x - b;
        //println!("Level {} residual norm: {}", l, residual_buffer.norm());
    }
}


