use log::warn;
use nalgebra_sparse::na::RealField;

use crate::iteratives::amg::restrict::build_r;

use super::{super::*, coarsen::coarsen, graph::strength_graph, interpolate::build_p, rap};

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
    loop {
        let (marks, coarse_of) = coarsen(&a, theta);
        let s = strength_graph(&a, theta);
        let p_candidate = build_p(&a, &marks, &coarse_of, &s); // P_candidate: N_fine x N_coarse_candidate

        // Determine if this is the coarsest level
        let is_coarsest_level = p_candidate.ncols() >= a.nrows() || a.nrows() <= n_min;

        if is_coarsest_level {
            let r_coarsest = build_r(&a, &p_candidate);
            let diag_coarsest = a.diagonal_as_csr();
            
            // Check condition before moving p_candidate
            let coarsening_failed = p_candidate.ncols() >= a.nrows() && a.nrows() > n_min;
            
            //if coarsening_failed {
            //    println!(
            //        "Level {}: Coarsening failed to reduce size (N_fine={}, N_coarse_candidate={}). This is the coarsest level.",
            //        levels.len(),
            //        a.nrows(),
            //        p_candidate.ncols()
            //    );
            //} else {
            //    println!(
            //        "Level {}: Reached n_min or effective coarsest level (N_fine={}, N_coarse_candidate={}). n_min={}",
            //        levels.len(),
            //        a.nrows(),
            //        p_candidate.ncols(),
            //        n_min
            //    );
            //}
            
            levels.push(Level {
                a: a.clone(),
                p: p_candidate,
                r: r_coarsest.transpose(),
                diag: diag_coarsest,
            });
            break; 
        }

        // This is an intermediate level
        let r_intermediate = build_r(&a, &p_candidate);
        let diag_intermediate = a.diagonal_as_csr();

        //println!("Level {}: {} rows (N_fine)", levels.len(), a.nrows());
        levels.push(Level {
            a: a.clone(),
            p: p_candidate.clone(), // Clone p_candidate for this level
            r: r_intermediate.clone(),
            diag: diag_intermediate.clone(),
        });
        //println!("  Fine A: {}x{}", a.nrows(), a.ncols());
        //println!("  P: {}x{} (N_fine x N_coarse)", p_candidate.nrows(), p_candidate.ncols());
        //println!("  R: {}x{} (N_coarse x N_fine)", r_intermediate.nrows(), r_intermediate.ncols());

        // Prepare A_coarse for the next iteration: A_coarse = R_intermediate * A_fine * P_candidate
        a = rap(&r_intermediate, &a, &p_candidate); // p_candidate is used here
        //println!("  Coarse A (for next level): {}x{}", a.nrows(), a.ncols());

        if a.nrows() == 0 {
            println!("Matrix A became empty after RAP. Stopping.");
            break;
        }
        if levels.len() > 20 { // Max levels
            warn!("Too many levels generated ({}). Stopping.", levels.len());
            break;
        }
        //println!("Coarsen for next level...");
    }
    //println!("Setup done");
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
        //    Store r_l in residual_buffer.
        //    Note: x now contains the pre-smoothed solution.
        residual_buffer.copy_from(b);
        residual_buffer.axpy(-N::one(), &(&lev.a * &*x), N::one());

        // 3. Check if current level 'l' is the coarsest level
        if l + 1 == self.levels.len() {
            // Coarsest level: Solve A_l u_l = b_l directly (or as accurately as possible).
            // Here, u_l is the vector `x` for this level (e.g., e_L, the error correction).
            // And b_l is the vector `b` for this level (e.g., r_L, the restricted residual from finer level).
            // Pre-smoothing (step 1) has already updated `x` using `b` as RHS, so `x` is an initial/intermediate approximation.
            // We continue solving A_l u_l = b_l using the current `x` as the initial guess.
            jacobi::solve_with_initial_guess(&lev.a, b, x, 1000, tol);
            return;
        }

        // 4. If NOT coarsest level:
        //   a. Restrict fine residual to create coarse residual: r_{l+1} = R_l * r_l
        //      lev.r is R_l. residual_buffer contains r_l.
        let residual_coarse = &lev.r * &*residual_buffer; // This is b_{l+1} for the next level

        //   b. Initialize coarse grid error vector e_{l+1} = 0
        let mut error_coarse = DVector::<N>::zeros(residual_coarse.len());
        //      Create a residual buffer for the next coarser level.
        let mut residual_buffer_coarse = DVector::<N>::zeros(residual_coarse.len());

        //   c. Recursively call vcycle for level l+1 to solve A_{l+1} * e_{l+1} = r_{l+1}
        //      After this call, error_coarse will contain the solved e_{l+1}.
        self.vcycle(
            l + 1,
            &residual_coarse,
            &mut error_coarse,
            &mut residual_buffer_coarse,
            tol,
            nu_pre,
            nu_post,
        );

        //   d. Prolongate coarse grid error and correct fine grid solution: x_l = x_l + P_l * e_{l+1}
        //      x contains the pre-smoothed solution. lev.p is P_l. error_coarse is e_{l+1}.
        x.axpy(N::one(), &(&lev.p * &error_coarse), N::one());

        // 5. Post-smooth: x_l gets updated by nu_post smoothing steps for A_l x_l = b_l
        gauss_seidel::solve_with_initial_guess(&lev.a, b, x, nu_post, tol);

        // Optional: For debugging, print residual after post-smoothing
        residual_buffer.copy_from(b);
        residual_buffer.axpy(-N::one(), &(&lev.a * &*x), N::one());
        //println!("Level {}: Residual after post-smoothing: {}", l, residual_buffer.amax());
    }
}
