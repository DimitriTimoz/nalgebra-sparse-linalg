use nalgebra_sparse::na::RealField;

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
        let p = build_p(&a, &marks, &coarse_of, &s);
        let r = build_p(&a, &marks, &coarse_of, &s);
        let diag   = a.diagonal_as_csr();

        if a.nrows() <= n_min {
            levels.push(Level { a: a.clone(),
                                p,
                                r,
                                diag});
            break;
        }

        println!("Level {}: {} rows", levels.len(), a.nrows());
        levels.push(Level { a: a.clone(), p: p.clone(), r: r.clone(), diag: diag.clone() });
        a = rap(&levels.last().unwrap().r, &a, &p);
    }
    println!("Setup done");
    Hierarchy { levels }
}


impl<N: RealField + Copy> Hierarchy<N> {
    pub fn vcycle(
        &self,
        l: usize,
        b: &DVector<N>,
        x: &mut DVector<N>,
        tmp: &mut DVector<N>,
        tol: N,
        nu_pre: usize,
        nu_post: usize,
    ) {
        let lev = &self.levels[l];
        // pre-smooth
        gauss_seidel::solve_with_initial_guess(&lev.a, b, x, nu_pre, tol);

        // residual r = b - A x
        tmp.copy_from(b);
        tmp.axpy(-N::one(), &(&lev.a * &*x), N::one());

        if l + 1 == self.levels.len() {
            // coarsest level: tiny dense solve
            jacobi::solve_with_initial_guess(&lev.a, tmp, x, 1000, tol); // or lapack
            return;
        }

        // restrict
        let b_c = &lev.r * &*tmp;

        // recurse
        let mut e_c = DVector::<N>::zeros(b_c.len());
        let mut tmp_c = DVector::<N>::zeros(b_c.len());
        self.vcycle(l + 1, &b_c, &mut e_c, &mut tmp_c, tol, nu_pre, nu_post);

        // prolongate & correct
        x.axpy(N::one(), &(&lev.p * &e_c), N::one());

        // post-smooth (backward)
        gauss_seidel::solve_with_initial_guess(&lev.a, b, x, nu_post, tol);
        println!("Last iteration residual: {}", tmp.amax());
    }
}
