// https://edoc.unibas.ch/server/api/core/bitstreams/d56e8bdd-9b91-49ec-a8f5-04eff6db51ca/content
pub mod graph;
pub mod coarsen;
pub mod interpolate;
pub mod restrict;
pub mod level;

pub(crate) use nalgebra_sparse::na::{DVector, RealField};
pub(crate) use super::*;

pub(crate) fn rap<N>(
    r: &CsrMatrix<N>,
    a: &CsrMatrix<N>,
    p: &CsrMatrix<N>,
) -> CsrMatrix<N>
where
    N: RealField + Copy,
{
    // Optimized triple product: R * A * P
    // First compute A * P (often much smaller than computing R * A first)
    let ap = a * p;
    // Then compute R * (A * P)
    r * &ap
}

pub fn solve_with_initial_guess<T>(a: CsrMatrix<T>, b: &DVector<T>, x: &mut DVector<T>, max_iter: usize, tol: T, theta: T) -> bool
where 
    T: RealField + Copy,
{
    let mut solver = Amg::new(tol, theta, max_iter);
    solver.init(&a, b, Some(x));
    let converged = solver.solve_iterations(&a, b, max_iter);
    *x = solver.x.clone();
    converged
}

pub fn solve<T>(a: CsrMatrix<T>, b: &DVector<T>, max_iter: usize, tol: T, theta: T) -> Option<DVector<T>> 
where 
    T: RealField + Copy,
{
    let mut x = DVector::<T>::zeros(a.nrows());
    if solve_with_initial_guess(a, b, &mut x, max_iter, tol, theta) {
        Some(x)
    } else {
        None
    }
}

/// AMG (Algebraic Multigrid) solver struct that implements the IterativeSolver trait
/// for customizable solving with configurable parameters.
pub struct Amg<T> {
    pub x: DVector<T>,
    pub tol: T,
    pub theta: T,
    pub max_iter: usize,
    pub iter: usize,
    pub converged: bool,
    pub nu_pre: usize,
    pub nu_post: usize,
    pub n_min: usize,
    hierarchy: Option<level::Hierarchy<T>>,
    residual_buffer: DVector<T>,
}

impl<T> Amg<T>
where
    T: RealField + Copy,
{
    /// Creates a new AMG solver with specified parameters
    pub fn new(tol: T, theta: T, max_iter: usize) -> Self {
        Self {
            x: DVector::zeros(0),
            tol,
            theta,
            max_iter,
            iter: 0,
            converged: false,
            nu_pre: 1,
            nu_post: 1,
            n_min: 100,
            hierarchy: None,
            residual_buffer: DVector::zeros(0),
        }
    }

    /// Creates a new AMG solver with custom smoothing parameters
    pub fn with_smoothing(tol: T, theta: T, max_iter: usize, nu_pre: usize, nu_post: usize) -> Self {
        Self {
            x: DVector::zeros(0),
            tol,
            theta,
            max_iter,
            iter: 0,
            converged: false,
            nu_pre,
            nu_post,
            n_min: 100,
            hierarchy: None,
            residual_buffer: DVector::zeros(0),
        }
    }

    /// Sets the minimum coarse level size
    pub fn with_coarse_size(mut self, n_min: usize) -> Self {
        self.n_min = n_min;
        self
    }

    /// Sets the pre and post smoothing iterations
    pub fn set_smoothing(&mut self, nu_pre: usize, nu_post: usize) {
        self.nu_pre = nu_pre;
        self.nu_post = nu_post;
    }

    /// Gets the current tolerance
    pub fn tolerance(&self) -> T {
        self.tol
    }

    /// Sets a new tolerance
    pub fn set_tolerance(&mut self, tol: T) {
        self.tol = tol;
    }

    /// Gets the theta parameter for strength-of-connection
    pub fn theta(&self) -> T {
        self.theta
    }

    /// Sets a new theta parameter (requires rebuilding hierarchy)
    pub fn set_theta(&mut self, theta: T) {
        self.theta = theta;
        // Clear hierarchy to force rebuild with new theta
        self.hierarchy = None;
    }

    /// Gets the minimum coarse level size
    pub fn coarse_size(&self) -> usize {
        self.n_min
    }

    /// Gets the pre-smoothing iterations
    pub fn pre_smoothing(&self) -> usize {
        self.nu_pre
    }

    /// Gets the post-smoothing iterations  
    pub fn post_smoothing(&self) -> usize {
        self.nu_post
    }

    /// Checks if the solver has converged
    pub fn has_converged(&self) -> bool {
        self.converged
    }

    /// Gets the number of levels in the AMG hierarchy
    pub fn num_levels(&self) -> usize {
        self.hierarchy.as_ref().map_or(0, |h| h.levels.len())
    }

    /// Get residual norm (absolute maximum)
    pub fn residual_norm(&self, a: &CsrMatrix<T>, b: &DVector<T>) -> T {
        let residual = a * &self.x - b;
        residual.amax()
    }
}

impl<T> IterativeSolver<CsrMatrix<T>, DVector<T>, T> for Amg<T>
where
    T: RealField + Copy,
{
    fn init(&mut self, a: &CsrMatrix<T>, _b: &DVector<T>, x0: Option<&DVector<T>>) {
        let n = a.nrows();
        self.x = match x0 {
            Some(x0) => x0.clone(),
            None => DVector::<T>::zeros(n),
        };
        self.residual_buffer = DVector::zeros(n);
        self.iter = 0;
        self.converged = false;
        
        // Build the AMG hierarchy
        self.hierarchy = Some(level::setup(a.clone(), self.theta, self.n_min));
    }

    fn step(&mut self, a: &CsrMatrix<T>, b: &DVector<T>) -> bool {
        if let Some(ref hierarchy) = self.hierarchy {
            // Check if we're already converged
            let current_residual = a * &self.x - b;
            let residual_norm = current_residual.amax();
            
            if residual_norm <= self.tol {
                self.converged = true;
                return true;
            }
            
            // Use adaptive tolerance for intermediate iterations
            let adaptive_tol = self.tol.max(residual_norm * T::from_f64(1e-3).unwrap());
            
            // Perform one V-cycle
            hierarchy.vcycle(
                0, 
                b, 
                &mut self.x, 
                &mut self.residual_buffer, 
                adaptive_tol, 
                self.nu_pre, 
                self.nu_post
            );
            
            self.iter += 1;
            
            // Check convergence after the V-cycle
            let new_residual = a * &self.x - b;
            let new_residual_norm = new_residual.amax();
            println!("Iteration {}: Residual norm = {}", self.iter, new_residual_norm);
            if new_residual_norm <= self.tol {
                self.converged = true;
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.x.fill(T::zero());
        self.residual_buffer.fill(T::zero());
        self.iter = 0;
        self.converged = false;
        // Keep hierarchy for reuse
    }

    fn hard_reset(&mut self) {
        self.x = DVector::<T>::zeros(0);
        self.residual_buffer = DVector::<T>::zeros(0);
        self.iter = 0;
        self.converged = false;
        self.hierarchy = None;
    }

    fn soft_reset(&mut self) {
        self.x.fill(T::zero());
        self.residual_buffer.fill(T::zero());
        self.iter = 0;
        self.converged = false;
        // Keep hierarchy for reuse
    }

    fn solution(&self) -> &DVector<T> {
        &self.x
    }

    fn iterations(&self) -> usize {
        self.iter
    }
}

/// Convenience function to create an AMG solver with default parameters
/// and solve the linear system Ax = b.
pub fn solve_amg<T>(a: &CsrMatrix<T>, b: &DVector<T>, max_iter: usize, tol: T, theta: T) -> Option<DVector<T>> 
where 
    T: RealField + Copy,
{
    let mut solver = Amg::new(tol, theta, max_iter);
    solver.init(a, b, None);
    if solver.solve_iterations(a, b, max_iter) {
        Some(solver.x.clone())
    } else {
        None
    }
}

/// Convenience function to create an AMG solver with default parameters
/// and solve with an initial guess.
pub fn solve_amg_with_initial_guess<T>(
    a: &CsrMatrix<T>, 
    b: &DVector<T>, 
    x: &mut DVector<T>, 
    max_iter: usize, 
    tol: T, 
    theta: T
) -> bool
where 
    T: RealField + Copy,
{
    let mut solver = Amg::new(tol, theta, max_iter);
    solver.init(a, b, Some(x));
    let converged = solver.solve_iterations(a, b, max_iter);
    *x = solver.x.clone();
    converged
}

#[cfg(test)]
mod tests {
    use crate::iteratives::amg::{coarsen::Mark, graph::strength_graph, interpolate::build_p};

    use super::*;
    use approx::assert_relative_eq;
    use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};

    /// 1-D Poisson: 2 on the diagonal, −1 on neighbours.
    fn poisson_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0);
            if i + 1 < n {
                coo.push(i, i + 1, -1.0);
                coo.push(i + 1, i, -1.0);
            }
        }
        CsrMatrix::from(&coo)
    }

    #[test]
    fn graph_neighbours() {
        let a = poisson_1d(5);
        let g = strength_graph(&a, 0.0);           // θ = 0 → full graph
        assert_eq!(g[0], [1]);
        let mut v = g[2].clone(); v.sort();
        assert_eq!(v, [1, 3]);
    }

    #[test]
    fn coarsen_basic() {
        let a = poisson_1d(6);
        let (marks, coarse_of) = coarsen::coarsen(&a, 0.0);
        let n_c = marks.iter().filter(|m| matches!(m, Mark::C)).count();
        let n_f = marks.iter().filter(|m| matches!(m, Mark::F)).count();
        assert!(n_c >= 1 && n_f >= 1);
        for (i, m) in marks.iter().enumerate() {
            match m {
                Mark::C => assert!(coarse_of[i] < n_c),
                Mark::F => assert_eq!(coarse_of[i], usize::MAX),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn p_shape_and_identity() {
        let a = poisson_1d(8);
        let (marks, coarse_of) = coarsen::coarsen(&a, 0.0);
        let s = strength_graph(&a, 0.0);
        let p = build_p(&a, &marks, &coarse_of, &s);

        assert_eq!(p.nrows(), a.nrows());
        let n_c = marks.iter().filter(|m| matches!(m, Mark::C)).count();
        assert_eq!(p.ncols(), n_c);

        for (i, m) in marks.iter().enumerate() {
            if matches!(m, Mark::C) {
                let j = coarse_of[i];
                let row = p.row(i);
                assert_eq!(row.nnz(), 1);
                assert_eq!(row.col_indices()[0], j);
                assert_relative_eq!(row.values()[0], 1.0);
            }
        }
    }

    #[test]
    fn amg_solver_struct() {
        let a = poisson_1d(20);
        let b = DVector::from_vec(vec![1.0; 20]);
        
        // Test with struct-based solver
        let mut solver = Amg::new(1e-6, 0.25, 100);
        solver.init(&a, &b, None);
        
        let converged = solver.solve_iterations(&a, &b, 100);
        assert!(converged, "AMG solver should converge");
        assert!(solver.iterations() > 0, "Should perform at least one iteration");
        
        // Verify solution quality
        let residual = &a * solver.solution() - &b;
        let residual_norm = residual.amax();
        assert!(residual_norm <= 1e-6, "Residual should be below tolerance");
    }

    #[test]
    fn amg_solver_with_custom_smoothing() {
        let a = poisson_1d(16);
        let b = DVector::from_vec(vec![1.0; 16]);
        
        // Test with custom smoothing parameters
        let mut solver = Amg::with_smoothing(1e-6, 0.25, 50, 2, 2);
        assert_eq!(solver.nu_pre, 2);
        assert_eq!(solver.nu_post, 2);
        
        solver.init(&a, &b, None);
        let converged = solver.solve_iterations(&a, &b, 50);
        assert!(converged, "AMG solver with custom smoothing should converge");
    }

    #[test]
    fn amg_convenience_functions() {
        let a = poisson_1d(12);
        let b = DVector::from_vec(vec![1.0; 12]);
        
        // Test convenience function
        let solution = solve_amg(&a, &b, 100, 1e-6, 0.25);
        assert!(solution.is_some(), "Convenience function should return a solution");
        
        // Test with initial guess
        let mut x = DVector::zeros(12);
        let converged = solve_amg_with_initial_guess(&a, &b, &mut x, 100, 1e-6, 0.25);
        assert!(converged, "Convenience function with initial guess should converge");
    }

    #[test]
    fn amg_customization_methods() {
        let mut solver = Amg::new(1e-6, 0.25, 100);
        
        // Test getters
        assert_eq!(solver.tolerance(), 1e-6);
        assert_eq!(solver.theta(), 0.25);
        assert_eq!(solver.coarse_size(), 100);
        assert_eq!(solver.pre_smoothing(), 1);
        assert_eq!(solver.post_smoothing(), 1);
        assert!(!solver.has_converged());
        assert_eq!(solver.num_levels(), 0);
        
        // Test setters
        solver.set_tolerance(1e-8);
        assert_eq!(solver.tolerance(), 1e-8);
        
        solver.set_theta(0.5);
        assert_eq!(solver.theta(), 0.5);
        
        solver.set_smoothing(3, 2);
        assert_eq!(solver.pre_smoothing(), 3);
        assert_eq!(solver.post_smoothing(), 2);
    }

    #[test] 
    fn amg_hierarchy_info() {
        let a = poisson_1d(20);
        let b = DVector::from_vec(vec![1.0; 20]);
        
        let mut solver = Amg::new(1e-6, 0.25, 100);
        solver.init(&a, &b, None);
        
        // After initialization, hierarchy should be built
        assert!(solver.num_levels() > 0, "Hierarchy should have levels after init");
        
        // Test theta change clears hierarchy
        solver.set_theta(0.1);
        solver.init(&a, &b, None);
        assert!(solver.num_levels() > 0, "Hierarchy should be rebuilt after theta change");
    }
}
