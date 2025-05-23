use super::{super::*, graph::strength_graph};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mark {
    Unmarked,
    C, // High degree
    F, //High neighbor 
}

#[derive(Clone)]
struct Node {
    index: usize,
    degree: usize,
}

pub(crate) fn coarsen<T>(a: &CsrMatrix<T>, theta: T) -> (Vec<Mark>, Vec<usize>)
where 
    T: RealField + Copy,
{
    let n = a.nrows();
    let graph = strength_graph(a, theta);
    
    // Create nodes with degrees in one pass, avoiding indexing
    let mut nodes: Vec<Node> = graph.iter()
        .enumerate()
        .map(|(index, neighbors)| Node {
            index,
            degree: neighbors.len(),
        })
        .collect();

    // Sort by degree (descending) - this is the correct order for Ruge-Stüben
    nodes.sort_unstable_by(|a, b| b.degree.cmp(&a.degree));
    
    // Pre-allocate marking vectors
    let mut marks = vec![Mark::Unmarked; n];
    let mut coarse_of = vec![usize::MAX; n];
    let mut next_coarse = 0usize;

    // Ruge-Stüben coarsening algorithm
    for node in &nodes {
        if marks[node.index] != Mark::Unmarked {
            continue;
        }
        
        // Mark the node as C (coarse)
        marks[node.index] = Mark::C;
        coarse_of[node.index] = next_coarse;
        next_coarse += 1;
        
        // Mark all strong neighbors as F (fine)
        for &neighbor in &graph[node.index] {
            if matches!(marks[neighbor], Mark::Unmarked) {
                marks[neighbor] = Mark::F;
            }
        }
    }
    
    (marks, coarse_of)
}
