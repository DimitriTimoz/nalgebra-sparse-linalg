use num_traits::Float;

use super::{super::*, graph::strength_graph};

#[derive(Clone, Copy)]
enum Mark {
    Unmarked,
    C, // High degree
    F, //High neighbor 
}


struct Node {
    index: usize,
    degree: usize,
}
impl Node {
    fn new(index: usize, degree: usize) -> Self {
        Node { index, degree }
    }
}
pub fn coarsen<T>(a: &CsrMatrix<T>, theta: T) -> Vec<Mark>
where 
    T: Float,
{
    let mut marks = vec![Mark::Unmarked; a.nrows()];

    let graph = strength_graph(a, theta);
    // Compute the degrees of the rows
    let mut nodes = Vec::with_capacity(a.nrows());
    for (row_i, row) in a.row_iter().enumerate() {
        nodes[row_i] = Node {
            index: row_i,
            degree: row.col_indices().len()
        }
    }

    nodes.sort_by(|a, b| b.degree.cmp(&a.degree)); //TODO If failed be sure to check the sort order
    // Select the coarsening strategy
    for node in nodes.iter_mut() {
        if !matches!(marks[node.index], Mark::Unmarked) {
            continue;
        }
        // Mark the node as C
        marks[node.index] = Mark::C;
        // Mark all neighbors as F
        for neighbor in graph[node.index].iter() {
            if matches!(marks[*neighbor], Mark::Unmarked) {
                marks[*neighbor] = Mark::F;
            }
        }
    }
    
    marks
}
