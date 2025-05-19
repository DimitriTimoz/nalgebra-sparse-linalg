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
    // Compute the degrees of the rows
    let mut nodes = vec![Node { index: 0, degree: 0 }; n];
    for row_i in 0..n {
        nodes[row_i] = Node {
            index: row_i,
            degree: graph[row_i].len(),
        }
    }

    nodes.sort_by(|a, b| b.degree.cmp(&a.degree)); //TODO If failed be sure to check the sort order
    // Select the coarsening strategy
    let mut marks = vec![Mark::Unmarked; n];
    let mut coarse_of  = vec![usize::MAX; n];

    let mut next_coarse = 0usize;

    for node in nodes.iter_mut() {
        if marks[node.index] != Mark::Unmarked {
            continue;
        }
        // Mark the node as C
        marks[node.index] = Mark::C;
        coarse_of[node.index] = next_coarse;
        next_coarse += 1;
        // Mark all neighbors as F
        for neighbor in graph[node.index].iter() {
            if matches!(marks[*neighbor], Mark::Unmarked) {
                marks[*neighbor] = Mark::F;
            }
        }
    }
    (marks, coarse_of)
}
