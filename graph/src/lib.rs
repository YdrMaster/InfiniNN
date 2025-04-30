mod topo;

pub use topo::{GraphTopo, NodeRef, TopoNode};

pub struct Graph<N, E> {
    pub topo: GraphTopo,
    pub nodes: Box<[N]>,
    pub edges: Box<[E]>,
}
