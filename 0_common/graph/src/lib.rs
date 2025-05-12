mod topo;

pub use topo::{GraphTopo, NodeRef, TopoNode};

#[derive(Clone)]
pub struct Graph<N, E> {
    pub topo: GraphTopo,
    pub nodes: Box<[N]>,
    pub edges: Box<[E]>,
}

#[derive(Clone, Debug)]
pub struct Named<T> {
    pub name: String,
    pub value: T,
}
