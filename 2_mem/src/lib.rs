mod analyze;
mod op;

use graph::GraphTopo;
use std::{iter::zip, rc::Rc};
use tensor::Tensor;

pub use analyze::{Action, BlobLifeTime, KeyWeak, MemRangeMap, print_lifetime};
pub use exec::{Exec, Node, Operator};

#[repr(transparent)]
pub struct Graph<T>(pub graph::Graph<Node, Edge<T>>);

pub type Edge<T> = Tensor<Rc<Info<T>>, 2>;

pub enum Info<T> {
    Internal(usize),
    External(External<T>),
}

#[derive(Clone)]
pub struct External<T> {
    pub name: String,
    pub item: T,
}

impl<T> Graph<T> {
    pub fn new(
        topo: GraphTopo,
        nodes: impl IntoIterator<Item = Node>,
        edges: impl IntoIterator<Item = Tensor<Info<T>, 2>>,
    ) -> Self {
        let mut nodes = nodes.into_iter().collect::<Box<_>>();
        let mut edges = edges
            .into_iter()
            .map(|t| t.map(Rc::new))
            .collect::<Box<_>>();
        for (node, topo) in zip(&mut nodes, topo.iter()) {
            match &*node.value.name {
                "split" => op::split(node, topo, &mut edges),
                "tile" => op::tile(node, topo, &mut edges),
                "transpose" => op::transpose(node, topo, &mut edges),
                "concat" => op::concat(node, topo, &mut edges),
                _ => {}
            }
        }
        Self(graph::Graph { topo, nodes, edges })
    }

    pub fn lower<U>(
        self,
        mut internal: impl FnMut(KeyWeak<Info<T>>) -> U,
        mut external: impl FnMut(&T) -> U,
    ) -> exec::Graph<U> {
        let Self(graph::Graph { topo, nodes, edges }) = self;
        let edges = edges
            .into_iter()
            .map(|tensor| match &**tensor.get() {
                Info::Internal(_) => tensor.as_ref().map(|_| internal(tensor.get().into())),
                Info::External(External { item, .. }) => tensor.as_ref().map(|_| external(item)),
            })
            .collect();
        exec::Graph(graph::Graph { topo, nodes, edges })
    }
}
