use arg::Arg;
use graph::NodeRef;
use std::iter::zip;
pub use tensor::Tensor;

#[repr(transparent)]
pub struct Graph<T>(pub graph::Graph<Node, Tensor<T, 2>>);

#[derive(Clone)]
pub struct Node {
    pub name: String,
    pub op: String,
    pub arg: Option<Arg>,
}

pub struct Exec<T> {
    pub node: Node,
    pub inputs: Box<[Tensor<T, 2>]>,
    pub outputs: Box<[Tensor<T, 2>]>,
}

impl<T: Clone> Graph<T> {
    pub fn into_exec(self) -> Box<[Exec<T>]> {
        let Self(graph::Graph { topo, nodes, edges }) = self;
        zip(topo.iter(), nodes)
            .map(|(topo, node)| {
                let NodeRef { inputs, outputs } = topo;
                Exec {
                    node,
                    inputs: inputs.iter().map(|&i| edges[i].clone()).collect(),
                    outputs: outputs.into_iter().map(|i| edges[i].clone()).collect(),
                }
            })
            .collect()
    }
}
