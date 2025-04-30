use super::TensorMeta;
use crate::Arg;
use mem::{External, Tensor};
use std::collections::HashMap;

#[repr(transparent)]
pub struct Graph<T>(pub graph::Graph<Node, Edge<T>>);

pub struct Node {
    pub name: String,
    pub op: String,
    pub arg: Option<Arg>,
}

pub struct Edge<T> {
    pub meta: TensorMeta,
    pub external: Option<External<T>>,
}

impl<T> Graph<T> {
    pub fn lower(self, value: &HashMap<&str, usize>) -> mem::Graph<T> {
        let Self(graph::Graph { topo, nodes, edges }) = self;
        let nodes = nodes.into_iter().map(|n| {
            let Node { name, op, arg } = n;
            mem::Node {
                name,
                op,
                arg: arg.map(|a| a.substitute(value)),
            }
        });
        let edges = edges.into_iter().map(|e| {
            let Edge { meta, external } = e;
            let shape = meta
                .shape
                .iter()
                .map(|d| d.substitute(value))
                .collect::<Vec<_>>();
            Tensor::from_dim_slice(meta.dt, &shape).map(|size| match external {
                Some(ext) => mem::Info::External(ext),
                None => mem::Info::Internal(size),
            })
        });
        mem::Graph::new(topo, nodes, edges)
    }
}
