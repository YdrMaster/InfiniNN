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
    /// 从逻辑连接图下降到存储管理图
    pub fn lower<U>(
        self,
        value: &HashMap<&str, usize>,
        mut map: impl FnMut(T) -> Tensor<U, 2>,
    ) -> mem::Graph<U> {
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
            match external {
                Some(External { name, item }) => {
                    let tensor = map(item);
                    assert_eq!(tensor.dt(), meta.dt());
                    assert_eq!(tensor.shape(), shape);
                    tensor.map(|item| mem::Info::External(External { name, item }))
                }
                None => Tensor::from_dim_slice(meta.dt, &shape).map(mem::Info::Internal),
            }
        });
        mem::Graph::new(topo, nodes, edges)
    }
}
