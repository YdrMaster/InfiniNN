use super::TensorMeta;
use arg::Arg;
use mem::{External, Node, Tensor};
use std::collections::HashMap;

#[repr(transparent)]
pub struct Graph<T>(pub graph::Graph<Node, Edge<T>>);

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
        let Self(graph::Graph {
            topo,
            mut nodes,
            edges,
        }) = self;
        for n in &mut nodes {
            if let Some(arg) = &mut n.arg {
                *arg = std::mem::replace(arg, Arg::Bool(false)).substitute(value)
            }
        }
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
                    assert_eq!(tensor.dt(), meta.dt(), "data type mismatch: {name}");
                    assert_eq!(tensor.shape(), shape, "shape mismatch: {name}");
                    tensor.map(|item| mem::Info::External(External { name, item }))
                }
                None => Tensor::from_dim_slice(meta.dt, &shape).map(mem::Info::Internal),
            }
        });
        mem::Graph::new(topo, nodes, edges)
    }
}
