mod ctx;
mod nn;

use std::collections::HashMap;

pub mod op;

pub use ::graph::{GraphTopo, NodeRef, TopoNode};
pub use arg::{Arg, Dim};
pub use mem::{BlobLifeTime, Exec, External, Info, Node, Tensor};
pub use op::{OpError, Operator};

pub use ctx::*;
pub use nn::*;

#[derive(Clone)]
#[repr(transparent)]
pub struct Graph<T>(pub graph::Graph<Node, Edge<TPTensor<T>>>);

#[derive(Clone)]
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
    ) -> mem::Graph<TPTensor<U>> {
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
                    let TPTensor { act, val } = item;
                    let tensor = map(val);
                    assert_eq!(tensor.dt(), meta.dt(), "data type mismatch: {name}");
                    assert_eq!(tensor.shape(), shape, "shape mismatch: {name}");
                    tensor.map(|val| {
                        mem::Info::External(External {
                            name,
                            item: TPTensor { act, val },
                        })
                    })
                }
                None => Tensor::from_dim_slice(meta.dt, &shape).map(mem::Info::Internal),
            }
        });
        mem::Graph::new(topo, nodes, edges)
    }
}
