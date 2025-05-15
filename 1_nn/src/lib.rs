mod ctx;
mod nn;

use std::collections::HashMap;

pub mod op;

pub use arg::{Arg, Dim};
pub use graph::{Graph, GraphTopo, Named, NodeRef, TopoNode};
pub use mem::{BlobLifeTime, Exec, External, Info, Node, Operator as OpInfo};
pub use tensor::{Tensor, digit_layout, ndarray_layout};

pub use ctx::*;
pub use nn::*;

#[derive(Clone)]
#[repr(transparent)]
pub struct NNGraph<T>(pub graph::Graph<Node, Edge<T>>);

#[derive(Clone)]
pub struct Edge<T> {
    pub meta: TensorMeta,
    pub external: Option<External<T>>,
}

impl<T> NNGraph<T> {
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
        for node in &mut nodes {
            if let Some(arg) = &mut node.value.arg {
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
