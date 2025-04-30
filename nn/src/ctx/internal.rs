use super::{Edge, External, GraphBuilder, Node, OpLib, Tensor, TensorMeta};
use crate::{Arg, Dim, Graph, GraphTopo, OpError, TopoNode};
use digit_layout::DigitLayout;
use std::{cell::RefCell, collections::HashMap, ops::Range, rc::Rc};

pub(super) struct GraphContext<T>(Rc<RefCell<Internal<T>>>);

impl GraphBuilder {
    pub(super) fn new_context<T>(
        &self,
        global_inputs: impl IntoIterator<Item = TensorMeta>,
    ) -> (GraphContext<T>, Vec<Tensor<T>>) {
        let tensors = global_inputs
            .into_iter()
            .map(|meta| Edge {
                meta,
                external: None,
            })
            .collect::<Vec<_>>();
        let n_inputs = tensors.len();
        let rc = Rc::new(RefCell::new(Internal {
            op_lib: self.op_lib.clone(),
            op_nodes: Default::default(),
            tensors,
            n_inputs,
        }));

        let tensors = (0..n_inputs)
            .map(|idx| Tensor {
                idx,
                ctx: Rc::downgrade(&rc),
            })
            .collect();

        (GraphContext(rc), tensors)
    }
}

pub(super) struct Internal<T> {
    op_lib: Rc<OpLib>,
    op_nodes: Vec<Node_>,
    tensors: Vec<Edge<T>>,
    n_inputs: usize,
}

impl<T> Internal<T> {
    pub fn tensor(&self, idx: usize) -> TensorMeta {
        self.tensors[idx].meta.clone()
    }

    pub fn into_graph(self, global_outputs: Vec<Tensor<T>>) -> Graph<Node, Edge<T>> {
        let Self {
            op_nodes,
            tensors,
            n_inputs,
            ..
        } = self;
        let global_outputs = global_outputs
            .into_iter()
            .map(|t| t.idx)
            .collect::<Vec<_>>();
        let n_outputs = global_outputs.len();

        let mut nodes = Vec::with_capacity(op_nodes.len());
        let mut topo_nodes = Vec::with_capacity(op_nodes.len());
        let mut edges = Vec::with_capacity(tensors.len());
        let mut connections =
            Vec::with_capacity(n_outputs + op_nodes.iter().map(|n| n.inputs.len()).sum::<usize>());

        let mut edge_map = vec![usize::MAX; tensors.len()];
        let mut tensors = tensors.into_iter().enumerate().collect::<HashMap<_, _>>();

        // 填入全图输入
        for (i, map) in edge_map.iter_mut().enumerate().take(n_inputs) {
            *map = i;
            edges.push(tensors.remove(&i).unwrap())
        }
        // 预留全图输出的空间
        connections.extend(std::iter::repeat_n(usize::MAX, n_outputs));
        // 遍历节点
        for op in op_nodes {
            let Node_ {
                node,
                inputs,
                outputs,
            } = op;
            // 记录输入
            let n_inputs = inputs.len();
            let n_outputs = outputs.len();
            let mut n_local = 0;
            connections.extend(inputs.into_iter().map(|i| match edge_map[i] {
                usize::MAX => {
                    // 未映射，应该是权重
                    let j = edges.len();
                    edge_map[i] = j;
                    n_local += 1;
                    edges.push(tensors.remove(&i).unwrap());
                    j
                }
                j => j,
            }));
            // 记录输出
            for i in outputs {
                assert_eq!(edge_map[i], usize::MAX);
                edge_map[i] = edges.len();
                edges.push(tensors.remove(&i).unwrap());
            }
            // 记录节点拓扑
            topo_nodes.push(TopoNode {
                n_local,
                n_inputs,
                n_outputs,
            });
            // 记录节点
            nodes.push(node)
        }
        // 回填全图输出
        for (i, j) in global_outputs.into_iter().enumerate() {
            connections[i] = edge_map[j]
        }
        Graph {
            topo: unsafe {
                GraphTopo::from_raw_parts(
                    n_inputs,
                    n_outputs,
                    connections.into(),
                    topo_nodes.into(),
                )
            },
            nodes: nodes.into(),
            edges: edges.into(),
        }
    }
}

struct Node_ {
    node: Node,
    inputs: Box<[usize]>,
    outputs: Range<usize>,
}

impl<T> GraphContext<T> {
    pub fn take(self) -> Internal<T> {
        let mut internal = self.0.borrow_mut();

        Internal {
            op_lib: internal.op_lib.clone(),
            n_inputs: internal.n_inputs,
            op_nodes: std::mem::take(&mut internal.op_nodes),
            tensors: std::mem::take(&mut internal.tensors),
        }
    }

    pub fn load_external(
        &self,
        name: String,
        dt: DigitLayout,
        shape: impl IntoIterator<Item = Dim>,
        item: T,
    ) -> Tensor<T> {
        let mut internal = self.0.borrow_mut();

        let idx = internal.tensors.len();
        internal.tensors.push(Edge {
            meta: TensorMeta::new(dt, shape),
            external: Some(External { name, item }),
        });
        self.tensor(idx)
    }

    pub fn save_external(&self, name: String, tensor: Tensor<T>, item: T) {
        let mut internal = self.0.borrow_mut();

        assert!(
            internal.tensors[tensor.idx]
                .external
                .replace(External { name, item })
                .is_none()
        )
    }

    pub fn call<'ctx>(
        &self,
        name: String,
        op: impl AsRef<str>,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        arg: Option<Arg>,
    ) -> Result<Vec<Tensor<T>>, OpError>
    where
        T: 'ctx,
    {
        let mut internal = self.0.borrow_mut();

        let op = op.as_ref();
        let Some(infer) = internal.op_lib.get(op) else {
            return Err(OpError::NotExist);
        };

        let inputs = inputs.into_iter().map(|t| t.idx).collect::<Box<_>>();
        let input_meta = inputs
            .iter()
            .map(|&i| internal.tensor(i))
            .collect::<Vec<_>>();

        let start = internal.tensors.len();
        internal.tensors.extend(
            infer
                .infer(&input_meta, arg.as_ref())?
                .into_iter()
                .map(|meta| Edge {
                    meta,
                    external: None,
                }),
        );
        let end = internal.tensors.len();

        internal.op_nodes.push(Node_ {
            node: Node {
                name,
                op: op.into(),
                arg,
            },
            inputs,
            outputs: start..end,
        });

        Ok((start..end).map(|idx| self.tensor(idx)).collect())
    }

    fn tensor(&self, idx: usize) -> Tensor<T> {
        Tensor {
            idx,
            ctx: Rc::downgrade(&self.0),
        }
    }
}
