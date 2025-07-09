use super::{GraphBuilder, OpLib, Tensor, TensorMeta};
use crate::{Arg, Dim, Edge, NNError, NNGraph, NuralNetwork, ctx::name::Namespace, op::OpError};
use graph::{GraphTopo, TopoNode};
use mem::{External, Node, Operator};
use std::{cell::RefCell, clone::Clone, collections::HashMap, fmt::Display, ops::Range, rc::Rc};
use tensor::digit_layout::DigitLayout;

pub struct Context<T>(Rc<RefCell<Internal<T>>>);

impl GraphBuilder {
    pub fn build<T: Clone, NN: NuralNetwork<T>>(
        &self,
        nn: NN,
        inputs: impl IntoIterator<Item = TensorMeta>,
    ) -> Result<NNGraph<T>, NNError> {
        let (ctx, inputs) = self.new_context(inputs);
        let outputs = nn.launch(inputs, ctx.clone()).map(|(_, outputs)| outputs)?;
        Ok(ctx.into_graph(outputs))
    }

    fn new_context<T: Clone>(
        &self,
        global_inputs: impl IntoIterator<Item = TensorMeta>,
    ) -> (Context<T>, Vec<Tensor<T>>) {
        let tensors = global_inputs
            .into_iter()
            .enumerate()
            .map(|(i, meta)| Tensor_ {
                name: format!("Ω.{i}"),
                meta,
                external: None,
            })
            .collect::<Vec<_>>();
        let n_inputs = tensors.len();
        let ctx = Context(Rc::new(RefCell::new(Internal {
            op_lib: self.op_lib.clone(),
            namespace: Namespace::new("Ω"),
            operators: Default::default(),
            tensors,
            n_inputs,
        })));

        let tensors = (0..n_inputs)
            .map(|idx| Tensor {
                idx,
                ctx: ctx.clone(),
            })
            .collect();

        (ctx, tensors)
    }
}

pub(super) struct Internal<T> {
    op_lib: Rc<OpLib>,
    namespace: Namespace,
    operators: Vec<Op_>,
    tensors: Vec<Tensor_<T>>,
    n_inputs: usize,
}

struct Op_ {
    name: String,
    operator: Operator,
    inputs: Box<[usize]>,
    outputs: Range<usize>,
}

struct Tensor_<T> {
    name: String,
    meta: TensorMeta,
    external: Option<T>,
}

impl<T: Clone> Context<T> {
    pub fn path(&self) -> String {
        self.0.borrow().namespace.top().path().to_string()
    }

    pub fn trap<NN: NuralNetwork<T>>(
        &mut self,
        name: impl ToString,
        nn: NN,
        inputs: impl IntoIterator<Item = Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, NNError> {
        self.0.borrow_mut().namespace.push(name);
        let outputs = nn
            .launch(inputs, Self(self.0.clone()))
            .map(|(_, outputs)| outputs);
        self.0.borrow_mut().namespace.pop();
        outputs
    }

    pub fn load_external(
        &mut self,
        name: impl Display,
        dt: DigitLayout,
        shape: impl IntoIterator<Item = Dim>,
        item: T,
    ) -> Result<Vec<Tensor<T>>, NNError> {
        let mut internal = self.0.borrow_mut();

        let top = internal.namespace.top_mut();
        assert!(top.tensor.check(&name));
        let name = format!("{}.{}", top.path(), name);

        let external_meta = TensorMeta::load_external(dt, shape);
        let mut tensors = Vec::with_capacity(external_meta.len());
        for meta in external_meta {
            let idx = internal.tensors.len();
            internal.tensors.push(Tensor_ {
                name: name.clone(),
                meta,
                external: Some(item.clone()),
            });
            tensors.push(Tensor {
                idx,
                ctx: Context(self.0.clone()),
            });
        }
        Ok(tensors)
    }

    pub fn bind_external(&mut self, tensor: Tensor<T>, item: T) {
        assert!(
            self.0.borrow_mut().tensors[tensor.idx]
                .external
                .replace(item)
                .is_none()
        )
    }

    pub fn call(
        &mut self,
        name: impl ToString,
        op: impl ToString,
        arg: Option<Arg>,
        inputs: impl IntoIterator<Item = Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, NNError> {
        let mut internal = self.0.borrow_mut();

        let top = internal.namespace.top_mut();
        let op = op.to_string();
        // 没有设置名字的，使用 op 名作为名字
        let mut name = name.to_string();
        if name.is_empty() {
            name = op.clone()
        }
        // 加序号去重
        let name = top.operator.decorate(name.clone());
        let name = format!("{}:{}", top.path(), name);

        let Some(infer) = internal.op_lib.get(&op) else {
            return Err(NNError {
                name,
                err: OpError::NotExist,
            });
        };

        let inputs = inputs.into_iter().map(|t| t.idx).collect::<Box<_>>();
        let meta = inputs
            .iter()
            .map(|&idx| internal.tensors[idx].meta.clone())
            .collect::<Vec<_>>();
        let meta = match infer.infer(&meta, arg.as_ref()) {
            Ok(meta) => meta,
            Err(err) => return Err(NNError { name, err }),
        };

        let start = internal.tensors.len();
        internal
            .tensors
            .extend(meta.into_iter().enumerate().map(|(i, meta)| Tensor_ {
                name: format!("{name}.output.{i}"),
                meta,
                external: None,
            }));
        let end = internal.tensors.len();

        internal.operators.push(Op_ {
            name,
            operator: Operator { name: op, arg },
            inputs,
            outputs: start..end,
        });

        Ok((start..end)
            .map(|idx| Tensor {
                idx,
                ctx: Context(self.0.clone()),
            })
            .collect())
    }
}

impl<T: Clone> Context<T> {
    pub(super) fn clone(&self) -> Self {
        Self(self.0.clone())
    }

    pub(super) fn get_meta(&self, i: usize) -> TensorMeta {
        self.0.borrow().tensors[i].meta.clone()
    }

    fn into_graph(self, global_outputs: Vec<Tensor<T>>) -> NNGraph<T> {
        let Internal {
            operators,
            tensors,
            n_inputs,
            ..
        } = self.0.replace(Internal {
            namespace: Namespace::new("Ω"),
            op_lib: Default::default(),
            operators: Default::default(),
            tensors: Default::default(),
            n_inputs: Default::default(),
        });

        let global_outputs = global_outputs
            .into_iter()
            .map(|t| t.idx)
            .collect::<Vec<_>>();
        let n_outputs = global_outputs.len();

        let mut nodes = Vec::with_capacity(operators.len());
        let mut topo_nodes = Vec::with_capacity(operators.len());
        let mut edges = Vec::with_capacity(tensors.len());
        let mut connections =
            Vec::with_capacity(n_outputs + operators.iter().map(|n| n.inputs.len()).sum::<usize>());

        let mut edge_map = vec![usize::MAX; tensors.len()];
        let mut tensors = tensors
            .into_iter()
            .map(|t| Edge {
                meta: t.meta,
                external: t.external.map(|item| External { name: t.name, item }),
            })
            .enumerate()
            .collect::<HashMap<_, _>>();

        // 填入全图输入
        for (i, map) in edge_map.iter_mut().enumerate().take(n_inputs) {
            *map = i;
            edges.push(tensors.remove(&i).unwrap())
        }
        // 预留全图输出的空间
        connections.extend(std::iter::repeat_n(usize::MAX, n_outputs));
        // 遍历节点
        for op in operators {
            let Op_ {
                name,
                operator,
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
            nodes.push(Node {
                name,
                value: operator,
            })
        }
        // 回填全图输出
        for (i, j) in global_outputs.into_iter().enumerate() {
            connections[i] = edge_map[j]
        }
        NNGraph(::graph::Graph {
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
        })
    }
}
