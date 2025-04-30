use arg::Arg;
use graph::{GraphTopo, NodeRef};
use std::{iter::zip, rc::Rc};
use tensor::Tensor;

#[repr(transparent)]
pub struct Graph<T>(pub graph::Graph<Node, Edge<T>>);

pub struct Node {
    pub name: String,
    pub op: String,
    pub arg: Option<Arg>,
}

pub struct Edge<T>(pub Tensor<Rc<Info<T>>, 2>);

pub enum Info<T> {
    Internal(usize),
    External(External<T>),
}

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
            .map(|t| Edge(t.map(Rc::new)))
            .collect::<Box<_>>();
        for (node, NodeRef { inputs, outputs }) in zip(&mut nodes, topo.iter()) {
            match &*node.op {
                "split" => {
                    // split 应该只有一个输入
                    let &[input] = inputs else { unreachable!() };
                    let input: Tensor<_, 2> = edges[input].0.clone();
                    // 提取属性
                    let Some(Arg::Dict(arg)) = &node.arg else {
                        unreachable!()
                    };
                    let axis = arg["axis"].to_usize();
                    // 计算步长变换
                    let mut start = 0;
                    for output in outputs {
                        let output = &mut edges[output];
                        let part = output.0.shape()[axis];
                        // 暂时不支持 output 是外部的，因为外部 output 需要添加 rearrange kernel
                        assert!(matches!(&**output.0.get(), Info::Internal(_)));
                        // 用 slice 实现 split，并替换原来的边
                        output.0 = input
                            .clone()
                            .transform(|layout| layout.slice(axis, start, 1, part));
                        start += part
                    }
                    // 算子擦除
                    node.op = "empty".to_string();
                    node.arg = None
                }
                "concat" => {
                    // concat 应该只有一个输出
                    assert_eq!(outputs.len(), 1);
                    let output: Tensor<_, 2> = edges[outputs.start].0.clone();
                    // 提取属性
                    let &Some(Arg::Int(axis)) = &node.arg else {
                        unreachable!()
                    };
                    let axis = axis as usize;
                    // 计算步长变换
                    let mut start = 0;
                    for &input in inputs {
                        let input = &mut edges[input];
                        let part = input.0.shape()[axis];
                        // 暂时不支持 input 是外部的，因为外部 input 需要添加 rearrange kernel
                        assert!(matches!(&**input.0.get(), Info::Internal(_)));
                        // 用 slice 实现 split，并替换原来的边
                        input.0 = output
                            .clone()
                            .transform(|layout| layout.slice(axis, start, 1, part));
                        start += part
                    }
                    // 算子擦除
                    node.op = "empty".to_string();
                    node.arg = None
                }
                _ => {}
            }
        }
        Self(graph::Graph { topo, nodes, edges })
    }
}
