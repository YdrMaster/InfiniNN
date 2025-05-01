use crate::{Edge, Info, Node};
use arg::Arg;
use graph::NodeRef;

pub(crate) fn split<T>(node: &mut Node, topo: NodeRef, edges: &mut [Edge<T>]) {
    let NodeRef { inputs, outputs } = topo;
    // split 应该只有一个输入
    let &[input] = inputs else { unreachable!() };
    let input = edges[input].0.clone();
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

pub(crate) fn concat<T>(node: &mut Node, topo: NodeRef, edges: &mut [Edge<T>]) {
    let NodeRef { inputs, outputs } = topo;
    // concat 应该只有一个输出
    assert_eq!(outputs.len(), 1);
    let output = edges[outputs.start].0.clone();
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
