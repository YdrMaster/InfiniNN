use crate::{Edge, Info, Node};
use arg::Arg;
use exec::Operator;
use graph::NodeRef;

pub(crate) fn split<T>(node: &mut Node, topo: NodeRef, edges: &mut [Edge<T>]) {
    let NodeRef { inputs, outputs } = topo;
    // split 应该只有一个输入
    let &[input] = inputs else { unreachable!() };
    let input = edges[input].clone();
    // 提取属性
    let Some(Arg::Dict(arg)) = &node.value.arg else {
        unreachable!()
    };
    let axis = arg["axis"].to_usize();
    // 计算步长变换
    let mut start = 0;
    for output in outputs {
        let output = &mut edges[output];
        let part = output.shape()[axis];
        // 暂时不支持 output 是外部的，因为外部 output 需要添加 rearrange kernel
        assert!(matches!(&**output.get(), Info::Internal(_)));
        // 用 slice 实现 split，并替换原来的边
        *output = input
            .clone()
            .transform(|layout| layout.slice(axis, start, 1, part));
        start += part
    }
    // 算子擦除
    node.value = Operator {
        name: "empty".to_string(),
        arg: None,
    }
}

pub(crate) fn tile<T>(node: &mut Node, topo: NodeRef, edges: &mut [Edge<T>]) {
    let NodeRef { inputs, outputs } = topo;
    // tile 应该只有一个输入
    let &[input] = inputs else { unreachable!() };
    let input = edges[input].clone();
    // 提取属性
    let Some(Arg::Dict(arg)) = &node.value.arg else {
        unreachable!()
    };
    let axis = arg["axis"].to_usize();
    let Some(Arg::Arr(tile)) = arg.get("tile") else {
        unreachable!()
    };
    let tile = tile
        .iter()
        .map(|p| match p {
            Arg::Dim(dim) => dim.to_usize(),
            Arg::Int(dim) => *dim as usize,
            _ => {
                unreachable!()
            }
        })
        .collect::<Vec<_>>();
    // 计算步长变换
    assert_eq!(outputs.len(), 1); // tile 应该只有一个输出
    for output in outputs {
        let output = &mut edges[output];
        // 暂时不支持 output 是外部的，因为外部 output 需要添加 rearrange kernel
        assert!(matches!(&**output.get(), Info::Internal(_)));
        // 用 tile_be 实现，并替换原来的边
        *output = input
            .clone()
            .transform(|layout| layout.tile_be(axis, &tile));
    }
    // 算子擦除
    node.value = Operator {
        name: "empty".to_string(),
        arg: None,
    }
}

pub(crate) fn transpose<T>(node: &mut Node, topo: NodeRef, edges: &mut [Edge<T>]) {
    let NodeRef { inputs, outputs } = topo;
    // transpose 应该只有一个输入
    let &[input] = inputs else { unreachable!() };
    let input = edges[input].clone();
    // 提取属性
    let Some(Arg::Dict(arg)) = &node.value.arg else {
        unreachable!()
    };
    let Some(Arg::Arr(perm)) = arg.get("perm") else {
        unreachable!()
    };
    let perm = perm
        .iter()
        .map(|p| {
            if let Arg::Int(perm) = p {
                *perm as usize
            } else {
                unreachable!()
            }
        })
        .collect::<Vec<_>>();
    // 计算步长变换
    assert_eq!(outputs.len(), 1); // transpose 应该只有一个输出
    for output in outputs {
        let output = &mut edges[output];
        // 暂时不支持 output 是外部的，因为外部 output 需要添加 rearrange kernel
        assert!(matches!(&**output.get(), Info::Internal(_)));
        // 用 transpose 实现，并替换原来的边
        *output = input.clone().transform(|layout| layout.transpose(&perm));
    }
    // 算子擦除
    node.value = Operator {
        name: "empty".to_string(),
        arg: None,
    }
}

pub(crate) fn concat<T>(node: &mut Node, topo: NodeRef, edges: &mut [Edge<T>]) {
    let NodeRef { inputs, outputs } = topo;
    // concat 应该只有一个输出
    assert_eq!(outputs.len(), 1);
    let output = edges[outputs.start].clone();
    // 提取属性
    let &Some(Arg::Int(axis)) = &node.value.arg else {
        unreachable!()
    };
    let axis = axis as usize;
    // 计算步长变换
    let mut start = 0;
    for &input in inputs {
        let input = &mut edges[input];
        let part = input.shape()[axis];
        // 暂时不支持 input 是外部的，因为外部 input 需要添加 rearrange kernel
        assert!(matches!(&**input.get(), Info::Internal(_)));
        // 用 slice 实现 split，并替换原来的边
        *input = output
            .clone()
            .transform(|layout| layout.slice(axis, start, 1, part));
        start += part
    }
    // 算子擦除
    node.value = Operator {
        name: "empty".to_string(),
        arg: None,
    }
}
