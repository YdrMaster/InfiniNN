use crate::tensor::{Tensor, Shape};
use std::marker::PhantomData;

/// RWKV 每层的状态统一封装在结构体中
#[derive(Clone)]
pub struct RWKVState<T> {
    pub layers: Vec<Tensor<T>>,
    _marker: PhantomData<T>,
}

impl<T: Clone> RWKVState<T> {
    /// 从每层状态列表构建
    pub fn from_layers(layers: Vec<Tensor<T>>) -> Self {
        Self { layers, _marker: PhantomData }
    }

    /// 从拼接的 Tensor 构建 RWKVState（假设沿 batch 维度堆叠）
    pub fn from_tensor(t: Tensor<T>, n_layer: usize) -> Self {
        let chunks = t.chunk(n_layer); // 假设 Tensor 实现了 chunk
        Self::from_layers(chunks)
    }

    /// 将所有状态合并为一个 Tensor（例如用于返回值）
    pub fn to_tensor(&self) -> Tensor<T> {
        Tensor::stack(&self.layers)
    }

    /// 获取第 i 层的状态
    pub fn slice(&self, i: usize) -> Tensor<T> {
        self.layers[i].clone()
    }

    /// 更新第 i 层的状态，返回新状态对象
    pub fn update_layer(&self, i: usize, new: Tensor<T>) -> Self {
        let mut new_layers = self.layers.clone();
        new_layers[i] = new;
        Self::from_layers(new_layers)
    }

    /// 初始化状态（每层复制相同初值）
    pub fn init(n_layer: usize, shape: Shape, init_value: T) -> Self {
        let init_tensor = Tensor::full(&shape, init_value);
        Self::from_layers(vec![init_tensor; n_layer])
    }

    /// 获取层数
    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }
}
