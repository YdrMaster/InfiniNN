use super::{Context, NNError, OpError, Tensor};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct RWKVState<T> {
    pub layers: Vec<Tensor<T>>,
    _marker: PhantomData<T>,
}

impl<T> RWKVState<T> {
    pub fn from_layers(layers: Vec<Tensor<T>>) -> Self {
        Self {
            layers,
            _marker: PhantomData,
        }
    }

    pub fn from_tensor(t: Tensor<T>, n_layer: usize) -> Result<Self, NNError> {
        // 假设状态张量在第0维按层数分割
        let shape = t.shape();
        let layer_size = shape[0].clone() / n_layer;
        let parts = vec![layer_size; n_layer];

        let chunks = t.split("split-state", 0, parts)?;
        Ok(Self::from_layers(chunks))
    }

    pub fn to_tensor(self, ctx: &mut Context<T>) -> Result<Tensor<T>, NNError> {
        // 使用 concat 操作将所有层状态拼接
        let result = ctx.call(
            "concat-state",
            "concat",
            Some(arg::Arg::dict([("axis".into(), arg::Arg::int(0))])),
            self.layers,
        )?;

        if result.len() != 1 {
            return Err(NNError {
                name: "RWKVState::to_tensor".into(),
                err: OpError::ShapeError,
            });
        }
        Ok(result.into_iter().next().unwrap())
    }

    pub fn into_layer(mut self, i: usize) -> (Tensor<T>, RWKVStateBuilder<T>) {
        let layer = self.layers.remove(i);
        (
            layer,
            RWKVStateBuilder {
                layers: self.layers,
                index: i,
                _marker: PhantomData,
            },
        )
    }

    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }

    pub fn empty() -> Self {
        Self::from_layers(Vec::new())
    }
}

pub struct RWKVStateBuilder<T> {
    layers: Vec<Tensor<T>>,
    index: usize,
    _marker: PhantomData<T>,
}

impl<T> RWKVStateBuilder<T> {
    pub fn with_layer(mut self, layer: Tensor<T>) -> RWKVState<T> {
        self.layers.insert(self.index, layer);
        RWKVState::from_layers(self.layers)
    }
}
