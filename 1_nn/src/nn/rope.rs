use arg::Dim;
use tensor::digit_layout::types;

use super::{Context, Distribution, NNError, NuralNetwork, TPTensor, Tensor, macros::*};

#[derive(Clone)]
pub struct RoPE<T> {
    pub type_: RoPEType,
    pub nctx: usize,
    pub dh: usize,
    pub sin: T,
    pub cos: T,
}

#[derive(Clone)]
pub enum RoPEType {
    Normal,

    Multimodal,
}

impl<T> RoPE<T> {
    pub fn tensor_parallel(self, _dist: Distribution) -> RoPE<TPTensor<T>> {
        let Self {
            type_,
            nctx,
            dh,
            sin,
            cos,
        } = self;
        RoPE {
            type_: match type_ {
                RoPEType::Normal => RoPEType::Normal,
                RoPEType::Multimodal => RoPEType::Multimodal,
            },
            nctx,
            dh,
            sin: sin.into(),
            cos: cos.into(),
        }
    }
}

impl<T> NuralNetwork<T> for RoPE<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        destruct!([x, pos] = inputs);

        let Self {
            type_,
            nctx,
            dh,
            sin,
            cos,
        } = self;

        let shape = [Dim::from(nctx), Dim::from(dh / 2)];
        let sin = ctx.load_external("rope.sin", types::F32, shape.clone(), sin);
        let cos = ctx.load_external("rope.cos", types::F32, shape, cos);
        let outputs = match type_ {
            RoPEType::Normal => ctx.call("", "rope", None, [x, pos, sin, cos]),
            RoPEType::Multimodal => ctx.call("", "mrope", None, [x, pos, sin, cos]),
        };

        Ok((ctx, outputs?))
    }
}
