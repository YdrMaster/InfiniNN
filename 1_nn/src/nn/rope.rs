use arg::Dim;
use tensor::digit_layout::types;

use super::{Context, Distribution, NNError, NuralNetwork, TPTensor, Tensor, macros::*};

#[derive(Clone)]
pub struct RoPE<T> {
    pub type_: RoPEType<T>,
    pub nctx: usize,
    pub dh: usize,
    pub sin: T,
    pub cos: T,
}

#[derive(Clone)]
pub enum RoPEType<T> {
    Normal,
    Long { short: T, long: T, s: f32 },
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
                RoPEType::Long { short, long, s } => RoPEType::Long {
                    short: short.into(),
                    long: long.into(),
                    s,
                },
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
            RoPEType::Long { short, long, s } => {
                let short = ctx.load_external("short", types::F32, todo!(), short);
                let long = ctx.load_external("long", types::F32, todo!(), long);
                todo!()
            }
        };

        Ok((ctx, outputs?))
    }
}
