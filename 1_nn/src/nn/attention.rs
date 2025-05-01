use super::{Context, Linear, NNError, NuralNetwork, Tensor, macros::*};
use crate::{Arg, Dim};
use digit_layout::types;

pub struct Attention<T> {
    pub nh: Dim,
    pub nkvh: Dim,
    pub qkv: Linear<T>,
    pub rope: Option<RoPE<T>>,
    pub output: Linear<T>,
}

pub struct RoPE<T> {
    pub nctx: Dim,
    pub sin: T,
    pub cos: T,
}

pub struct Cache<T> {
    pub pos: Dim,
    pub items: [T; 2],
}

impl<T> NuralNetwork<T> for Attention<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        destruct!([x, pos] = inputs);

        let Self {
            nh,
            nkvh,
            qkv,
            rope,
            output,
        } = self;
        let residual = x.clone();

        destruct!([x] = ctx.trap("attn-qkv", qkv, [x])?);
        dims!([_, dqkv] = x);
        let dh = dqkv.clone() / (nh.clone() + nkvh.clone() + nkvh.clone());

        destruct!(
            [q, k, v] = ctx.call(
                "split-qkv",
                "split",
                Some(Arg::dict([
                    ("axis".into(), Arg::int(1)),
                    (
                        "parts".into(),
                        Arg::arr([nh, nkvh.clone(), nkvh.clone()].map(Arg::from))
                    )
                ])),
                [x],
            )?
        );

        let [q, k] = match rope {
            Some(RoPE { nctx, sin, cos }) => {
                let shape = [nctx, dh.clone() / 2];
                let sin = ctx.load_external("rope.sin", types::F32, shape.clone(), sin);
                let cos = ctx.load_external("rope.cos", types::F32, shape.clone(), cos);
                destruct!(
                    [q_] = ctx.call(
                        "attn-q-rope",
                        "rope",
                        None,
                        [q, pos.clone(), sin.clone(), cos.clone()]
                    )?
                );
                destruct!([k_] = ctx.call("attn-k-rope", "rope", None, [k, pos, sin, cos])?);
                [q_, k_]
            }
            None => [q, k],
        };

        destruct!([o] = ctx.call("", "attention", Some(dh.clone().into()), [q, k, v,])?);

        let outputs = ctx.trap("attn-output", output, [o, residual]);

        Ok((ctx, outputs?))
    }
}
