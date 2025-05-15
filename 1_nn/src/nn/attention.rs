use super::{Context, Distribution, Linear, NNError, NuralNetwork, TPTensor, Tensor, macros::*};
use crate::{
    Arg, TPAction,
    weight_types::{AttnQKV, RowTPWeight},
};
use tensor::digit_layout::types;

#[derive(Clone)]
pub struct Attention<T> {
    pub nh: usize,
    pub nkvh: usize,
    pub qkv: Linear<T>,
    pub rope: Option<RoPE<T>>,
    pub output: Linear<T>,
}

#[derive(Clone)]
pub struct RoPE<T> {
    pub nctx: usize,
    pub sin: T,
    pub cos: T,
}

impl<T> Attention<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> Attention<TPTensor<T>> {
        let Self {
            nh,
            nkvh,
            qkv,
            rope,
            output,
        } = self;
        assert_eq!(nh % dist.total, 0);
        assert_eq!(nkvh % dist.total, 0);
        Attention {
            nh: nh / dist.total * dist.len,
            nkvh: nkvh / dist.total * dist.len,
            qkv: qkv.parallel(TPAction::new(AttnQKV(nh / nkvh), dist)),
            rope: rope.map(|RoPE { nctx, sin, cos }| RoPE {
                nctx,
                sin: sin.into(),
                cos: cos.into(),
            }),
            output: output.parallel(TPAction::new(RowTPWeight, dist)),
        }
    }
}

impl<T> NuralNetwork<T> for Attention<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        destruct!([x, pos, residual] = inputs);

        let Self {
            nh,
            nkvh,
            qkv,
            rope,
            output,
        } = self;

        destruct!([x] = ctx.trap("attn-qkv", qkv, [x])?);
        dims!([_, dqkv] = x);
        let dh = dqkv.clone() / (nh + nkvh + nkvh);

        destruct!(
            [q, k, v] = ctx.call(
                "split-qkv",
                "split",
                Some(Arg::dict([
                    ("axis".into(), Arg::int(1)),
                    (
                        "parts".into(),
                        Arg::arr([Arg::dim(nh), Arg::dim(nkvh), Arg::dim(nkvh)])
                    )
                ])),
                [x],
            )?
        );

        let [q, k] = match rope {
            Some(RoPE { nctx, sin, cos }) => {
                let shape = [nctx.into(), dh.clone() / 2];
                let sin = ctx.load_external("rope.sin", types::F32, shape.clone(), sin);
                let cos = ctx.load_external("rope.cos", types::F32, shape, cos);
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

        destruct!([o] = ctx.call("", "attention", Some(dh.into()), [q, k, v,])?);

        let outputs = ctx.trap("attn-output", output, [o, residual]);

        Ok((ctx, outputs?))
    }
}
