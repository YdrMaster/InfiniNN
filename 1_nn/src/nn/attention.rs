use super::{
    Context, Distribution, Linear, NNError, Normalization, NuralNetwork, TPTensor, Tensor,
    macros::*,
};
use crate::{
    TPAction,
    weight_types::{AttnQKV, RowTPWeight},
};
use tensor::digit_layout::types;

#[derive(Clone)]
pub struct Attention<T> {
    pub nh: usize,
    pub nkvh: usize,
    pub qkv: Linear<T>,
    pub q_norm: Option<Normalization<T>>,
    pub k_norm: Option<Normalization<T>>,
    pub rope: Option<RoPE<T>>,
    pub output: Linear<T>,
}

#[derive(Clone)]
pub struct RoPE<T> {
    pub multimodal: bool,
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
            q_norm,
            k_norm,
            rope,
            output,
        } = self;
        assert_eq!(nh % dist.total, 0);
        assert_eq!(nkvh % dist.total, 0);
        Attention {
            nh: nh / dist.total * dist.len,
            nkvh: nkvh / dist.total * dist.len,
            qkv: qkv.parallel(TPAction::new(AttnQKV(nh / nkvh), dist)),
            q_norm: q_norm.map(|norm| norm.tensor_parallel()),
            k_norm: k_norm.map(|norm| norm.tensor_parallel()),
            rope: rope.map(
                |RoPE {
                     multimodal,
                     nctx,
                     sin,
                     cos,
                 }| RoPE {
                    multimodal,
                    nctx,
                    sin: sin.into(),
                    cos: cos.into(),
                },
            ),
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
            q_norm,
            k_norm,
            rope,
            output,
        } = self;
        destruct!([x] = ctx.trap("attn-qkv", qkv, [x])?);
        dims!([_, dqkv] = x);
        let dh = dqkv.clone() / (nh + nkvh + nkvh);

        destruct!([q, k, v] = x.split("split-qkv", 1, [nh.into(), nkvh.into(), nkvh.into()])?);

        // Apply normalization to q and k if they exist
        let q = match q_norm {
            Some(norm) => {
                let q = q.tile("", 1, [nh.into(), dh.clone()])?;
                destruct!([q] = ctx.trap("attn-q-norm", norm, [q])?);
                q.merge("", 1, 2)?
            }
            None => q,
        };

        let k = match k_norm {
            Some(norm) => {
                let k = k.tile("", 1, [nkvh.into(), dh.clone()])?;
                destruct!([k] = ctx.trap("attn-k-norm", norm, [k])?);
                k.merge("", 1, 2)?
            }
            None => k,
        };

        let [q, k] = match rope {
            Some(RoPE {
                multimodal,
                nctx,
                sin,
                cos,
            }) => {
                let shape = [nctx.into(), dh.clone() / 2];
                let sin = ctx.load_external("rope.sin", types::F32, shape.clone(), sin);
                let cos = ctx.load_external("rope.cos", types::F32, shape, cos);

                let op = if multimodal { "mrope" } else { "rope" };
                destruct!(
                    [q_] = ctx.call(
                        "attn-q-rope",
                        op,
                        None,
                        [q, pos.clone(), sin.clone(), cos.clone()]
                    )?
                );
                destruct!([k_] = ctx.call("attn-k-rope", op, None, [k, pos, sin, cos])?);
                [q_, k_]
            }
            None => [q, k],
        };

        destruct!([o] = ctx.call("", "attention", Some(dh.into()), [q, k, v,])?);

        let outputs = ctx.trap("attn-output", output, [o, residual]);

        Ok((ctx, outputs?))
    }
}
