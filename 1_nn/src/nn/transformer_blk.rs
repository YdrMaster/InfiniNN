use super::{
    Attention, Context, Distribution, Mlp, NNError, Normalization, NuralNetwork, TPTensor, Tensor,
    macros::destruct,
};

#[derive(Clone)]
pub struct TransformerBlk<T: Clone> {
    pub attn_norm: Normalization<T>,
    pub attn: Attention<T>,
    pub ffn_norm: Normalization<T>,
    pub ffn: Mlp<T>,
    pub all_reduce: bool,
}

impl<T: Clone> TransformerBlk<T> {
    #[inline]
    pub const fn new(
        attn_norm: Normalization<T>,
        attn: Attention<T>,
        ffn_norm: Normalization<T>,
        ffn: Mlp<T>,
    ) -> Self {
        Self {
            attn_norm,
            attn,
            ffn_norm,
            ffn,
            all_reduce: false,
        }
    }

    pub fn tensor_parallel(self, dist: Distribution) -> TransformerBlk<TPTensor<T>> {
        let Self {
            attn_norm,
            attn,
            ffn_norm,
            ffn,
            ..
        } = self;
        TransformerBlk {
            attn_norm: attn_norm.tensor_parallel(),
            attn: attn.tensor_parallel(dist),
            ffn_norm: ffn_norm.tensor_parallel(),
            ffn: ffn.tensor_parallel(dist),
            all_reduce: !dist.is_mono(),
        }
    }
}

impl<T: Clone> NuralNetwork<T> for TransformerBlk<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            attn_norm,
            attn,
            ffn_norm,
            ffn,
            all_reduce,
        } = self;

        destruct!([x, pos] = inputs);
        let residual = x.clone();
        let tensors = ctx.trap("attn-norm", attn_norm, [x])?;
        destruct!([x] = tensors);
        let tensors = ctx.trap("attn", attn, [x, pos, residual])?;
        let tensors = if all_reduce {
            ctx.call("", "all-reduce", Some("sum".into()), tensors)?
        } else {
            tensors
        };

        destruct!([x] = tensors);
        let residual = x.clone();
        let tensors = ctx.trap("ffn-norm", ffn_norm, [x])?;
        destruct!([x] = tensors);
        let tensors = ctx.trap("ffn", ffn, [x, residual])?;
        let tensors = if all_reduce {
            ctx.call("", "all-reduce", Some("sum".into()), tensors)?
        } else {
            tensors
        };

        Ok((ctx, tensors))
    }
}
