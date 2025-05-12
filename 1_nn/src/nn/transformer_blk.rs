use super::{
    Attention, Context, Distribution, Mlp, NNError, Normalization, NuralNetwork, TPTensor, Tensor,
    macros::destruct,
};

#[derive(Clone)]
pub struct TransformerBlk<T> {
    pub attn_norm: Normalization<T>,
    pub attn: Attention<T>,
    pub ffn_norm: Normalization<T>,
    pub ffn: Mlp<T>,
}

impl<T> TransformerBlk<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> TransformerBlk<TPTensor<T>> {
        let Self {
            attn_norm,
            attn,
            ffn_norm,
            ffn,
        } = self;
        TransformerBlk {
            attn_norm: attn_norm.tensor_parallel(),
            attn: attn.tensor_parallel(dist),
            ffn_norm: ffn_norm.tensor_parallel(),
            ffn: ffn.tensor_parallel(dist),
        }
    }
}

impl<T> NuralNetwork<T> for TransformerBlk<T> {
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
        } = self;
        destruct!([x, pos] = inputs);
        let residual = x.clone();
        let tensors = ctx.trap("attn-norm", attn_norm, [x])?;
        destruct!([x] = tensors);
        let tensors = ctx.trap("attn", attn, [x, pos, residual])?;
        destruct!([x] = tensors);
        let residual = x.clone();
        let tensors = ctx.trap("ffn-norm", ffn_norm, [x])?;
        destruct!([x] = tensors);
        let tensors = ctx.trap("ffn", ffn, [x, residual])?;

        Ok((ctx, tensors))
    }
}
