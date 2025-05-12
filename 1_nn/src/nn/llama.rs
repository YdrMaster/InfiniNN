use super::{
    Context, Distribution, Embedding, Linear, NNError, Normalization, NuralNetwork, TPAction,
    TPTensor, Tensor, TransformerBlk, macros::destruct, weight_types::ColumnTPWeight,
};

#[derive(Clone)]
pub struct LLaMA<T> {
    pub embedding: Embedding<T>,
    pub blks: Box<[TransformerBlk<T>]>,
    pub out_norm: Normalization<T>,
    pub lm_head: Linear<T>,
}

impl<T> LLaMA<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> LLaMA<TPTensor<T>> {
        let Self {
            embedding,
            blks,
            out_norm,
            lm_head,
        } = self;
        LLaMA {
            embedding: embedding.tensor_parallel(),
            blks: blks
                .into_iter()
                .map(|blk| blk.tensor_parallel(dist))
                .collect(),
            out_norm: out_norm.tensor_parallel(),
            lm_head: lm_head.parallel(TPAction::new(ColumnTPWeight, Distribution::MONO)),
        }
    }
}

impl<T> NuralNetwork<T> for LLaMA<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            embedding,
            blks,
            out_norm,
            lm_head,
        } = self;

        destruct!([tokens, pos, out_idx] = inputs);
        destruct!([x] = ctx.trap("embedding", embedding, [tokens])?);

        let x = blks.into_iter().enumerate().try_fold(x, |x, (i, blk)| {
            destruct!([x] = ctx.trap(format!("blk{i}"), blk, [x, pos.clone()])?);
            Ok(x)
        })?;

        destruct!([x] = ctx.call("out-gather", "embedding", None, [x, out_idx])?);
        destruct!([x] = ctx.trap("out-norm", out_norm, [x])?);
        destruct!([x] = ctx.trap("lm-head", lm_head, [x])?);

        Ok((ctx, vec![x]))
    }
}
