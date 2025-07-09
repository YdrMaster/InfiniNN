use super::{
    Context, Distribution, Embedding, NNError, NuralNetwork, TPTensor, Tensor, TransformerBlk,
    macros::destruct, output_head::OutputHead,
};

#[derive(Clone)]
pub struct LLaMA<T: Clone> {
    pub embedding: Embedding<T>,
    pub blks: Box<[TransformerBlk<T>]>,
    pub output_head: Option<OutputHead<T>>,
}

impl<T: Clone> LLaMA<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> LLaMA<TPTensor<T>> {
        let Self {
            embedding,
            blks,
            output_head,
        } = self;
        LLaMA {
            embedding: embedding.tensor_parallel(),
            blks: blks
                .into_iter()
                .map(|blk| blk.tensor_parallel(dist))
                .collect(),
            output_head: output_head.map(OutputHead::tensor_parallel),
        }
    }
}

impl<T: Clone> NuralNetwork<T> for LLaMA<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            embedding,
            blks,
            output_head,
        } = self;

        let mut inputs = inputs.into_iter();
        let tokens = inputs.next().unwrap();
        let pos = inputs.next().unwrap();

        destruct!([x] = ctx.trap("embedding", embedding, [tokens])?);

        let x = blks.into_iter().enumerate().try_fold(x, |x, (i, blk)| {
            destruct!([x] = ctx.trap(format!("blk{i}"), blk, [x, pos.clone()])?);
            Ok(x)
        })?;

        let x = if let Some(OutputHead { out_norm, lm_head }) = output_head {
            let out_idx = inputs.next().unwrap();
            destruct!([x] = ctx.call("out-gather", "embedding", None, [x, out_idx])?);
            destruct!([x] = ctx.trap("out-norm", out_norm, [x])?);
            destruct!([x] = ctx.trap("lm-head", lm_head, [x])?);
            x
        } else {
            x
        };

        Ok((ctx, vec![x]))
    }
}
