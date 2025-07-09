use super::{
    Context, Distribution, Merger, Mlp, NNError, NuralNetwork, PatchEmbd, TPTensor, Tensor,
    TransformerBlk, macros::destruct,
};

#[derive(Clone)]
pub struct CogVLM<T: Clone> {
    pub patch_embd: PatchEmbd<T>,
    pub vision_blks: Box<[TransformerBlk<T>]>,
    pub glu_proj: Mlp<T>,
    pub merger: Merger<T>,
}

impl<T: Clone> CogVLM<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> CogVLM<TPTensor<T>> {
        let Self {
            patch_embd,
            vision_blks,
            glu_proj,
            merger,
        } = self;
        CogVLM {
            patch_embd: patch_embd.tensor_parallel(dist),
            vision_blks: vision_blks
                .into_iter()
                .map(|blk| blk.tensor_parallel(dist))
                .collect(),
            glu_proj: glu_proj.tensor_parallel(dist),
            merger: merger.tensor_parallel(dist),
        }
    }
}

impl<T: Clone> NuralNetwork<T> for CogVLM<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            patch_embd,
            vision_blks,
            glu_proj,
            merger,
        } = self;

        let mut inputs = inputs.into_iter();
        let image = inputs.next().unwrap();
        let pos = inputs.next().unwrap();

        destruct!([x] = ctx.trap("patch_embd", patch_embd, [image])?);

        let x = vision_blks
            .into_iter()
            .enumerate()
            .try_fold(x, |x, (i, blk)| {
                destruct!([x] = ctx.trap(format!("blk{i}"), blk, [x, pos.clone()])?);
                Ok(x)
            })?;

        destruct!([x] = ctx.trap("glu_proj", glu_proj, [x])?);
        let output = ctx.trap("merger", merger, [x])?;

        Ok((ctx, output))
    }
}
