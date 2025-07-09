use super::{
    Context, Distribution, Merger, NNError, NuralNetwork, TPTensor, Tensor, TransformerBlk,
    macros::destruct, patch_embd::PatchEmbd,
};

#[derive(Clone)]
pub struct Qwen2VLmmproj<T: Clone> {
    pub patch_embd: PatchEmbd<T>,
    pub vision_blks: Box<[TransformerBlk<T>]>,
    pub merger: Merger<T>,
}

impl<T: Clone> Qwen2VLmmproj<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> Qwen2VLmmproj<TPTensor<T>> {
        let Self {
            patch_embd,
            vision_blks,
            merger,
        } = self;
        Qwen2VLmmproj {
            patch_embd: patch_embd.tensor_parallel(dist),
            vision_blks: vision_blks
                .into_iter()
                .map(|blk| blk.tensor_parallel(dist))
                .collect(),
            merger: merger.tensor_parallel(dist),
        }
    }
}

impl<T: Clone> NuralNetwork<T> for Qwen2VLmmproj<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            patch_embd,
            vision_blks,
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

        let output = ctx.trap("merger", merger, [x])?;

        Ok((ctx, output))
    }
}
