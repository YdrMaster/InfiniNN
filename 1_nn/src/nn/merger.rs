use super::{
    Context, Distribution, Mlp, NNError, Normalization, NuralNetwork, TPTensor, Tensor,
    macros::destruct,
};

#[derive(Clone)]
pub struct Merger<T: Clone> {
    pub post_norm: Normalization<T>,
    pub mlp: Mlp<T>,
}

impl<T: Clone> Merger<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> Merger<TPTensor<T>> {
        let Self { post_norm, mlp } = self;
        Merger {
            post_norm: post_norm.tensor_parallel(),
            mlp: mlp.tensor_parallel(dist),
        }
    }
}

impl<T: Clone> NuralNetwork<T> for Merger<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self { post_norm, mlp } = self;

        destruct!([x] = inputs);
        let tensors = ctx.trap("post-norm", post_norm, [x])?;
        destruct!([x] = tensors);
        let tensors = ctx.trap("mlp", mlp, [x])?;

        Ok((ctx, tensors))
    }
}
