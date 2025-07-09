use super::{
    Context, Distribution, Mlp, NNError, Normalization, NuralNetwork, TPTensor, Tensor,
    macros::destruct,
};
use crate::macros::dims;
use arg::Dim;

#[derive(Clone)]
pub struct Merger<T> {
    pub post_norm: Normalization<T>,
    pub mlp: Mlp<T>,
}

impl<T> Merger<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> Merger<TPTensor<T>> {
        let Self { post_norm, mlp } = self;
        Merger {
            post_norm: post_norm.tensor_parallel(),
            mlp: mlp.tensor_parallel(dist),
        }
    }
}

impl<T> NuralNetwork<T> for Merger<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self { post_norm, mlp } = self;

        destruct!([x] = inputs);
        let tensors = ctx.trap("post-norm", post_norm, [x])?;
        destruct!([x] = tensors);

        // 每 4 个图像特征合为 1 个，x: [np, d] -> [np/4, 4*d]
        dims!([np, _d] = x);
        destruct!([x] = x.tile("", 0, [np.clone() / 4, Dim::from(4)]));
        destruct!([x] = x.merge("", 1, 2));

        let output = ctx.trap("mlp", mlp, [x])?;

        Ok((ctx, output))
    }
}
