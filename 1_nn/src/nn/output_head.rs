use super::{
    Context, Distribution, Linear, NNError, Normalization, NuralNetwork, TPAction, TPTensor,
    Tensor, macros::destruct, weight_types::ColumnTPWeight,
};

#[derive(Clone)]
pub struct OutputHead<T> {
    pub out_norm: Normalization<T>,
    pub lm_head: Linear<T>,
}

impl<T> OutputHead<T> {
    pub fn tensor_parallel(self) -> OutputHead<TPTensor<T>> {
        let Self { out_norm, lm_head } = self;
        OutputHead {
            out_norm: out_norm.tensor_parallel(),
            lm_head: lm_head.parallel(TPAction::new(ColumnTPWeight, Distribution::MONO)),
        }
    }
}

impl<T> NuralNetwork<T> for OutputHead<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self { out_norm, lm_head } = self;
        destruct!([x] = inputs);
        destruct!([x] = ctx.trap("out-norm", out_norm, [x])?);
        destruct!([x] = ctx.trap("lm-head", lm_head, [x])?);
        Ok((ctx, vec![x]))
    }
}
