use super::{
    Activation, Context, Distribution, Linear, NNError, NuralNetwork, TPAction, TPTensor, Tensor,
    macros::destruct,
    weight_types::{ColumnTPWeight, FfnGateUp, RowTPWeight},
};

#[derive(Clone)]
pub struct Mlp<T> {
    pub up: Linear<T>,
    pub act: Activation,
    pub down: Linear<T>,
}

impl<T> Mlp<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> Mlp<TPTensor<T>> {
        let Self { up, act, down } = self;
        Mlp {
            up: up.parallel(match act {
                Activation::SwiGLU => TPAction::new(FfnGateUp, dist),
                Activation::GeLU => TPAction::new(ColumnTPWeight, dist),
            }),
            act,
            down: down.parallel(TPAction::new(RowTPWeight, dist)),
        }
    }
}

impl<T> NuralNetwork<T> for Mlp<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self { up, act, down } = self;

        destruct!([x, residual] = inputs);
        destruct!([x] = ctx.trap("ffn-up", up, [x])?);
        destruct!([x] = ctx.trap("activation", act, [x])?);
        destruct!([x] = ctx.trap("ffn-down", down, [x, residual])?);

        Ok((ctx, vec![x]))
    }
}
