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
                Activation::SiLU => TPAction::new(FfnGateUp, dist),
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

        let mut inputs = inputs.into_iter();
        let x = inputs.next().unwrap();
        destruct!([x] = ctx.trap("ffn-up", up, [x])?);
        destruct!([x] = ctx.trap("activation", act, [x])?);
        let outputs = match inputs.next() {
            Some(residual) => {
                destruct!([x] = ctx.trap("ffn-down", down, [x, residual])?);
                x
            }
            None => {
                destruct!([x] = ctx.trap("ffn-down", down, [x])?);
                x
            }
        };

        Ok((ctx, vec![outputs]))
    }
}
