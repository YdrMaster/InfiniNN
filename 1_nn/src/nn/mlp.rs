use super::{
    Activation, Context, Distribution, Linear, NNError, NuralNetwork, TPAction, TPTensor, Tensor,
    macros::destruct,
    weight_types::{ColumnTPWeight, FfnGateUp, RowTPWeight},
};

#[derive(Clone)]
pub struct Mlp<T: Clone> {
    pub up: FFNUpFormat<T>,
    pub act: Activation,
    pub down: Linear<T>,
}

#[derive(Clone)]
pub enum FFNUpFormat<T: Clone> {
    Combined(Linear<T>),
    Separated { gate: Linear<T>, up: Linear<T> },
}

impl<T: Clone> Mlp<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> Mlp<TPTensor<T>> {
        let Self { up, act, down } = self;
        Mlp {
            up: match up {
                FFNUpFormat::Combined(up) => FFNUpFormat::Combined(up.parallel(match act {
                    Activation::SwiGLU => TPAction::new(FfnGateUp, dist),
                    Activation::GeLU => TPAction::new(ColumnTPWeight, dist),
                })),
                FFNUpFormat::Separated { gate, up } => FFNUpFormat::Separated {
                    gate: gate.parallel(TPAction::new(ColumnTPWeight, dist)),
                    up: up.parallel(TPAction::new(ColumnTPWeight, dist)),
                },
            },
            act,
            down: down.parallel(TPAction::new(RowTPWeight, dist)),
        }
    }
}

impl<T: Clone> NuralNetwork<T> for Mlp<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self { up, act, down } = self;

        destruct!([x, residual] = inputs);
        let x = match up {
            FFNUpFormat::Combined(up) => {
                destruct!([x] = ctx.trap("ffn-up", up, [x])?);
                destruct!([x] = ctx.trap("activation", act, [x])?);
                x
            }
            FFNUpFormat::Separated { gate, up } => {
                destruct!([gate] = ctx.trap("ffn-gate", gate, [x.clone()])?);
                destruct!([up] = ctx.trap("ffn-up", up, [x])?);
                destruct!([x] = ctx.call("", "swiglu", None, [gate, up])?);
                x
            }
        };
        destruct!([x] = ctx.trap("ffn-down", down, [x, residual])?);

        Ok((ctx, vec![x]))
    }
}
