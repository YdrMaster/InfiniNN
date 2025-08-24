use super::{Context, NNError, NuralNetwork, Tensor, macros::*};

#[derive(Clone, Copy)]
pub enum Activation {
    SwiGLU,
    SiLU,
    GeLU,
}

impl<T> NuralNetwork<T> for Activation {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        destruct!([x] = inputs);
        dims!([_, d] = x);

        let outputs = match self {
            Self::SwiGLU => {
                let d = d.clone() / 2;
                destruct!([gate, up] = x.split("split-gate-up", 1, [d.clone(), d])?);
                ctx.call("", "swiglu", None, [gate, up])
            }
            Self::SiLU => ctx.call("", "silu", None, [x]),
            Self::GeLU => {
                // format
                ctx.call("", "gelu", None, [x])
            }
        };

        Ok((ctx, outputs?))
    }
}
