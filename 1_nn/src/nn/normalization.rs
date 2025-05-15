use super::{Context, NNError, NuralNetwork, TPTensor, Tensor, macros::destruct};
use tensor::digit_layout::DigitLayout;

#[derive(Clone)]
pub struct Normalization<T> {
    pub d: usize,
    pub epsilon: f64,
    pub items: Type<T>,
}

#[derive(Clone)]
pub enum Type<T> {
    RmsNorm {
        dt: DigitLayout,
        scale: T,
    },
    LayerNorm {
        dt_scale: DigitLayout,
        scale: T,
        dt_bias: DigitLayout,
        bias: T,
    },
}

impl<T> Normalization<T> {
    pub fn tensor_parallel(self) -> Normalization<TPTensor<T>> {
        let Self { d, epsilon, items } = self;
        Normalization {
            d,
            epsilon,
            items: match items {
                Type::RmsNorm { dt, scale } => Type::RmsNorm {
                    dt,
                    scale: scale.into(),
                },
                Type::LayerNorm {
                    dt_scale,
                    scale,
                    dt_bias,
                    bias,
                } => Type::LayerNorm {
                    dt_scale,
                    scale: scale.into(),
                    dt_bias,
                    bias: bias.into(),
                },
            },
        }
    }
}

impl<T> NuralNetwork<T> for Normalization<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        destruct!([x] = inputs);

        let Self { d, epsilon, items } = self;
        let outputs = match items {
            Type::RmsNorm { dt, scale } => {
                let scale = ctx.load_external("scale", dt, [d.into()], scale);
                ctx.call("", "rms-norm", Some(epsilon.into()), [x, scale])
            }
            Type::LayerNorm {
                dt_scale,
                scale,
                dt_bias,
                bias,
            } => {
                let scale = ctx.load_external("scale", dt_scale, [d.into()], scale);
                let bias = ctx.load_external("bias", dt_bias, [d.into()], bias);
                ctx.call("", "layer-norm", Some(epsilon.into()), [x, scale, bias])
            }
        };

        Ok((ctx, outputs?))
    }
}
