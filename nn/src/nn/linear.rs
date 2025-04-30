use super::{Context, NNError, NuralNetwork, Tensor};
use crate::Dim;
use digit_layout::DigitLayout;

pub struct Linear<T> {
    pub dt: DigitLayout,
    pub shape: [Dim; 2],
    pub weight: T,
    pub bias: Option<(DigitLayout, T)>,
}

impl<T> NuralNetwork<T> for Linear<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            dt,
            shape,
            weight,
            bias,
        } = self;
        let [r, c] = shape;
        let w = ctx.load_external("weight", dt, [r, c.clone()], weight);

        let mut inputs = inputs.into_iter();
        let x = inputs.next().unwrap();
        let outputs = match inputs.next() {
            Some(residual) => match bias {
                Some((dt, bias)) => {
                    let b = ctx.load_external("bias", dt, [c], bias);
                    ctx.call("", "linear", Some(true.into()), [x, residual, w, b])
                }
                None => {
                    // format
                    ctx.call("", "linear", Some(true.into()), [x, residual, w])
                }
            },
            None => match bias {
                Some((dt, bias)) => {
                    let b = ctx.load_external("bias", dt, [c], bias);
                    ctx.call("", "linear", Some(false.into()), [x, w, b])
                }
                None => {
                    // format
                    ctx.call("", "linear", Some(false.into()), [x, w])
                }
            },
        };

        Ok((ctx, outputs?))
    }
}
