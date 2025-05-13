use super::{
    Context, Distribution, NNError, NuralNetwork, TPAction, TPTensor, Tensor,
    weight_types::RowTPWeight,
};
use digit_layout::DigitLayout;
use std::any::Any;

#[derive(Clone)]
pub struct Linear<T> {
    pub dt: DigitLayout,
    pub shape: [usize; 2],
    pub weight: T,
    pub bias: Option<(DigitLayout, T)>,
    pub allow_residual: bool,
}

impl<T> Linear<T> {
    pub const fn new(
        dt: DigitLayout,
        shape: [usize; 2],
        weight: T,
        bias: Option<(DigitLayout, T)>,
    ) -> Self {
        Self {
            dt,
            shape,
            weight,
            bias,
            allow_residual: true,
        }
    }

    pub fn parallel(self, tp_action: TPAction) -> Linear<TPTensor<T>> {
        let Self {
            dt,
            mut shape,
            weight,
            bias,
            allow_residual,
        } = self;
        let (act, allow_residual) = if !tp_action.dist.is_mono() {
            let [r, c] = &mut shape;
            let Distribution { start, len, total } = tp_action.dist;
            if (*tp_action.wt).type_id() == RowTPWeight.type_id() {
                *c = *c / total * len
            } else {
                *r = *r / total * len
            }
            (Some(tp_action), allow_residual && start == 0)
        } else {
            (None, allow_residual)
        };
        Linear {
            dt,
            shape,
            weight: TPTensor {
                act: act.clone(),
                val: weight,
            },
            bias: bias.map(|(dt, val)| (dt, TPTensor { act, val })),
            allow_residual,
        }
    }
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
            allow_residual,
        } = self;
        let [r, c] = shape;
        let w = ctx.load_external("weight", dt, [r.into(), c.into()], weight);

        let mut inputs = inputs.into_iter();
        let x = inputs.next().unwrap();
        let outputs = match inputs.next() {
            Some(residual) if allow_residual => match bias {
                Some((dt, bias)) => {
                    let b = ctx.load_external("bias", dt, [r.into()], bias);
                    ctx.call("", "linear", Some(true.into()), [x, residual, w, b])
                }
                None => {
                    // format
                    ctx.call("", "linear", Some(true.into()), [x, residual, w])
                }
            },
            _ => match bias {
                Some((dt, bias)) => {
                    let b = ctx.load_external("bias", dt, [r.into()], bias);
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
