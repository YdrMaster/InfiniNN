use std::any::Any;

use super::{
    Context, Distribution, NNError, NuralNetwork, TPAction, TPTensor, Tensor,
    weight_types::RowTPWeight,
};
use digit_layout::DigitLayout;

pub struct Linear<T> {
    pub dt: DigitLayout,
    pub shape: [usize; 2],
    pub weight: T,
    pub bias: Option<(DigitLayout, T)>,
    pub parallel: Option<TPAction>,
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
            parallel: None,
        }
    }

    pub fn parallel(self, tp_action: TPAction) -> Self {
        if tp_action.dist.is_mono() {
            return self;
        }
        let Self {
            dt,
            shape,
            weight,
            bias,
            parallel,
        } = self;
        assert!(parallel.is_none());
        Self {
            dt,
            shape,
            weight,
            bias,
            parallel: Some(tp_action),
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
            parallel,
        } = self;
        let [mut r, mut c] = shape;
        if let Some(tp_act) = &parallel {
            let Distribution { len, total, .. } = tp_act.dist;
            if tp_act.wt.type_id() == RowTPWeight.type_id() {
                c = c / total * len
            } else {
                r = r / total * len
            }
        }
        let w = ctx.load_external(
            "weight",
            dt,
            [r.into(), c.into()],
            TPTensor {
                act: parallel.clone(),
                val: weight,
            },
        );

        let mut inputs = inputs.into_iter();
        let x = inputs.next().unwrap();
        let outputs = match inputs.next() {
            Some(residual) => match bias {
                Some((dt, bias)) => {
                    let b = ctx.load_external(
                        "bias",
                        dt,
                        [r.into()],
                        TPTensor {
                            act: parallel,
                            val: bias,
                        },
                    );
                    ctx.call("", "linear", Some(true.into()), [x, residual, w, b])
                }
                None => {
                    // format
                    ctx.call("", "linear", Some(true.into()), [x, residual, w])
                }
            },
            None => match bias {
                Some((dt, bias)) => {
                    let b = ctx.load_external(
                        "bias",
                        dt,
                        [r.into()],
                        TPTensor {
                            act: parallel,
                            val: bias,
                        },
                    );
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
