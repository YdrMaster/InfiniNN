use crate::{Arg, macros::destruct};

use super::{
    Context, Distribution, NNError, NuralNetwork, TPAction, TPTensor, Tensor,
    weight_types::RowTPWeight,
};
use std::{any::Any, collections::HashMap};
use tensor::digit_layout::DigitLayout;

#[derive(Clone)]
pub struct Linear<T: Clone> {
    pub dt: DigitLayout,
    pub shape: [usize; 2],
    pub weight: T,
    pub bias: Option<(DigitLayout, T)>,
    pub allow_residual: bool,
}

impl<T: Clone> Linear<T> {
    #[inline]
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

impl<T: Clone> NuralNetwork<T> for Linear<T> {
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

        let mut inputs = inputs.into_iter();
        let x = inputs.next().unwrap();
        let outputs = if dt.group_size() > 1 {
            let w = ctx.load_external("weight", dt, [r.into(), c.into()], weight)?;

            match inputs.next() {
                Some(residual) if allow_residual => {
                    let arg = Some(
                        HashMap::<String, Arg>::from([
                            ("allow_residual".to_string(), true.into()),
                            ("allow_bias".to_string(), bias.is_some().into()),
                        ])
                        .into(),
                    );
                    match bias {
                        Some((dt, bias)) => {
                            destruct!([b] = ctx.load_external("bias", dt, [r.into()], bias)?);
                            let inputs = vec![x, residual, b]
                                .into_iter()
                                .chain(w)
                                .collect::<Vec<_>>();
                            ctx.call("", "quant-linear", arg, inputs)
                        }
                        None => {
                            let inputs = vec![x, residual].into_iter().chain(w).collect::<Vec<_>>();
                            ctx.call("", "quant-linear", arg, inputs)
                        }
                    }
                }
                _ => {
                    let arg = Some(
                        HashMap::<String, Arg>::from([
                            ("allow_residual".to_string(), false.into()),
                            ("allow_bias".to_string(), bias.is_some().into()),
                        ])
                        .into(),
                    );
                    match bias {
                        Some((dt, bias)) => {
                            destruct!([b] = ctx.load_external("bias", dt, [r.into()], bias)?);
                            let inputs = vec![x, b].into_iter().chain(w).collect::<Vec<_>>();
                            ctx.call("", "quant-linear", arg, inputs)
                        }
                        None => {
                            let inputs = vec![x].into_iter().chain(w).collect::<Vec<_>>();
                            ctx.call("", "quant-linear", arg, inputs)
                        }
                    }
                }
            }
        } else {
            destruct!([w] = ctx.load_external("weight", dt, [r.into(), c.into()], weight)?);
            match inputs.next() {
                Some(residual) if allow_residual => match bias {
                    Some((dt, bias)) => {
                        destruct!([b] = ctx.load_external("bias", dt, [r.into()], bias)?);
                        ctx.call("", "linear", Some(true.into()), [x, residual, w, b])
                    }
                    None => {
                        // format
                        ctx.call("", "linear", Some(true.into()), [x, residual, w])
                    }
                },
                _ => match bias {
                    Some((dt, bias)) => {
                        destruct!([b] = ctx.load_external("bias", dt, [r.into()], bias)?);
                        ctx.call("", "linear", Some(false.into()), [x, w, b])
                    }
                    None => {
                        // format
                        ctx.call("", "linear", Some(false.into()), [x, w])
                    }
                },
            }
        };

        Ok((ctx, outputs?))
    }
}
