use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};
use arg::make_eq;

pub struct CausalConv1d;

impl Operator for CausalConv1d {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Arr(_)) = args else {
            return Err(OpError::ArgError);
        };

        destruct!([x, w, b] = inputs);

        dims!([l, d_in] = x);
        dims!([d_in2, _k] = w);
        dims!([d_in3] = b);

        make_eq(&[d_in, d_in2]).ok_or(OpError::ShapeMismatch)?;
        make_eq(&[d_in, d_in3]).ok_or(OpError::ShapeMismatch)?;

        Ok(vec![TensorMeta::new(x.dt, [l.clone(), d_in.clone()])])
    }
}

pub struct SelectiveScan;

impl Operator for SelectiveScan {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }

        destruct!([x, delta, a, b, c, d] = inputs);
        dims!([l, d_in] = x);
        dims!([l2, d_in2] = delta);
        dims!([d_in3, d_state] = a);
        dims!([l3, d_state2] = b);
        dims!([l4, d_state3] = c);
        dims!([d_in4] = d);

        make_eq(&[d_in, d_in2]).ok_or(OpError::ShapeMismatch)?;
        make_eq(&[d_in, d_in3]).ok_or(OpError::ShapeMismatch)?;
        make_eq(&[d_in, d_in4]).ok_or(OpError::ShapeMismatch)?;
        make_eq(&[l, l2]).ok_or(OpError::ShapeMismatch)?;
        make_eq(&[l, l3]).ok_or(OpError::ShapeMismatch)?;
        make_eq(&[l, l4]).ok_or(OpError::ShapeMismatch)?;
        make_eq(&[d_state, d_state2]).ok_or(OpError::ShapeMismatch)?;
        make_eq(&[d_state, d_state3]).ok_or(OpError::ShapeMismatch)?;

        Ok(vec![TensorMeta::new(x.dt, [l.clone(), d_in.clone()])])
    }
}
