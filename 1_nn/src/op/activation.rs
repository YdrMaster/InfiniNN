use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};
use arg::make_eq;

pub struct SwiGLU;

impl Operator for SwiGLU {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }

        destruct!([gate, up] = inputs);

        dims!([_n, _d] = gate);
        dims!([n_up, d_up] = up);

        let n_up = make_eq(&[&gate.shape[0], n_up]).ok_or(OpError::ShapeMismatch)?;
        let d_up = make_eq(&[&gate.shape[1], d_up]).ok_or(OpError::ShapeMismatch)?;

        Ok(vec![TensorMeta::new(gate.dt, [n_up, d_up])])
    }
}
pub struct SiLU;

impl Operator for SiLU {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }

        destruct!([x] = inputs);
        dims!([_n, _d] = x);

        Ok(vec![x.clone()])
    }
}

pub struct GeLU;

impl Operator for GeLU {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }

        destruct!([x] = inputs);
        dims!([_n, _d] = x);

        Ok(vec![x.clone()])
    }
}
