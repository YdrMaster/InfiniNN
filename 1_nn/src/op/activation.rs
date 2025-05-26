use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct SwiGLU;

impl Operator for SwiGLU {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }

        destruct!([gate, up] = inputs);

        dims!([_n, _d] = gate);
        dims!([n_up, d_up] = up);
        
        let mut gate = gate.clone();
        if !gate.shape[0].check_eq(n_up) {
            return Err(OpError::ShapeMismatch);
        }
        if !gate.shape[1].check_eq(d_up) {
            return Err(OpError::ShapeMismatch);
        }

        Ok(vec![gate])
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
