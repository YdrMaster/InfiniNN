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
        dims!([_n, _d] = up);

        // TODO 判断正确性

        Ok(vec![gate.clone()])
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
