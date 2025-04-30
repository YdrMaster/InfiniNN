use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct RmsNorm;

impl Operator for RmsNorm {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let _epsilon = args.ok_or(OpError::ArgError)?;

        // epsilon是浮点数

        match inputs {
            [x, scale] => {
                dims!([_n, _d] = x);
                dims!([_d] = scale);

                // TODO 判断正确性

                Ok(vec![x.clone()])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}

pub struct LayerNorm;

impl Operator for LayerNorm {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let _epsilon = args.ok_or(OpError::ArgError)?;
        // epsilon是浮点数

        match inputs {
            [x, scale, bias] => {
                dims!([_n, _d] = x);
                dims!([_d] = scale);
                dims!([_d] = bias);

                // TODO 判断正确性

                Ok(vec![x.clone()])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
