use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct RmsNorm;

impl Operator for RmsNorm {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let _epsilon = args.ok_or(OpError::ArgError)?;

        // epsilon是浮点数

        match inputs {
            [x, scale] => {
                dims!([_n, d_x] = x);
                dims!([d_scale] = scale);

                if d_x != d_scale {
                    return Err(OpError::ShapeMismatch);
                }


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
                dims!([_n, d_x] = x);
                dims!([d_scale] = scale);
                dims!([d_bias] = bias);

                if d_x != d_scale {
                    return Err(OpError::ShapeMismatch);
                }
                if d_x != d_bias {
                    return Err(OpError::ShapeMismatch);
                }

                Ok(vec![x.clone()])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
