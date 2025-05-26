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

                let mut x = x.clone();
                if !x.shape[1].check_eq(&scale.shape[0]) {
                    return Err(OpError::ShapeMismatch);
                }

                Ok(vec![x])
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

                let mut x = x.clone();
                if !x.shape[1].check_eq(&scale.shape[0]) {
                    return Err(OpError::ShapeMismatch);
                }
                if !x.shape[1].check_eq(&bias.shape[0]) {
                    return Err(OpError::ShapeMismatch);
                }   

                Ok(vec![x])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
