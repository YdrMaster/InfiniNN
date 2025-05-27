use arg::make_eq;

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

                let _d = make_eq(&[&x.shape[1], &scale.shape[0]]).ok_or(OpError::ShapeMismatch)?;

                Ok(vec![TensorMeta::new(x.dt, [_n.clone(), _d])])
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

                let _d = make_eq(&[&x.shape[1], &scale.shape[0], &bias.shape[0]])
                    .ok_or(OpError::ShapeMismatch)?;
                Ok(vec![TensorMeta::new(x.dt, [_n.clone(), _d])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
