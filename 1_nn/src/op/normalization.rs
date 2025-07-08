use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};
use arg::make_eq;

pub struct RmsNorm;

impl Operator for RmsNorm {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let _epsilon = args.ok_or(OpError::ArgError)?;

        // epsilon是浮点数

        match inputs {
            [x, scale] => {
                let (x_d, scale_d) = match x.shape().len() {
                    2 => {
                        dims!([_n, d] = x);
                        dims!([d_] = scale);
                        (d, d_)
                    }
                    3 => {
                        dims!([_n, _, d] = x);
                        dims!([d_] = scale);
                        (d, d_)
                    }
                    _ => {
                        return Err(OpError::ShapeError);
                    }
                };
                let _d = make_eq(&[x_d, scale_d]).ok_or(OpError::ShapeMismatch)?;
                Ok(vec![TensorMeta::new(x.dt, x.shape().to_vec())])
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
