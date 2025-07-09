use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};
use arg::make_eq;

pub struct Linear;

impl Operator for Linear {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Bool(residual)) = args else {
            return Err(OpError::ArgError);
        };
        match inputs {
            [x, w] if !*residual => {
                dims!([m, k_x] = x);
                dims!([n, k_w] = w);

                if k_x != k_w {
                    return Err(OpError::ShapeMismatch);
                }

                Ok(vec![TensorMeta::new(x.dt, [m.clone(), n.clone()])])
            }
            [x, w, b] if !*residual => {
                dims!([m, k_x] = x);
                dims!([n, k_w] = w);
                dims!([_n] = b);

                if k_x != k_w {
                    return Err(OpError::ShapeMismatch);
                }
                let n = make_eq(&[n, _n]).ok_or(OpError::ShapeMismatch)?;
                Ok(vec![TensorMeta::new(x.dt, [m.clone(), n])])
            }
            [x, residual, w] => {
                dims!([_m, k_x] = x);
                dims!([_n, k_w] = w);
                dims!([m, n] = residual);

                if k_x != k_w {
                    return Err(OpError::ShapeMismatch);
                }

                let m = make_eq(&[m, _m]).ok_or(OpError::ShapeMismatch)?;
                let n = make_eq(&[n, _n]).ok_or(OpError::ShapeMismatch)?;
                Ok(vec![TensorMeta::new(x.dt, [m, n])])
            }
            [x, residual, w, b] => {
                dims!([_m, k_x] = x);
                dims!([_n, k_w] = w);
                dims!([_n] = b);
                dims!([m, n] = residual);

                if k_x != k_w {
                    return Err(OpError::ShapeMismatch);
                }

                let m = make_eq(&[m, _m]).ok_or(OpError::ShapeMismatch)?;
                let n = make_eq(&[n, _n]).ok_or(OpError::ShapeMismatch)?;
                Ok(vec![TensorMeta::new(x.dt, [m, n])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
