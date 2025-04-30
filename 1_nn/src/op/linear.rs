use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct Linear;

impl Operator for Linear {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Bool(residual)) = args else {
            return Err(OpError::ArgError);
        };
        match inputs {
            [x, w] if !*residual => {
                dims!([m, _k] = x);
                dims!([n, _k] = w);

                // TODO 判断正确性

                Ok(vec![TensorMeta::new(x.dt, [m.clone(), n.clone()])])
            }
            [x, w, b] if !*residual => {
                dims!([m, _k] = x);
                dims!([n, _k] = w);
                dims!([_n] = b);

                // TODO 判断正确性

                Ok(vec![TensorMeta::new(x.dt, [m.clone(), n.clone()])])
            }
            [x, residual, w] => {
                dims!([_m, _k] = x);
                dims!([_n, _k] = w);
                dims!([m, n] = residual);

                // TODO 判断正确性

                Ok(vec![TensorMeta::new(x.dt, [m.clone(), n.clone()])])
            }
            [x, residual, w, b] => {
                dims!([_m, _k] = x);
                dims!([_n, _k] = w);
                dims!([_n] = b);
                dims!([m, n] = residual);

                // TODO 判断正确性

                Ok(vec![TensorMeta::new(x.dt, [m.clone(), n.clone()])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
