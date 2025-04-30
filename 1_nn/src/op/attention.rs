use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct Attention;

impl Operator for Attention {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Dim(_dh)) = args else {
            return Err(OpError::ArgError);
        };

        match inputs {
            [q, k, v] => {
                dims!([_n, _dq] = q);
                dims!([_n, _dk] = k);
                dims!([_n, _dv] = v);

                // TODO 判断正确性

                Ok(vec![q.clone()])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
