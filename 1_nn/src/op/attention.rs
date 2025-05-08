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
                dims!([n_q, _dq] = q);
                dims!([n_k, _dk] = k);
                dims!([n_v, _dv] = v);

                // Check if all inputs have the same batch size
                if n_q != n_k || n_k != n_v {
                    return Err(OpError::ShapeMismatch);
                }

                Ok(vec![q.clone()])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
