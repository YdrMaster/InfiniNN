use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct Rope;

impl Operator for Rope {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }

        match inputs {
            [x, pos, sin, cos] => {
                dims!([_n, _d] = x);
                dims!([_n] = pos);
                dims!([_nctx, _dh_2] = sin);
                dims!([_nctx, _dh_2] = cos);

                // TODO 判断正确性

                Ok(vec![x.clone()])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
