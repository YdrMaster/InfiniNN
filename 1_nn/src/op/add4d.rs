use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};
use arg::make_eq;

pub struct Add4d;

impl Operator for Add4d {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }

        match inputs {
            [a, b] => {
                dims!([d1a, d2a, d3a, d4a] = a);
                dims!([d1b, d2b, d3b, d4b] = b);

                let d1c = make_eq(&[d1a, d1b]).ok_or(OpError::ShapeMismatch)?;
                let d2c = make_eq(&[d2a, d2b]).ok_or(OpError::ShapeMismatch)?;
                let d3c = make_eq(&[d3a, d3b]).ok_or(OpError::ShapeMismatch)?;
                let d4c = make_eq(&[d4a, d4b]).ok_or(OpError::ShapeMismatch)?;

                Ok(vec![TensorMeta::new(a.dt, [d1c, d2c, d3c, d4c])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
