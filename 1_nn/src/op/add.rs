use super::{OpError, Operator};
use crate::{Arg, TensorMeta};
use arg::make_eq;

pub struct Add;

impl Operator for Add {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }

        match inputs {
            [a, b] => {
                let a_shape = a.shape();
                let b_shape = b.shape();
                let c_shape = a_shape
                    .iter()
                    .zip(b_shape.iter())
                    .map(|(a, b)| make_eq(&[a, b]).ok_or(OpError::ShapeMismatch))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(vec![TensorMeta::new(a.dt, c_shape)])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
