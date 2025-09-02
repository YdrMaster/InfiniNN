use super::{OpError, Operator};
use crate::{Arg, TensorMeta};
use arg::make_eq;

pub struct ElementMul;

impl Operator for ElementMul {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }

        match inputs {
            [a, b] => {
                let a_shape = a.shape();
                let b_shape = b.shape();
                if a_shape.len() != b_shape.len() {
                    return Err(OpError::ShapeMismatch);
                }
                let c_shape = a_shape
                    .iter()
                    .zip(b_shape.iter())
                    .map(|(da, db)| make_eq(&[da, db]).ok_or(OpError::ShapeMismatch))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(vec![TensorMeta::new(a.dt, c_shape)])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
