use super::{OpError, Operator};
use crate::{Arg, TensorMeta};

pub struct AllReduce;

impl Operator for AllReduce {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(&Arg::Str(_op)) = args else {
            return Err(OpError::ArgError);
        };

        match inputs {
            [x] => Ok(vec![x.clone()]),
            _ => Err(OpError::ShapeError),
        }
    }
}
