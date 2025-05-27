use arg::make_eq;

use super::{OpError, Operator};
use crate::{Arg, Dim, TensorMeta};

pub struct Concat;

impl Operator for Concat {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Int(axis)) = args else {
            return Err(OpError::ArgError);
        };
        let axis = *axis as usize;

        // TODO 判定其他维度相等

        let dt = inputs[0].dt;
        let mut concat_shape = vec![];

        for i in 0..inputs[0].shape.len() {
            if i == axis {
                let axis_sum = inputs
                    .iter()
                    .map(|t| t.shape[axis].clone())
                    .fold(Dim::from(0), |acc, d| acc + d);
                concat_shape.push(axis_sum);
            } else {
                let dim = make_eq(&inputs.iter().map(|t| &t.shape[i]).collect::<Vec<_>>())
                    .ok_or(OpError::ShapeMismatch)?;
                concat_shape.push(dim);
            }
        }

        Ok(vec![TensorMeta::new(dt, concat_shape)])
    }
}
