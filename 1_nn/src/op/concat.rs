use super::{OpError, Operator};
use crate::{Arg, Dim, TensorMeta};
use arg::make_eq;

pub struct Concat;

impl Operator for Concat {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Int(axis)) = args else {
            return Err(OpError::ArgError);
        };
        let axis = *axis as usize;

        // TODO 判定其他维度相等

        let dt = inputs[0].dt;
        let concat_shape = (0..inputs[0].shape.len())
            .map(|i| {
                if i == axis {
                    Ok(inputs
                        .iter()
                        .map(|t| t.shape[axis].clone())
                        .fold(Dim::from(0), |acc, d| acc + d))
                } else {
                    make_eq(&inputs.iter().map(|t| &t.shape[i]).collect::<Vec<_>>())
                        .ok_or(OpError::ShapeMismatch)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(vec![TensorMeta::new(dt, concat_shape)])
    }
}
