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
        let mut origin_shape = inputs[0].shape.to_vec();

        for i in 0..inputs.len() {
            if i == axis {
                continue;
            }
            if inputs.iter().any(|t| !origin_shape[i].check_eq(&t.shape[i])) {
                return Err(OpError::ShapeMismatch);
            }
        }

        let axis_sum = inputs
            .iter()
            .map(|t| t.shape[axis].clone())
            .fold(Dim::from(0), |acc, d| acc + d);

        origin_shape[axis] = axis_sum;

        Ok(vec![TensorMeta::new(dt, origin_shape)])
    }
}
