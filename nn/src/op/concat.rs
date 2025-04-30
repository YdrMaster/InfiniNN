use super::{OpError, Operator};
use crate::{Arg, Dim, TensorMeta};

pub struct Concat;

impl Operator for Concat {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Dim(Dim::Constant(axis))) = args else {
            return Err(OpError::ArgError);
        };

        // TODO 判定其他维度相等

        let dt = inputs[0].dt;
        let mut origin_shape = inputs[0].shape.to_vec();

        let axis_sum = inputs
            .iter()
            .map(|t| t.shape[*axis].clone())
            .fold(Dim::Constant(0), |acc, d| acc + d);

        origin_shape[*axis] = axis_sum;

        Ok(vec![TensorMeta::new(dt, origin_shape)])
    }
}
