use super::{OpError, Operator, macros::*};
use crate::{Arg, Dim, TensorMeta};

pub struct Merge;

impl Operator for Merge {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Dict(args)) = args else {
            return Err(OpError::ArgError);
        };
        let Some(Arg::Int(start)) = args.get("start") else {
            return Err(OpError::ArgError);
        };
        let Some(Arg::Int(len)) = args.get("len") else {
            return Err(OpError::ArgError);
        };

        let start = *start as usize;
        let end = start + *len as usize;

        destruct!([x] = inputs);

        let shape = x.shape();

        if end > shape.len() {
            return Err(OpError::ShapeError);
        }

        let merged_dim = shape[start..end]
            .iter()
            .fold(Dim::from(1), |acc, dim| acc * dim.clone());

        let mut new_shape = shape[..start].to_vec();
        new_shape.push(merged_dim);
        new_shape.extend_from_slice(&shape[end..]);

        Ok(vec![TensorMeta::new(x.dt, new_shape)])
    }
}
