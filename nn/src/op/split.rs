use super::{OpError, Operator, macros::*};
use crate::{Arg, Dim, TensorMeta};

pub struct Split;

impl Operator for Split {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Dict(args)) = args else {
            return Err(OpError::ArgError);
        };
        let Some(Arg::Dim(Dim::Constant(axis))) = args.get("axis") else {
            return Err(OpError::ArgError);
        };
        let Some(Arg::Arr(parts)) = args.get("parts") else {
            return Err(OpError::ArgError);
        };
        let parts = parts
            .iter()
            .map(|p| {
                if let Arg::Dim(dim) = p {
                    Ok(dim.clone())
                } else {
                    Err(OpError::ArgError)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        destruct!([x] = inputs);

        let shape = x.shape();

        if *axis >= shape.len() {
            return Err(OpError::ShapeError);
        }

        let sum = parts
            .iter()
            .fold(Dim::Constant(0), |acc, p| acc + p.clone());

        let c = shape[*axis].clone() / sum;
        //TODO 需要检查parts的和是否等于shape[axis]
        Ok(parts
            .into_iter()
            .map(|p| {
                let mut shape = shape.to_vec();
                shape[*axis] = p * c.clone();
                TensorMeta::new(x.dt, shape)
            })
            .collect::<Vec<_>>())
    }
}
