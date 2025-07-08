use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct Transpose;

impl Operator for Transpose {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Dict(args)) = args else {
            return Err(OpError::ArgError);
        };
        let Some(Arg::Arr(perm)) = args.get("perm") else {
            return Err(OpError::ArgError);
        };

        let perm = perm
            .iter()
            .map(|p| {
                if let Arg::Int(perm) = p {
                    Ok(*perm as usize)
                } else {
                    Err(OpError::ArgError)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        destruct!([x] = inputs);

        let shape = x.shape().to_vec();

        if perm.len() != shape.len() {
            return Err(OpError::ShapeError);
        }

        let new_shape = perm.iter().map(|&p| shape[p].clone()).collect::<Vec<_>>();

        Ok(vec![TensorMeta::new(x.dt, new_shape)])
    }
}
