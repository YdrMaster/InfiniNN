use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};
use arg::make_eq;

pub struct Embedding;

impl Operator for Embedding {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }
        match inputs {
            [wte, tokens] => {
                dims!([_, d] = wte);
                dims!([n] = tokens);
                Ok(vec![TensorMeta::new(wte.dt, [n.clone(), d.clone()])])
            }
            [wte, tokens, wpe, pos] => {
                dims!([_, d] = wte);
                dims!([n] = tokens);
                dims!([_, _d] = wpe);
                dims!([_n] = pos);

                let d = make_eq(&[d, _d]).ok_or(OpError::ShapeMismatch)?;
                let n = make_eq(&[n, _n]).ok_or(OpError::ShapeMismatch)?;

                Ok(vec![TensorMeta::new(wte.dt, [n, d])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
