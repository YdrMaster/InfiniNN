use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

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

                // Check if embedding dimensions match
                if d != _d {
                    return Err(OpError::ShapeMismatch);
                }

                // Check if sequence lengths match
                if n != _n {
                    return Err(OpError::ShapeMismatch);
                }

                println!("Embedding dimensions match");

                Ok(vec![TensorMeta::new(wte.dt, [n.clone(), d.clone()])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
