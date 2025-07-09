use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct QuantEmbedding;

impl Operator for QuantEmbedding {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Bool(pe)) = args else {
            return Err(OpError::ArgError);
        };
        if *pe {
            todo!()
        } else {
            let (w, tokens) = inputs.split_at(inputs.len() - 1);
            dims!([_, d] = w[0]);
            dims!([n] = tokens[0]);
            Ok(vec![TensorMeta::new(
                tensor::digit_layout::types::F32,
                [n.clone(), d.clone() * w[0].dt.group_size()],
            )])
        }
    }
}
