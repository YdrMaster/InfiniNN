use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};
use arg::make_eq;

pub struct QuantLinear;

impl Operator for QuantLinear {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Dict(args)) = args else {
            return Err(OpError::ArgError);
        };
        let Some(Arg::Bool(allow_residual)) = args.get("allow_residual") else {
            return Err(OpError::ArgError);
        };
        let Some(Arg::Bool(allow_bias)) = args.get("allow_bias") else {
            return Err(OpError::ArgError);
        };

        match (allow_residual, allow_bias) {
            (false, false) => {
                let (inputs, w) = inputs.split_at(1);
                destruct!([x] = inputs);
                dims!([m, k_x] = x);
                dims!([n, k_w] = w[0]);

                if k_x.clone() != k_w.clone() * w[0].dt.group_size() {
                    return Err(OpError::ShapeMismatch);
                }

                Ok(vec![TensorMeta::new(x.dt, [m.clone(), n.clone()])])
            }
            (false, true) => {
                let (inputs, w) = inputs.split_at(2);
                destruct!([x, b] = inputs);
                dims!([m, k_x] = x);
                dims!([n, k_w] = w[0]);
                dims!([_n] = b);

                if k_x.clone() != k_w.clone() * w[0].dt.group_size() {
                    return Err(OpError::ShapeMismatch);
                }
                let m = m.clone();
                let n = make_eq(&[n, _n]).ok_or(OpError::ShapeMismatch)?;
                Ok(vec![TensorMeta::new(x.dt, [m, n])])
            }
            (true, false) => {
                let (inputs, w) = inputs.split_at(2);
                destruct!([x, residual] = inputs);
                dims!([_m, k_x] = x);
                dims!([_n, k_w] = w[0]);
                dims!([m, n] = residual);

                if k_x.clone() != k_w.clone() * w[0].dt.group_size() {
                    return Err(OpError::ShapeMismatch);
                }
                let m = make_eq(&[m, _m]).ok_or(OpError::ShapeMismatch)?;
                let n = make_eq(&[n, _n]).ok_or(OpError::ShapeMismatch)?;
                Ok(vec![TensorMeta::new(x.dt, [m, n])])
            }
            (true, true) => {
                let (inputs, w) = inputs.split_at(3);

                destruct!([x, residual, b] = inputs);
                dims!([_m, k_x] = x);
                dims!([_n, k_w] = w[0]);
                dims!([_n] = b);
                dims!([m, n] = residual);

                if k_x.clone() != k_w.clone() * w[0].dt.group_size() {
                    return Err(OpError::ShapeMismatch);
                }
                let m = make_eq(&[m, _m]).ok_or(OpError::ShapeMismatch)?;
                let n = make_eq(&[n, _n]).ok_or(OpError::ShapeMismatch)?;
                Ok(vec![TensorMeta::new(x.dt, [m, n])])
            }
        }
    }
}
