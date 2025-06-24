use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};
use arg::make_eq;

pub struct Conv;

impl Operator for Conv {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Bool(bias)) = args else {
            return Err(OpError::ArgError);
        };
        match inputs {
            [x, w] if !*bias => {
                dims!([n, c, height, width] = x);
                dims!([m, ck, hk, wk] = w);

                // Check if channel match
                if c != ck {
                    return Err(OpError::ShapeMismatch);
                }

                let ny = n.clone();
                let my = m.clone();
                let hy = height.clone() / hk.clone();
                let wy = width.clone() / wk.clone();

                Ok(vec![TensorMeta::new(x.dt, [ny, my, hy, wy])])
            }
            [x, w, b] if *bias => {
                dims!([n, c, height, width] = x);
                dims!([m, ck, hk, wk] = w);
                dims!([mb] = b);

                // Check if channel match
                if c != ck {
                    return Err(OpError::ShapeMismatch);
                }

                // Check if embd_dims match
                if m != mb {
                    return Err(OpError::ShapeMismatch);
                }

                let ny = n.clone();
                let my = make_eq(&[m, mb]).ok_or(OpError::ShapeMismatch)?;
                let hy = height.clone() / hk.clone();
                let wy = width.clone() / wk.clone();

                Ok(vec![TensorMeta::new(x.dt, [ny, my, hy, wy])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
