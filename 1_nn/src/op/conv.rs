use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};
use arg::make_eq;

pub struct Conv;

impl Operator for Conv {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }

        destruct!([x, w, b] = inputs);

        dims!([n, _c_x, height, width] = x);
        dims!([embd_w, _c_w, d_patch, d_patch_1] = w);
        dims!([embd_b] = b);

        // Check if embd dims match
        if embd_w != embd_b {
            return Err(OpError::ShapeMismatch);
        }

        // Check if patch match
        if d_patch != d_patch_1 {
            return Err(OpError::ShapeMismatch);
        }

        let embd = make_eq(&[embd_w, embd_b]).ok_or(OpError::ShapeMismatch)?;
        let d_patch = make_eq(&[d_patch, d_patch_1]).ok_or(OpError::ShapeMismatch)?;
        let h_dim = height.clone() / d_patch.clone();
        let w_dim = width.clone() / d_patch.clone();

        Ok(vec![TensorMeta::new(x.dt, [n.clone(), embd, h_dim, w_dim])])
    }
}
