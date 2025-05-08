use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct Linear;

impl Operator for Linear {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Bool(residual)) = args else {
            return Err(OpError::ArgError);
        };
        match inputs {
            [x, w] if !*residual => {
                dims!([m, k_x] = x);
                dims!([n, k_w] = w);

                if k_x != k_w {
                    return Err(OpError::ShapeMismatch);
                }

                Ok(vec![TensorMeta::new(x.dt, [m.clone(), n.clone()])])
            }
            [x, w, b] if !*residual => {
                dims!([m, k_x] = x);
                dims!([n_w, k_w] = w);
                dims!([n_b] = b);

                if k_x != k_w {
                    return Err(OpError::ShapeMismatch);
                }
                if n_w != n_b {
                    return Err(OpError::ShapeMismatch);
                }

                Ok(vec![TensorMeta::new(x.dt, [m.clone(), n_w.clone()])])
            }
            [x, residual, w] => {
                dims!([m_x, k_x] = x);
                dims!([n_w, k_w] = w);
                dims!([m_r, n_r] = residual);

                if k_x != k_w {
                    return Err(OpError::ShapeMismatch);
                }
                if m_x != m_r {
                    return Err(OpError::ShapeMismatch);
                }
                if n_r != n_w {
                    return Err(OpError::ShapeMismatch);
                }

                Ok(vec![TensorMeta::new(x.dt, [m_x.clone(), n_w.clone()])])
            }
            [x, residual, w, b] => {
                dims!([m_x, k_x] = x);
                dims!([n_w, k_w] = w);
                dims!([n_b] = b);
                dims!([m_r, n_r] = residual);

                if k_x != k_w {
                    return Err(OpError::ShapeMismatch);
                }
                if n_w != n_b {
                    return Err(OpError::ShapeMismatch);
                }
                if m_x != m_r {
                    return Err(OpError::ShapeMismatch);
                }
                if n_r != n_w {
                    return Err(OpError::ShapeMismatch);
                }

                Ok(vec![TensorMeta::new(x.dt, [m_x.clone(), n_w.clone()])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
