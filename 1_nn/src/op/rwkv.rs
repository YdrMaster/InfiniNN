use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};
use arg::make_eq;

pub struct RWKVTimeMix;

impl Operator for RWKVTimeMix {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Int(_layer_id)) = args else {
            return Err(OpError::ArgError);
        };

        match inputs {
            [x, k, v, r, time_mix_k, time_mix_v, time_mix_r] => {
                // 检查所有输入的形状一致性
                dims!([batch, seq_len, d_model] = x);
                dims!([batch_k, seq_len_k, d_model_k] = k);
                dims!([batch_v, seq_len_v, d_model_v] = v);
                dims!([batch_r, seq_len_r, d_model_r] = r);

                // 确保所有张量的形状匹配
                let batch = make_eq(&[
                    &x.shape[0],
                    batch,
                    &k.shape[0],
                    batch_k,
                    &v.shape[0],
                    batch_v,
                    &r.shape[0],
                    batch_r,
                    &time_mix_k.shape[0],
                    &time_mix_v.shape[0],
                    &time_mix_r.shape[0],
                ])
                .ok_or(OpError::ShapeMismatch)?;

                let seq_len = make_eq(&[
                    &x.shape[1],
                    seq_len,
                    &k.shape[1],
                    seq_len_k,
                    &v.shape[1],
                    seq_len_v,
                    &r.shape[1],
                    seq_len_r,
                    &time_mix_k.shape[1],
                    &time_mix_v.shape[1],
                    &time_mix_r.shape[1],
                ])
                .ok_or(OpError::ShapeMismatch)?;

                let d_model = make_eq(&[
                    &x.shape[2],
                    d_model,
                    &k.shape[2],
                    d_model_k,
                    &v.shape[2],
                    d_model_v,
                    &r.shape[2],
                    d_model_r,
                    &time_mix_k.shape[2],
                    &time_mix_v.shape[2],
                    &time_mix_r.shape[2],
                ])
                .ok_or(OpError::ShapeMismatch)?;

                // 检查数据类型一致性
                if ![
                    k.dt,
                    v.dt,
                    r.dt,
                    time_mix_k.dt,
                    time_mix_v.dt,
                    time_mix_r.dt,
                ]
                .iter()
                .all(|&dt| dt == x.dt)
                {
                    return Err(OpError::DataTypeMismatch);
                }

                // 输出形状与输入x相同
                Ok(vec![TensorMeta::new(x.dt, [batch, seq_len, d_model])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}

pub struct RWKVChannelMix;

impl Operator for RWKVChannelMix {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Int(_layer_id)) = args else {
            return Err(OpError::ArgError);
        };

        match inputs {
            [x, k, r, time_mix_k, time_mix_r] => {
                // 检查所有输入的形状一致性
                dims!([batch, seq_len, d_model] = x);
                dims!([batch_k, seq_len_k, d_model_k] = k);
                dims!([batch_r, seq_len_r, _d_model_r] = r);

                // 确保所有张量的形状匹配
                let batch = make_eq(&[
                    &x.shape[0],
                    batch,
                    &k.shape[0],
                    batch_k,
                    &r.shape[0],
                    batch_r,
                    &time_mix_k.shape[0],
                    &time_mix_r.shape[0],
                ])
                .ok_or(OpError::ShapeMismatch)?;

                let seq_len = make_eq(&[
                    &x.shape[1],
                    seq_len,
                    &k.shape[1],
                    seq_len_k,
                    &r.shape[1],
                    seq_len_r,
                    &time_mix_k.shape[1],
                    &time_mix_r.shape[1],
                ])
                .ok_or(OpError::ShapeMismatch)?;

                let _d_model = make_eq(&[
                    &x.shape[2],
                    d_model,
                    &time_mix_k.shape[2],
                    &time_mix_r.shape[2],
                ])
                .ok_or(OpError::ShapeMismatch)?;

                // k和r可能有不同的维度（由于linear层的输出维度）
                // 但batch和seq_len必须匹配

                // 检查数据类型一致性
                if ![k.dt, r.dt, time_mix_k.dt, time_mix_r.dt]
                    .iter()
                    .all(|&dt| dt == x.dt)
                {
                    return Err(OpError::DataTypeMismatch);
                }

                // 输出形状与k相同（因为这是channel mix的中间结果）
                Ok(vec![TensorMeta::new(
                    x.dt,
                    [batch, seq_len, d_model_k.clone()],
                )])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
