use super::{OpError, Operator, macros::*};
use crate::{Arg, Dim, TensorMeta};

pub struct Tile;

impl Operator for Tile {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let Some(Arg::Dict(args)) = args else {
            return Err(OpError::ArgError);
        };
        let Some(Arg::Int(axis)) = args.get("axis") else {
            return Err(OpError::ArgError);
        };
        let Some(Arg::Arr(tile)) = args.get("tile") else {
            return Err(OpError::ArgError);
        };

        let axis = *axis as usize;
        let tile = tile
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

        if axis >= shape.len() {
            return Err(OpError::ShapeError);
        }

        let tile_product = tile.iter().fold(Dim::from(1), |acc, t| acc * t.clone());

        if tile_product != shape[axis] {
            return Err(OpError::ShapeError);
        }

        let mut new_shape = shape[..axis].to_vec();
        new_shape.extend_from_slice(tile.as_slice());
        new_shape.extend_from_slice(&shape[axis + 1..]);

        Ok(vec![TensorMeta::new(x.dt, new_shape)])
    }
}
