use crate::{Backend, LayoutManager, MemManager, StorageTensor, Tensor};
use digit_layout::DigitLayout;

pub enum Normalization {
    LayerNorm {
        y: Tensor,
        x: Tensor,
        w: Tensor,
        b: Tensor,
    },
    RmsNorm {
        y: Tensor,
        x: Tensor,
        w: Tensor,
        epsilon: f32,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct Meta {
    ty: Type,
    dt_a: DigitLayout,
    dt_w: DigitLayout,
    d: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum Type {
    LayerNorm,
    RmsNorm { epsilon: f32 },
}

#[derive(Clone, Copy)]
pub enum Arg {
    Y,
    X,
    W,
    B,
}

impl Meta {
    pub fn build(&self, env: &impl LayoutManager<Arg>, batch_size: usize) -> Normalization {
        let shape_a = [batch_size, self.d];
        let shape_wb = [self.d];
        match self.ty {
            Type::LayerNorm => Normalization::LayerNorm {
                y: env.tensor(Arg::Y, self.dt_a, &shape_a),
                x: env.tensor(Arg::X, self.dt_a, &shape_a),
                w: env.tensor(Arg::W, self.dt_w, &shape_wb),
                b: env.tensor(Arg::B, self.dt_w, &shape_wb),
            },
            Type::RmsNorm { epsilon } => Normalization::RmsNorm {
                y: env.tensor(Arg::Y, self.dt_a, &shape_a),
                x: env.tensor(Arg::X, self.dt_a, &shape_a),
                w: env.tensor(Arg::W, self.dt_w, &shape_wb),
                epsilon,
            },
        }
    }
}

pub trait Env<B: Backend>: MemManager<Arg, B> {
    fn layer_norm(
        &self,
        y: &mut StorageTensor<B>,
        x: &StorageTensor<B>,
        w: &StorageTensor<B>,
        b: &StorageTensor<B>,
    );
    fn rms_norm(
        &self,
        y: &mut StorageTensor<B>,
        x: &StorageTensor<B>,
        w: &StorageTensor<B>,
        theta: f32,
    );
}

impl Normalization {
    pub fn launch<B: Backend>(&self, env: impl Env<B>) {
        match self {
            Self::LayerNorm { y, x, w, b } => {
                let mut y = env.load_tensor_mut(Arg::Y, y);
                let x = env.load_tensor(Arg::X, x);
                let w = env.load_tensor(Arg::W, w);
                let b = env.load_tensor(Arg::B, b);
                env.layer_norm(&mut y, &x, &w, &b)
            }
            Self::RmsNorm { y, x, w, epsilon } => {
                let mut y = env.load_tensor_mut(Arg::Y, y);
                let x = env.load_tensor(Arg::X, x);
                let w = env.load_tensor(Arg::W, w);
                env.rms_norm(&mut y, &x, &w, *epsilon)
            }
        }
    }
}
