use crate::{
    LayoutManage, MemManage, StorageTensor, Tensor,
    ext::{LayoutManageExt, MemManageExt},
};
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
    pub fn build(&self, env: &impl LayoutManage, batch_size: usize) -> Normalization {
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

pub trait Env: MemManage {
    fn layer_norm(
        &self,
        y: &mut StorageTensor<Self::B>,
        x: &StorageTensor<Self::B>,
        w: &StorageTensor<Self::B>,
        b: &StorageTensor<Self::B>,
    );
    fn rms_norm(
        &self,
        y: &mut StorageTensor<Self::B>,
        x: &StorageTensor<Self::B>,
        w: &StorageTensor<Self::B>,
        theta: f32,
    );
}

impl Normalization {
    pub fn launch(&self, env: &impl Env) {
        match self {
            Self::LayerNorm { y, x, w, b } => {
                let mut y = env.tensor(Arg::Y, y, true);
                let x = env.tensor(Arg::X, x, false);
                let w = env.tensor(Arg::W, w, false);
                let b = env.tensor(Arg::B, b, false);
                env.layer_norm(&mut y, &x, &w, &b)
            }
            Self::RmsNorm { y, x, w, epsilon } => {
                let mut y = env.tensor(Arg::Y, y, true);
                let x = env.tensor(Arg::X, x, false);
                let w = env.tensor(Arg::W, w, false);
                env.rms_norm(&mut y, &x, &w, *epsilon)
            }
        }
    }
}
