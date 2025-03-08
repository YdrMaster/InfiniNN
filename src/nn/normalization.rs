use crate::{
    LayerNorm, LayoutManage, RmsNorm, Tensor,
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
    pub fn build(&self, env: &impl LayoutManage) -> Normalization {
        let &Self { ty, dt_a, dt_w, d } = self;
        let n = env.get_dim(Arg::Y, 0);

        match ty {
            Type::LayerNorm => Normalization::LayerNorm {
                y: env.tensor(Arg::Y, dt_a, &[n, d]),
                x: env.tensor(Arg::X, dt_a, &[n, d]),
                w: env.tensor(Arg::W, dt_w, &[d]),
                b: env.tensor(Arg::B, dt_w, &[d]),
            },
            Type::RmsNorm { epsilon } => Normalization::RmsNorm {
                y: env.tensor(Arg::Y, dt_a, &[n, d]),
                x: env.tensor(Arg::X, dt_a, &[n, d]),
                w: env.tensor(Arg::W, dt_w, &[d]),
                epsilon,
            },
        }
    }
}

pub trait Env: LayerNorm + RmsNorm {}

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

#[cfg(test)]
mod test {
    use super::{Arg, Env, Meta, Type};
    use crate::{
        Tensor,
        test_recorder::{TestLayoutManager, TestMemManager, TestMemManagerLoader},
    };
    use digit_layout::types as ty;

    impl Env for TestMemManager {}

    #[test]
    fn test() {
        let dt_a = ty::F16;
        let dt_w = ty::F32;
        let d = 2048;
        let batch_size = 7;

        let meta = Meta {
            ty: Type::RmsNorm { epsilon: 1e-5 },
            dt_a,
            dt_w,
            d,
        };
        let a = Tensor::new(dt_a, &[batch_size, d]).layout;
        let w = Tensor::new(dt_w, &[d]).layout;

        let lm = TestLayoutManager::from([(Arg::Y, a.clone()), (Arg::X, a), (Arg::W, w)]);
        let act = meta.build(&lm);

        let mm = TestMemManagerLoader::new([Arg::Y], [Arg::X, Arg::W]).build();
        act.launch(&mm);

        println!("{mm}")
    }
}
