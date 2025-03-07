use super::activation::{self, Activation, Type};
use crate::{
    Add, LayoutManage, StorageTensor, Tensor,
    ext::{LayoutManageExt, MemManageExt, TrapTraceExt},
    operators::{MatMul, Rearrange},
    split,
};

pub struct Mlp {
    y: Tensor,
    x: Tensor,
    mid_up: Tensor,
    mid_down: Tensor,
    up: Tensor,
    up_bias: Option<Tensor>,
    down: Tensor,
    down_bias: Option<Tensor>,
    act: Activation,
    residual: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Meta {
    act: activation::Meta,
    d: usize,
    up_bias: bool,
    down_bias: bool,
}

#[derive(Clone, Copy)]
pub struct Sub {}

#[derive(Clone, Copy)]
pub enum Arg {
    Y,
    X,
    Up,
    UpBias,
    Down,
    DownBias,
}

impl Meta {
    pub fn build(&self, env: &impl LayoutManage, residual: bool) -> Mlp {
        let &Self {
            act,
            d,
            up_bias,
            down_bias,
        } = self;
        let activation::Meta { ty, dt, di } = act;
        let n = env.get_dim(Arg::Y, 0);

        let trap = env.trap(Sub {});
        let (mid_up, mid_down, act) = match ty {
            Type::SwiGLU => {
                let mid = Tensor::new(dt, &[n, di * 2]);
                split!(mid => gate, up; [di, di] @ 1);
                env.set_tensor(activation::Arg::Gate, &gate);
                env.set_tensor(activation::Arg::Up, &up);
                (mid, gate, act.build(env))
            }
            Type::GeLU => {
                let mid = Tensor::new(dt, &[n, di]);
                env.set_tensor(activation::Arg::Up, &mid);
                (mid.clone(), mid, act.build(env))
            }
        };
        drop(trap);

        let shape = [n, d];
        let d_up = mid_up.layout.shape()[1];
        Mlp {
            y: env.tensor(Arg::Y, dt, &shape),
            x: env.tensor(Arg::X, dt, &shape),
            mid_up,
            mid_down,

            up: env.tensor(Arg::Up, dt, &[d, d_up]),
            up_bias: if up_bias {
                Some(env.tensor(Arg::UpBias, dt, &[1, d_up]))
            } else {
                None
            },

            down: env.tensor(Arg::Down, dt, &[di, d]),
            down_bias: if down_bias {
                Some(env.tensor(Arg::DownBias, dt, &[1, d]))
            } else {
                None
            },

            act,
            residual,
        }
    }
}

pub trait Env: activation::Env + Rearrange + MatMul + Add {}

impl Mlp {
    pub fn launch(&self, env: &impl Env, scale: f32) {
        let Self {
            act: activation,
            y,
            x,
            mid_up,
            mid_down,
            up,
            up_bias,
            down,
            down_bias,
            residual,
        } = self;

        let mut x = env.tensor(Arg::X, x, true);
        let mut mid = env.workspace(mid_up);
        {
            let up = env.tensor(Arg::Up, up, false);
            if let Some(up_bias) = up_bias {
                let up_bias = env.tensor(Arg::UpBias, up_bias, false);
                env.rearrange(&mut mid, &up_bias);
                env.mat_mul(&mut mid, 1., &x, &up, 1.)
            } else {
                env.mat_mul(&mut mid, 0., &x, &up, 1.)
            }
        }

        {
            use activation::Arg as Act;
            let _trap = env.trap_with(Sub {}, &[(Act::Gate, mid.ptr), (Act::Up, mid.ptr)]);
            activation.launch(env);
        }

        let mid = StorageTensor {
            tensor: mid_down,
            ptr: mid.ptr,
        };
        let down = env.tensor(Arg::Down, down, false);
        let mut y = env.tensor(Arg::Y, y, true);
        if let Some(down_bias) = down_bias {
            {
                let x1 = if *residual { &mut x } else { &mut y };
                {
                    let down_bias = env.tensor(Arg::DownBias, down_bias, false);
                    env.rearrange(x1, &down_bias);
                }
                env.mat_mul(x1, scale, &mid, &down, scale);
            }
            if *residual {
                env.add(&mut y, &x)
            }
        } else {
            let beta = if *residual { 1. } else { 0. };
            env.mat_mul(&mut y, beta, &mid, &down, scale)
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Arg, Env, Meta, Type, activation};
    use crate::{
        Tensor,
        test_recorder::{TestLayoutManager, TestMemManager, TestMemManagerLoader},
    };
    use digit_layout::types as ty;

    impl Env for TestMemManager {}

    #[test]
    fn test() {
        let dt = ty::F16;
        let d = 1024;
        let di = 1536;
        let batch_size = 7;

        let meta = Meta {
            act: activation::Meta {
                ty: Type::SwiGLU,
                dt,
                di,
            },
            d,
            up_bias: false,
            down_bias: false,
        };
        let xy = Tensor::new(dt, &[batch_size, d]).layout;
        let up = Tensor::new(dt, &[d, di * 2]).layout;
        let down = Tensor::new(dt, &[di, d]).layout;

        let lm = TestLayoutManager::from([
            (Arg::Y, xy.clone()),
            (Arg::X, xy),
            (Arg::Up, up),
            (Arg::Down, down),
        ]);
        let mlp = meta.build(&lm, true);

        let mm = TestMemManagerLoader::new([Arg::Y, Arg::X], [Arg::Up, Arg::Down]).build();
        mlp.launch(&mm, 1.);

        println!("{mm}")
    }
}
