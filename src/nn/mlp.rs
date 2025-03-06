use super::activation::{self, Activation, Type};
use crate::{Backend, LayoutManager, MemManager, MemManagerExt, StorageTensor, Tensor, split};

pub struct Mlp {
    y: Tensor,
    x: Tensor,
    mid: Tensor,
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
pub enum Arg {
    Y,
    X,
    Up,
    UpBias,
    Down,
    DownBias,
    Residual,
}

impl Meta {
    pub fn build<Env>(&self, env: &mut Env, batch_size: usize, residual: bool) -> Mlp
    where
        Env: LayoutManager<Arg> + LayoutManager<activation::Arg>,
    {
        let &Self {
            act,
            d,
            up_bias,
            down_bias,
        } = self;

        let activation::Meta { ty, dt, di } = act;
        let (mid, act) = match ty {
            Type::SwiGLU => {
                let mid = Tensor::new(dt, &[batch_size, di * 2]);
                split!(mid => gate, up; [di, di] @ 1);
                env.set_tensor(activation::Arg::Gate, &gate);
                env.set_tensor(activation::Arg::Up, &up);
                (mid, act.build(env, batch_size))
            }
            Type::GeLU => {
                let mid = Tensor::new(dt, &[batch_size, di]);
                env.set_tensor(activation::Arg::Up, &mid);
                (mid, act.build(env, batch_size))
            }
        };

        let shape = [batch_size, d];
        let d_up = mid.layout.shape()[1];
        Mlp {
            y: env.tensor(Arg::Y, dt, &shape),
            x: env.tensor(Arg::X, dt, &shape),
            mid,

            up: env.tensor(Arg::Up, dt, &[d, d_up]),
            up_bias: if up_bias {
                Some(
                    env.tensor(Arg::UpBias, dt, &[1, d_up])
                        .broadcast(0, batch_size),
                )
            } else {
                None
            },

            down: env.tensor(Arg::Down, dt, &[di, d]),
            down_bias: if down_bias {
                Some(
                    env.tensor(Arg::DownBias, dt, &[1, d])
                        .broadcast(0, batch_size),
                )
            } else {
                None
            },

            act,
            residual,
        }
    }
}

pub trait Env<B: Backend>: MemManager<Arg, B> {
    fn rearrange(&self, y: &mut StorageTensor, x: &StorageTensor);
    fn mat_mul(
        &self,
        c: &mut StorageTensor,
        beta: f32,
        a: &StorageTensor,
        b: &StorageTensor,
        alpha: f32,
    );
    fn swiglu(&self, gate: &mut StorageTensor, up: &StorageTensor);
    fn gelu(&self, up: &mut StorageTensor);
    fn add(&self, y: &mut StorageTensor, x: &StorageTensor);
}

impl Mlp {
    pub fn launch<B: Backend, Env_>(&self, env: &Env_, scale: f32)
    where
        Env_: Env<B>,
        Env_: activation::Env<B>,
    {
        let Self {
            act: activation,
            y,
            x,
            mid,
            up,
            up_bias,
            down,
            down_bias,
            residual,
        } = self;

        let mut x = env.load_tensor_mut(Arg::X, x);
        let mut mid = <Env_ as MemManagerExt<Arg, B>>::workspace(env, &mid);
        {
            let up = env.load_tensor(Arg::Up, up);
            if let Some(up_bias) = up_bias {
                let up_bias = env.load_tensor(Arg::UpBias, up_bias);
                env.rearrange(&mut mid, &up_bias);
                env.mat_mul(&mut mid, 1., &x, &up, 1.)
            } else {
                env.mat_mul(&mut mid, 0., &x, &up, 1.)
            }
        }

        activation.launch(env);

        let mut y = env.load_tensor_mut(Arg::Y, &y);
        {
            let down = env.load_tensor(Arg::Down, down);
            if let Some(down_bias) = down_bias {
                {
                    let x1 = if *residual { &mut x } else { &mut y };
                    {
                        let down_bias = env.load_tensor(Arg::DownBias, down_bias);
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
}
