use crate::{Backend, LayoutManager, MemManager, StorageTensor, Tensor, split};
use digit_layout::DigitLayout;

pub struct Mlp {
    activation: MlpExtra,
    y: Tensor,
    x: Tensor,
    mid: Tensor,
    up: Tensor,
    up_bias: Option<Tensor>,
    down: Tensor,
    down_bias: Option<Tensor>,
    residual: bool,
}

enum MlpExtra {
    SwiGLU { gate: Tensor, up: Tensor },
    GeLU,
}

#[derive(Clone, Copy, Debug)]
pub struct Meta {
    ty: Type,
    dt: DigitLayout,
    d: usize,
    di: usize,
    up_bias: bool,
    down_bias: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum Type {
    SwiGLU,
    Gelu,
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
    pub fn build(&self, env: &impl LayoutManager<Arg>, batch_size: usize, residual: bool) -> Mlp {
        let &Self {
            ty,
            dt,
            d,
            di,
            up_bias,
            down_bias,
        } = self;

        let (mid, extra) = match ty {
            Type::SwiGLU => {
                let mid = Tensor::new(dt, &[batch_size, di * 2]);
                split!(mid => gate, up; [di, di] @ 1);
                (mid, MlpExtra::SwiGLU { gate, up })
            }
            Type::Gelu => {
                let mid = Tensor::new(dt, &[batch_size, di]);
                (mid, MlpExtra::GeLU)
            }
        };

        let shape = [batch_size, d];
        let d_up = mid.layout.shape()[1];
        Mlp {
            y: env.tensor(Arg::Y, dt, &shape),
            x: env.tensor(Arg::X, dt, &shape),

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

            mid,
            activation: extra,
            residual,
        }
    }
}

pub trait Env<B: Backend>: MemManager<Arg, B> {
    fn rearrange(&self, y: &mut StorageTensor<B>, x: &StorageTensor<B>);
    fn mat_mul(
        &self,
        c: &mut StorageTensor<B>,
        beta: f32,
        a: &StorageTensor<B>,
        b: &StorageTensor<B>,
        alpha: f32,
    );
    fn swiglu(&self, gate: &mut StorageTensor<B>, up: &StorageTensor<B>);
    fn gelu(&self, up: &mut StorageTensor<B>);
    fn add(&self, y: &mut StorageTensor<B>, x: &StorageTensor<B>);
}

impl Mlp {
    pub fn launch<B: Backend>(&self, env: impl Env<B>, scale: f32) {
        let Self {
            activation,
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
        let mut mid = env.workspace(mid);
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

        match activation {
            MlpExtra::SwiGLU { gate, up } => {
                let mut gate = StorageTensor {
                    tensor: gate,
                    ptr: mid.ptr,
                };
                let up = StorageTensor {
                    tensor: up,
                    ptr: mid.ptr,
                };
                env.swiglu(&mut gate, &up)
            }
            MlpExtra::GeLU => env.gelu(&mut mid),
        }

        let mut y = env.load_tensor_mut(Arg::Y, &y);
        {
            let down = env.load_tensor(Arg::Down, down);
            if let Some(down_bias) = down_bias {
                let down_bias = env.load_tensor(Arg::DownBias, down_bias);

                if *residual {
                    env.rearrange(&mut x, &down_bias);
                    env.mat_mul(&mut x, scale, &mid, &down, scale);
                    env.add(&mut y, &x)
                } else {
                    env.rearrange(&mut y, &down_bias);
                    env.mat_mul(&mut y, scale, &mid, &down, scale)
                }
            } else if *residual {
                env.mat_mul(&mut y, 1., &mid, &down, scale)
            } else {
                env.mat_mul(&mut y, 0., &mid, &down, scale)
            }
        }
    }
}
