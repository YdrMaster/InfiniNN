use super::activation::{self, Activation, Type};
use crate::{
    LayoutManage, MemManage, StorageTensor, Tensor,
    ext::{LayoutManageExt, MemManageExt, TrapTraceExt},
    split,
};

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
    pub fn build(&self, env: &impl LayoutManage, batch_size: usize, residual: bool) -> Mlp {
        let &Self {
            act,
            d,
            up_bias,
            down_bias,
        } = self;

        let activation::Meta { ty, dt, di } = act;

        let trap = env.trap(Sub {});
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
        drop(trap);

        let shape = [batch_size, d];
        Mlp {
            y: env.tensor(Arg::Y, dt, &shape),
            x: env.tensor(Arg::X, dt, &shape),
            mid,

            up: env.tensor(Arg::Up, dt, &[d, di]),
            up_bias: if up_bias {
                Some(env.tensor(Arg::UpBias, dt, &[1, di]))
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

pub trait Env: MemManage {
    fn rearrange(&self, y: &mut StorageTensor<Self::B>, x: &StorageTensor<Self::B>);
    fn mat_mul(
        &self,
        c: &mut StorageTensor<Self::B>,
        beta: f32,
        a: &StorageTensor<Self::B>,
        b: &StorageTensor<Self::B>,
        alpha: f32,
    );
    fn add(&self, y: &mut StorageTensor<Self::B>, x: &StorageTensor<Self::B>);
}

impl Mlp {
    pub fn launch<Env_>(&self, env: &Env_, scale: f32)
    where
        Env_: Env + activation::Env,
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

        let mut x = env.tensor(Arg::X, x, true);
        let mut mid = env.workspace(&mid);
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
            let _trap = env.trap(Sub {});
            env.push_arg(activation::Arg::Gate, mid.ptr);
            env.push_arg(activation::Arg::Up, mid.ptr);
            activation.launch(env);
            env.pop_arg(activation::Arg::Gate);
            env.pop_arg(activation::Arg::Up);
        }

        let mut y = env.tensor(Arg::Y, &y, true);
        {
            let down = env.tensor(Arg::Down, down, false);
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
}

#[cfg(test)]
mod test {
    use super::{Arg, Env, Meta, Type, activation};
    use crate::{
        LayoutManage, StorageTensor,
        test_recorder::{TestLayoutManager, TestMemManager},
    };
    use digit_layout::types as ty;
    use ndarray_layout::{ArrayLayout, Endian::BigEndian};

    impl Env for TestMemManager {
        fn rearrange(&self, y: &mut StorageTensor<Self::B>, x: &StorageTensor<Self::B>) {
            self.launch(format!(
                "rearrange(mut %{}, %{})",
                y.ptr.address(),
                x.ptr.address(),
            ))
        }

        fn mat_mul(
            &self,
            c: &mut StorageTensor<Self::B>,
            beta: f32,
            a: &StorageTensor<Self::B>,
            b: &StorageTensor<Self::B>,
            alpha: f32,
        ) {
            self.launch(format!(
                "mat-mul(mut %{}, {beta:.2e}, %{}, %{}, {alpha:.2e})",
                c.ptr.address(),
                a.ptr.address(),
                b.ptr.address(),
            ))
        }

        fn add(&self, y: &mut StorageTensor<Self::B>, x: &StorageTensor<Self::B>) {
            self.launch(format!(
                "add(mut %{}, %{})",
                y.ptr.address(),
                x.ptr.address(),
            ))
        }
    }

    #[test]
    fn test() {
        let meta = Meta {
            act: activation::Meta {
                ty: Type::SwiGLU,
                dt: ty::F16,
                di: 1536,
            },
            d: 1024,
            up_bias: false,
            down_bias: false,
        };
        let xy = ArrayLayout::new_contiguous(&[7, 1024], BigEndian, 2);
        let up = ArrayLayout::new_contiguous(&[1024, 1536], BigEndian, 2);
        let down = ArrayLayout::new_contiguous(&[1536, 1024], BigEndian, 2);

        let mut lm = TestLayoutManager::default();
        lm.set(Arg::Y, xy.clone());
        lm.set(Arg::X, xy);
        lm.set(Arg::Up, up);
        lm.set(Arg::Down, down);

        let act = meta.build(&mut lm, 7, true);

        let mm = TestMemManager::default();

        mm.put_arg(Arg::Y);
        mm.put_arg(Arg::X);
        mm.put_arg(Arg::Up);
        mm.put_arg(Arg::Down);
        act.launch(&mm, 1.);

        println!("{mm}")
    }
}
