use crate::{Backend, LayoutManager, MemManager, MemManagerExt, StorageTensor, Tensor};
use digit_layout::DigitLayout;

pub enum Activation {
    SwiGLU { gate: Tensor, up: Tensor },
    GeLU { up: Tensor },
}

#[derive(Clone, Copy, Debug)]
pub struct Meta {
    pub ty: Type,
    pub dt: DigitLayout,
    pub di: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum Type {
    SwiGLU,
    GeLU,
}

#[derive(Clone, Copy)]
pub enum Arg {
    Gate,
    Up,
}

impl Meta {
    pub fn build(&self, env: &impl LayoutManager<Arg>, batch_size: usize) -> Activation {
        let &Self { ty, dt, di } = self;
        let shape = [batch_size, di];
        match ty {
            Type::SwiGLU => Activation::SwiGLU {
                gate: env.tensor(Arg::Gate, dt, &shape),
                up: env.tensor(Arg::Up, dt, &shape),
            },
            Type::GeLU => Activation::GeLU {
                up: env.tensor(Arg::Up, dt, &shape),
            },
        }
    }
}

pub trait Env<B: Backend>: MemManager<Arg, B> {
    fn swiglu(&self, gate: &mut StorageTensor, up: &StorageTensor);
    fn gelu(&self, up: &mut StorageTensor);
}

impl Activation {
    pub fn launch<B: Backend>(&self, env: &impl Env<B>) {
        match self {
            Self::SwiGLU { gate, up } => {
                let mut gate = env.load_tensor_mut(Arg::Gate, gate);
                let up = env.load_tensor_mut(Arg::Up, up);
                env.swiglu(&mut gate, &up)
            }
            Self::GeLU { up } => {
                let mut up = env.load_tensor_mut(Arg::Up, up);
                env.gelu(&mut up)
            }
        }
    }
}
