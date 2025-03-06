use crate::{
    LayoutManage, MemManage, StorageTensor, Tensor,
    ext::{LayoutManageExt, MemManageExt},
};
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Arg {
    Gate,
    Up,
}

impl Meta {
    pub fn build(&self, env: &impl LayoutManage, batch_size: usize) -> Activation {
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

pub trait Env: MemManage {
    fn swiglu(&self, gate: &mut StorageTensor<Self::B>, up: &StorageTensor<Self::B>);
    fn gelu(&self, up: &mut StorageTensor<Self::B>);
}

impl Activation {
    pub fn launch(&self, env: &impl Env) {
        match self {
            Self::SwiGLU { gate, up } => {
                let mut gate = env.tensor(Arg::Gate, gate, true);
                let up = env.tensor(Arg::Up, up, false);
                env.swiglu(&mut gate, &up)
            }
            Self::GeLU { up } => {
                let mut up = env.tensor(Arg::Up, up, true);
                env.gelu(&mut up)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Arg, Env, Meta, Type};
    use crate::{
        LayoutManage, StorageTensor,
        test_recorder::{TestLayoutManager, TestMemManager},
    };
    use digit_layout::types as ty;
    use ndarray_layout::{ArrayLayout, Endian::BigEndian};

    impl Env for TestMemManager {
        fn swiglu(&self, gate: &mut StorageTensor<Self::B>, up: &StorageTensor<Self::B>) {
            self.launch(format!(
                "swiglu(mut %{}, %{})",
                gate.ptr.address(),
                up.ptr.address()
            ))
        }

        fn gelu(&self, up: &mut StorageTensor<Self::B>) {
            self.launch(format!("gelu(mut %{})", up.ptr.address()))
        }
    }

    #[test]
    fn test() {
        let meta = Meta {
            ty: Type::SwiGLU,
            dt: ty::F16,
            di: 2048,
        };
        let layout = ArrayLayout::new_contiguous(&[7, 2048], BigEndian, 2);

        let mut lm = TestLayoutManager::default();
        lm.set(Arg::Gate, layout.clone());
        lm.set(Arg::Up, layout);
        let act = meta.build(&mut lm, 7);

        let mm = TestMemManager::default();

        mm.put_arg(Arg::Gate);
        mm.put_arg(Arg::Up);
        act.launch(&mm);

        println!("{mm}")
    }
}
