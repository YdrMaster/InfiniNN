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
    pub fn build(&self, env: &impl LayoutManage) -> Activation {
        let &Self { ty, dt, di } = self;
        let n = env.get_dim(Arg::Up, 0);

        match ty {
            Type::SwiGLU => Activation::SwiGLU {
                gate: env.tensor(Arg::Gate, dt, &[n, di]),
                up: env.tensor(Arg::Up, dt, &[n, di]),
            },
            Type::GeLU => Activation::GeLU {
                up: env.tensor(Arg::Up, dt, &[n, di]),
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
        Ptr, StorageTensor, Tensor,
        ext::MemManageExt,
        test_recorder::{TestLayoutManager, TestMemManager},
    };
    use digit_layout::types as ty;

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
        let dt = ty::F16;
        let di = 2048;
        let batch_size = 7;

        let meta = Meta {
            ty: Type::SwiGLU,
            dt,
            di,
        };
        let layout = Tensor::new(dt, &[batch_size, di]).layout;

        let lm = TestLayoutManager::from([(Arg::Gate, layout.clone()), (Arg::Up, layout)]);
        let act = meta.build(&lm);

        let mm = TestMemManager::default();
        let _trap = mm.trap_with(
            (),
            &[(Arg::Gate, Ptr::Mut(0 as _)), (Arg::Up, Ptr::Const(1 as _))],
        );
        act.launch(&mm);

        println!("{mm}")
    }
}
