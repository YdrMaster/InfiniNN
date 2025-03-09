use super::NuralNetwork;
use crate::{
    Context, VirtualMachine,
    op::{GeLU, SwiGLU},
    tensor::Tensor,
};

#[derive(Clone, Copy, Debug)]
pub struct Activation;

pub enum Args<VM>
where
    VM: VirtualMachine + ?Sized,
{
    SwiGLU {
        gate: Tensor<VM::Blob>,
        up: Tensor<VM::Blob>,
    },
    GeLU {
        up: Tensor<VM::Blob>,
    },
}

pub trait Ops: SwiGLU + GeLU {}

impl<VM> NuralNetwork<VM> for Activation
where
    VM: VirtualMachine + ?Sized + Ops,
{
    type Args = Args<VM>;
    type Weight = ();
    type Sub = ();

    fn launch(ctx: Context<VM, Self>, args: Self::Args) {
        match args {
            Args::SwiGLU { mut gate, up } => ctx.swiglu(&mut gate, &up),
            Args::GeLU { mut up } => ctx.gelu(&mut up),
        }
    }
}

// impl Meta {
//     pub fn build(&self, env: &impl LayoutManage) -> Activation {
//         let &Self { ty, dt, di } = self;
//         let n = env.get_dim(Arg::Up, 0);

//         match ty {
//             Type::SwiGLU => Activation::SwiGLU {
//                 gate: env.tensor(Arg::Gate, dt, &[n, di]),
//                 up: env.tensor(Arg::Up, dt, &[n, di]),
//             },
//             Type::GeLU => Activation::GeLU {
//                 up: env.tensor(Arg::Up, dt, &[n, di]),
//             },
//         }
//     }
// }

// pub trait Env: SwiGLU + GeLU {}

// impl Activation {
//     pub fn launch(&self, env: &impl Env) {
//         match self {
//             Self::SwiGLU { gate, up } => {
//                 let mut gate = env.tensor(Arg::Gate, gate, true);
//                 let up = env.tensor(Arg::Up, up, false);
//                 env.swiglu(&mut gate, &up)
//             }
//             Self::GeLU { up } => {
//                 let mut up = env.tensor(Arg::Up, up, true);
//                 env.gelu(&mut up)
//             }
//         }
//     }
// }

// #[cfg(test)]
// mod test {
//     use super::{Arg, Env, Meta, Type};
//     use crate::{
//         Tensor,
//         test_recorder::{TestLayoutManager, TestMemManager, TestMemManagerLoader},
//     };
//     use digit_layout::types as ty;

//     impl Env for TestMemManager {}

//     #[test]
//     fn test() {
//         let dt = ty::F16;
//         let di = 2048;
//         let batch_size = 7;

//         let meta = Meta {
//             ty: Type::SwiGLU,
//             dt,
//             di,
//         };
//         let layout = Tensor::new(dt, &[batch_size, di]).layout;

//         let lm = TestLayoutManager::from([(Arg::Gate, layout.clone()), (Arg::Up, layout)]);
//         let act = meta.build(&lm);

//         let mm = TestMemManagerLoader::new([Arg::Gate, Arg::Up], []).build();
//         act.launch(&mm);

//         println!("{mm}")
//     }
// }
