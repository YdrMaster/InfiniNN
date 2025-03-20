use super::{Access, Args, Operator};
use crate::VirtualMachine;
use std::{fmt, marker::PhantomData};

pub trait Gemm<VM: VirtualMachine>: 'static + Sized {
    fn new() -> Self;
    fn launch(&self, c: &VM::Tensor, beta: f32, a: &VM::Tensor, b: &VM::Tensor, alpha: f32);

    fn op() -> Box<dyn Operator<VM>> {
        Box::new(GemmOp(Self::new(), PhantomData))
    }
}

pub struct GemmOp<VM: VirtualMachine, T: Gemm<VM>>(T, PhantomData<VM>);

pub struct Scale {
    pub alpha: f32,
    pub beta: f32,
}
impl Args for Scale {}
impl fmt::Display for Scale {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Self { alpha, beta } = self;
        writeln!(f, "α: {alpha}, β: {beta}")
    }
}

pub const NAME: &str = "gemm";

impl<T: Gemm<VM>, VM: VirtualMachine> Operator<VM> for GemmOp<VM, T> {
    fn name(&self) -> String {
        NAME.to_string()
    }

    fn args(&self) -> &[super::Access] {
        // c, a, b
        &[Access::RW, Access::R, Access::R]
    }

    fn launch(&self, tensors: &[&VM::Tensor], args: Box<dyn Args>) {
        let [c, a, b] = tensors else { unreachable!() };
        let &Scale { alpha, beta } = args.downcast_ref::<Scale>().unwrap();
        self.0.launch(c, beta, a, b, alpha)
    }
}
