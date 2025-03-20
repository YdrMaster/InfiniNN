use super::{Access, Args, Empty, Operator};
use crate::VirtualMachine;
use std::marker::PhantomData;

pub trait Swiglu<VM: VirtualMachine>: 'static + Sized {
    fn new() -> Self;
    fn launch(&self, out: &VM::Tensor, gate: &VM::Tensor, up: &VM::Tensor);

    fn op() -> Box<dyn Operator<VM>> {
        Box::new(SwigluOp(Self::new(), PhantomData))
    }
}

pub struct SwigluOp<VM: VirtualMachine, T: Swiglu<VM>>(T, PhantomData<VM>);

pub const NAME: &str = "swiglu";

impl<T: Swiglu<VM>, VM: VirtualMachine> Operator<VM> for SwigluOp<VM, T> {
    fn name(&self) -> String {
        NAME.to_string()
    }

    fn args(&self) -> &[super::Access] {
        // out, gate, up
        &[Access::W, Access::R, Access::R]
    }

    fn launch(&self, tensors: &[&VM::Tensor], args: Box<dyn Args>) {
        let [out, gate, up] = tensors else {
            unreachable!()
        };
        args.downcast_ref::<Empty>().unwrap();
        self.0.launch(out, gate, up)
    }
}
