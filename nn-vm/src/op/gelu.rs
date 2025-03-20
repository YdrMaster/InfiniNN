use super::{Access, Args, Empty, Operator};
use crate::VirtualMachine;
use std::marker::PhantomData;

pub trait Gelu<VM: VirtualMachine>: 'static + Sized {
    fn new() -> Self;
    fn launch(&self, y: &VM::Tensor, x: &VM::Tensor);

    fn op() -> Box<dyn Operator<VM>> {
        Box::new(GeluOp(Self::new(), PhantomData))
    }
}

pub struct GeluOp<VM: VirtualMachine, T: Gelu<VM>>(T, PhantomData<VM>);

pub const NAME: &str = "swiglu";

impl<T: Gelu<VM>, VM: VirtualMachine> Operator<VM> for GeluOp<VM, T> {
    fn name(&self) -> String {
        NAME.to_string()
    }

    fn args(&self) -> &[super::Access] {
        // y, x
        &[Access::W, Access::R]
    }

    fn launch(&self, tensors: &[&VM::Tensor], args: Box<dyn Args>) {
        let [y, x] = tensors else { unreachable!() };
        args.downcast_ref::<Empty>().unwrap();
        self.0.launch(y, x)
    }
}
