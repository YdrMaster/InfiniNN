use super::{Access, Args, Empty, Operator};
use crate::VirtualMachine;
use std::marker::PhantomData;

pub trait Add<VM: VirtualMachine>: 'static + Sized {
    fn new() -> Self;
    fn launch(&self, c: &VM::Tensor, a: &VM::Tensor, b: &VM::Tensor);

    fn op() -> Box<dyn Operator<VM>> {
        Box::new(AddOp(Self::new(), PhantomData))
    }
}

pub struct AddOp<VM: VirtualMachine, T: Add<VM>>(T, PhantomData<VM>);

pub const NAME: &str = "add";

impl<T: Add<VM>, VM: VirtualMachine> Operator<VM> for AddOp<VM, T> {
    fn name(&self) -> String {
        NAME.to_string()
    }

    fn args(&self) -> &[super::Access] {
        // c, a, b
        &[Access::W, Access::R, Access::R]
    }

    fn launch(&self, tensors: &[&VM::Tensor], args: Box<dyn Args>) {
        let [c, a, b] = tensors else { unreachable!() };
        args.downcast_ref::<Empty>().unwrap();
        self.0.launch(c, a, b)
    }
}
