use super::{Access, Args, Empty, Operator};
use crate::VirtualMachine;
use std::marker::PhantomData;

pub trait Rearrange<VM: VirtualMachine>: 'static + Sized {
    fn new() -> Self;
    fn launch(&self, dst: &VM::Tensor, src: &VM::Tensor);

    fn op() -> Box<dyn Operator<VM>> {
        Box::new(RearrangeOp(Self::new(), PhantomData))
    }
}

pub struct RearrangeOp<VM: VirtualMachine, T: Rearrange<VM>>(T, PhantomData<VM>);

pub const NAME: &str = "rearrange";

impl<T: Rearrange<VM>, VM: VirtualMachine> Operator<VM> for RearrangeOp<VM, T> {
    fn name(&self) -> String {
        NAME.to_string()
    }

    fn args(&self) -> &[super::Access] {
        // dst <- src
        &[Access::W, Access::R]
    }

    fn launch(&self, tensors: &[&VM::Tensor], args: Box<dyn Args>) {
        let [dst, src] = tensors else { unreachable!() };
        args.downcast_ref::<Empty>().unwrap();
        self.0.launch(dst, src)
    }
}
