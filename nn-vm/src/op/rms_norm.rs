use super::{Access, Operator};
use crate::VirtualMachine;
use std::marker::PhantomData;

pub trait RmsNorm<VM: VirtualMachine>: 'static + Sized {
    fn new() -> Self;
    fn launch(&self, y: &VM::Tensor, x: &VM::Tensor, scale: &VM::Tensor, epsilon: f32);

    fn op() -> Box<dyn Operator<VM>> {
        Box::new(RmsNormOp(Self::new(), PhantomData))
    }
}

pub struct RmsNormOp<VM: VirtualMachine, T: RmsNorm<VM>>(T, PhantomData<VM>);

pub const NAME: &str = "rms-norm";

impl<T: RmsNorm<VM>, VM: VirtualMachine> Operator<VM> for RmsNormOp<VM, T> {
    fn name(&self) -> String {
        NAME.to_string()
    }

    fn args(&self) -> &[super::Access] {
        // y, x, scale, bias
        &[Access::W, Access::R, Access::R]
    }

    fn launch(&self, tensors: &[&VM::Tensor], args: Box<dyn super::Args>) {
        let [y, x, scale] = tensors else {
            unreachable!()
        };
        let epsilon = *args.downcast_ref::<f32>().unwrap();
        self.0.launch(y, x, scale, epsilon)
    }
}
