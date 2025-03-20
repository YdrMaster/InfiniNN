use super::{Access, Operator};
use crate::{VirtualMachine, op::Empty};
use std::marker::PhantomData;

pub trait LayerNorm<VM: VirtualMachine>: 'static + Sized {
    fn new() -> Self;
    fn launch(&self, y: &VM::Tensor, x: &VM::Tensor, scale: &VM::Tensor, bias: &VM::Tensor);

    fn op() -> Box<dyn Operator<VM>> {
        Box::new(LayerNormOp(Self::new(), PhantomData))
    }
}

pub struct LayerNormOp<VM: VirtualMachine, T: LayerNorm<VM>>(T, PhantomData<VM>);

pub const NAME: &str = "layer-norm";

impl<T: LayerNorm<VM>, VM: VirtualMachine> Operator<VM> for LayerNormOp<VM, T> {
    fn name(&self) -> String {
        NAME.to_string()
    }

    fn args(&self) -> &[super::Access] {
        // y, x, scale, bias
        &[Access::W, Access::R, Access::R, Access::R]
    }

    fn launch(&self, tensors: &[&VM::Tensor], args: Box<dyn super::Args>) {
        let [y, x, scale, bias] = tensors else {
            unreachable!()
        };
        args.downcast_ref::<Empty>().unwrap();
        self.0.launch(y, x, scale, bias)
    }
}
