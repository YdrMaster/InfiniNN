use crate::{Context, VirtualMachine, tensor::Tensor};

pub trait Rearrange: VirtualMachine {
    fn rearrange(&self, y: &mut Tensor<Self::Blob>, x: &Tensor<Self::Blob>);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: Rearrange + ?Sized,
{
    pub fn rearrange(&self, y: &mut Tensor<VM::Blob>, x: &Tensor<VM::Blob>) {
        self.vm.rearrange(y, x);
    }
}
