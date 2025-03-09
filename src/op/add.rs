use crate::{Context, VirtualMachine, tensor::Tensor};

pub trait Add: VirtualMachine {
    fn add(&self, y: &mut Tensor<Self::Blob>, x: &Tensor<Self::Blob>);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: Add + ?Sized,
{
    pub fn add(&mut self, y: &mut Tensor<VM::Blob>, x: &Tensor<VM::Blob>) {
        self.vm.add(y, x)
    }
}
