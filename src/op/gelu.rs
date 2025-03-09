use crate::{Context, VirtualMachine, tensor::Tensor};

pub trait GeLU: VirtualMachine {
    fn gelu(&self, up: &mut Tensor<Self::Blob>);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: GeLU + ?Sized,
{
    pub fn gelu(&self, up: &mut Tensor<VM::Blob>) {
        self.vm.gelu(up)
    }
}
