use crate::{Context, Tensor, VirtualMachine};

pub trait GeLU: VirtualMachine {
    fn gelu(&self, up: &mut Tensor<Self>);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: GeLU + ?Sized,
{
    pub fn gelu(&self, up: &mut Tensor<VM>) {
        self.vm.gelu(up)
    }
}
