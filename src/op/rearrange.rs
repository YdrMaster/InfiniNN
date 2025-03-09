use crate::{Context, Tensor, VirtualMachine};

pub trait Rearrange: VirtualMachine {
    fn rearrange(&self, y: &mut Tensor<Self>, x: &Tensor<Self>);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: Rearrange + ?Sized,
{
    pub fn rearrange(&self, y: &mut Tensor<VM>, x: &Tensor<VM>) {
        self.vm.rearrange(y, x);
    }
}
