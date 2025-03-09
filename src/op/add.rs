use crate::{Context, Tensor, VirtualMachine};

pub trait Add: VirtualMachine {
    fn add(&self, y: &mut Tensor<Self>, x: &Tensor<Self>);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: Add + ?Sized,
{
    pub fn add(&mut self, y: &mut Tensor<VM>, x: &Tensor<VM>) {
        self.vm.add(y, x)
    }
}
