use crate::Context;
use vm::{Tensor, op::Add};

impl<VM, NN> Context<'_, VM, NN>
where
    VM: Add + ?Sized,
{
    pub fn add(&self, y: &mut Tensor<VM>, x: &Tensor<VM>) {
        self.vm().add(self.stack(), y, x)
    }
}
