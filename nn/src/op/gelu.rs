use crate::Context;
use vm::{Tensor, op::GeLU};

impl<VM, NN> Context<'_, VM, NN>
where
    VM: GeLU + ?Sized,
{
    pub fn gelu(&self, up: &mut Tensor<VM>) {
        self.vm().gelu(self.stack(), up)
    }
}
