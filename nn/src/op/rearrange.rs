use crate::Context;
use vm::{Tensor, op::Rearrange};

impl<VM, NN> Context<'_, VM, NN>
where
    VM: Rearrange + ?Sized,
{
    pub fn rearrange(&self, y: &mut Tensor<VM>, x: &Tensor<VM>) {
        self.vm().rearrange(self.stack(), y, x);
    }
}
