use crate::Context;
use vm::{Tensor, op::RoPE};

impl<VM, NN> Context<'_, VM, NN>
where
    VM: RoPE + ?Sized,
{
    pub fn rope(&self, x: &mut Tensor<VM>, pos: &Tensor<VM>, sin: &Tensor<VM>, cos: &Tensor<VM>) {
        self.vm().rope(self.stack(), x, pos, sin, cos)
    }
}
