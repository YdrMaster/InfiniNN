use crate::Context;
use vm::{Tensor, op::RmsNorm};

impl<VM, NN> Context<'_, VM, NN>
where
    VM: RmsNorm + ?Sized,
{
    pub fn rms_norm(&self, y: &mut Tensor<VM>, x: &Tensor<VM>, w: &Tensor<VM>, epsilon: f32) {
        self.vm().rms_norm(self.stack(), y, x, w, epsilon)
    }
}
