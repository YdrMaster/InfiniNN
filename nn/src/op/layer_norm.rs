use crate::Context;
use vm::{Tensor, op::LayerNorm};

impl<VM, NN> Context<'_, VM, NN>
where
    VM: LayerNorm + ?Sized,
{
    pub fn layer_norm(&self, y: &mut Tensor<VM>, x: &Tensor<VM>, w: &Tensor<VM>, b: &Tensor<VM>) {
        self.vm().layer_norm(self.stack(), y, x, w, b)
    }
}
