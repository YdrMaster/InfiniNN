use crate::Context;
use vm::{Tensor, op::SwiGLU};

impl<VM, NN> Context<'_, VM, NN>
where
    VM: SwiGLU + ?Sized,
{
    pub fn swiglu(&self, gate: &mut Tensor<VM>, up: &Tensor<VM>) {
        self.vm().swiglu(self.stack(), gate, up)
    }
}
