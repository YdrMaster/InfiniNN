use crate::Context;
use vm::{Tensor, op::MatMul};

impl<VM, NN> Context<'_, VM, NN>
where
    VM: MatMul + ?Sized,
{
    pub fn mat_mul(
        &self,
        c: &mut Tensor<VM>,
        beta: f32,
        a: &Tensor<VM>,
        b: &Tensor<VM>,
        alpha: f32,
    ) {
        self.vm().mat_mul(self.stack(), c, beta, a, b, alpha)
    }
}
