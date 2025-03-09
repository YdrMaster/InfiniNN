use crate::{Context, Tensor, VirtualMachine};

pub trait MatMul: VirtualMachine {
    fn mat_mul(
        &self,
        c: &mut Tensor<Self>,
        beta: f32,
        a: &Tensor<Self>,
        b: &Tensor<Self>,
        alpha: f32,
    );
}

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
        self.vm.mat_mul(c, beta, a, b, alpha)
    }
}
