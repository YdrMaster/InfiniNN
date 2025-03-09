use crate::{Context, VirtualMachine, tensor::Tensor};

pub trait MatMul: VirtualMachine {
    fn mat_mul(
        &self,
        c: &mut Tensor<Self::Blob>,
        beta: f32,
        a: &Tensor<Self::Blob>,
        b: &Tensor<Self::Blob>,
        alpha: f32,
    );
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: MatMul + ?Sized,
{
    pub fn mat_mul(
        &self,
        c: &mut Tensor<VM::Blob>,
        beta: f32,
        a: &Tensor<VM::Blob>,
        b: &Tensor<VM::Blob>,
        alpha: f32,
    ) {
        self.vm.mat_mul(c, beta, a, b, alpha)
    }
}
