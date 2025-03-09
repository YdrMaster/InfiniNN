use crate::{Context, VirtualMachine, tensor::Tensor};

pub trait RmsNorm: VirtualMachine {
    fn rms_norm(
        &self,
        y: &mut Tensor<Self::Blob>,
        x: &Tensor<Self::Blob>,
        w: &Tensor<Self::Blob>,
        epsilon: f32,
    );
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: RmsNorm + ?Sized,
{
    pub fn rms_norm(
        &mut self,
        y: &mut Tensor<VM::Blob>,
        x: &Tensor<VM::Blob>,
        w: &Tensor<VM::Blob>,
        epsilon: f32,
    ) {
        self.vm.rms_norm(y, x, w, epsilon)
    }
}
