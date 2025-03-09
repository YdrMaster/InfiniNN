use crate::{Context, VirtualMachine, tensor::Tensor};

pub trait LayerNorm: VirtualMachine {
    fn layer_norm(
        &self,
        y: &mut Tensor<Self::Blob>,
        x: &Tensor<Self::Blob>,
        w: &Tensor<Self::Blob>,
        b: &Tensor<Self::Blob>,
    );
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: LayerNorm + ?Sized,
{
    pub fn layer_norm(
        &mut self,
        y: &mut Tensor<VM::Blob>,
        x: &Tensor<VM::Blob>,
        w: &Tensor<VM::Blob>,
        b: &Tensor<VM::Blob>,
    ) {
        self.vm.layer_norm(y, x, w, b)
    }
}
