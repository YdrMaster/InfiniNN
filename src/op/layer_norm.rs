use crate::{Context, Tensor, VirtualMachine};

pub trait LayerNorm: VirtualMachine {
    fn layer_norm(
        &self,
        y: &mut Tensor<Self>,
        x: &Tensor<Self>,
        w: &Tensor<Self>,
        b: &Tensor<Self>,
    );
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: LayerNorm + ?Sized,
{
    pub fn layer_norm(
        &mut self,
        y: &mut Tensor<VM>,
        x: &Tensor<VM>,
        w: &Tensor<VM>,
        b: &Tensor<VM>,
    ) {
        self.vm.layer_norm(y, x, w, b)
    }
}
