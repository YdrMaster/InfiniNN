use crate::{Context, Tensor, VirtualMachine};

pub trait RmsNorm: VirtualMachine {
    fn rms_norm(&self, y: &mut Tensor<Self>, x: &Tensor<Self>, w: &Tensor<Self>, epsilon: f32);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: RmsNorm + ?Sized,
{
    pub fn rms_norm(&mut self, y: &mut Tensor<VM>, x: &Tensor<VM>, w: &Tensor<VM>, epsilon: f32) {
        self.vm.rms_norm(y, x, w, epsilon)
    }
}
