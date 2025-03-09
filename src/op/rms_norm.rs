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

#[cfg(test)]
impl RmsNorm for crate::test::TestVM {
    fn rms_norm(&self, y: &mut Tensor<Self>, x: &Tensor<Self>, w: &Tensor<Self>, epsilon: f32) {
        assert_eq!(y.dt(), x.dt());
        assert_eq!(y.shape(), x.shape());

        self.launch(format!(
            "rms_norm(mut %{}, %{}, %{}, {epsilon:.2e})",
            y.blob().id(),
            x.blob().id(),
            w.blob().id(),
        ))
    }
}
