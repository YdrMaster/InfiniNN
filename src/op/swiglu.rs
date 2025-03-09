use crate::{Context, Tensor, VirtualMachine};

pub trait SwiGLU: VirtualMachine {
    fn swiglu(&self, gate: &mut Tensor<Self>, up: &Tensor<Self>);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: SwiGLU + ?Sized,
{
    pub fn swiglu(&self, gate: &mut Tensor<VM>, up: &Tensor<VM>) {
        self.vm.swiglu(gate, up)
    }
}

#[cfg(test)]
impl SwiGLU for crate::test::TestVM {
    fn swiglu(&self, gate: &mut Tensor<Self>, up: &Tensor<Self>) {
        assert_eq!(gate.dt(), up.dt());
        assert_eq!(gate.shape(), up.shape());
        assert_eq!(gate.shape().len(), 2);

        self.launch(format!(
            "swiglu(mut %{}, %{})",
            gate.blob().id(),
            up.blob().id()
        ))
    }
}
