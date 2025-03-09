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
