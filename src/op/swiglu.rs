use crate::{Context, VirtualMachine, tensor::Tensor};

pub trait SwiGLU: VirtualMachine {
    fn swiglu(&self, gate: &mut Tensor<Self::Blob>, up: &Tensor<Self::Blob>);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: SwiGLU + ?Sized,
{
    pub fn swiglu(&self, gate: &mut Tensor<VM::Blob>, up: &Tensor<VM::Blob>) {
        self.vm.swiglu(gate, up)
    }
}
