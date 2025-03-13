use crate::{ObjId, Tensor, VirtualMachine};

pub trait SwiGLU: VirtualMachine {
    fn swiglu(&self, stack: ObjId, gate: &mut Tensor<Self>, up: &Tensor<Self>);
}
