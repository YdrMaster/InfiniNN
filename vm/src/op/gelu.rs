use crate::{ObjId, Tensor, VirtualMachine};

pub trait GeLU: VirtualMachine {
    fn gelu(&self, stack: ObjId, up: &mut Tensor<Self>);
}
