use crate::{ObjId, Tensor, VirtualMachine};

pub trait Add: VirtualMachine {
    fn add(&self, stack: ObjId, y: &mut Tensor<Self>, x: &Tensor<Self>);
}
