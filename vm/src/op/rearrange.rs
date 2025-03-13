use crate::{ObjId, Tensor, VirtualMachine};

pub trait Rearrange: VirtualMachine {
    fn rearrange(&self, stack: ObjId, y: &mut Tensor<Self>, x: &Tensor<Self>);
}
