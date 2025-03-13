use crate::{ObjId, Tensor, VirtualMachine};

pub trait RoPE: VirtualMachine {
    fn rope(
        &self,
        stack: ObjId,
        x: &mut Tensor<Self>,
        pos: &Tensor<Self>,
        sin: &Tensor<Self>,
        cos: &Tensor<Self>,
    );
}
