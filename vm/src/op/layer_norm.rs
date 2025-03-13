use crate::{ObjId, Tensor, VirtualMachine};

pub trait LayerNorm: VirtualMachine {
    fn layer_norm(
        &self,
        stack: ObjId,
        y: &mut Tensor<Self>,
        x: &Tensor<Self>,
        w: &Tensor<Self>,
        b: &Tensor<Self>,
    );
}
