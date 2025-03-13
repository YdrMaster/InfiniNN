use crate::{ObjId, Tensor, VirtualMachine};

pub trait RmsNorm: VirtualMachine {
    fn rms_norm(
        &self,
        stack: ObjId,
        y: &mut Tensor<Self>,
        x: &Tensor<Self>,
        w: &Tensor<Self>,
        epsilon: f32,
    );
}
