use crate::{ObjId, Tensor, VirtualMachine};

pub trait MatMul: VirtualMachine {
    fn mat_mul(
        &self,
        stack: ObjId,
        c: &mut Tensor<Self>,
        beta: f32,
        a: &Tensor<Self>,
        b: &Tensor<Self>,
        alpha: f32,
    );
}
