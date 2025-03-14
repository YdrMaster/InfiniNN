use super::ptr;
use crate::CpuVM;
use operators::{
    Operator, TensorLayout,
    common_cpu::{Cpu, ThisThread},
    mat_mul::{self, common_cpu::Operator as MatMulCpu},
};
use vm::{ObjId, Tensor, op::MatMul};

impl MatMul for crate::CpuVM {
    fn mat_mul(
        &self,
        _stack: ObjId,
        c: &mut Tensor<Self>,
        beta: f32,
        a: &Tensor<Self>,
        b: &Tensor<Self>,
        alpha: f32,
    ) {
        MatMulCpu::new(&Cpu)
            .launch(
                &mat_mul::Args {
                    c_layout: layout(c),
                    c_base: ptr(c).cast_mut(),
                    beta,
                    a_layout: layout(a),
                    a_base: ptr(a),
                    b_layout: layout(b),
                    b_base: ptr(b),
                    alpha,
                },
                &mut [],
                &ThisThread,
            )
            .unwrap()
    }
}

fn layout(tensor: &Tensor<CpuVM>) -> TensorLayout {
    TensorLayout::new(tensor.dt(), tensor.shape(), tensor.strides())
}
