use vm::{ObjId, Tensor, op::GeLU};

impl GeLU for crate::CpuVM {
    fn gelu(&self, _stack: ObjId, up: &mut Tensor<Self>) {
        assert_eq!(up.shape().len(), 2);

        todo!()
    }
}
