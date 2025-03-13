use vm::{ObjId, Tensor, op::GeLU};

impl GeLU for crate::TestVM {
    fn gelu(&self, stack: ObjId, up: &mut Tensor<Self>) {
        assert_eq!(up.shape().len(), 2);

        self.launch(stack, format!("gelu(mut %{})", up.blob().id()))
    }
}
