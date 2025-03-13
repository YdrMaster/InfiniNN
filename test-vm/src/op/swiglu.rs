use vm::{ObjId, Tensor, op::SwiGLU};

impl SwiGLU for crate::TestVM {
    fn swiglu(&self, stack: ObjId, gate: &mut Tensor<Self>, up: &Tensor<Self>) {
        assert_eq!(gate.dt(), up.dt());
        assert_eq!(gate.shape(), up.shape());
        assert_eq!(gate.shape().len(), 2);

        self.launch(
            stack,
            format!("swiglu(mut %{}, %{})", gate.blob().id(), up.blob().id()),
        )
    }
}
