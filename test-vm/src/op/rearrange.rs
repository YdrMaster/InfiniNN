use vm::{ObjId, Tensor, op::Rearrange};

impl Rearrange for crate::TestVM {
    fn rearrange(&self, stack: ObjId, y: &mut Tensor<Self>, x: &Tensor<Self>) {
        assert_eq!(y.dt(), x.dt());
        assert_eq!(y.shape(), x.shape());

        self.launch(
            stack,
            format!("rearrange(mut %{}, %{})", y.blob().id(), x.blob().id(),),
        )
    }
}
