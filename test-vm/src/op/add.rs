use vm::{ObjId, Tensor, op::Add};

impl Add for crate::TestVM {
    fn add(&self, stack: ObjId, y: &mut Tensor<Self>, x: &Tensor<Self>) {
        assert_eq!(y.dt(), x.dt());
        assert_eq!(y.shape(), x.shape());

        self.launch(
            stack,
            format!("add(mut %{}, %{})", y.blob().id(), x.blob().id(),),
        )
    }
}
