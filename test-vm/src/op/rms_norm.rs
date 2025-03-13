use vm::{ObjId, Tensor, op::RmsNorm};

impl RmsNorm for crate::TestVM {
    fn rms_norm(
        &self,
        stack: ObjId,
        y: &mut Tensor<Self>,
        x: &Tensor<Self>,
        w: &Tensor<Self>,
        epsilon: f32,
    ) {
        assert_eq!(y.dt(), x.dt());
        assert_eq!(y.shape(), x.shape());

        self.launch(
            stack,
            format!(
                "rms_norm(mut %{}, %{}, %{}, {epsilon:.2e})",
                y.blob().id(),
                x.blob().id(),
                w.blob().id(),
            ),
        )
    }
}
