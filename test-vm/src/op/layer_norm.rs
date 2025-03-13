use vm::{ObjId, Tensor, op::LayerNorm};

impl LayerNorm for crate::TestVM {
    fn layer_norm(
        &self,
        stack: ObjId,
        y: &mut Tensor<Self>,
        x: &Tensor<Self>,
        w: &Tensor<Self>,
        b: &Tensor<Self>,
    ) {
        assert_eq!(y.dt(), x.dt());
        assert_eq!(y.shape(), x.shape());
        assert_eq!(w.dt(), b.dt());

        self.launch(
            stack,
            format!(
                "layer_norm(mut %{}, %{}, %{}, %{})",
                y.blob().id(),
                x.blob().id(),
                w.blob().id(),
                b.blob().id(),
            ),
        )
    }
}
