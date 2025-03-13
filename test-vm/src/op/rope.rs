use vm::{ObjId, Tensor, op::RoPE};

impl RoPE for crate::TestVM {
    fn rope(
        &self,
        stack: ObjId,
        x: &mut Tensor<Self>,
        pos: &Tensor<Self>,
        sin: &Tensor<Self>,
        cos: &Tensor<Self>,
    ) {
        let &[_, seq, dh] = x.shape() else { panic!() };
        let &[seq_] = pos.shape() else { panic!() };
        let &[_, dh_sin] = sin.shape() else { panic!() };
        let &[_, dh_cos] = sin.shape() else { panic!() };
        assert_eq!(seq, seq_);
        assert_eq!(dh, dh_sin * 2);
        assert_eq!(dh, dh_cos * 2);

        self.launch(
            stack,
            format!(
                "rope(mut %{}, %{}, %{}, %{})",
                x.blob().id(),
                pos.blob().id(),
                sin.blob().id(),
                cos.blob().id(),
            ),
        )
    }
}
