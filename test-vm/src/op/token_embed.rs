use vm::{ObjId, Tensor, op::TokenEmbed};

impl TokenEmbed for crate::TestVM {
    fn token_embed(
        &self,
        stack: ObjId,
        embd: &mut Tensor<Self>,
        tok: &Tensor<Self>,
        table: &Tensor<Self>,
    ) {
        assert_eq!(embd.dt(), table.dt());
        let [n, d] = embd.shape() else { panic!() };
        let [n_] = tok.shape() else { panic!() };
        let [_, d_] = table.shape() else { panic!() };
        assert_eq!(n, n_);
        assert_eq!(d, d_);

        self.launch(
            stack,
            format!(
                "token_embed(mut %{}, %{}, %{})",
                embd.blob().id(),
                tok.blob().id(),
                table.blob().id(),
            ),
        )
    }
}
