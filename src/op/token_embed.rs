use crate::{Context, ObjId, Tensor, VirtualMachine};

pub trait TokenEmbed: VirtualMachine {
    fn token_embed(
        &self,
        stack: ObjId,
        embd: &mut Tensor<Self>,
        tok: &Tensor<Self>,
        table: &Tensor<Self>,
    );
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: TokenEmbed + ?Sized,
{
    pub fn token_embed(&self, embd: &mut Tensor<VM>, tok: &Tensor<VM>, table: &Tensor<VM>) {
        self.vm().token_embed(self.stack(), embd, tok, table)
    }
}

#[cfg(test)]
impl TokenEmbed for crate::test::TestVM {
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
