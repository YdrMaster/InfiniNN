use crate::Context;
use vm::{Tensor, op::TokenEmbed};

impl<VM, NN> Context<'_, VM, NN>
where
    VM: TokenEmbed + ?Sized,
{
    pub fn token_embed(&self, embd: &mut Tensor<VM>, tok: &Tensor<VM>, table: &Tensor<VM>) {
        self.vm().token_embed(self.stack(), embd, tok, table)
    }
}
