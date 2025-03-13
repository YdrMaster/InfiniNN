use crate::{ObjId, Tensor, VirtualMachine};

pub trait TokenEmbed: VirtualMachine {
    fn token_embed(
        &self,
        stack: ObjId,
        embd: &mut Tensor<Self>,
        tok: &Tensor<Self>,
        table: &Tensor<Self>,
    );
}
