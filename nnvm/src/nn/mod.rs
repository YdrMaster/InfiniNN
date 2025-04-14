mod embedding;

use crate::{Context, NNTensor, VirtualMachine};

pub trait NeuralNetwork<VM: VirtualMachine> {
    type Init;
    type Args;

    fn init(init: Self::Init, ctx: &mut Context<VM>);

    fn forward(
        args: &Self::Args,
        inputs: impl IntoIterator<Item = NNTensor<VM>>,
        ctx: &mut Context<VM>,
    ) -> Vec<NNTensor<VM>>;
}
