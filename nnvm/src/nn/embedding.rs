use super::NeuralNetwork;
use crate::{
    Context, NNTensor, VirtualMachine,
    op::embedding::{Backward, Forward},
};

pub struct Embedding;

impl<VM: VirtualMachine> NeuralNetwork<VM> for Embedding {
    type Init = (NNTensor<VM>, Option<NNTensor<VM>>);
    type Args = ();

    fn init(init: Self::Init, ctx: &mut Context<VM>) {
        let (wte, wpe) = init;
        ctx.save_weight("wte", wte);
        if let Some(wpe) = wpe {
            ctx.save_weight("wpe", wpe)
        }
    }

    fn forward(
        (): &Self::Args,
        inputs: impl IntoIterator<Item = NNTensor<VM>>,
        ctx: &mut Context<VM>,
    ) -> Vec<NNTensor<VM>> {
        let mut inputs = inputs.into_iter();
        let tokens = inputs.next().unwrap();
        let pos = inputs.next();
        let wte = ctx.load_weight("wte").unwrap();
        let wpe = ctx.load_weight("wpe");

        if let Some(mut backward) = ctx.take_backward_builder() {
            let dwte = wte.grad();
            let tokens_save = tokens.save();

            if let Some(wpe) = wpe {
                let pos = pos.unwrap();

                let dwpe = wpe.grad();
                let pos_save = pos.save();

                let outputs = ctx.call::<Forward>((), [wte, tokens, wpe, pos]);
                let dy = outputs[0].grad();

                backward.call::<Backward>((), [dwte, dwpe, dy, tokens_save, pos_save], []);
                ctx.put_backward_builder(backward);

                outputs
            } else {
                let outputs = ctx.call::<Forward>((), [wte, tokens]);
                let dy = outputs[0].grad();

                backward.call::<Backward>((), [dwte, dy, tokens_save], []);
                ctx.put_backward_builder(backward);

                outputs
            }
        } else {
            if let Some(wpe) = wpe {
                let pos = pos.unwrap();

                ctx.call::<Forward>((), [wte, tokens, wpe, pos])
            } else {
                ctx.call::<Forward>((), [wte, tokens])
            }
        }
    }
}
