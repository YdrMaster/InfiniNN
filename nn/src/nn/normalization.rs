use super::{WeightBias, data::Data};
use crate::{Context, NuralNetwork, VirtualMachine, call, child, fetch_data, forward, init};

pub enum Normalization {
    RmsNorm { scale: Data, epsilon: f32 },
    LayerNorm { scale: Data, bias: Data },
}

child!(Normalization[0] = scale: Data);
child!(Normalization[1] = bias : Data);

#[derive(Clone, Copy)]
pub enum Meta {
    RmsNorm { epsilon: f32 },
    LayerNorm,
}

pub struct Args<VM: VirtualMachine> {
    pub y: VM::Tensor,
    pub x: VM::Tensor,
}

impl<VM> NuralNetwork<VM> for Normalization
where
    VM: VirtualMachine,
{
    const NAME: &str = "normalization";
    type Meta = Meta;
    type Args = Args<VM>;
    type Data = WeightBias<VM>;

    fn init(meta: &Self::Meta, data: Self::Data, mut ctx: Context<VM, Self>) -> Self {
        let Self::Data { weight, bias } = data;
        match *meta {
            Meta::RmsNorm { epsilon } => {
                let scale = init!(0: None, (), weight; ctx);
                Self::RmsNorm { scale, epsilon }
            }
            Meta::LayerNorm => {
                let scale = init!(0: None, (), weight; ctx);
                let bias = init!(1: None, (), bias.unwrap(); ctx);
                Self::LayerNorm { scale, bias }
            }
        }
    }

    fn forward(&self, args: Self::Args, mut ctx: Context<VM, Self>) {
        let Args { y, x } = args;

        match self {
            Self::RmsNorm { scale, epsilon } => {
                let scale = fetch_data!(0: scale; ctx);
                call!(rms_norm: [&y, &x, &scale], *epsilon; ctx)
            }
            Self::LayerNorm { scale, bias } => {
                let scale = fetch_data!(0: scale; ctx);
                let bias = fetch_data!(1: bias; ctx);
                call!(layer_norm: [&y, &x, &scale, &bias]; ctx)
            }
        }
    }
}
