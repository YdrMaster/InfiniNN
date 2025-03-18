use super::{WeightBias, WithChild, data::Data};
use crate::{Context, NuralNetwork, VirtualMachine, call, forward_child, init_child, op::Empty};

pub struct Normalization {
    meta: Meta,
    scale: Data,
    bias: Option<Data>,
}

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
        let scale = init_child!(0: None, (), weight; ctx);
        let bias = bias.map(|bias| init_child!(1: None, (), bias; ctx));
        Self {
            meta: *meta,
            scale,
            bias,
        }
    }

    fn forward(&self, args: Self::Args, mut ctx: Context<VM, Self>) {
        let Self { meta, scale, bias } = self;
        let Args { y, x } = args;

        let scale_ = ctx.tensor(None);
        forward_child!(0: None, scale, scale_.clone(); ctx);

        match *meta {
            Meta::RmsNorm { epsilon } => {
                call!(rms_norm: [y, x, scale_], epsilon; ctx)
            }
            Meta::LayerNorm => {
                let bias_ = ctx.tensor(None);
                forward_child!(1: None, bias.as_ref().unwrap(), bias_.clone(); ctx);
                call!(layer_norm: [y, x, scale_, bias_], Empty; ctx)
            }
        }
    }
}

impl<VM> WithChild<VM, 0> for Normalization
where
    VM: VirtualMachine,
{
    type Type = Data;
    const NAME: &str = "scale";
}

impl<VM> WithChild<VM, 1> for Normalization
where
    VM: VirtualMachine,
{
    type Type = Data;
    const NAME: &str = "bias";
}
