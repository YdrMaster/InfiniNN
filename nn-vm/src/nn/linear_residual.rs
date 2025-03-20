use super::{NuralNetwork, WeightBias, data::Data};
use crate::{Context, Tensor, VirtualMachine, call, child, fetch_data, forward, gemm::Scale, init};

pub struct LinearResidual {
    weight: Data,
    bias: Option<Data>,
}

child!(LinearResidual[0] = weight: Data);
child!(LinearResidual[1] = bias  : Data);

pub struct Args<VM: VirtualMachine> {
    pub y: VM::Tensor,
    pub x: VM::Tensor,
    pub scale: f32,
    pub residual: bool,
}

impl<VM: VirtualMachine> NuralNetwork<VM> for LinearResidual {
    const NAME: &str = "linear-residual";
    type Meta = ();
    type Args = Args<VM>;
    type Data = WeightBias<VM>;

    fn init(_: &Self::Meta, data: Self::Data, mut ctx: Context<VM, Self>) -> Self {
        let Self::Data { weight, bias } = data;
        let weight = init!(0: None, (), weight; ctx);
        let bias = bias.map(|bias| init!(1: None, (), bias; ctx));
        Self { weight, bias }
    }

    fn forward(&self, args: Self::Args, mut ctx: Context<VM, Self>) {
        let Self { weight, bias } = self;
        let Args {
            y,
            x,
            scale,
            residual,
        } = args;

        if let Some(bias) = bias {
            let y_ = if residual {
                ctx.tensor(y.meta())
            } else {
                y.clone()
            };
            {
                let b = fetch_data!(1: bias; ctx);
                call!(rearrange: [&y_, &b]; ctx)
            }
            {
                let w = fetch_data!(0: weight; ctx);
                call!(gemm: [&y_, &x, &w], Scale{alpha: scale, beta: scale}; ctx)
            }
            if residual {
                call!(add: [&y, &y, &y_]; ctx)
            }
        } else {
            let w = fetch_data!(0: weight; ctx);
            call!(gemm: [&y, &x, &w], Scale{alpha: scale, beta: if residual { 1. } else { 0. }}; ctx)
        }
    }
}
