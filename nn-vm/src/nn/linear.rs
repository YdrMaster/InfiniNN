use super::{NuralNetwork, WeightBias, data::Data};
use crate::{
    Context, Tensor, VirtualMachine, call, child, fetch_data, forward, gemm::Scale, init, shape,
};

pub struct Linear {
    weight: Data,
    bias: Option<Data>,
}

child!(Linear[0] = weight: Data);
child!(Linear[1] = bias  : Data);

pub struct Args<VM: VirtualMachine> {
    pub y: VM::Tensor,
    pub x: VM::Tensor,
    pub scale: f32,
}

impl<VM> NuralNetwork<VM> for Linear
where
    VM: VirtualMachine,
{
    const NAME: &str = "linear";
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
        let Args { y, x, scale } = args;

        let beta = bias.as_ref().map_or(0., |bias| {
            let b = fetch_data!(1: bias; ctx);

            shape!(x => [n, _]);
            shape!(b => [di]);

            let b = b.tile(0, &[1, di]).broadcast(0, n);
            call!(rearrange: [&y, &b]; ctx);
            1.
        });

        let w = fetch_data!(0: weight; ctx);
        call!(gemm: [&y, &x, &w], Scale { alpha: scale, beta }; ctx)
    }
}
