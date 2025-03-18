use super::NuralNetwork;
use crate::{Context, Tensor, VirtualMachine};

pub struct Data;

impl<VM> NuralNetwork<VM> for Data
where
    VM: VirtualMachine,
{
    const NAME: &str = "data";
    type Meta = ();
    type Args = VM::Tensor;
    type Data = VM::Tensor;

    fn init(_: &Self::Meta, tensor: Self::Data, ctx: Context<VM, Self>) -> Self {
        ctx.save_data(tensor);
        Self
    }

    fn forward(&self, tensor: Self::Args, ctx: Context<VM, Self>) {
        tensor.assign(ctx.load_data())
    }
}
