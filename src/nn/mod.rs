use crate::{Context, Id, VirtualMachine};

pub mod attention;
pub mod linear;
pub mod linear_residual;
pub mod mlp;
pub mod normalization;
pub mod self_attn;
pub mod transformer;

pub trait NuralNetwork<VM>: Sized
where
    VM: VirtualMachine + ?Sized,
{
    type Args<'vm>
    where
        VM: 'vm;
    type Obj: Id;
    type Sub: Id;

    fn launch(&self, args: Self::Args<'_>, ctx: Context<VM, Self>);
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WeightBias {
    Weight,
    Bias,
}

impl Id for WeightBias {
    fn name(&self) -> &str {
        match self {
            Self::Weight => "weight",
            Self::Bias => "bias",
        }
    }
}
