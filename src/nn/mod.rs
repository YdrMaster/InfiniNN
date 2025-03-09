use crate::{Context, Id, VirtualMachine};

pub mod activation;
// pub mod attention;
pub mod mlp;
// pub mod normalization;

pub trait NuralNetwork<VM>: Sized
where
    VM: VirtualMachine + ?Sized,
{
    type Args;
    type Weight: Id;
    type Sub: Id;

    fn launch(ctx: Context<VM, Self>, args: Self::Args);
}
