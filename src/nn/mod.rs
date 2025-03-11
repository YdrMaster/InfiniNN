use crate::{Context, Id, VirtualMachine};

pub mod attention;
pub mod linear;
pub mod mlp;
pub mod normalization;
pub mod self_attn;

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
