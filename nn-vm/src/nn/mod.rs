use crate::{Context, VirtualMachine};

pub mod data;
pub mod normalization;

pub trait NuralNetwork<VM: VirtualMachine>: Sized {
    const NAME: &str;
    type Meta;
    type Data;
    type Args;

    fn init(meta: &Self::Meta, data: Self::Data, ctx: Context<VM, Self>) -> Self;
    fn forward(&self, args: Self::Args, ctx: Context<VM, Self>);
}

pub trait WithChild<VM, const ID: usize>: NuralNetwork<VM>
where
    VM: VirtualMachine,
{
    type Type: NuralNetwork<VM>;
    const NAME: &str;
    const LOOP: bool = false;

    fn init_child(
        loop_idx: Option<usize>,
        meta: &<Self::Type as NuralNetwork<VM>>::Meta,
        data: <Self::Type as NuralNetwork<VM>>::Data,
        ctx: &mut Context<VM, Self>,
    ) -> Self::Type {
        assert_eq!(Self::LOOP, loop_idx.is_some());
        let ctx = ctx.trap(ID, loop_idx, <Self as WithChild<VM, ID>>::NAME);
        Self::Type::init(meta, data, ctx)
    }

    fn forward_child(
        loop_idx: Option<usize>,
        child: &Self::Type,
        args: <Self::Type as NuralNetwork<VM>>::Args,
        ctx: &mut Context<VM, Self>,
    ) {
        assert_eq!(Self::LOOP, loop_idx.is_some());
        let ctx = ctx.trap(ID, loop_idx, <Self as WithChild<VM, ID>>::NAME);
        child.forward(args, ctx)
    }
}

pub struct WeightBias<VM: VirtualMachine> {
    pub weight: VM::Tensor,
    pub bias: Option<VM::Tensor>,
}

#[macro_export]
macro_rules! init_child {
    ($child_id:literal: $loop_idx:expr, $meta:expr, $data:expr; $ctx:expr) => {
        <Self as WithChild<VM, $child_id>>::init_child($loop_idx, &$meta, $data, &mut $ctx)
    };
}

#[macro_export]
macro_rules! forward_child {
    ($child_id:literal: $loop_idx:expr, $child:expr, $args:expr; $ctx:expr) => {
        <Self as WithChild<VM, $child_id>>::forward_child($loop_idx, &$child, $args, &mut $ctx)
    };
}

#[macro_export]
macro_rules! call {
    ($op:ident: $tensors:expr, $args:expr; $ctx:expr) => {
        $ctx.call($crate::op::$op::NAME, &$tensors, Box::new($args))
    };
}
