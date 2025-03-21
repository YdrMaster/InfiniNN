mod ctx;
mod nn;

use nnvm::VirtualMachine;

pub use ctx::{Context, Domain};

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
macro_rules! child {
    ($ty:ty[$child_id:literal] = $name:ident: $child:ty) => {
        impl<VM> $crate::WithChild<VM, $child_id> for $ty
        where
            VM: $crate::VirtualMachine,
        {
            type Type = $child;
            const NAME: &str = stringify!($name);
        }
    };
}

#[macro_export]
macro_rules! init {
    ($child_id:literal: $loop_idx:expr, $meta:expr, $data:expr; $ctx:expr) => {
        <Self as $crate::WithChild<VM, $child_id>>::init_child($loop_idx, &$meta, $data, &mut $ctx)
    };
}

#[macro_export]
macro_rules! forward {
    ($child_id:literal: $loop_idx:expr, $child:expr, $args:expr; $ctx:expr) => {
        <Self as $crate::WithChild<VM, $child_id>>::forward_child(
            $loop_idx, &$child, $args, &mut $ctx,
        )
    };
}

#[macro_export]
macro_rules! fetch_data {
    ($child_id:literal: $data:expr; $ctx:expr) => {{
        let tensor = $ctx.tensor(None);
        forward!($child_id: None, $data, tensor.clone(); $ctx);
        tensor
    }};
}

#[macro_export]
macro_rules! call {
    ($op:ident: $tensors:expr, $args:expr; $ctx:expr) => {
        $ctx.call($crate::op::$op::NAME, &$tensors, Box::new($args))
    };

    ($op:ident: $tensors:expr; $ctx:expr) => {
        call!($op: $tensors, $crate::op::Empty; $ctx)
    };
}
