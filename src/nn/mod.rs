use crate::{Context, Id, VirtualMachine};
use macros::def;

pub mod attention;
pub mod mlp;
pub mod normalization;

pub trait NuralNetwork<VM>: Sized
where
    VM: VirtualMachine + ?Sized,
{
    type Args<'ctx, 'vm: 'ctx>
    where
        VM: 'vm;
    type Obj: Id;
    type Sub: Id;

    fn launch(&self, args: Self::Args<'_, '_>, ctx: Context<VM, Self>);
}

mod macros {
    /// 用于定义参数结构体的宏。由于参数上有比较复杂的生命周期约束，可以使用宏来简化定义。
    macro_rules! def {
        ($name:ident: $(<mut: $($t_mut:ident),*>)? $(<ref: $($t_ref:ident),*>)? $( $arg:ident : $ty:ty ),* ) => {
            pub struct $name<'ctx, 'vm, VM>
            where
                VM: VirtualMachine + ?Sized,
            {
                $(
                    $(
                        pub $t_mut: &'ctx mut Tensor<'vm, VM>,
                    )*
                )?
                $(
                    $(
                        pub $t_ref: &'ctx     Tensor<'vm, VM>,
                    )*
                )?
                $(
                    pub $arg: $ty,
                )*
            }
        };
    }

    pub(super) use def;
}
