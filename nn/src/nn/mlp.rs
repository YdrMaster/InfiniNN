use super::{NuralNetwork, WeightBias, linear::Linear, linear_residual::LinearResidual};
use crate::{
    Context, Tensor, VirtualMachine, call, child, forward, init, linear, linear_residual, shape,
    split,
};

pub struct Mlp {
    act: Activation,
    up: Linear,
    down: LinearResidual,
}

child!(Mlp[0] = up  : Linear);
child!(Mlp[1] = down: LinearResidual);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Activation {
    SwiGLU,
    GeLU,
}

pub struct Args<VM: VirtualMachine> {
    pub y: VM::Tensor,
    pub x: VM::Tensor,
    pub scale: f32,
    pub residual: bool,
}

pub struct Data<VM: VirtualMachine> {
    pub up: WeightBias<VM>,
    pub down: WeightBias<VM>,
}

impl<VM: VirtualMachine> NuralNetwork<VM> for Mlp {
    const NAME: &str = "mlp";
    type Meta = Activation;
    type Args = Args<VM>;
    type Data = Data<VM>;

    fn init(meta: &Self::Meta, data: Self::Data, mut ctx: Context<VM, Self>) -> Self {
        let Self::Data { up, down } = data;
        Self {
            act: *meta,
            up: init!(0: None, (), up; ctx),
            down: init!(1: None, (), down; ctx),
        }
    }

    fn forward(&self, args: Self::Args, mut ctx: Context<VM, Self>) {
        let Self { act, up, down } = self;
        let Args {
            y,
            x,
            scale,
            residual,
        } = args;

        let mid = ctx.tensor(None);
        forward!(0: None, up, linear::Args{y: mid.clone(), x}; ctx);

        let mid = match act {
            Activation::SwiGLU => {
                shape!(mid => [_, di]);
                split!(mid => gate, up; [di / 2, di / 2] @ 1);
                call!(gelu: [&gate, &gate, &up]; ctx);
                gate
            }
            Activation::GeLU => {
                call!(gelu: [&mid, &mid]; ctx);
                mid
            }
        };

        forward!(1: None, down, linear_residual::Args{y, x: mid, scale, residual}; ctx)
    }
}
