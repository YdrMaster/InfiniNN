use super::{
    NuralNetwork,
    mlp::{self, Mlp},
    normalization::{self as norm, Normalization},
    self_attn::{self, Request, SelfAttn},
};
use crate::{Context, Id, Tensor, VirtualMachine};

pub struct TransformerBlk {
    pub norm: Normalization,
    pub self_attn: SelfAttn,
    pub mlp: Mlp,
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub embd: Tensor<'vm, VM>, // [n, d]
    pub pos: Tensor<'vm, VM>,  // [d]
    pub n_sin: usize,
    pub n_cos: usize,
    pub reqs: Vec<Request<'vm, VM>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Sub {
    PreNorm,
    SelfAttn,
    PostNorm,
    Mlp,
}

impl Id for Sub {
    fn name(&self) -> &str {
        match self {
            Sub::PreNorm => "pre-norm",
            Sub::SelfAttn => "self-attn",
            Sub::PostNorm => "post-norm",
            Sub::Mlp => "mlp",
        }
    }
}

pub trait Ops: norm::Ops + self_attn::Ops + mlp::Ops {}
impl<VM> Ops for VM where VM: norm::Ops + self_attn::Ops + mlp::Ops + ?Sized {}

impl<VM> NuralNetwork<VM> for TransformerBlk
where
    VM: VirtualMachine + ?Sized + Ops,
{
    type Args<'vm>
        = Args<'vm, VM>
    where
        VM: 'vm;
    type Obj = ();
    type Sub = Sub;

    fn launch(&self, args: Self::Args<'_>, mut ctx: Context<VM, Self>) {
        let Self {
            norm,
            self_attn,
            mlp,
        } = self;
        let Args {
            embd: x,
            pos,
            n_sin,
            n_cos,
            reqs,
        } = args;

        let x1 = ctx.workspace(x.dt(), x.shape());

        ctx.trap(
            Sub::PreNorm,
            norm,
            norm::Args {
                y: x1.clone(),
                x: x.clone(),
            },
        );

        ctx.trap(
            Sub::SelfAttn,
            self_attn,
            self_attn::Args {
                y: x.clone(),
                x: x1.clone(),
                pos,
                n_sin,
                n_cos,
                reqs,
            },
        );

        ctx.trap(
            Sub::PostNorm,
            norm,
            norm::Args {
                y: x1.clone(),
                x: x.clone(),
            },
        );

        ctx.trap(
            Sub::Mlp,
            mlp,
            mlp::Args {
                y: x,
                x: x1,
                scale: 1.,
                residual: true,
            },
        )
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test() {}
}
