mod context;
mod op;

use std::{borrow::Cow, ops::Deref};
use vm::{Id, VirtualMachine};

pub mod attention;
pub mod linear;
pub mod linear_residual;
pub mod lm_output;
pub mod mlp;
pub mod normalization;
pub mod self_attn;
pub mod token_embed;
pub mod transformer_blk;

pub use context::{Context, Mapping, VirtualMachineExt};

pub trait NuralNetwork<VM>: Sized
where
    VM: VirtualMachine + ?Sized,
{
    type Args<'vm>
    where
        VM: 'vm;
    type Data;
    type Obj: Id;
    type Sub: Id;

    fn init(data: Self::Data, mapping: Mapping<VM, Self>);
    fn forward(&self, args: Self::Args<'_>, ctx: Context<VM, Self>);
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WeightBias {
    Weight,
    Bias,
}

impl Id for WeightBias {
    fn name(&self) -> Cow<str> {
        match self {
            Self::Weight => "weight".into(),
            Self::Bias => "bias".into(),
        }
    }
}

pub struct WeightBiasData {
    pub weight: Box<dyn Deref<Target = [u8]>>,
    pub bias: Option<Box<dyn Deref<Target = [u8]>>>,
}

impl WeightBiasData {
    fn map<NN, VM>(self, mut mapping: Mapping<VM, NN>)
    where
        VM: VirtualMachine + ?Sized,
        NN: NuralNetwork<VM, Data = Self, Obj = WeightBias>,
    {
        let NN::Data { weight, bias } = self;
        mapping.map_host(WeightBias::Weight, weight);
        if let Some(bias) = bias {
            mapping.map_host(WeightBias::Bias, bias);
        }
    }
}
