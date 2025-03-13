use std::borrow::Cow;

use crate::{
    Context, Mapping, NuralNetwork, WeightBiasData,
    linear::{self, Linear},
    linear_residual::{self, LinearResidual},
};
use digit_layout::DigitLayout;
use vm::{
    Id, Tensor, VirtualMachine,
    op::{GeLU, SwiGLU},
    split,
};

#[derive(Clone)]
pub struct Mlp {
    pub act: Activation,
    pub dt_w: DigitLayout,
    pub di: usize,
    pub up_bias: bool,
    pub down_bias: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Activation {
    SwiGLU,
    GeLU,
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub y: Tensor<'vm, VM>,
    pub x: Tensor<'vm, VM>,
    pub scale: f32,
    pub residual: bool,
}

pub struct Data {
    pub up: WeightBiasData,
    pub down: WeightBiasData,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Sub {
    UpLinear,
    DownLinear,
}

impl Id for Sub {
    fn name(&self) -> Cow<str> {
        match self {
            Self::UpLinear => "up".into(),
            Self::DownLinear => "down".into(),
        }
    }
}

pub trait Ops: linear_residual::Ops + SwiGLU + GeLU {}
impl<VM> Ops for VM where VM: linear_residual::Ops + SwiGLU + GeLU + ?Sized {}

impl<VM> NuralNetwork<VM> for Mlp
where
    VM: VirtualMachine + ?Sized + Ops,
{
    type Args<'vm>
        = Args<'vm, VM>
    where
        VM: 'vm;
    type Data = Data;
    type Obj = ();
    type Sub = Sub;

    fn init(weights: Self::Data, mut mapping: Mapping<VM, Self>) {
        let Self::Data { up, down } = weights;
        mapping
            .trap::<Linear>(Sub::UpLinear, up)
            .trap::<LinearResidual>(Sub::DownLinear, down);
    }

    fn forward(&self, args: Self::Args<'_>, mut ctx: Context<VM, Self>) {
        let &Self {
            act: ty,
            dt_w,
            di,
            up_bias,
            down_bias,
        } = self;
        let Args {
            y,
            x,
            scale,
            residual,
        } = args;

        let dt_a = Tensor::check_dt_same(&[&y, &x]).unwrap();
        assert_eq!(y.shape(), x.shape());
        let &[n, _] = y.shape() else { panic!() };

        let d_up = match ty {
            Activation::SwiGLU => di * 2,
            Activation::GeLU => di,
        };
        let mut mid = ctx.workspace(dt_a, &[n, d_up]);

        ctx.trap(
            Sub::UpLinear,
            &Linear {
                dt_w,
                bias: up_bias,
            },
            linear::Args {
                y: mid.clone(),
                x: x.clone(),
            },
        );

        match ty {
            Activation::SwiGLU => {
                split!(mid => gate, up; [di, di] @ 1);
                mid = gate;
                ctx.swiglu(&mut mid, &up)
            }
            Activation::GeLU => ctx.gelu(&mut mid),
        }

        ctx.trap(
            Sub::DownLinear,
            &LinearResidual {
                dt_w,
                bias: down_bias,
            },
            linear_residual::Args {
                y,
                x: mid,
                y_: x,
                scale,
                residual,
            },
        );
    }
}

#[cfg(test)]
mod test {
    use super::{Activation, Args, Data, Mlp};
    use crate::{VirtualMachineExt, WeightBiasData};
    use digit_layout::{DigitLayout, types};
    use test_vm::{TestVM, test_data};
    use vm::{VirtualMachine, device_id};

    const DEVICE: device_id = 0;
    const DT: DigitLayout = types::F16;
    const D: usize = 1024;
    const DI: usize = 1536;
    const N: usize = 7;

    #[test]
    fn test() {
        let vm = TestVM::default();
        let pid = vm.register("linear");

        vm.init::<Mlp>(
            pid,
            DEVICE,
            Data {
                up: WeightBiasData {
                    weight: test_data(DT, &[D, DI * 2]),
                    bias: None,
                },
                down: WeightBiasData {
                    weight: test_data(DT, &[DI, D]),
                    bias: None,
                },
            },
        )
        .forward(
            pid,
            DEVICE,
            &Mlp {
                act: Activation::SwiGLU,
                dt_w: DT,
                di: 1536,
                up_bias: false,
                down_bias: false,
            },
            Args {
                y: vm.workspace(DT, &[N, D]),
                x: vm.workspace(DT, &[N, D]),
                scale: 1.,
                residual: true,
            },
        );

        vm.unregister(pid)
    }
}
