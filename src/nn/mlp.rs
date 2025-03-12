use super::{NuralNetwork, linear_residual};
use crate::{
    Context, Id, Tensor, VirtualMachine,
    nn::{
        linear::{self, Linear},
        linear_residual::LinearResidual,
    },
    op::{GeLU, SwiGLU},
    split,
};
use digit_layout::DigitLayout;

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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Sub {
    UpLinear,
    DownLinear,
}

impl Id for Sub {
    fn name(&self) -> &str {
        match self {
            Self::UpLinear => "up",
            Self::DownLinear => "down",
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
    type Obj = ();
    type Sub = Sub;

    fn launch(&self, args: Self::Args<'_>, mut ctx: Context<VM, Self>) {
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
                scale,
                residual,
            },
            linear_residual::Args { y, x: mid, y_: x },
        )
    }
}

#[cfg(test)]
mod test {
    use super::{Activation, Args, Mlp, Sub};
    use crate::{
        Exec, Map, VirtualMachine,
        nn::{WeightBias, linear::Linear},
        test::TestVM,
    };
    use digit_layout::types as ty;

    #[test]
    fn test() {
        let vm = TestVM::default();
        let pid = vm.register("mlp");
        let device = 0;

        {
            let wup = vec![0u8; 1024 * 1536 * 2 * 2];
            let wdown = vec![0u8; 1024 * 1536 * 2];
            let mlp = vm.map::<Mlp>(pid, device);
            mlp.step_into::<Linear>(Sub::UpLinear)
                .map_host(WeightBias::Weight, Box::new(wup));
            mlp.step_into::<Linear>(Sub::DownLinear)
                .map_host(WeightBias::Weight, Box::new(wdown));

            let y = vm.workspace(Some(device), ty::F16, &[7, 1024]);
            let x = vm.workspace(Some(device), ty::F16, &[7, 1024]);

            vm.exec(
                pid,
                device,
                &Mlp {
                    act: Activation::SwiGLU,
                    dt_w: ty::F16,
                    di: 1536,
                    up_bias: false,
                    down_bias: false,
                },
                Args {
                    y,
                    x,
                    scale: 1.,
                    residual: true,
                },
            )
        }

        vm.unregister(pid)
    }
}
