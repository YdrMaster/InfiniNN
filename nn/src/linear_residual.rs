use crate::{Context, Mapping, NuralNetwork, WeightBias, WeightBiasData};
use digit_layout::DigitLayout;
use vm::{
    Tensor, VirtualMachine,
    op::{Add, MatMul, Rearrange},
};

pub struct LinearResidual {
    pub dt_w: DigitLayout,
    pub bias: bool,
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub y: Tensor<'vm, VM>,
    pub x: Tensor<'vm, VM>,
    pub y_: Tensor<'vm, VM>,
    pub scale: f32,
    pub residual: bool,
}

pub trait Ops: Rearrange + MatMul + Add {}
impl<VM> Ops for VM where VM: Rearrange + MatMul + Add + ?Sized {}

impl<VM> NuralNetwork<VM> for LinearResidual
where
    VM: VirtualMachine + ?Sized + Ops,
{
    type Args<'vm>
        = Args<'vm, VM>
    where
        VM: 'vm;
    type Data = WeightBiasData;
    type Obj = WeightBias;
    type Sub = ();

    fn init(weights: Self::Data, mapping: Mapping<VM, Self>) {
        weights.map(mapping)
    }

    fn forward(&self, args: Self::Args<'_>, ctx: Context<VM, Self>) {
        let &Self { dt_w, bias } = self;
        let Args {
            mut y,
            x,
            mut y_,
            scale,
            residual,
        } = args;

        let _dt = Tensor::check_dt_same(&[&y, &x, &y_]).unwrap();
        let &[n, d] = y.shape() else { panic!() };
        let &[n_, d_] = x.shape() else { panic!() };
        assert_eq!(y.shape(), y_.shape());
        assert_eq!(n, n_);

        let w = ctx
            .get_mapped(WeightBias::Weight, dt_w, &[d, d_])
            .transpose(&[1, 0]);
        if bias {
            {
                let x1 = if residual { &mut y_ } else { &mut y };
                {
                    let bias = ctx
                        .get_mapped(WeightBias::Bias, dt_w, &[1, d])
                        .broadcast(0, n);
                    ctx.rearrange(x1, &bias)
                }
                ctx.mat_mul(x1, scale, &x, &w, scale)
            }
            if residual {
                ctx.add(&mut y, &y_)
            }
        } else {
            let beta = if residual { 1. } else { 0. };
            ctx.mat_mul(&mut y, beta, &x, &w, scale)
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Args, LinearResidual};
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
        let pid = vm.register("linear-residual");

        vm.init::<LinearResidual>(
            pid,
            DEVICE,
            WeightBiasData {
                weight: test_data(DT, &[D, DI]),
                bias: Some(test_data(DT, &[DI])),
            },
        )
        .forward(
            pid,
            DEVICE,
            &LinearResidual {
                dt_w: DT,
                bias: true,
            },
            Args {
                y: vm.workspace(DT, &[N, DI]),
                x: vm.workspace(DT, &[N, D]),
                y_: vm.workspace(DT, &[N, DI]),
                scale: 1.,
                residual: true,
            },
        );

        vm.unregister(pid)
    }
}
