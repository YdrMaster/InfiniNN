use crate::{Context, Mapping, NuralNetwork, WeightBias, WeightBiasData};
use digit_layout::DigitLayout;
use vm::{
    Tensor, VirtualMachine,
    op::{MatMul, Rearrange},
};

pub struct Linear {
    pub dt_w: DigitLayout,
    pub bias: bool,
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub y: Tensor<'vm, VM>,
    pub x: Tensor<'vm, VM>,
}

pub trait Ops: Rearrange + MatMul {}
impl<VM> Ops for VM where VM: Rearrange + MatMul + ?Sized {}

impl<VM> NuralNetwork<VM> for Linear
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

    fn init(data: Self::Data, mapping: Mapping<VM, Self>) {
        data.map(mapping)
    }

    fn forward(&self, args: Self::Args<'_>, ctx: Context<VM, Self>) {
        let &Self { dt_w, bias } = self;
        let Args { mut y, x } = args;

        let _dt = Tensor::check_dt_same(&[&y, &x]).unwrap();
        let &[n_, d_] = y.shape() else { panic!() };
        let &[n, d] = x.shape() else { panic!() };
        assert_eq!(n, n_);

        let beta = if bias {
            let b = ctx
                .get_mapped(WeightBias::Bias, dt_w, &[1, d_])
                .broadcast(0, n);
            ctx.rearrange(&mut y, &b);
            1.
        } else {
            0.
        };
        let w = ctx
            .get_mapped(WeightBias::Weight, dt_w, &[d_, d])
            .transpose(&[1, 0]);
        ctx.mat_mul(&mut y, beta, &x, &w, 1.)
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Linear};
    use crate::{VirtualMachineExt, WeightBiasData};
    use digit_layout::types as ty;
    use test_vm::TestVM;
    use vm::{VirtualMachine, device_id};

    const DEVICE: device_id = 0;
    const D: usize = 1024;
    const DI: usize = 1536;
    const N: usize = 7;

    #[test]
    fn test() {
        let vm = TestVM::default();
        let pid = vm.register("linear");

        {
            let w = vec![0u8; D * DI * 2];
            let b = vec![0u8; DI * 2];
            vm.init::<Linear>(
                pid,
                DEVICE,
                WeightBiasData {
                    weight: Box::new(w),
                    bias: Some(Box::new(b)),
                },
            )
            .forward(
                pid,
                DEVICE,
                &Linear {
                    dt_w: ty::F16,
                    bias: true,
                },
                Args {
                    y: vm.workspace(ty::F16, &[N, DI]),
                    x: vm.workspace(ty::F16, &[N, D]),
                },
            );
        }

        vm.unregister(pid)
    }
}
