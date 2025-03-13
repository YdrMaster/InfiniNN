use crate::{Context, Mapping, NuralNetwork, WeightBias, WeightBiasData};
use digit_layout::DigitLayout;
use vm::{
    Tensor, VirtualMachine,
    op::{LayerNorm, RmsNorm},
};

#[derive(Clone)]
pub struct Normalization {
    pub ty: Type,
    pub dt_w: DigitLayout,
}

#[derive(Clone, Copy, Debug)]
pub enum Type {
    RmsNorm { epsilon: f32 },
    LayerNorm,
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub y: Tensor<'vm, VM>,
    pub x: Tensor<'vm, VM>,
}

pub trait Ops: LayerNorm + RmsNorm {}
impl<VM> Ops for VM where VM: LayerNorm + RmsNorm + ?Sized {}

impl<VM> NuralNetwork<VM> for Normalization
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
        let &Self { ty, dt_w } = self;
        let Args { mut y, x } = args;

        let _dt = Tensor::check_dt_same(&[&y, &x]).unwrap();
        assert_eq!(y.shape(), x.shape());
        let &[_, d] = y.shape() else { panic!() };

        match ty {
            Type::RmsNorm { epsilon } => {
                let w = ctx.get_mapped(WeightBias::Weight, dt_w, &[d]);
                ctx.rms_norm(&mut y, &x, &w, epsilon)
            }
            Type::LayerNorm => {
                let w = ctx.get_mapped(WeightBias::Weight, dt_w, &[d]);
                let b = ctx.get_mapped(WeightBias::Bias, dt_w, &[d]);
                ctx.layer_norm(&mut y, &x, &w, &b)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Normalization, Type};
    use crate::{VirtualMachineExt, WeightBiasData};
    use digit_layout::types as ty;
    use test_vm::TestVM;
    use vm::{VirtualMachine, device_id};

    const DEVICE: device_id = 0;
    const D: usize = 1024;
    const N: usize = 7;

    #[test]
    fn test() {
        let vm = TestVM::default();
        let pid = vm.register("norm");

        {
            let w = vec![0u8; D * 4];
            let b = vec![0u8; D * 4];
            vm.init::<Normalization>(
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
                &Normalization {
                    ty: Type::RmsNorm { epsilon: 1e-5 },
                    dt_w: ty::F32,
                },
                Args {
                    y: vm.workspace(ty::F16, &[N, D]),
                    x: vm.workspace(ty::F16, &[N, D]),
                },
            );
        }

        vm.unregister(pid)
    }
}
