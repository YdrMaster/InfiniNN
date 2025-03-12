use super::{NuralNetwork, WeightBias};
use crate::{
    Context, Tensor, VirtualMachine,
    op::{LayerNorm, RmsNorm},
};
use digit_layout::DigitLayout;

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
    type Obj = WeightBias;
    type Sub = ();

    fn launch(&self, args: Self::Args<'_>, ctx: Context<VM, Self>) {
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
    use super::{Args, Normalization, Type, WeightBias};
    use crate::{Exec, Map, VirtualMachine, test::TestVM};
    use digit_layout::types as ty;

    #[test]
    fn test() {
        let vm = TestVM::default();
        let pid = vm.register("norm");
        let device = 0;

        let w = vec![0u8; 1024 * 4];
        let norm = vm.map::<Normalization>(pid, device);
        norm.map_host(WeightBias::Weight, Box::new(w));

        let y = vm.workspace(Some(device), ty::F16, &[7, 1024]);
        let x = vm.workspace(Some(device), ty::F16, &[7, 1024]);

        vm.exec(
            pid,
            device,
            &Normalization {
                ty: Type::RmsNorm { epsilon: 1e-5 },
                dt_w: ty::F32,
            },
            Args { y, x },
        )
    }
}
