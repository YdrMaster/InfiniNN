use super::{NuralNetwork, WeightBias};
use crate::{
    Context, Tensor, VirtualMachine,
    op::{Add, MatMul, Rearrange},
};
use digit_layout::DigitLayout;

pub struct LinearResidual {
    pub dt_w: DigitLayout,
    pub bias: bool,
    pub scale: f32,
    pub residual: bool,
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub y: Tensor<'vm, VM>,
    pub x: Tensor<'vm, VM>,
    pub y_: Tensor<'vm, VM>,
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
    type Obj = WeightBias;
    type Sub = ();

    fn launch(&self, args: Self::Args<'_>, ctx: Context<VM, Self>) {
        let &Self {
            dt_w,
            bias,
            scale,
            residual,
        } = self;
        let Args { mut y, x, mut y_ } = args;

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

// #[cfg(test)]
// mod test {
//     use super::{Args, Linear, WeightBias};
//     use crate::{Exec, Map, VirtualMachine, test::TestVM};
//     use digit_layout::types as ty;

//     #[test]
//     fn test() {
//         let vm = TestVM::default();
//         let pid = vm.register("linear");
//         let device = 0;

//         let w = vec![0u8; 1024 * 1536 * 2];
//         let b = vec![0u8; 1536 * 2];
//         let norm = vm.map::<Linear>(pid, device);
//         norm.map_host(WeightBias::Weight, Box::new(w));
//         norm.map_host(WeightBias::Bias, Box::new(b));

//         let y = vm.workspace(Some(device), ty::F16, &[7, 1536]);
//         let x = vm.workspace(Some(device), ty::F16, &[7, 1024]);

//         vm.exec(
//             pid,
//             device,
//             &Linear {
//                 dt_w: ty::F16,
//                 bias: true,
//             },
//             Args { y, x },
//         )
//     }
// }
