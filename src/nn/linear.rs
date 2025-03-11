use super::NuralNetwork;
use crate::{
    Context, Tensor, VirtualMachine,
    op::{MatMul, Rearrange},
};
use digit_layout::DigitLayout;

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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Obj {
    Weight,
    Bias,
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
    type Obj = Obj;
    type Sub = ();

    fn launch(&self, args: Self::Args<'_>, ctx: Context<VM, Self>) {
        let &Self { dt_w, bias } = self;
        let Args { mut y, x } = args;

        let _dt = Tensor::check_dt_same(&[&y, &x]).unwrap();
        let &[n_, d_] = y.shape() else { panic!() };
        let &[n, d] = x.shape() else { panic!() };
        assert_eq!(n, n_);

        let beta = if bias {
            let b = ctx.get_mapped(Obj::Bias, dt_w, &[1, d_]).broadcast(0, n);
            ctx.rearrange(&mut y, &b);
            1.
        } else {
            0.
        };
        let w = ctx
            .get_mapped(Obj::Weight, dt_w, &[d_, d])
            .transpose(&[1, 0]);
        ctx.mat_mul(&mut y, beta, &x, &w, 1.)
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Linear, Obj};
    use crate::{Exec, Map, VirtualMachine, test::TestVM};
    use digit_layout::types as ty;

    #[test]
    fn test() {
        let vm = TestVM::default();
        let pid = vm.register("linear");
        let device = 0;

        let w = vec![0u8; 1024 * 1536 * 2];
        let b = vec![0u8; 1536 * 2];
        let norm = vm.map::<Linear>(pid, device);
        norm.map_host(Obj::Weight, Box::new(w));
        norm.map_host(Obj::Bias, Box::new(b));

        let y = vm.workspace(Some(device), ty::F16, &[7, 1536]);
        let x = vm.workspace(Some(device), ty::F16, &[7, 1024]);

        vm.exec(
            pid,
            device,
            &Linear {
                dt_w: ty::F16,
                bias: true,
            },
            Args { y, x },
        )
    }
}
