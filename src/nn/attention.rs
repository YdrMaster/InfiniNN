use super::NuralNetwork;
use crate::{
    Blob, Context, Tensor, VirtualMachine,
    op::{AttnMask, MatMul, Rearrange, Softmax},
};

pub struct Attention {
    mask: AttnMask,
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    q: Tensor<'vm, VM>,
    k: Tensor<'vm, VM>,
    v: Tensor<'vm, VM>,
    o: Tensor<'vm, VM>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Obj {
    Qbuf,
    Attn,
}

pub trait Ops: Rearrange + MatMul + Softmax {}
impl<VM> Ops for VM where VM: Rearrange + MatMul + Softmax {}

impl<VM> NuralNetwork<VM> for Attention
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
        let &Self { mask } = self;
        let Args { q, k, v, o } = args;

        let dt = Tensor::check_dt_same(&[&q, &k, &v, &o]).unwrap();
        assert_eq!(q.shape(), o.shape());
        assert_eq!(k.shape(), v.shape());

        let &[nh, n_seq, dh] = q.shape() else {
            panic!()
        };
        let &[nkvh, n_att, dh_] = k.shape() else {
            panic!()
        };
        let gh = nh / nkvh;
        assert_eq!(dh, dh_);

        let mut qx = if gh == 1 {
            q
        } else if let Some(q_) = q.clone().merge(0, 2) {
            q_.tile(0, &[nkvh, gh * n_seq])
        } else {
            let mut qx = ctx.workspace(dt, &[nkvh, gh * n_seq, dh]);
            ctx.rearrange(&mut qx, &q);
            qx
        };
        {
            let k = k.transpose(&[2, 1]);
            let mut att = ctx.workspace(dt, &[nkvh, gh * n_seq, n_att]);
            ctx.mat_mul(&mut att, 0., &qx, &k, (dh as f32).sqrt().recip());
            ctx.softmax(&mut att, mask);
            ctx.mat_mul(&mut qx, 0., &att, &v, 1.)
        }
        if !VM::Blob::eq(qx.blob(), o.blob()) {
            let mut o = o.tile(0, &[nkvh, gh]);
            qx = qx.tile(1, &[gh, n_seq]);
            ctx.rearrange(&mut o, &qx)
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Attention};
    use crate::{Exec, VirtualMachine, op::AttnMask, test::TestVM};
    use digit_layout::types as ty;

    #[test]
    fn test() {
        let vm = TestVM::default();
        let pid = vm.register("norm");
        let device = 0;

        {
            let qo = [32, 7, 64];
            let kv = [4, 777, 64];
            let q = vm.workspace(Some(device), ty::F16, &qo);
            let k = vm.workspace(Some(device), ty::F16, &kv);
            let v = vm.workspace(Some(device), ty::F16, &kv);
            let o = vm.workspace(Some(device), ty::F16, &qo);

            vm.exec(
                pid,
                0,
                &Attention {
                    mask: AttnMask::Causal,
                },
                Args { q, k, v, o },
            )
        }

        vm.unregister(pid)
    }
}
