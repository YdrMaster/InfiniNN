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
    q: Tensor<'vm, VM>, // [nh, n_seq, dh]
    k: Tensor<'vm, VM>, // [nkvh, kv_seq, dh]
    v: Tensor<'vm, VM>, // [nkvh, kv_seq, dh]
    o: Tensor<'vm, VM>, // [nh, n_seq, dh]
    cache: Option<KVCache<'vm, VM>>,
}

pub struct KVCache<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    k_cache: Tensor<'vm, VM>, // [nkvh, k_buf, dh]
    v_cache: Tensor<'vm, VM>, // [nkvh, v_buf, dh]
    pos: usize,
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
        let Args {
            q,
            mut k,
            mut v,
            o,
            cache,
        } = args;

        let dt = Tensor::check_dt_same(&[&q, &k, &v, &o]).unwrap();
        assert_eq!(q.shape(), o.shape());
        assert_eq!(k.shape(), v.shape());

        let &[nh, n_seq, dh] = q.shape() else {
            panic!()
        };
        let &[nkvh, kv_seq, dh_] = k.shape() else {
            panic!()
        };

        let gh = nh / nkvh;
        assert_eq!(dh, dh_);

        let n_att = match cache {
            Some(KVCache {
                k_cache,
                v_cache,
                pos,
            }) => {
                assert_eq!(kv_seq, n_seq);
                k = concat(&ctx, k_cache, k, pos, n_seq);
                v = concat(&ctx, v_cache, v, pos, n_seq);
                pos + n_seq
            }
            None => {
                assert!(kv_seq >= n_seq);
                kv_seq
            }
        };

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

fn concat<'vm, VM>(
    ctx: &Context<VM, Attention>,
    cache: Tensor<'vm, VM>,
    seq: Tensor<'vm, VM>,
    pos: usize,
    n_seq: usize,
) -> Tensor<'vm, VM>
where
    VM: ?Sized + VirtualMachine + Ops,
{
    let cache = cache.transpose(&[1, 0]);
    let mut concat = cache.clone().slice(1, pos, n_seq);
    ctx.rearrange(&mut concat, &seq);
    cache.slice(1, 0, pos + n_seq)
}

#[cfg(test)]
mod test {
    use super::{Args, Attention, KVCache};
    use crate::{Exec, VirtualMachine, op::AttnMask, test::TestVM};
    use digit_layout::types as ty;

    #[test]
    fn test_no_cache() {
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
                Args {
                    q,
                    k,
                    v,
                    o,
                    cache: None,
                },
            )
        }

        vm.unregister(pid)
    }

    #[test]
    fn test_cached() {
        let vm = TestVM::default();
        let pid = vm.register("norm");
        let device = 0;

        {
            let qo = [32, 1, 64];
            let kv = [4, 1, 64];
            let kv_cache = [2048, 4, 64];
            let q = vm.workspace(Some(device), ty::F16, &qo);
            let k = vm.workspace(Some(device), ty::F16, &kv);
            let v = vm.workspace(Some(device), ty::F16, &kv);
            let o = vm.workspace(Some(device), ty::F16, &qo);

            let k_cache = vm.workspace(Some(device), ty::F16, &kv_cache);
            let v_cache = vm.workspace(Some(device), ty::F16, &kv_cache);

            vm.exec(
                pid,
                0,
                &Attention {
                    mask: AttnMask::Causal,
                },
                Args {
                    q,
                    k,
                    v,
                    o,
                    cache: Some(KVCache {
                        k_cache,
                        v_cache,
                        pos: 100,
                    }),
                },
            )
        }

        vm.unregister(pid)
    }
}
