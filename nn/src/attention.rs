use crate::{Context, Mapping, NuralNetwork};
use vm::{
    Blob as _, Tensor, VirtualMachine,
    op::{AttnMask, MatMul, Rearrange, Softmax},
};

pub struct Attention {
    pub mask: AttnMask,
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub q: Tensor<'vm, VM>, // [nh, n_seq, dh]
    pub k: Tensor<'vm, VM>, // [nkvh, kv_seq, dh]
    pub v: Tensor<'vm, VM>, // [nkvh, kv_seq, dh]
    pub o: Tensor<'vm, VM>, // [nh, n_seq, dh]
    pub cache: Option<KVCache<'vm, VM>>,
}

pub struct KVCache<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub k_cache: Tensor<'vm, VM>, // [k_buf, nkvh, dh]
    pub v_cache: Tensor<'vm, VM>, // [v_buf, nkvh, dh]
    pub pos: usize,
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
    type Data = ();
    type Obj = ();
    type Sub = ();

    fn init(_data: Self::Data, _mapping: Mapping<VM, Self>) {}

    fn forward(&self, args: Self::Args<'_>, ctx: Context<VM, Self>) {
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
            let mut qx = ctx.workspace(dt, &[nh, n_seq, dh]);
            ctx.rearrange(&mut qx, &q);
            qx.merge(0, 2).unwrap().tile(0, &[nkvh, gh * n_seq])
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
    use crate::VirtualMachineExt;
    use digit_layout::types as ty;
    use test_vm::TestVM;
    use vm::{VirtualMachine, device_id, op::AttnMask};

    const DEVICE: device_id = 0;

    #[test]
    fn test_no_cache() {
        let vm = TestVM::default();
        let pid = vm.register("attention");

        let qo = [32, 7, 64];
        let kv = [4, 777, 64];
        vm.forward(
            pid,
            DEVICE,
            &Attention {
                mask: AttnMask::Causal,
            },
            Args {
                q: vm.workspace(ty::F16, &qo),
                k: vm.workspace(ty::F16, &kv),
                v: vm.workspace(ty::F16, &kv),
                o: vm.workspace(ty::F16, &qo),
                cache: None,
            },
        );

        vm.unregister(pid)
    }

    #[test]
    fn test_cached() {
        let vm = TestVM::default();
        let pid = vm.register("attention");

        let qo = [32, 1, 64];
        let kv = [4, 1, 64];
        let kv_cache = [2048, 4, 64];
        vm.forward(
            pid,
            DEVICE,
            &Attention {
                mask: AttnMask::Causal,
            },
            Args {
                q: vm.workspace(ty::F16, &qo),
                k: vm.workspace(ty::F16, &kv),
                v: vm.workspace(ty::F16, &kv),
                o: vm.workspace(ty::F16, &qo),
                cache: Some(KVCache {
                    k_cache: vm.workspace(ty::F16, &kv_cache),
                    v_cache: vm.workspace(ty::F16, &kv_cache),
                    pos: 100,
                }),
            },
        );

        vm.unregister(pid)
    }
}
