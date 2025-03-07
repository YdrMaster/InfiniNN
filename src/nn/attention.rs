use crate::{
    LayoutManage, MemManage, Tensor,
    ext::{LayoutManageExt, MemManageExt},
    operators::{AttnMask, MatMul, Rearrange, Softmax},
};
use digit_layout::DigitLayout;

pub struct Attention {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,

    alpha: f32,
    attn_mask: AttnMask,
    qx: Option<Tensor>,
    att: Tensor,
}

#[derive(Clone, Copy, Debug)]
pub struct Meta {
    dt: DigitLayout,
    nh: usize,
    nkvh: usize,
    dh: usize,
    attn_mask: AttnMask,
}

#[derive(Clone, Copy)]
pub enum Arg {
    Q,
    K,
    V,
    O,
}

impl Meta {
    pub fn build(&self, env: &impl LayoutManage, n_seq: usize, n_att: usize) -> Attention {
        let &Self {
            dt,
            nh,
            nkvh,
            dh,
            attn_mask,
        } = self;
        let gh = nh / nkvh;

        let qo = [nh, n_seq, dh];
        let kv = [nkvh, n_att, dh];
        let mut q = env.tensor(Arg::Q, dt, &qo);
        let k = env.tensor(Arg::K, dt, &kv).transpose(&[2, 1]);
        let v = env.tensor(Arg::V, dt, &kv);
        let o = env.tensor(Arg::O, dt, &qo);

        let qx = if gh == 1 {
            None
        } else if let Some(q_) = q.merge(0, 2) {
            q = q_.tile(0, &[nkvh, gh * n_seq]);
            None
        } else {
            Some(Tensor::new(dt, &[nkvh, gh * n_seq, nh]))
        };

        let alpha = (dh as f32).sqrt().recip();
        let att = Tensor::new(dt, &[nkvh, n_seq, n_att]);
        Attention {
            q,
            k,
            v,
            o,
            alpha,
            attn_mask,
            qx,
            att,
        }
    }
}

pub trait Env: MemManage + Rearrange + MatMul + Softmax {}

impl Attention {
    pub fn launch(&self, env: &impl Env) {
        let Self {
            q,
            k,
            v,
            o,

            alpha,
            attn_mask,
            qx,
            att,
        } = self;

        let mut att = env.workspace(att);
        let mut ox = {
            let k = env.tensor(Arg::K, k, false);
            if let Some(qx) = qx {
                let q = env.tensor(Arg::Q, q, false);
                let mut qx = env.workspace(qx);
                env.rearrange(&mut qx, &q);
                env.mat_mul(&mut att, 0.0, &qx, &k, *alpha);
                qx
            } else {
                let q = env.tensor(Arg::Q, q, true);
                env.mat_mul(&mut att, 0.0, &q, &k, *alpha);
                q
            }
        };

        env.softmax(&mut att, *attn_mask);

        {
            let v = env.tensor(Arg::V, v, false);
            env.mat_mul(&mut ox, 0., &att, &v, 1.)
        }

        let mut o = env.tensor(Arg::O, o, true);
        if o.ptr.address() != ox.ptr.address() {
            env.rearrange(&mut o, &ox);
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Arg, Env, Meta};
    use crate::{
        LayoutManage, Ptr,
        ext::MemManageExt,
        operators::AttnMask,
        test_recorder::{TestLayoutManager, TestMemManager},
    };
    use digit_layout::types as ty;
    use ndarray_layout::{ArrayLayout, Endian::BigEndian};

    impl Env for TestMemManager {}

    #[test]
    fn test() {
        let meta = Meta {
            dt: ty::F16,
            nh: 8,
            nkvh: 2,
            dh: 64,
            attn_mask: AttnMask::Causal,
        };
        let qo = ArrayLayout::new_contiguous(&[8, 7, 64], BigEndian, 2);
        let kv = ArrayLayout::new_contiguous(&[2, 777, 64], BigEndian, 2);

        let mut lm = TestLayoutManager::default();
        lm.set(Arg::Q, qo.clone());
        lm.set(Arg::K, kv.clone());
        lm.set(Arg::V, kv);
        lm.set(Arg::O, qo);

        let att = meta.build(&mut lm, 7, 777);

        let mm = TestMemManager::default();
        let _trap = mm.trap_with(
            (),
            &[
                (Arg::Q, Ptr::Mut(0 as _)),
                (Arg::K, Ptr::Const(1 as _)),
                (Arg::V, Ptr::Const(2 as _)),
                (Arg::O, Ptr::Mut(3 as _)),
            ],
        );
        att.launch(&mm);

        println!("{mm}")
    }
}
