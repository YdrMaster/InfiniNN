use crate::{
    LayoutManage, MemManage, StorageTensor, Tensor,
    ext::{LayoutManageExt, MemManageExt},
    operators::{AttnMask, MatMul, Rearrange, Softmax},
};
use digit_layout::DigitLayout;

pub struct Attention {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,

    qx: Tensor,
    att: Tensor,

    alpha: f32,
    rearrange: bool,
    attn_mask: AttnMask,
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
    pub fn build(&self, env: &impl LayoutManage) -> Attention {
        let &Self {
            dt,
            nh,
            nkvh,
            dh,
            attn_mask,
        } = self;
        let gh = nh / nkvh;
        let n_seq = env.get_dim(Arg::Q, 1);
        let n_att = env.get_dim(Arg::K, 1);

        let qo = [nh, n_seq, dh];
        let kv = [nkvh, n_att, dh];
        let q = env.tensor(Arg::Q, dt, &qo);
        let k = env.tensor(Arg::K, dt, &kv).transpose(&[2, 1]);
        let v = env.tensor(Arg::V, dt, &kv);
        let o = env.tensor(Arg::O, dt, &qo);

        let (qx, rearrange) = if gh == 1 {
            (q.clone(), false)
        } else if let Some(q_) = q.merge(0, 2) {
            (q_.tile(0, &[nkvh, gh * n_seq]), false)
        } else {
            (Tensor::new(dt, &[nkvh, gh * n_seq, nh]), true)
        };

        let alpha = (dh as f32).sqrt().recip();
        let att = Tensor::new(dt, &[nkvh, gh * n_seq, n_att]);
        Attention {
            q,
            k,
            v,
            o,

            qx,
            att,

            alpha,
            rearrange,
            attn_mask,
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

            qx,
            att,

            alpha,
            rearrange,
            attn_mask,
        } = self;

        let mut att = env.workspace(att);
        let mut ox = {
            let k = env.tensor(Arg::K, k, false);
            let qx = if *rearrange {
                let q = env.tensor(Arg::Q, q, false);
                let mut qx = env.workspace(qx);
                env.rearrange(&mut qx, &q);
                qx
            } else {
                env.tensor(Arg::Q, qx, true)
            };
            env.mat_mul(&mut att, 0.0, &qx, &k, *alpha);
            qx
        };

        env.softmax(&mut att, *attn_mask);

        {
            let att = att;
            let v = env.tensor(Arg::V, v, false);
            env.mat_mul(&mut ox, 0., &att, &v, 1.)
        }

        let mut o = env.tensor(Arg::O, o, true);
        if o.ptr.address() != ox.ptr.address() {
            env.rearrange(
                &mut o,
                &StorageTensor {
                    tensor: q,
                    ptr: ox.ptr,
                },
            )
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Arg, Env, Meta};
    use crate::{
        Tensor,
        operators::AttnMask,
        test_recorder::{TestLayoutManager, TestMemManager, TestMemManagerLoader},
    };
    use digit_layout::types as ty;

    impl Env for TestMemManager {}

    #[test]
    fn test() {
        let dt = ty::F16;
        let nh = 8;
        let nkvh = 2;
        let dh = 64;
        let n_seq = 7;
        let n_att = 777;

        let meta = Meta {
            dt,
            nh,
            nkvh,
            dh,
            attn_mask: AttnMask::Causal,
        };
        let qo = Tensor::new(dt, &[nh, n_seq, dh]).layout;
        let kv = Tensor::new(dt, &[nkvh, n_att, dh]).layout;

        let lm = TestLayoutManager::from([
            (Arg::Q, qo.clone()),
            (Arg::K, kv.clone()),
            (Arg::V, kv),
            (Arg::O, qo),
        ]);
        let att = meta.build(&lm);

        let mm = TestMemManagerLoader::new([Arg::Q, Arg::O], [Arg::K, Arg::V]).build();
        att.launch(&mm);

        println!("{mm}")
    }
}
