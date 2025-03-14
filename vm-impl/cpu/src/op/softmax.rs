use super::Exp_;
use crate::op::ptr;
use digit_layout::types;
use half::f16;
use std::{
    cmp::Ordering,
    ops::{AddAssign, DivAssign, Sub},
};
use vm::{
    ObjId, Tensor,
    op::{AttnMask, Softmax},
};

impl Softmax for crate::CpuVM {
    fn softmax(&self, _stack: ObjId, att: &mut Tensor<Self>, mask: AttnMask) {
        let &[nh, n_seq, n_att] = att.shape() else {
            panic!()
        };
        let &[sh, ss, sa] = att.strides() else {
            panic!()
        };
        assert_eq!(sa, att.dt().nbytes() as isize);

        let scheme = Scheme {
            nh,
            n_seq,
            n_att,
            sh,
            ss,
            att: ptr(att).cast_mut(),
            mask,
        };

        match att.dt() {
            types::F16 => scheme.compute::<f16>(),
            _ => todo!(),
        }
    }
}

struct Scheme {
    nh: usize,
    n_seq: usize,
    n_att: usize,
    sh: isize,
    ss: isize,
    att: *mut u8,
    mask: AttnMask,
}

trait Data: Copy + AddAssign + DivAssign + Sub<Output = Self> + Exp_ {
    const ZERO_: Self;
    fn compare(a: &Self, b: &Self) -> Ordering;
}

impl Data for f16 {
    const ZERO_: Self = f16::ZERO;
    fn compare(a: &Self, b: &Self) -> Ordering {
        a.total_cmp(b)
    }
}

impl Scheme {
    fn compute<T: Data>(&self) {
        let &Self {
            nh,
            n_seq,
            n_att,
            sh,
            ss,
            att,
            mask,
        } = self;

        for i in 0..nh * n_seq {
            let ih = (i / n_seq) as isize;
            let is = (i % n_seq) as isize;

            let att = unsafe { att.byte_offset(ih * sh + is * ss).cast::<T>() };
            let att = unsafe { std::slice::from_raw_parts_mut(att, n_att) };

            let split = match mask {
                AttnMask::None => n_att,
                AttnMask::Causal => n_att - n_seq + is as usize + 1,
            };

            let (att, tail) = att.split_at_mut(split);
            tail.fill(T::ZERO_);

            let max = att.iter().copied().max_by(T::compare).unwrap();
            let mut sum = T::ZERO_;
            for x in &mut *att {
                *x = (*x - max).exp_();
                sum += *x
            }
            for x in att {
                *x /= sum
            }
        }
    }
}
