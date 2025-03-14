use super::Exp_;
use crate::op::ptr;
use digit_layout::types;
use half::f16;
use std::ops::{Mul, MulAssign};
use vm::{ObjId, Tensor, op::SwiGLU};

impl SwiGLU for crate::CpuVM {
    fn swiglu(&self, _stack: ObjId, gate: &mut Tensor<Self>, up: &Tensor<Self>) {
        assert_eq!(gate.dt(), up.dt());
        assert_eq!(gate.shape(), up.shape());

        let &[n, d] = gate.shape() else { panic!() };
        let &[sgn, sgd] = gate.strides() else {
            panic!()
        };
        let &[sun, sud] = up.strides() else { panic!() };

        let scheme = Scheme {
            n,
            d,
            sgn,
            sgd,
            sun,
            sud,
            gate: ptr(gate).cast_mut(),
            up: ptr(up),
        };

        match gate.dt() {
            types::F16 => scheme.calculate::<f16>(),
            types::F32 => scheme.calculate::<f32>(),
            types::F64 => scheme.calculate::<f64>(),
            _ => unimplemented!(),
        }
    }
}

struct Scheme {
    n: usize,
    d: usize,
    sgn: isize,
    sgd: isize,
    sun: isize,
    sud: isize,
    gate: *mut u8,
    up: *const u8,
}

trait Date: Exp_ + Mul<Output = Self> + MulAssign {}
impl<T: Exp_ + Mul<Output = T> + MulAssign> Date for T {}

impl Scheme {
    fn calculate<T: Date>(&self) {
        let &Self {
            n,
            d,
            sgn,
            sgd,
            sun,
            sud,
            gate,
            up,
        } = self;

        for i in 0..n * d {
            let i0 = (i / d) as isize;
            let i1 = (i % d) as isize;

            let gate = unsafe { &mut *gate.byte_offset(i0 * sgn + i1 * sgd).cast::<T>() };
            let up = unsafe { *up.byte_offset(i0 * sun + i1 * sud).cast::<T>() };
            *gate *= gate.sigmoid() * up
        }
    }
}
