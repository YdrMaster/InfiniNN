use crate::op::ptr;
use digit_layout::types;
use half::f16;
use itertools::izip;
use std::{
    iter::Sum,
    ops::Mul,
    slice::{from_raw_parts, from_raw_parts_mut},
};
use vm::{ObjId, Tensor, op::RmsNorm};

impl RmsNorm for crate::CpuVM {
    fn rms_norm(
        &self,
        _stack: ObjId,
        y: &mut Tensor<Self>,
        x: &Tensor<Self>,
        w: &Tensor<Self>,
        epsilon: f32,
    ) {
        let &[n, d] = y.shape() else { panic!() };
        let &[d_] = w.shape() else { panic!() };
        assert_eq!(y.shape(), x.shape());
        assert_eq!(d, d_);

        let dt = Tensor::check_dt_same(&[y, x]).unwrap();
        let &[sny, sdy] = y.strides() else { panic!() };
        let &[snx, sdx] = x.strides() else { panic!() };
        let &[sdw] = w.strides() else { panic!() };
        assert_eq!(sdy, dt.nbytes() as isize);
        assert_eq!(sdx, dt.nbytes() as isize);
        assert_eq!(sdw, w.dt().nbytes() as isize);

        let scheme = Scheme {
            n,
            d,
            sny,
            snx,
            y: ptr(y).cast_mut(),
            x: ptr(x),
            w: ptr(w),
            epsilon,
        };

        match (dt, w.dt()) {
            (types::F16, types::F16) => scheme.compute::<f16, f16>(),
            (types::F16, types::F32) => scheme.compute::<f16, f32>(),
            (types::F16, types::F64) => scheme.compute::<f16, f64>(),
            (types::F32, types::F32) => scheme.compute::<f32, f32>(),
            (types::F32, types::F64) => scheme.compute::<f32, f64>(),
            (types::F64, types::F64) => scheme.compute::<f64, f64>(),
            (_, _) => unimplemented!(),
        }
    }
}

struct Scheme {
    n: usize,
    d: usize,
    sny: isize,
    snx: isize,
    y: *mut u8,
    x: *const u8,
    w: *const u8,
    epsilon: f32,
}

trait Data<T>: Copy {
    fn compute(self) -> T;
    fn store(&mut self, val: T);
}

macro_rules! data {
    ($t:ty => $u:ty; $compute:expr, $store:expr) => {
        impl Data<$u> for $t {
            fn compute(self) -> $u {
                $compute(self)
            }
            fn store(&mut self, val: $u) {
                *self = $store(val)
            }
        }
    };
}

data!(f16 => f16; |x| x, |x| x);
data!(f32 => f32; |x| x, |x| x);
data!(f64 => f64; |x| x, |x| x);
data!(f16 => f32; f16::to_f32, f16::from_f32);
data!(f16 => f64; f16::to_f64, f16::from_f64);
data!(f32 => f64; |x| x as _, |x| x as _);

trait Compute: Mul<Output = Self> + Sum<Self> + Copy {
    fn mean(self, d: usize, epsilon: f32) -> Self;
}

impl Compute for f16 {
    fn mean(self, d: usize, epsilon: f32) -> Self {
        f16::from_f32((self.to_f32() / d as f32 + epsilon).powf(-0.5))
    }
}
impl Compute for f32 {
    fn mean(self, d: usize, epsilon: f32) -> Self {
        (self / d as f32 + epsilon).powf(-0.5)
    }
}
impl Compute for f64 {
    fn mean(self, d: usize, epsilon: f32) -> Self {
        (self / d as f64 + epsilon as f64).powf(-0.5)
    }
}

impl Scheme {
    fn compute<T, U>(&self)
    where
        T: Data<U>,
        U: Compute,
    {
        let &Self {
            n,
            d,
            sny,
            snx,
            y,
            x,
            w,
            epsilon,
        } = self;

        let w = unsafe { from_raw_parts(w.cast::<U>(), d) };
        for i in 0..n as isize {
            let y = unsafe { from_raw_parts_mut(y.byte_offset(i * sny).cast::<T>(), d) };
            let x = unsafe { from_raw_parts(x.byte_offset(i * snx).cast::<T>(), d) };
            let sum = x
                .iter()
                .map(|x| {
                    let x = x.compute();
                    x * x
                })
                .sum::<U>();
            let mean = sum.mean(d, epsilon);
            for (y, x, w) in izip!(y, x, w) {
                y.store(mean * x.compute() * *w)
            }
        }
    }
}
